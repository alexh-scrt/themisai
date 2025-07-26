"""
Embedding Service - Business Logic Layer

This module provides the business logic layer for embedding generation and management
in the Patexia Legal AI Chatbot. It orchestrates embedding operations, manages model
switching, enforces business rules, and provides service layer abstraction.

Key Features:
- Multi-model embedding generation with automatic fallback
- Business rule enforcement for embedding operations
- Model health monitoring and automatic switching
- Batch processing optimization for legal documents
- Performance monitoring and resource management
- Cache management for frequently embedded texts
- Integration with document processing workflows
- Real-time progress tracking and notifications

Model Management:
- Primary model: mxbai-embed-large (1000-dimensional legal-optimized)
- Fallback model: nomic-embed-text (768-dimensional reliability backup)
- Runtime model switching with validation and warm-up
- Model availability monitoring and health checks
- Automatic model selection based on availability and performance

Business Rules:
- Embedding quality validation and normalization
- Batch size optimization based on model specifications
- Concurrent processing limits and queue management
- Error handling with retry mechanisms and fallback strategies
- Resource monitoring and capacity management
- Performance metrics collection and analysis

Architecture Integration:
- Integrates with OllamaClient for direct model operations
- Coordinates with EmbeddingProcessor for batch processing
- Uses DocumentChunk domain models for legal document chunks
- Provides service layer abstraction for document processing
- Implements business logic for embedding-related operations
- Supports WebSocket notifications for progress tracking
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from config.settings import get_settings
from ..core.ollama_client import OllamaClient, get_ollama_client
from ..core.websocket_manager import WebSocketManager
from ..models.domain.document import DocumentChunk
from ..core.exceptions import (
    EmbeddingError, ModelError, ValidationError, ResourceError,
    ErrorCode, raise_embedding_error, raise_model_error, raise_validation_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class EmbeddingModel(str, Enum):
    """Supported embedding models with specifications."""
    MXBAI_EMBED_LARGE = "mxbai-embed-large"
    NOMIC_EMBED_TEXT = "nomic-embed-text"
    E5_LARGE_V2 = "e5-large-v2"  # Future support


class EmbeddingQuality(str, Enum):
    """Embedding quality levels for validation."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


@dataclass
class ModelSpecification:
    """Embedding model specifications and capabilities."""
    name: str
    dimensions: int
    max_tokens: int
    optimal_batch_size: int
    max_batch_size: int
    gpu_memory_mb: int
    avg_inference_time_ms: float
    domain_optimization: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    is_primary: bool = False
    is_fallback: bool = False
    
    def __post_init__(self):
        """Validate model specification after creation."""
        if self.dimensions <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    texts: List[str]
    model_name: Optional[str] = None
    normalize: bool = True
    use_cache: bool = True
    priority: int = 5  # 1=highest, 10=lowest
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None
    request_id: str = field(default_factory=lambda: f"emb_{int(time.time() * 1000)}")


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    embeddings: List[List[float]]
    model_used: str
    dimensions: int
    processing_time_ms: float
    tokens_processed: int
    quality_score: float
    fallback_used: bool
    cached_count: int = 0
    request_id: str = ""


@dataclass
class ModelHealth:
    """Model health and performance metrics."""
    model_name: str
    is_available: bool
    avg_response_time_ms: float
    success_rate: float
    error_count: int
    last_error: Optional[str] = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EmbeddingService:
    """
    Business logic service for embedding generation and management.
    
    Orchestrates embedding operations, manages model switching, enforces
    business rules, and provides service layer abstraction for embedding-related
    operations in the legal document processing pipeline.
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        websocket_manager: Optional[WebSocketManager] = None
    ):
        """
        Initialize embedding service with dependencies.
        
        Args:
            ollama_client: Ollama client for model operations
            websocket_manager: WebSocket manager for real-time updates
        """
        self.ollama_client = ollama_client or get_ollama_client()
        self.websocket_manager = websocket_manager
        
        # Load configuration
        self.settings = get_settings()
        
        # Model specifications
        self.model_specs = self._initialize_model_specs()
        
        # Service state
        self._current_primary_model = self.settings.ollama.embedding_model
        self._current_fallback_model = self.settings.ollama.fallback_model
        self._model_health: Dict[str, ModelHealth] = {}
        
        # Performance tracking
        self._performance_metrics = {
            "total_embeddings_generated": 0,
            "total_processing_time_ms": 0,
            "model_usage_stats": {model.value: 0 for model in EmbeddingModel},
            "fallback_usage_count": 0,
            "cache_hit_rate": 0.0,
            "average_batch_size": 0.0,
            "error_counts": {},
            "quality_distribution": {quality.value: 0 for quality in EmbeddingQuality}
        }
        
        # Processing state
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._request_queue = asyncio.Queue()
        self._processing_semaphore = asyncio.Semaphore(
            self.settings.ollama.concurrent_requests
        )
        
        # Cache for embedding results (simple in-memory cache)
        self._embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._cache_ttl = timedelta(hours=24)  # 24-hour cache TTL
        self._max_cache_size = 10000
        
        logger.info(
            "EmbeddingService initialized",
            primary_model=self._current_primary_model,
            fallback_model=self._current_fallback_model,
            concurrent_requests=self.settings.ollama.concurrent_requests
        )
    
    @property
    def current_model(self) -> str:
        """Get the current primary embedding model."""
        return self._current_primary_model
    
    @property
    def fallback_model(self) -> str:
        """Get the current fallback embedding model."""
        return self._current_fallback_model
    
    async def generate_embeddings(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        Generate embeddings for the given request.
        
        Args:
            request: Embedding generation request
            
        Returns:
            Embedding response with generated vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If request validation fails
        """
        async with performance_context(
            "embedding_service_generate",
            model_name=request.model_name or self._current_primary_model,
            text_count=len(request.texts)
        ):
            # Validate request
            self._validate_embedding_request(request)
            
            # Determine model to use
            model_name = request.model_name or self._current_primary_model
            
            # Check cache first if enabled
            cached_embeddings, cache_hits = [], 0
            if request.use_cache:
                cached_embeddings, cache_hits = await self._check_cache(
                    request.texts, model_name
                )
            
            # Determine texts that need processing
            texts_to_process = []
            cache_indices = []
            
            for i, text in enumerate(request.texts):
                if i < len(cached_embeddings) and cached_embeddings[i] is not None:
                    cache_indices.append(i)
                else:
                    texts_to_process.append((i, text))
            
            start_time = time.time()
            fallback_used = False
            
            try:
                # Generate embeddings for non-cached texts
                new_embeddings = []
                if texts_to_process:
                    # Use semaphore to limit concurrent requests
                    async with self._processing_semaphore:
                        new_embeddings = await self._generate_embeddings_internal(
                            [text for _, text in texts_to_process],
                            model_name,
                            request.normalize
                        )
                
                # Combine cached and new embeddings
                all_embeddings = self._combine_embeddings(
                    cached_embeddings, new_embeddings, texts_to_process
                )
                
                # Validate embedding quality
                quality_score = self._validate_embedding_quality(all_embeddings, model_name)
                
                # Update cache with new embeddings
                if request.use_cache and new_embeddings:
                    await self._update_cache(
                        [text for _, text in texts_to_process],
                        new_embeddings,
                        model_name
                    )
                
            except Exception as e:
                # Attempt fallback if primary model fails
                if model_name != self._current_fallback_model:
                    logger.warning(
                        "Primary model failed, attempting fallback",
                        primary_model=model_name,
                        fallback_model=self._current_fallback_model,
                        error=str(e)
                    )
                    
                    try:
                        async with self._processing_semaphore:
                            all_embeddings = await self._generate_embeddings_internal(
                                request.texts,
                                self._current_fallback_model,
                                request.normalize
                            )
                        
                        model_name = self._current_fallback_model
                        fallback_used = True
                        quality_score = self._validate_embedding_quality(
                            all_embeddings, model_name
                        )
                        
                        self._performance_metrics["fallback_usage_count"] += 1
                        
                    except Exception as fallback_error:
                        logger.error(
                            "Both primary and fallback models failed",
                            primary_error=str(e),
                            fallback_error=str(fallback_error)
                        )
                        raise_embedding_error(
                            f"Embedding generation failed: {str(fallback_error)}",
                            ErrorCode.EMBEDDING_GENERATION_FAILED,
                            {"primary_error": str(e), "fallback_error": str(fallback_error)}
                        )
                else:
                    raise_embedding_error(
                        f"Embedding generation failed: {str(e)}",
                        ErrorCode.EMBEDDING_GENERATION_FAILED,
                        {"model": model_name, "error": str(e)}
                    )
            
            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            tokens_processed = sum(len(text.split()) for text in request.texts)
            
            # Update performance metrics
            self._update_performance_metrics(
                len(all_embeddings), processing_time_ms, model_name, cache_hits
            )
            
            # Send progress update if WebSocket available
            if self.websocket_manager and request.user_id:
                await self._send_progress_update(request, "Completed", 100)
            
            # Create response
            response = EmbeddingResponse(
                embeddings=all_embeddings,
                model_used=model_name,
                dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                processing_time_ms=processing_time_ms,
                tokens_processed=tokens_processed,
                quality_score=quality_score,
                fallback_used=fallback_used,
                cached_count=cache_hits,
                request_id=request.request_id
            )
            
            logger.info(
                "Embeddings generated successfully",
                request_id=request.request_id,
                model_used=model_name,
                embedding_count=len(all_embeddings),
                processing_time_ms=processing_time_ms,
                cached_count=cache_hits,
                fallback_used=fallback_used
            )
            
            return response
    
    async def generate_document_chunk_embeddings(
        self,
        chunks: List[DocumentChunk],
        user_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            user_id: User ID for progress tracking
            model_name: Optional model override
            
        Returns:
            List of chunks with embeddings attached
        """
        if not chunks:
            return chunks
        
        # Create embedding request
        request = EmbeddingRequest(
            texts=[chunk.content for chunk in chunks],
            model_name=model_name,
            normalize=True,
            use_cache=True,
            priority=3,  # Medium priority for chunk processing
            user_id=user_id,
            document_id=chunks[0].document_id if chunks else None,
            chunk_ids=[chunk.chunk_id for chunk in chunks]
        )
        
        # Generate embeddings
        response = await self.generate_embeddings(request)
        
        # Attach embeddings to chunks
        updated_chunks = []
        for chunk, embedding in zip(chunks, response.embeddings):
            # Create new chunk with embedding (assuming immutable chunks)
            updated_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding_vector=embedding,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                paragraph_number=chunk.paragraph_number,
                legal_citations=chunk.legal_citations
            )
            updated_chunks.append(updated_chunk)
        
        logger.info(
            "Document chunk embeddings generated",
            document_id=chunks[0].document_id if chunks else None,
            chunk_count=len(updated_chunks),
            model_used=response.model_used,
            dimensions=response.dimensions,
            processing_time_ms=response.processing_time_ms
        )
        
        return updated_chunks
    
    async def validate_model_availability(self, model_name: str) -> bool:
        """
        Validate that a model is available for embedding generation.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if model is in our specifications
            if model_name not in [spec.name for spec in self.model_specs.values()]:
                return False
            
            # Check with Ollama client
            is_available = await self.ollama_client.validate_model_availability(
                model_name, auto_pull=False
            )
            
            # Update model health
            await self._update_model_health(model_name, is_available)
            
            return is_available
            
        except Exception as e:
            logger.error(
                "Model availability check failed",
                model_name=model_name,
                error=str(e)
            )
            await self._update_model_health(model_name, False, str(e))
            return False
    
    async def switch_primary_model(self, new_model: str) -> bool:
        """
        Switch the primary embedding model with validation.
        
        Args:
            new_model: New primary model name
            
        Returns:
            True if switch was successful, False otherwise
        """
        # Validate new model
        if not await self.validate_model_availability(new_model):
            logger.error(
                "Cannot switch to unavailable model",
                requested_model=new_model
            )
            return False
        
        old_model = self._current_primary_model
        
        # Perform model warm-up
        if not await self._warm_up_model(new_model):
            logger.error(
                "Model warm-up failed, cancelling switch",
                model_name=new_model
            )
            return False
        
        # Update current model
        self._current_primary_model = new_model
        
        # Clear cache for consistency (optional, based on business rules)
        # self._clear_model_cache(old_model)
        
        logger.info(
            "Primary embedding model switched",
            old_model=old_model,
            new_model=new_model
        )
        
        return True
    
    async def get_model_health(self) -> Dict[str, ModelHealth]:
        """
        Get health status for all configured models.
        
        Returns:
            Dictionary of model health information
        """
        # Update health for all models
        for model_name in [spec.name for spec in self.model_specs.values()]:
            await self.validate_model_availability(model_name)
        
        return self._model_health.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics and statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate cache hit rate
        total_requests = (
            self._performance_metrics["total_embeddings_generated"] + 
            sum(self._performance_metrics.get("cache_hits", {}).values())
        )
        cache_hits = sum(self._performance_metrics.get("cache_hits", {}).values())
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            **self._performance_metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._embedding_cache),
            "active_requests": len(self._active_requests),
            "current_primary_model": self._current_primary_model,
            "current_fallback_model": self._current_fallback_model,
            "model_specifications": {
                name: {
                    "dimensions": spec.dimensions,
                    "max_tokens": spec.max_tokens,
                    "optimal_batch_size": spec.optimal_batch_size,
                    "quality_score": spec.quality_score
                }
                for name, spec in self.model_specs.items()
            }
        }
    
    def get_model_specifications(self) -> Dict[str, ModelSpecification]:
        """Get all model specifications."""
        return self.model_specs.copy()
    
    # Private helper methods
    
    def _initialize_model_specs(self) -> Dict[str, ModelSpecification]:
        """Initialize model specifications."""
        specs = {
            EmbeddingModel.MXBAI_EMBED_LARGE.value: ModelSpecification(
                name=EmbeddingModel.MXBAI_EMBED_LARGE.value,
                dimensions=1000,
                max_tokens=8192,
                optimal_batch_size=16,
                max_batch_size=32,
                gpu_memory_mb=2048,
                avg_inference_time_ms=150.0,
                domain_optimization=["legal", "technical", "academic"],
                quality_score=0.95,
                is_primary=True
            ),
            EmbeddingModel.NOMIC_EMBED_TEXT.value: ModelSpecification(
                name=EmbeddingModel.NOMIC_EMBED_TEXT.value,
                dimensions=768,
                max_tokens=8192,
                optimal_batch_size=24,
                max_batch_size=48,
                gpu_memory_mb=1536,
                avg_inference_time_ms=120.0,
                domain_optimization=["general", "web"],
                quality_score=0.85,
                is_fallback=True
            )
        }
        
        return specs
    
    def _validate_embedding_request(self, request: EmbeddingRequest) -> None:
        """Validate embedding request parameters."""
        if not request.texts:
            raise_validation_error(
                "Empty text list provided for embedding",
                ErrorCode.EMBEDDING_EMPTY_INPUT
            )
        
        if len(request.texts) > 1000:  # Reasonable batch limit
            raise_validation_error(
                f"Too many texts in request: {len(request.texts)} (max: 1000)",
                ErrorCode.EMBEDDING_BATCH_TOO_LARGE,
                {"batch_size": len(request.texts), "max_size": 1000}
            )
        
        # Validate individual text lengths
        for i, text in enumerate(request.texts):
            if not text or not text.strip():
                raise_validation_error(
                    f"Empty or whitespace-only text at index {i}",
                    ErrorCode.EMBEDDING_EMPTY_TEXT,
                    {"index": i}
                )
            
            if len(text) > 32000:  # Conservative token limit
                raise_validation_error(
                    f"Text too long at index {i}: {len(text)} characters",
                    ErrorCode.EMBEDDING_TEXT_TOO_LONG,
                    {"index": i, "length": len(text), "max_length": 32000}
                )
        
        # Validate priority
        if not 1 <= request.priority <= 10:
            raise_validation_error(
                f"Invalid priority: {request.priority} (must be 1-10)",
                ErrorCode.EMBEDDING_INVALID_PRIORITY,
                {"priority": request.priority}
            )
    
    async def _generate_embeddings_internal(
        self,
        texts: List[str],
        model_name: str,
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings using Ollama client."""
        try:
            # Generate embeddings using Ollama client
            embeddings = await self.ollama_client.generate_embeddings(
                model_name=model_name,
                texts=texts,
                normalize=normalize
            )
            
            # Ensure we got embeddings for all texts
            if len(embeddings) != len(texts):
                raise_embedding_error(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}",
                    ErrorCode.EMBEDDING_COUNT_MISMATCH,
                    {"expected": len(texts), "actual": len(embeddings)}
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "Internal embedding generation failed",
                model_name=model_name,
                text_count=len(texts),
                error=str(e)
            )
            raise
    
    async def _check_cache(
        self,
        texts: List[str],
        model_name: str
    ) -> Tuple[List[Optional[List[float]]], int]:
        """Check cache for existing embeddings."""
        cached_embeddings = []
        cache_hits = 0
        
        for text in texts:
            cache_key = self._get_cache_key(text, model_name)
            
            if cache_key in self._embedding_cache:
                embedding, timestamp = self._embedding_cache[cache_key]
                
                # Check if cache entry is still valid
                if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                    cached_embeddings.append(embedding)
                    cache_hits += 1
                else:
                    # Remove expired entry
                    del self._embedding_cache[cache_key]
                    cached_embeddings.append(None)
            else:
                cached_embeddings.append(None)
        
        return cached_embeddings, cache_hits
    
    async def _update_cache(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model_name: str
    ) -> None:
        """Update cache with new embeddings."""
        current_time = datetime.now(timezone.utc)
        
        for text, embedding in zip(texts, embeddings):
            # Check cache size limit
            if len(self._embedding_cache) >= self._max_cache_size:
                # Remove oldest entries (simple LRU approximation)
                oldest_key = min(
                    self._embedding_cache.keys(),
                    key=lambda k: self._embedding_cache[k][1]
                )
                del self._embedding_cache[oldest_key]
            
            cache_key = self._get_cache_key(text, model_name)
            self._embedding_cache[cache_key] = (embedding, current_time)
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model_name}:{text_hash}"
    
    def _combine_embeddings(
        self,
        cached_embeddings: List[Optional[List[float]]],
        new_embeddings: List[List[float]],
        text_indices: List[Tuple[int, str]]
    ) -> List[List[float]]:
        """Combine cached and newly generated embeddings."""
        result = cached_embeddings.copy()
        new_embedding_idx = 0
        
        for original_idx, _ in text_indices:
            if original_idx < len(result):
                result[original_idx] = new_embeddings[new_embedding_idx]
                new_embedding_idx += 1
        
        # Filter out None values (should not happen if logic is correct)
        return [emb for emb in result if emb is not None]
    
    def _validate_embedding_quality(
        self,
        embeddings: List[List[float]],
        model_name: str
    ) -> float:
        """Validate embedding quality and return quality score."""
        if not embeddings:
            return 0.0
        
        # Get expected dimensions for model
        model_spec = self.model_specs.get(model_name)
        if not model_spec:
            return 0.5  # Unknown model, medium quality
        
        # Check dimension consistency
        expected_dims = model_spec.dimensions
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dims:
                logger.warning(
                    "Embedding dimension mismatch",
                    embedding_index=i,
                    expected_dims=expected_dims,
                    actual_dims=len(embedding),
                    model_name=model_name
                )
                return 0.3  # Low quality due to dimension mismatch
        
        # Check for valid values (no NaN, Inf, or all zeros)
        for i, embedding in enumerate(embeddings):
            np_embedding = np.array(embedding)
            
            if np.any(np.isnan(np_embedding)) or np.any(np.isinf(np_embedding)):
                logger.warning(
                    "Invalid embedding values detected",
                    embedding_index=i,
                    model_name=model_name
                )
                return 0.2  # Low quality due to invalid values
            
            if np.allclose(np_embedding, 0.0):
                logger.warning(
                    "Zero embedding detected",
                    embedding_index=i,
                    model_name=model_name
                )
                return 0.4  # Medium-low quality due to zero embeddings
        
        return model_spec.quality_score
    
    async def _warm_up_model(self, model_name: str) -> bool:
        """Warm up a model by generating a test embedding."""
        try:
            test_text = "This is a test embedding for model warm-up."
            await self._generate_embeddings_internal([test_text], model_name, True)
            
            logger.info(
                "Model warm-up successful",
                model_name=model_name
            )
            return True
            
        except Exception as e:
            logger.error(
                "Model warm-up failed",
                model_name=model_name,
                error=str(e)
            )
            return False
    
    async def _update_model_health(
        self,
        model_name: str,
        is_available: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Update model health status."""
        current_time = datetime.now(timezone.utc)
        
        if model_name not in self._model_health:
            self._model_health[model_name] = ModelHealth(
                model_name=model_name,
                is_available=is_available,
                avg_response_time_ms=0.0,
                success_rate=1.0 if is_available else 0.0,
                error_count=0 if is_available else 1,
                last_error=error_message,
                last_check=current_time
            )
        else:
            health = self._model_health[model_name]
            health.is_available = is_available
            health.last_check = current_time
            
            if not is_available:
                health.error_count += 1
                health.last_error = error_message
    
    def _update_performance_metrics(
        self,
        embedding_count: int,
        processing_time_ms: float,
        model_name: str,
        cache_hits: int
    ) -> None:
        """Update performance tracking metrics."""
        self._performance_metrics["total_embeddings_generated"] += embedding_count
        self._performance_metrics["total_processing_time_ms"] += processing_time_ms
        self._performance_metrics["model_usage_stats"][model_name] += embedding_count
        
        # Update cache hit tracking
        if "cache_hits" not in self._performance_metrics:
            self._performance_metrics["cache_hits"] = {}
        if model_name not in self._performance_metrics["cache_hits"]:
            self._performance_metrics["cache_hits"][model_name] = 0
        self._performance_metrics["cache_hits"][model_name] += cache_hits
    
    async def _send_progress_update(
        self,
        request: EmbeddingRequest,
        message: str,
        progress_percent: int
    ) -> None:
        """Send progress update via WebSocket."""
        if not self.websocket_manager or not request.user_id:
            return
        
        update = {
            "type": "embedding_progress",
            "data": {
                "request_id": request.request_id,
                "message": message,
                "progress_percent": progress_percent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        await self.websocket_manager.send_to_user(request.user_id, update)
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Cancel active requests
        for task in self._active_requests.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._active_requests:
            await asyncio.gather(*self._active_requests.values(), return_exceptions=True)
        
        # Clear cache
        self._embedding_cache.clear()
        
        logger.info("EmbeddingService cleanup completed")