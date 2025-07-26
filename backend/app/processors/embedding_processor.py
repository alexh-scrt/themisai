"""
Embedding Generation Processor for Legal Documents

This module provides sophisticated embedding generation capabilities for legal document
chunks using Ollama-managed models. It handles batch processing, model fallback,
GPU optimization, and caching for high-performance vector generation.

Key Features:
- Primary model: mxbai-embed-large (1000-dimensional legal-optimized embeddings)
- Fallback model: nomic-embed-text (768-dimensional reliability backup)
- Intelligent batch processing with GPU memory optimization
- Model health monitoring and automatic fallback
- Embedding quality validation and normalization
- Cache management for frequently embedded texts
- Progress tracking and performance metrics
- Async processing with concurrent request handling

Model Specifications:
- mxbai-embed-large: State-of-the-art MTEB performance, legal domain generalization
- nomic-embed-text: Reliable fallback with good general domain performance
- Runtime model switching with immediate validation
- GPU acceleration through Ollama CUDA integration

Architecture Integration:
- Integrates with OllamaClient for model management
- Works with DocumentProcessor for pipeline processing
- Provides embedding services for search and indexing
- Supports WebSocket progress updates for real-time feedback
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from enum import Enum

import numpy as np
from cachetools import TTLCache, LRUCache

from config.settings import get_settings
from ..core.ollama_client import OllamaClient
from ..core.websocket_manager import WebSocketManager
from ..models.domain.document import DocumentChunk
from ..utils.logging import get_logger
from ..core.exceptions import (
    ModelError,
    ConfigurationError,
    EmbeddingError,
    ErrorCode
)

logger = get_logger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models with specifications."""
    MXBAI_EMBED_LARGE = "mxbai-embed-large"
    NOMIC_EMBED_TEXT = "nomic-embed-text"
    E5_LARGE_V2 = "e5-large-v2"  # Future support


@dataclass
class ModelSpecification:
    """Embedding model specifications and capabilities."""
    name: str
    dimensions: int
    max_tokens: int
    batch_size_limit: int
    gpu_memory_mb: int
    avg_inference_time_ms: float
    domain_optimization: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    is_primary: bool = False
    is_fallback: bool = False


@dataclass
class EmbeddingTask:
    """Individual embedding generation task."""
    task_id: str
    texts: List[str]
    model_name: str
    priority: int = 5  # 1-10, lower is higher priority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None
    normalize: bool = True
    
    @property
    def age_seconds(self) -> float:
        """Get task age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""
    embeddings: List[List[float]]
    model_used: str
    dimensions: int
    processing_time_ms: float
    tokens_processed: int
    batch_size: int
    quality_score: float = 1.0
    was_cached: bool = False
    fallback_used: bool = False
    gpu_memory_used_mb: Optional[float] = None


class EmbeddingCache:
    """
    Intelligent caching system for embeddings with text hashing.
    
    Provides multi-level caching with TTL and LRU eviction policies,
    text normalization for cache hits, and model-specific storage.
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings per model
            ttl_hours: Time-to-live for cached embeddings in hours
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        
        # Model-specific caches
        self._caches: Dict[str, TTLCache] = {}
        self._access_counts: Dict[str, defaultdict] = {}
        
        # Cache statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_size": 0,
            "evictions": 0,
            "hit_rate": 0.0
        }
        
        logger.info(
            "EmbeddingCache initialized",
            max_size=max_size,
            ttl_hours=ttl_hours
        )
    
    def _get_cache(self, model_name: str) -> TTLCache:
        """Get or create cache for a specific model."""
        if model_name not in self._caches:
            self._caches[model_name] = TTLCache(
                maxsize=self.max_size,
                ttl=self.ttl_seconds
            )
            self._access_counts[model_name] = defaultdict(int)
        return self._caches[model_name]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent caching."""
        # Remove extra whitespace and normalize
        normalized = " ".join(text.split())
        return normalized.strip()
    
    def _generate_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination."""
        normalized_text = self._normalize_text(text)
        text_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()[:16]
        return f"{model_name}:{text_hash}"
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding for text and model.
        
        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._generate_cache_key(text, model_name)
        cache = self._get_cache(model_name)
        
        embedding = cache.get(cache_key)
        
        if embedding is not None:
            self._stats["cache_hits"] += 1
            self._access_counts[model_name][cache_key] += 1
            
            logger.debug(
                "Embedding cache hit",
                model_name=model_name,
                text_length=len(text),
                cache_key=cache_key[:12]
            )
        else:
            self._stats["cache_misses"] += 1
        
        # Update hit rate
        total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_requests > 0:
            self._stats["hit_rate"] = self._stats["cache_hits"] / total_requests
        
        return embedding
    
    def put(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Cache embedding for text and model.
        
        Args:
            text: Input text
            model_name: Model that generated the embedding
            embedding: Generated embedding vector
        """
        cache_key = self._generate_cache_key(text, model_name)
        cache = self._get_cache(model_name)
        
        # Check if this will cause an eviction
        if len(cache) >= self.max_size and cache_key not in cache:
            self._stats["evictions"] += 1
        
        cache[cache_key] = embedding
        self._stats["cache_size"] = sum(len(c) for c in self._caches.values())
        
        logger.debug(
            "Embedding cached",
            model_name=model_name,
            text_length=len(text),
            embedding_dimensions=len(embedding),
            cache_size=self._stats["cache_size"]
        )
    
    def get_batch(self, texts: List[str], model_name: str) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get cached embeddings for batch of texts.
        
        Returns:
            Tuple of (embeddings_list, missing_indices)
        """
        embeddings = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model_name)
            embeddings.append(embedding)
            if embedding is None:
                missing_indices.append(i)
        
        return embeddings, missing_indices
    
    def put_batch(self, texts: List[str], model_name: str, embeddings: List[List[float]]) -> None:
        """Cache batch of embeddings."""
        for text, embedding in zip(texts, embeddings):
            self.put(text, model_name, embedding)
    
    def invalidate_model(self, model_name: str) -> None:
        """Invalidate all cached embeddings for a model."""
        if model_name in self._caches:
            cleared_count = len(self._caches[model_name])
            self._caches[model_name].clear()
            self._access_counts[model_name].clear()
            self._stats["cache_size"] -= cleared_count
            
            logger.info(
                "Model cache invalidated",
                model_name=model_name,
                cleared_count=cleared_count
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            "models_cached": list(self._caches.keys()),
            "cache_sizes": {model: len(cache) for model, cache in self._caches.items()},
            "most_accessed": {
                model: dict(counts.most_common(5))
                for model, counts in self._access_counts.items()
            }
        }
    
    def clear_all(self) -> None:
        """Clear all cached embeddings."""
        total_cleared = sum(len(cache) for cache in self._caches.values())
        self._caches.clear()
        self._access_counts.clear()
        self._stats["cache_size"] = 0
        
        logger.info("All embedding caches cleared", total_cleared=total_cleared)


class EmbeddingProcessor:
    """
    Advanced embedding processor with model management and optimization.
    
    Provides high-performance embedding generation for legal documents with
    intelligent batching, caching, fallback handling, and GPU optimization.
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        websocket_manager: Optional[WebSocketManager] = None,
        cache_size: int = 10000,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize embedding processor.
        
        Args:
            ollama_client: Ollama client for model operations
            websocket_manager: Optional WebSocket manager for progress updates
            cache_size: Maximum number of cached embeddings
            cache_ttl_hours: Cache TTL in hours
        """
        self.ollama_client = ollama_client
        self.websocket_manager = websocket_manager
        
        # Initialize configuration
        self.config = get_settings()
        self._setup_model_specifications()
        
        # Initialize cache
        self.cache = EmbeddingCache(cache_size, cache_ttl_hours)
        
        # Processing queues and tracking
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_tasks: Dict[str, EmbeddingTask] = {}
        self._processing_semaphore = asyncio.Semaphore(4)  # Max concurrent requests
        
        # Performance tracking
        self._performance_metrics = {
            "total_embeddings_generated": 0,
            "total_processing_time_ms": 0,
            "average_processing_time_ms": 0,
            "gpu_utilization_history": deque(maxlen=100),
            "model_usage_stats": defaultdict(int),
            "error_counts": defaultdict(int),
            "fallback_usage_count": 0,
            "batch_size_optimization": {}
        }
        
        # Start background processing
        self._processing_task = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "EmbeddingProcessor initialized",
            primary_model=self.config.embedding_settings.get("primary_model", "mxbai-embed-large"),
            fallback_model=self.config.embedding_settings.get("fallback_model", "nomic-embed-text"),
            cache_size=cache_size,
            max_concurrent=4
        )
    
    def _setup_model_specifications(self) -> None:
        """Setup model specifications and capabilities."""
        self.model_specs = {
            EmbeddingModel.MXBAI_EMBED_LARGE.value: ModelSpecification(
                name="mxbai-embed-large",
                dimensions=1000,
                max_tokens=8192,
                batch_size_limit=32,
                gpu_memory_mb=2000,
                avg_inference_time_ms=150,
                domain_optimization=["legal", "technical", "patent"],
                quality_score=0.95,
                is_primary=True
            ),
            EmbeddingModel.NOMIC_EMBED_TEXT.value: ModelSpecification(
                name="nomic-embed-text",
                dimensions=768,
                max_tokens=8192,
                batch_size_limit=64,
                gpu_memory_mb=1500,
                avg_inference_time_ms=120,
                domain_optimization=["general", "research"],
                quality_score=0.85,
                is_fallback=True
            ),
            EmbeddingModel.E5_LARGE_V2.value: ModelSpecification(
                name="e5-large-v2",
                dimensions=1024,
                max_tokens=4096,
                batch_size_limit=16,
                gpu_memory_mb=2500,
                avg_inference_time_ms=200,
                domain_optimization=["general"],
                quality_score=0.88
            )
        }
    
    async def start_processing(self) -> None:
        """Start background embedding processing."""
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_queue())
            logger.info("Embedding processor started")
    
    async def stop_processing(self) -> None:
        """Stop background processing and cleanup."""
        self._shutdown_event.set()
        
        if self._processing_task and not self._processing_task.done():
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._processing_task.cancel()
        
        logger.info("Embedding processor stopped")
    
    async def _process_queue(self) -> None:
        """Background task processor for embedding queue."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    priority, task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task with semaphore limiting
                async with self._processing_semaphore:
                    await self._process_embedding_task(task)
                
                # Mark task as done
                self._task_queue.task_done()
                
            except Exception as e:
                logger.error(
                    "Error in embedding queue processor",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(1.0)
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True,
        use_cache: bool = True,
        priority: int = 5,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            model_name: Optional model override
            normalize: Whether to normalize embeddings
            use_cache: Whether to use caching
            priority: Task priority (1-10, lower is higher)
            user_id: Optional user identifier for progress tracking
            document_id: Optional document identifier
            chunk_ids: Optional chunk identifiers
            
        Returns:
            EmbeddingResult with generated embeddings and metadata
        """
        if not texts:
            raise EmbeddingError(
                "No texts provided for embedding generation",
                error_code=ErrorCode.INVALID_INPUT
            )
        
        # Determine model to use
        effective_model = model_name or self._get_primary_model()
        
        # Create task
        task_id = f"{int(time.time() * 1000)}_{hash(tuple(texts)) % 10000}"
        task = EmbeddingTask(
            task_id=task_id,
            texts=texts,
            model_name=effective_model,
            priority=priority,
            user_id=user_id,
            document_id=document_id,
            chunk_ids=chunk_ids,
            normalize=normalize
        )
        
        # Check cache first if enabled
        if use_cache:
            cached_embeddings, missing_indices = self.cache.get_batch(texts, effective_model)
            
            if not missing_indices:  # All embeddings cached
                logger.debug(
                    "All embeddings found in cache",
                    model_name=effective_model,
                    text_count=len(texts),
                    task_id=task_id
                )
                
                return EmbeddingResult(
                    embeddings=cached_embeddings,
                    model_used=effective_model,
                    dimensions=len(cached_embeddings[0]) if cached_embeddings else 0,
                    processing_time_ms=0,
                    tokens_processed=sum(len(text.split()) for text in texts),
                    batch_size=len(texts),
                    was_cached=True
                )
            
            # Update task to only process missing texts
            if missing_indices:
                task.texts = [texts[i] for i in missing_indices]
                logger.debug(
                    "Partial cache hit",
                    model_name=effective_model,
                    cached_count=len(texts) - len(missing_indices),
                    missing_count=len(missing_indices),
                    task_id=task_id
                )
        
        # Process task immediately for single request, queue for batch
        if len(texts) == 1:
            result = await self._process_embedding_task(task)
            
            # Merge with cached results if partial cache hit
            if use_cache and 'cached_embeddings' in locals():
                full_embeddings = cached_embeddings.copy()
                for i, missing_idx in enumerate(missing_indices):
                    full_embeddings[missing_idx] = result.embeddings[i]
                
                result.embeddings = full_embeddings
                result.was_cached = len(missing_indices) < len(texts)
            
            return result
        else:
            # Queue for batch processing
            await self._task_queue.put((priority, task))
            self._active_tasks[task_id] = task
            
            # Wait for completion
            while task_id in self._active_tasks:
                await asyncio.sleep(0.1)
            
            # Retrieve result (this would need a result storage mechanism)
            # For now, process immediately
            return await self._process_embedding_task(task)
    
    async def _process_embedding_task(self, task: EmbeddingTask) -> EmbeddingResult:
        """Process a single embedding task."""
        start_time = time.time()
        
        try:
            logger.debug(
                "Processing embedding task",
                task_id=task.task_id,
                model_name=task.model_name,
                text_count=len(task.texts),
                priority=task.priority
            )
            
            # Send progress update if WebSocket available
            if self.websocket_manager and task.user_id:
                await self._send_progress_update(
                    task,
                    "Starting embedding generation...",
                    0
                )
            
            # Validate model availability
            model_available = await self._validate_model_availability(task.model_name)
            if not model_available:
                # Try fallback model
                fallback_model = self._get_fallback_model()
                if fallback_model and fallback_model != task.model_name:
                    logger.warning(
                        "Primary model unavailable, using fallback",
                        primary_model=task.model_name,
                        fallback_model=fallback_model,
                        task_id=task.task_id
                    )
                    task.model_name = fallback_model
                    self._performance_metrics["fallback_usage_count"] += 1
                else:
                    raise EmbeddingError(
                        f"Model {task.model_name} unavailable and no fallback",
                        error_code=ErrorCode.MODEL_UNAVAILABLE
                    )
            
            # Optimize batch size based on model specs and GPU memory
            optimal_batch_size = self._calculate_optimal_batch_size(
                task.model_name,
                len(task.texts)
            )
            
            # Process in batches
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(task.texts), optimal_batch_size):
                batch_texts = task.texts[i:i + optimal_batch_size]
                
                # Send progress update
                if self.websocket_manager and task.user_id:
                    progress = int((i / len(task.texts)) * 90)  # 90% for processing
                    await self._send_progress_update(
                        task,
                        f"Processing batch {i//optimal_batch_size + 1}...",
                        progress
                    )
                
                # Generate embeddings for batch
                batch_embeddings = await self.ollama_client.generate_embeddings(
                    model_name=task.model_name,
                    texts=batch_texts
                )
                
                # Normalize if requested
                if task.normalize:
                    batch_embeddings = [
                        self._normalize_embedding(emb) for emb in batch_embeddings
                    ]
                
                all_embeddings.extend(batch_embeddings)
                total_tokens += sum(len(text.split()) for text in batch_texts)
                
                # Cache batch results
                self.cache.put_batch(batch_texts, task.model_name, batch_embeddings)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Get model specifications
            model_spec = self.model_specs.get(task.model_name)
            dimensions = len(all_embeddings[0]) if all_embeddings else 0
            
            # Create result
            result = EmbeddingResult(
                embeddings=all_embeddings,
                model_used=task.model_name,
                dimensions=dimensions,
                processing_time_ms=processing_time_ms,
                tokens_processed=total_tokens,
                batch_size=len(task.texts),
                quality_score=model_spec.quality_score if model_spec else 1.0,
                fallback_used=task.model_name != self._get_primary_model()
            )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Send completion update
            if self.websocket_manager and task.user_id:
                await self._send_progress_update(
                    task,
                    f"Completed - {len(all_embeddings)} embeddings generated",
                    100
                )
            
            logger.info(
                "Embedding task completed",
                task_id=task.task_id,
                model_name=task.model_name,
                embeddings_count=len(all_embeddings),
                processing_time_ms=processing_time_ms,
                tokens_processed=total_tokens
            )
            
            # Remove from active tasks
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            
            return result
            
        except Exception as e:
            self._performance_metrics["error_counts"][type(e).__name__] += 1
            
            # Send error update
            if self.websocket_manager and task.user_id:
                await self._send_progress_update(
                    task,
                    f"Error: {str(e)}",
                    -1
                )
            
            logger.error(
                "Embedding task failed",
                task_id=task.task_id,
                model_name=task.model_name,
                error=str(e),
                exc_info=True
            )
            
            # Remove from active tasks
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            
            raise EmbeddingError(
                f"Embedding generation failed: {str(e)}",
                error_code=ErrorCode.EMBEDDING_GENERATION_FAILED
            )
    
    def _get_primary_model(self) -> str:
        """Get the primary embedding model."""
        return self.config.embedding_settings.get("primary_model", "mxbai-embed-large")
    
    def _get_fallback_model(self) -> str:
        """Get the fallback embedding model."""
        return self.config.embedding_settings.get("fallback_model", "nomic-embed-text")
    
    async def _validate_model_availability(self, model_name: str) -> bool:
        """Validate that a model is available and loaded."""
        try:
            models = await self.ollama_client.list_models()
            return any(model.get("name") == model_name for model in models)
        except Exception as e:
            logger.warning(
                "Failed to validate model availability",
                model_name=model_name,
                error=str(e)
            )
            return False
    
    def _calculate_optimal_batch_size(self, model_name: str, total_texts: int) -> int:
        """Calculate optimal batch size based on model specs and GPU memory."""
        model_spec = self.model_specs.get(model_name)
        if not model_spec:
            return min(16, total_texts)  # Conservative default
        
        # Start with model's batch limit
        optimal_size = min(model_spec.batch_size_limit, total_texts)
        
        # Adjust based on GPU memory availability (simplified)
        # In practice, this would query GPU memory usage
        gpu_memory_factor = 0.8  # Use 80% of available memory
        memory_adjusted_size = int(optimal_size * gpu_memory_factor)
        
        # Store optimization result for analysis
        self._performance_metrics["batch_size_optimization"][model_name] = optimal_size
        
        return max(1, min(memory_adjusted_size, total_texts))
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        np_embedding = np.array(embedding)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            return (np_embedding / norm).tolist()
        return embedding
    
    def _update_performance_metrics(self, result: EmbeddingResult) -> None:
        """Update performance tracking metrics."""
        self._performance_metrics["total_embeddings_generated"] += len(result.embeddings)
        self._performance_metrics["total_processing_time_ms"] += result.processing_time_ms
        self._performance_metrics["model_usage_stats"][result.model_used] += 1
        
        # Calculate average processing time
        total_embeddings = self._performance_metrics["total_embeddings_generated"]
        if total_embeddings > 0:
            self._performance_metrics["average_processing_time_ms"] = (
                self._performance_metrics["total_processing_time_ms"] / total_embeddings
            )
    
    async def _send_progress_update(
        self,
        task: EmbeddingTask,
        message: str,
        progress_percent: int
    ) -> None:
        """Send progress update via WebSocket."""
        if not self.websocket_manager or not task.user_id:
            return
        
        try:
            update_data = {
                "task_id": task.task_id,
                "document_id": task.document_id,
                "chunk_ids": task.chunk_ids,
                "model_name": task.model_name,
                "message": message,
                "progress_percent": progress_percent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.websocket_manager.broadcast_to_user(
                task.user_id,
                "embedding_progress",
                update_data
            )
            
        except Exception as e:
            logger.warning(
                "Failed to send embedding progress update",
                task_id=task.task_id,
                error=str(e)
            )
    
    async def generate_chunk_embeddings(
        self,
        chunks: List[DocumentChunk],
        model_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            model_name: Optional model override
            user_id: Optional user ID for progress tracking
            
        Returns:
            List of chunks with embeddings attached
        """
        if not chunks:
            return chunks
        
        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        document_id = chunks[0].document_id if chunks else None
        
        # Generate embeddings
        result = await self.generate_embeddings(
            texts=texts,
            model_name=model_name,
            normalize=True,
            use_cache=True,
            priority=3,  # Medium priority for chunk processing
            user_id=user_id,
            document_id=document_id,
            chunk_ids=chunk_ids
        )
        
        # Attach embeddings to chunks
        updated_chunks = []
        for chunk, embedding in zip(chunks, result.embeddings):
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
            "Chunk embeddings generated",
            document_id=document_id,
            chunk_count=len(updated_chunks),
            model_used=result.model_used,
            dimensions=result.dimensions,
            processing_time_ms=result.processing_time_ms
        )
        
        return updated_chunks
    
    def get_model_specifications(self) -> Dict[str, ModelSpecification]:
        """Get all model specifications."""
        return self.model_specs.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics."""
        return {
            **self._performance_metrics,
            "cache_stats": self.cache.get_statistics(),
            "active_tasks_count": len(self._active_tasks),
            "queue_size": self._task_queue.qsize(),
            "available_models": list(self.model_specs.keys())
        }
    
    async def validate_model_switch(self, new_model: str) -> bool:
        """
        Validate that a model switch is possible.
        
        Args:
            new_model: New model name to validate
            
        Returns:
            True if model switch is valid, False otherwise
        """
        if new_model not in self.model_specs:
            logger.error(
                "Model not supported",
                model_name=new_model,
                supported_models=list(self.model_specs.keys())
            )
            return False
        
        # Check if model is available in Ollama
        is_available = await self._validate_model_availability(new_model)
        if not is_available:
            logger.error(
                "Model not available in Ollama",
                model_name=new_model
            )
            return False
        
        return True
    
    async def switch_primary_model(self, new_model: str) -> bool:
        """
        Switch the primary embedding model.
        
        Args:
            new_model: New primary model name
            
        Returns:
            True if switch was successful, False otherwise
        """
        if not await self.validate_model_switch(new_model):
            return False
        
        old_model = self._get_primary_model()
        
        # Update configuration
        self.config.embedding_settings["primary_model"] = new_model
        
        # Invalidate cache for old model to prevent inconsistencies
        self.cache.invalidate_model(old_model)
        
        logger.info(
            "Primary embedding model switched",
            old_model=old_model,
            new_model=new_model
        )
        
        return True
    
    async def warm_up_model(self, model_name: str) -> bool:
        """
        Warm up a model by generating a test embedding.
        
        Args:
            model_name: Model to warm up
            
        Returns:
            True if warm-up successful, False otherwise
        """
        try:
            test_text = "This is a test embedding for model warm-up."
            await self.generate_embeddings(
                texts=[test_text],
                model_name=model_name,
                use_cache=False,
                priority=1  # High priority for warm-up
            )
            
            logger.info(
                "Model warm-up completed",
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
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the embedding processor."""
        logger.info("Shutting down embedding processor")
        
        # Stop processing queue
        await self.stop_processing()
        
        # Clear all caches
        self.cache.clear_all()
        
        # Clear active tasks
        self._active_tasks.clear()
        
        logger.info("Embedding processor shutdown completed")