"""
Ollama client integration for Patexia Legal AI Chatbot.

This module provides:
- Ollama API client with connection management
- Embedding model operations for legal documents
- Model availability checking and validation
- Automatic model pulling and caching
- Error handling and retry logic
- Performance monitoring and optimization
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

import httpx
from httpx import AsyncClient, HTTPStatusError, RequestError, TimeoutException

from backend.app.core.exceptions import (
    ModelError,
    ErrorCode,
    raise_model_error
)
from backend.app.utils.logging import (
    get_logger,
    model_logger,
    performance_context
)
from backend.config.settings import get_settings

logger = get_logger(__name__)


class OllamaModelInfo:
    """
    Information about an available Ollama model.
    
    Contains model metadata, capabilities, and status information
    for tracking and validation purposes.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str,
        size: Optional[int] = None,
        digest: Optional[str] = None,
        modified_at: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model information.
        
        Args:
            name: Model name identifier
            model_type: Type of model (embedding, language, etc.)
            size: Model size in bytes
            digest: Model digest/hash for version tracking
            modified_at: Last modification timestamp
            details: Additional model details and parameters
        """
        self.name = name
        self.model_type = model_type
        self.size = size
        self.digest = digest
        self.modified_at = modified_at
        self.details = details or {}
    
    @property
    def size_mb(self) -> Optional[float]:
        """Get model size in megabytes."""
        return self.size / (1024 * 1024) if self.size else None
    
    @property
    def is_embedding_model(self) -> bool:
        """Check if this is an embedding model."""
        return self.model_type == "embedding" or "embed" in self.name.lower()
    
    @property
    def is_language_model(self) -> bool:
        """Check if this is a language generation model."""
        return self.model_type == "language" or not self.is_embedding_model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "size": self.size,
            "size_mb": self.size_mb,
            "digest": self.digest,
            "modified_at": self.modified_at,
            "details": self.details,
            "is_embedding_model": self.is_embedding_model,
            "is_language_model": self.is_language_model,
        }


class OllamaClient:
    """
    Async Ollama API client for model operations.
    
    Provides high-level interface for embedding generation, model management,
    and health monitoring with robust error handling and performance optimization.
    """
    
    def __init__(self):
        """Initialize Ollama client."""
        self._client: Optional[AsyncClient] = None
        self._base_url: Optional[str] = None
        self._timeout: int = 30
        self._is_connected: bool = False
        self._available_models: Dict[str, OllamaModelInfo] = {}
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """
        Establish connection to Ollama service.
        
        Raises:
            ModelError: If connection fails or service is unavailable
        """
        if self._is_connected:
            return
        
        async with self._connection_lock:
            if self._is_connected:
                return
            
            settings = get_settings()
            self._base_url = settings.ollama.base_url.rstrip('/')
            self._timeout = settings.ollama.timeout
            
            try:
                with performance_context("ollama_connection"):
                    # Create HTTP client with appropriate timeouts
                    timeout = httpx.Timeout(
                        connect=5.0,
                        read=float(self._timeout),
                        write=5.0,
                        pool=10.0
                    )
                    
                    self._client = AsyncClient(
                        base_url=self._base_url,
                        timeout=timeout,
                        limits=httpx.Limits(
                            max_keepalive_connections=5,
                            max_connections=10
                        )
                    )
                    
                    # Test connection with a simple API call
                    response = await self._client.get("/api/version")
                    response.raise_for_status()
                    
                    version_info = response.json()
                    
                    self._is_connected = True
                    
                    model_logger.model_loaded(
                        model_name="ollama_service",
                        model_type="service",
                        load_time=None
                    )
                    
                    logger.info(
                        "Ollama connection established",
                        base_url=self._base_url,
                        version=version_info.get("version", "unknown")
                    )
                    
                    # Refresh available models
                    await self._refresh_available_models()
                    
            except (HTTPStatusError, RequestError, TimeoutException) as e:
                model_logger.model_error("ollama_service", "connect", str(e))
                raise_model_error(
                    f"Failed to connect to Ollama service at {self._base_url}: {e}",
                    model_name="ollama_service",
                    operation="connect",
                    error_code=ErrorCode.MODEL_NOT_AVAILABLE
                )
            except Exception as e:
                model_logger.model_error("ollama_service", "connect", str(e))
                raise_model_error(
                    f"Unexpected error connecting to Ollama: {e}",
                    model_name="ollama_service",
                    operation="connect"
                )
    
    async def disconnect(self) -> None:
        """Close Ollama client connection."""
        if self._client and self._is_connected:
            try:
                await self._client.aclose()
                self._is_connected = False
                logger.info("Ollama connection closed")
            except Exception as e:
                logger.error(f"Error closing Ollama connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Ollama service health check.
        
        Returns:
            Health status information including available models
        """
        if not self._is_connected or not self._client:
            return {
                "status": "disconnected",
                "error": "Not connected to Ollama service"
            }
        
        try:
            start_time = time.time()
            
            # Check service version
            response = await self._client.get("/api/version")
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000
            
            version_info = response.json()
            
            # Get available models
            await self._refresh_available_models()
            
            embedding_models = [
                model.name for model in self._available_models.values()
                if model.is_embedding_model
            ]
            
            language_models = [
                model.name for model in self._available_models.values()
                if model.is_language_model
            ]
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "version": version_info.get("version", "unknown"),
                "base_url": self._base_url,
                "total_models": len(self._available_models),
                "embedding_models": embedding_models,
                "language_models": language_models,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _refresh_available_models(self) -> None:
        """Refresh the list of available models from Ollama."""
        if not self._client:
            return
        
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            self._available_models.clear()
            
            for model_info in models_data.get("models", []):
                name = model_info.get("name", "unknown")
                
                # Determine model type based on name patterns
                model_type = "embedding" if "embed" in name.lower() else "language"
                
                model = OllamaModelInfo(
                    name=name,
                    model_type=model_type,
                    size=model_info.get("size"),
                    digest=model_info.get("digest"),
                    modified_at=model_info.get("modified_at"),
                    details=model_info.get("details", {})
                )
                
                self._available_models[name] = model
            
            logger.debug(
                "Available models refreshed",
                total_models=len(self._available_models),
                embedding_models=len([m for m in self._available_models.values() if m.is_embedding_model]),
                language_models=len([m for m in self._available_models.values() if m.is_language_model])
            )
            
        except Exception as e:
            logger.warning(f"Failed to refresh available models: {e}")
    
    async def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        if not self._is_connected:
            await self.connect()
        
        await self._refresh_available_models()
        return model_name in self._available_models
    
    async def get_available_models(self) -> List[OllamaModelInfo]:
        """
        Get list of available models.
        
        Returns:
            List of available model information
        """
        if not self._is_connected:
            await self.connect()
        
        await self._refresh_available_models()
        return list(self._available_models.values())
    
    async def get_embedding_models(self) -> List[OllamaModelInfo]:
        """
        Get list of available embedding models.
        
        Returns:
            List of embedding model information
        """
        models = await self.get_available_models()
        return [model for model in models if model.is_embedding_model]
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ModelError: If model pull fails
        """
        if not self._is_connected:
            await self.connect()
        
        if not self._client:
            raise_model_error(
                "Ollama client not connected",
                model_name=model_name,
                operation="pull"
            )
        
        try:
            with performance_context("ollama_model_pull", model_name=model_name):
                logger.info(f"Pulling model {model_name} from Ollama registry")
                
                # Start model pull request
                response = await self._client.post(
                    "/api/pull",
                    json={"name": model_name},
                    timeout=600.0  # 10 minutes for model pulls
                )
                response.raise_for_status()
                
                # Process streaming response for pull progress
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            progress_data = json.loads(line)
                            status = progress_data.get("status", "")
                            
                            if "error" in progress_data:
                                raise_model_error(
                                    f"Model pull failed: {progress_data['error']}",
                                    model_name=model_name,
                                    operation="pull"
                                )
                            
                            logger.debug(f"Model pull progress: {status}")
                            
                        except json.JSONDecodeError:
                            continue
                
                # Refresh available models after successful pull
                await self._refresh_available_models()
                
                if model_name in self._available_models:
                    model_logger.model_loaded(
                        model_name=model_name,
                        model_type="pulled_model"
                    )
                    logger.info(f"Model {model_name} pulled successfully")
                    return True
                else:
                    logger.warning(f"Model {model_name} pull completed but not found in available models")
                    return False
                
        except (HTTPStatusError, RequestError, TimeoutException) as e:
            model_logger.model_error(model_name, "pull", str(e))
            raise_model_error(
                f"Failed to pull model {model_name}: {e}",
                model_name=model_name,
                operation="pull"
            )
        except Exception as e:
            model_logger.model_error(model_name, "pull", str(e))
            raise_model_error(
                f"Unexpected error pulling model {model_name}: {e}",
                model_name=model_name,
                operation="pull"
            )
    
    async def generate_embeddings(
        self,
        model_name: str,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s) using specified model.
        
        Args:
            model_name: Name of the embedding model to use
            texts: Single text string or list of texts
            normalize: Whether to normalize embedding vectors
            
        Returns:
            Single embedding vector or list of embedding vectors
            
        Raises:
            ModelError: If embedding generation fails
        """
        if not self._is_connected:
            await self.connect()
        
        if not self._client:
            raise_model_error(
                "Ollama client not connected",
                model_name=model_name,
                operation="embedding"
            )
        
        # Ensure model is available
        if not await self.is_model_available(model_name):
            raise_model_error(
                f"Embedding model {model_name} not available",
                model_name=model_name,
                operation="embedding",
                error_code=ErrorCode.MODEL_NOT_AVAILABLE
            )
        
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = []
        
        try:
            with performance_context(
                "ollama_generate_embeddings",
                model_name=model_name,
                text_count=len(texts)
            ):
                for i, text in enumerate(texts):
                    start_time = time.time()
                    
                    # Make embedding request
                    response = await self._client.post(
                        "/api/embeddings",
                        json={
                            "model": model_name,
                            "prompt": text
                        },
                        timeout=float(self._timeout)
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if "embedding" not in result:
                        raise_model_error(
                            f"No embedding returned for text {i+1}",
                            model_name=model_name,
                            operation="embedding"
                        )
                    
                    embedding = result["embedding"]
                    
                    # Normalize embedding if requested
                    if normalize and embedding:
                        magnitude = sum(x * x for x in embedding) ** 0.5
                        if magnitude > 0:
                            embedding = [x / magnitude for x in embedding]
                    
                    embeddings.append(embedding)
                    
                    duration = time.time() - start_time
                    
                    model_logger.model_inference(
                        model_name=model_name,
                        operation="embedding",
                        input_size=len(text),
                        output_size=len(embedding),
                        duration=duration
                    )
                    
                    logger.debug(
                        "Embedding generated",
                        model_name=model_name,
                        text_index=i,
                        text_length=len(text),
                        embedding_dimensions=len(embedding),
                        duration=duration
                    )
        
        except (HTTPStatusError, RequestError, TimeoutException) as e:
            model_logger.model_error(model_name, "embedding", str(e))
            raise_model_error(
                f"Failed to generate embeddings with {model_name}: {e}",
                model_name=model_name,
                operation="embedding"
            )
        except Exception as e:
            model_logger.model_error(model_name, "embedding", str(e))
            raise_model_error(
                f"Unexpected error generating embeddings: {e}",
                model_name=model_name,
                operation="embedding"
            )
        
        # Return single embedding for single input
        if single_input:
            return embeddings[0] if embeddings else []
        
        return embeddings
    
    async def generate_embeddings_batch(
        self,
        model_name: str,
        texts: List[str],
        batch_size: int = 10,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with automatic batching.
        
        Args:
            model_name: Name of the embedding model to use
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            normalize: Whether to normalize embedding vectors
            
        Returns:
            List of embedding vectors corresponding to input texts
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            logger.debug(
                "Processing embedding batch",
                model_name=model_name,
                batch_start=i,
                batch_size=len(batch),
                total_texts=len(texts)
            )
            
            batch_embeddings = await self.generate_embeddings(
                model_name=model_name,
                texts=batch,
                normalize=normalize
            )
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid overwhelming the service
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    async def validate_model_availability(self, model_name: str, auto_pull: bool = False) -> bool:
        """
        Validate that a model is available, optionally pulling it if not found.
        
        Args:
            model_name: Name of the model to validate
            auto_pull: Whether to automatically pull the model if not available
            
        Returns:
            True if model is available (after pulling if needed), False otherwise
        """
        try:
            if await self.is_model_available(model_name):
                return True
            
            if auto_pull:
                logger.info(f"Model {model_name} not found, attempting to pull")
                return await self.pull_model(model_name)
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[OllamaModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information if available, None otherwise
        """
        if not self._is_connected:
            await self.connect()
        
        await self._refresh_available_models()
        return self._available_models.get(model_name)


# Global Ollama client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """
    Get the global Ollama client instance.
    
    Returns:
        OllamaClient instance
    """
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def init_ollama() -> None:
    """Initialize Ollama client and validate configured models."""
    client = get_ollama_client()
    await client.connect()
    
    settings = get_settings()
    
    # Validate primary embedding model
    primary_model = settings.ollama.embedding_model
    if not await client.validate_model_availability(primary_model, auto_pull=False):
        logger.warning(f"Primary embedding model {primary_model} not available")
    
    # Validate fallback embedding model
    fallback_model = settings.ollama.fallback_model
    if not await client.validate_model_availability(fallback_model, auto_pull=False):
        logger.warning(f"Fallback embedding model {fallback_model} not available")


async def close_ollama() -> None:
    """Close Ollama client connection."""
    global _ollama_client
    if _ollama_client:
        await _ollama_client.disconnect()
        _ollama_client = None


@asynccontextmanager
async def get_ollama():
    """
    Async context manager for Ollama operations.
    
    Usage:
        async with get_ollama() as client:
            embeddings = await client.generate_embeddings("mxbai-embed-large", "text")
    """
    client = get_ollama_client()
    if not client._is_connected:
        await client.connect()
    
    try:
        yield client
    except Exception as e:
        logger.error(f"Ollama operation error: {e}")
        raise


# FastAPI dependency function
async def get_ollama_client_dependency() -> OllamaClient:
    """FastAPI dependency to get Ollama client."""
    client = get_ollama_client()
    if not client._is_connected:
        await client.connect()
    return client


# Convenience functions for common operations

async def generate_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generate a single embedding using the configured model.
    
    Args:
        text: Text to embed
        model_name: Model to use (uses configured primary model if None)
        
    Returns:
        Embedding vector
    """
    if model_name is None:
        settings = get_settings()
        model_name = settings.ollama.embedding_model
    
    async with get_ollama() as client:
        return await client.generate_embeddings(model_name, text)


async def generate_embeddings_for_chunks(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 10
) -> List[List[float]]:
    """
    Generate embeddings for multiple text chunks with batching.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use (uses configured primary model if None)
        batch_size: Batch size for processing
        
    Returns:
        List of embedding vectors
    """
    if model_name is None:
        settings = get_settings()
        model_name = settings.ollama.embedding_model
    
    async with get_ollama() as client:
        return await client.generate_embeddings_batch(
            model_name=model_name,
            texts=texts,
            batch_size=batch_size
        )


async def validate_embedding_model(model_name: str) -> bool:
    """
    Validate that an embedding model is available for use.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        True if model is available, False otherwise
    """
    async with get_ollama() as client:
        return await client.validate_model_availability(model_name)


async def get_available_embedding_models() -> List[str]:
    """
    Get list of available embedding model names.
    
    Returns:
        List of embedding model names
    """
    async with get_ollama() as client:
        models = await client.get_embedding_models()
        return [model.name for model in models]