"""
Dependency injection module for API routes.

This module provides dependency injection functions for FastAPI routes,
ensuring proper service instantiation and dependency management.
"""

import threading
from typing import Optional, Any
from ..utils.logging import get_logger
from ..services.config_service import ConfigurationService
from ..core.component_manager import ComponentConfigurationManager

from ..services.case_service import CaseService
from ..services.document_service import DocumentService
from ..services.search_service import SearchService
from ..services.embedding_service import EmbeddingService
from ..core.websocket_manager import get_websocket_manager
from ..core.ollama_client import get_ollama_client
from ..services.config_service import ConfigurationService
from ..core.component_manager import ComponentConfigurationManager
from ..core.config_watcher import ConfigurationWatcher  # If implemented

import threading
import asyncio
from typing import Optional, Any, Dict
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Global instances for dependency injection
_config_service_instance: Optional[ConfigurationService] = None
_config_service_lock = threading.Lock()


from ..services.config_service import ConfigurationService
from ..core.component_manager import ComponentConfigurationManager
from ..core.config_watcher import ConfigurationWatcher  # If implemented


# Global instances for dependency injection
_config_service_instance: Optional[ConfigurationService] = None
_config_service_lock = threading.Lock()

_document_service_instance: Optional[Any] = None
_document_service_lock = threading.Lock()

_search_service_instance: Optional[Any] = None
_search_service_lock = threading.Lock()

_embedding_service_instance: Optional[Any] = None
_embedding_service_lock = threading.Lock()

_notification_service_instance: Optional[Any] = None
_notification_service_lock = threading.Lock()


async def get_config_service() -> ConfigurationService:
    """
    Get configuration service instance (FastAPI dependency).
    
    This function provides a singleton ConfigurationService instance that can be
    used as a FastAPI dependency for dependency injection. It ensures that
    the same configuration service is used across all route handlers.
    
    Returns:
        ConfigurationService: The global configuration service instance
        
    Raises:
        RuntimeError: If the configuration service has not been initialized
        
    Usage:
        @router.post("/admin/config")
        async def update_config(
            request: ConfigurationChangeRequest,
            config_service: ConfigurationService = Depends(get_config_service)
        ):
            return await config_service.apply_configuration_changes(request)
    """
    global _config_service_instance
    
    if _config_service_instance is None:
        # Initialize configuration service if not already done
        _config_service_instance = await initialize_config_service()
    
    return _config_service_instance


async def initialize_config_service(
    websocket_manager: Optional[Any] = None,
    config_watcher: Optional[Any] = None
) -> ConfigurationService:
    """
    Initialize and return a new ConfigurationService instance.
    
    This function creates a new ConfigurationService with the specified dependencies
    and initializes it. It's typically called during application startup.
    
    Args:
        websocket_manager: Optional WebSocket manager for real-time notifications
        config_watcher: Optional configuration file watcher
        
    Returns:
        ConfigurationService: Initialized configuration service instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    global _config_service_instance
    
    with _config_service_lock:
        if _config_service_instance is not None:
            return _config_service_instance
        
        try:
            # Get WebSocket manager if not provided
            if websocket_manager is None:
                try:
                    websocket_manager = get_websocket_manager()
                except RuntimeError:
                    # WebSocket manager not initialized yet, will use None
                    websocket_manager = None
            
            # Initialize configuration service
            service = ConfigurationService(
                websocket_manager=websocket_manager,
                config_watcher=config_watcher
            )
            
            _config_service_instance = service
            
            logger.info("ConfigurationService initialized successfully")
            return service
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationService: {e}")
            raise RuntimeError(f"ConfigurationService initialization failed: {e}")


def set_config_service(service: ConfigurationService) -> None:
    """
    Set the global ConfigurationService instance.
    
    This function is typically called during application startup to set the
    ConfigurationService instance that will be used throughout the application.
    
    Args:
        service: The ConfigurationService instance to set as global
        
    Thread Safety:
        This function is thread-safe and uses a lock to prevent race conditions
        during initialization.
    """
    global _config_service_instance
    
    with _config_service_lock:
        _config_service_instance = service
        logger.info("Global ConfigurationService instance set")


async def cleanup_all_services() -> None:
    """
    Clean up all global service instances.
    
    This function is typically called during application shutdown to properly
    clean up all service resources.
    """
    global _config_service_instance, _document_service_instance, _search_service_instance
    global _embedding_service_instance, _notification_service_instance
    
    cleanup_tasks = []
    
    # Cleanup configuration service
    if _config_service_instance is not None:
        cleanup_tasks.append(_cleanup_service("ConfigurationService", _config_service_instance))
    
    # Cleanup document service
    if _document_service_instance is not None:
        cleanup_tasks.append(_cleanup_service("DocumentService", _document_service_instance))
    
    # Cleanup search service
    if _search_service_instance is not None:
        cleanup_tasks.append(_cleanup_service("SearchService", _search_service_instance))
    
    # Cleanup embedding service
    if _embedding_service_instance is not None:
        cleanup_tasks.append(_cleanup_service("EmbeddingService", _embedding_service_instance))
    
    # Cleanup notification service
    if _notification_service_instance is not None:
        cleanup_tasks.append(_cleanup_service("NotificationService", _notification_service_instance))
    
    # Run all cleanup tasks concurrently
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    # Clear all instances
    with _config_service_lock:
        _config_service_instance = None
    with _document_service_lock:
        _document_service_instance = None
    with _search_service_lock:
        _search_service_instance = None
    with _embedding_service_lock:
        _embedding_service_instance = None
    with _notification_service_lock:
        _notification_service_instance = None
    
    logger.info("All services cleaned up successfully")


async def _cleanup_service(service_name: str, service_instance: Any) -> None:
    """Helper function to cleanup individual service."""
    try:
        if hasattr(service_instance, 'cleanup'):
            await service_instance.cleanup()
        elif hasattr(service_instance, 'stop'):
            await service_instance.stop()
        elif hasattr(service_instance, 'close'):
            await service_instance.close()
        
        logger.info(f"{service_name} cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during {service_name} cleanup: {e}")


def get_service_status() -> Dict[str, bool]:
    """
    Get initialization status of all services.
    
    Returns:
        Dict mapping service names to their initialization status
    """
    return {
        "config_service": _config_service_instance is not None,
        "document_service": _document_service_instance is not None,
        "search_service": _search_service_instance is not None,
        "embedding_service": _embedding_service_instance is not None,
        "notification_service": _notification_service_instance is not None,
    }


def is_config_service_initialized() -> bool:
    """
    Check if the ConfigurationService has been initialized.
    
    Returns:
        bool: True if ConfigurationService is initialized, False otherwise
    """
    return _config_service_instance is not None


# Additional dependency functions for other missing services

async def get_notification_service():
    """Get notification service instance."""
    from ..services.notification_service import NotificationService
    
    # Get WebSocket manager for notifications
    websocket_manager = get_websocket_manager()
    
    return NotificationService(
        websocket_manager=websocket_manager,
        max_pending_notifications=10000,
        default_expiry_hours=24
    )


async def get_component_manager():
    """Get component configuration manager instance."""
    return ComponentConfigurationManager()


# Enhanced service factories with proper dependency injection (now using singletons)

async def get_case_service() -> CaseService:
    """Get case service instance with full dependencies."""
    from ..repositories.mongodb.case_repository import CaseRepository
    from ..repositories.weaviate.vector_repository import VectorRepository
    
    # Note: CaseService can be instantiated fresh each time as it's lightweight
    # and doesn't hold expensive resources like database connections
    
    # Initialize repositories
    case_repository = CaseRepository()
    vector_repository = VectorRepository()
    
    # Get WebSocket manager for real-time notifications
    try:
        websocket_manager = get_websocket_manager()
    except RuntimeError:
        websocket_manager = None
    
    return CaseService(
        case_repository=case_repository,
        vector_repository=vector_repository,
        websocket_manager=websocket_manager
    )


async def get_document_service() -> DocumentService:
    """Get document service instance with full dependencies."""
    from ..services.document_service import DocumentService
    from ..processors.document_processor import DocumentProcessor
    from ..repositories.mongodb.document_repository import DocumentRepository
    from ..repositories.weaviate.vector_repository import VectorRepository
    
    # Initialize dependencies
    document_repository = DocumentRepository()
    vector_repository = VectorRepository()
    websocket_manager = get_websocket_manager()
    embedding_service = await get_embedding_service()
    document_processor = DocumentProcessor(
        embedding_service=embedding_service,        # ✅ Required
        vector_repository=vector_repository,        # ✅ Required
        document_repository=document_repository,    # ✅ Required
        websocket_manager=websocket_manager,        # ✅ Required
        settings=None                               # ✅ Optional (can be None)
    )
    
    try:
        websocket_manager = get_websocket_manager()
    except RuntimeError:
        websocket_manager = None
    
    return DocumentService(
        document_processor=document_processor,
        document_repository=document_repository,
        vector_repository=vector_repository,
        embedding_service=embedding_service,
        websocket_manager=websocket_manager
    )


async def get_search_service() -> SearchService:
    """Get search service instance with full dependencies."""
    from ..repositories.weaviate.vector_repository import VectorRepository
    from ..repositories.mongodb.document_repository import DocumentRepository
    from ..repositories.mongodb.search_history_repository import SearchHistoryRepository
    
    # Initialize dependencies
    vector_repository = VectorRepository()
    document_repository = DocumentRepository()
    search_history_repository = SearchHistoryRepository()
    embedding_service = await get_embedding_service()
    
    try:
        notification_service = await get_notification_service()
    except Exception:
        notification_service = None
    
    return SearchService(
        vector_repository=vector_repository,
        document_repository=document_repository,
        search_history_repository=search_history_repository,
        embedding_service=embedding_service,
        notification_service=notification_service
    )


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance with Ollama client."""
    from ..services.embedding_service import EmbeddingService
    
    # Get Ollama client
    ollama_client = get_ollama_client()
    
    try:
        websocket_manager = get_websocket_manager()
    except RuntimeError:
        websocket_manager = None
    
    return EmbeddingService(
        ollama_client=ollama_client,
        websocket_manager=websocket_manager
    )


# Repository factory functions

def get_case_repository():
    """Get case repository instance."""
    from ..repositories.mongodb.case_repository import CaseRepository
    return CaseRepository()


def get_document_repository():
    """Get document repository instance."""
    from ..repositories.mongodb.document_repository import DocumentRepository
    return DocumentRepository()


def get_vector_repository():
    """Get vector repository instance."""
    from ..repositories.weaviate.vector_repository import VectorRepository
    return VectorRepository()


def get_search_history_repository():
    """Get search history repository instance."""
    from ..repositories.mongodb.search_history_repository import SearchHistoryRepository
    return SearchHistoryRepository()


async def cleanup_config_service() -> None:
    """
    Clean up the global ConfigurationService instance.
    
    This function is typically called during application shutdown to properly
    clean up configuration service resources.
    """
    global _config_service_instance
    
    if _config_service_instance is not None:
        try:
            # Perform any necessary cleanup
            if hasattr(_config_service_instance, 'cleanup'):
                await _config_service_instance.cleanup()
            
            logger.info("ConfigurationService cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during ConfigurationService cleanup: {e}")
        finally:
            with _config_service_lock:
                _config_service_instance = None

