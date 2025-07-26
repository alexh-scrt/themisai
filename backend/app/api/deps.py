"""
Dependency injection module for API routes.

This module provides dependency injection functions for FastAPI routes,
ensuring proper service instantiation and dependency management.
"""

from ..services.case_service import CaseService
from ..services.document_service import DocumentService
from ..services.search_service import SearchService
from ..services.embedding_service import EmbeddingService
from ..repositories.mongodb.case_repository import get_case_repository
from ..repositories.mongodb.document_repository import get_document_repository
from ..repositories.weaviate.vector_repository import get_vector_repository
from ..core.database import get_database_manager
from ..core.websocket_manager import get_websocket_manager
from ..core.ollama_client import get_ollama_client


async def get_case_service() -> CaseService:
    """Get case service instance."""
    return CaseService(
        case_repository=get_case_repository(),
        embedding_service=get_embedding_service()
    )


async def get_document_service() -> DocumentService:
    """Get document service instance."""
    return DocumentService()


async def get_search_service() -> SearchService:
    """Get search service instance.""" 
    return SearchService()


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService()