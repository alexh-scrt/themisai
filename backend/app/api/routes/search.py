"""
Search API Routes for LegalAI Document Processing System

This module provides REST API endpoints for search and retrieval operations in the 
Patexia Legal AI Chatbot. It enables semantic search, hybrid search, similarity search,
and comprehensive search analytics with real-time query processing.

Key Features:
- Hybrid search combining semantic and keyword search capabilities
- Semantic vector search using advanced embedding models
- Traditional keyword search with BM25 scoring
- Document similarity search and clustering
- Search history tracking and analytics
- Query suggestions and auto-completion
- Real-time search performance monitoring
- Legal citation search and extraction

Search Operations:
- Search: Primary search interface with configurable modes
- Similarity: Find similar documents or chunks
- Citations: Search for legal citations and references
- Suggestions: Get query suggestions based on history
- History: Track and analyze search patterns
- Analytics: Comprehensive search performance metrics

Search Types:
- Semantic: Vector similarity search using embeddings
- Keyword: Traditional BM25-based text search
- Hybrid: Configurable combination of semantic and keyword
- Citation: Legal citation and reference search
- Similarity: Document/chunk similarity analysis

Business Rules:
- Search scope validation (case, document, global access)
- User access control and case permissions
- Search result ranking and relevance scoring
- Query history tracking and analytics
- Performance optimization and caching

Architecture Integration:
- Uses SearchService for business logic and query processing
- Integrates with VectorRepository for semantic search
- Connects to LlamaIndex query engines for complex reasoning
- Coordinates with SearchHistoryRepository for analytics
- Implements comprehensive error handling and logging
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Path,
    Body,
    BackgroundTasks,
    Request,
    Response,
    status
)
from fastapi.responses import JSONResponse

from ...core.config import get_settings
from ...core.websocket_manager import WebSocketManager, get_websocket_manager
from ...services.search_service import SearchService, SearchOperationResult
from ...services.case_service import CaseService
from ...models.api.search_schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultChunk,
    SimilaritySearchRequest,
    CitationSearchRequest,
    SearchSuggestionRequest,
    SearchHistoryResponse,
    SearchAnalyticsResponse,
    SearchType,
    SearchScope,
    SortBy,
    SortOrder,
    ApiResponse,
    ErrorResponse
)
from ...models.domain.document import DocumentType
from ...utils.logging import (
    get_logger,
    performance_context,
    log_business_event,
    log_route_entry,
    log_route_exit,
    log_search_query
)
from ...exceptions import (
    SearchError,
    CaseManagementError,
    ValidationError,
    ResourceError,
    ErrorCode,
    get_exception_response_data
)
from ..deps import get_search_service, get_case_service


logger = get_logger(__name__)
router = APIRouter()


# Primary Search Operations

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Search Documents",
    description="Perform semantic, keyword, or hybrid search across documents"
)
async def search_documents(
    request: Request,
    search_request: SearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> SearchResponse:
    """Perform document search with configurable search types."""
    log_route_entry(
        request,
        query=search_request.query[:100],  # Truncate for logging
        search_type=search_request.search_type,
        search_scope=search_request.search_scope,
        case_id=search_request.case_id
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "search_documents",
            query=search_request.query[:50],
            search_type=search_request.search_type,
            search_scope=search_request.search_scope
        ):
            # Validate search scope and access
            await _validate_search_access(
                search_request,
                user_id,
                case_service
            )
            
            # Execute search via service
            search_result = await search_service.search_documents(
                search_request,
                user_id
            )
            
            if search_result.success:
                # Log search query for analytics
                log_search_query(
                    query=search_request.query,
                    case_id=search_request.case_id or "global",
                    search_type=search_request.search_type.value,
                    result_count=search_result.response.total_results,
                    duration=search_result.response.search_time_ms / 1000.0,
                    user_id=user_id
                )
                
                # Send search analytics in background
                background_tasks.add_task(
                    _track_search_analytics,
                    search_service,
                    search_request,
                    search_result.response,
                    user_id
                )
                
                log_business_event(
                    "search_executed",
                    request,
                    query=search_request.query[:100],
                    search_type=search_request.search_type,
                    result_count=search_result.response.total_results,
                    case_id=search_request.case_id
                )
                
                log_route_exit(request, search_result.response)
                return search_result.response
            else:
                raise SearchError(
                    message=search_result.error or "Search operation failed",
                    error_code=ErrorCode.SEARCH_ENGINE_ERROR,
                    query=search_request.query,
                    case_id=search_request.case_id
                )
                
    except Exception as exc:
        logger.error(
            "Search operation failed",
            query=search_request.query[:100],
            search_type=search_request.search_type,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (SearchError, ValidationError, CaseManagementError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search operation failed"}
        )


@router.post(
    "/similarity",
    response_model=SearchResponse,
    summary="Find Similar Documents",
    description="Find documents or chunks similar to a reference document or text"
)
async def find_similar_documents(
    request: Request,
    similarity_request: SimilaritySearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service),
    case_service: CaseService = Depends(get_case_service)
) -> SearchResponse:
    """Find similar documents or chunks."""
    log_route_entry(
        request,
        reference_id=similarity_request.reference_document_id or similarity_request.reference_chunk_id,
        reference_text=similarity_request.reference_text[:100] if similarity_request.reference_text else None,
        case_id=similarity_request.case_id
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "similarity_search",
            case_id=similarity_request.case_id,
            has_reference_text=bool(similarity_request.reference_text)
        ):
            # Validate access to case if specified
            if similarity_request.case_id:
                case = await case_service.get_case(similarity_request.case_id, user_id)
                if not case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": "Case not found"}
                    )
            
            # Execute similarity search
            search_result = await search_service.find_similar_documents(
                similarity_request,
                user_id
            )
            
            if search_result.success:
                # Track similarity search analytics
                background_tasks.add_task(
                    _track_similarity_analytics,
                    search_service,
                    similarity_request,
                    search_result.response,
                    user_id
                )
                
                log_business_event(
                    "similarity_search_executed",
                    request,
                    case_id=similarity_request.case_id,
                    result_count=search_result.response.total_results,
                    similarity_threshold=similarity_request.similarity_threshold
                )
                
                log_route_exit(request, search_result.response)
                return search_result.response
            else:
                raise SearchError(
                    message=search_result.error or "Similarity search failed",
                    error_code=ErrorCode.SEARCH_ENGINE_ERROR
                )
                
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Similarity search failed",
            case_id=similarity_request.case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (SearchError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Similarity search failed"}
        )


@router.post(
    "/citations",
    response_model=SearchResponse,
    summary="Search Legal Citations",
    description="Search for legal citations and references within documents"
)
async def search_citations(
    request: Request,
    citation_request: CitationSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    case_service: CaseService = Depends(get_case_service)
) -> SearchResponse:
    """Search for legal citations and references."""
    log_route_entry(
        request,
        citation_query=citation_request.citation_query,
        citation_types=citation_request.citation_types,
        case_id=citation_request.case_id
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "citation_search",
            citation_query=citation_request.citation_query,
            case_id=citation_request.case_id
        ):
            # Validate case access if specified
            if citation_request.case_id:
                case = await case_service.get_case(citation_request.case_id, user_id)
                if not case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": "Case not found"}
                    )
            
            # Execute citation search
            search_result = await search_service.search_citations(
                citation_request,
                user_id
            )
            
            if search_result.success:
                log_business_event(
                    "citation_search_executed",
                    request,
                    citation_query=citation_request.citation_query,
                    case_id=citation_request.case_id,
                    result_count=search_result.response.total_results
                )
                
                log_route_exit(request, search_result.response)
                return search_result.response
            else:
                raise SearchError(
                    message=search_result.error or "Citation search failed",
                    error_code=ErrorCode.SEARCH_ENGINE_ERROR
                )
                
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Citation search failed",
            citation_query=citation_request.citation_query,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (SearchError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Citation search failed"}
        )


# Search Assistance and Analytics

@router.get(
    "/suggestions",
    response_model=ApiResponse,
    summary="Get Search Suggestions",
    description="Get query suggestions based on search history and context"
)
async def get_search_suggestions(
    request: Request,
    partial_query: str = Query(..., description="Partial query text"),
    case_id: Optional[str] = Query(None, description="Case context for suggestions"),
    limit: int = Query(5, description="Maximum number of suggestions", ge=1, le=20),
    search_service: SearchService = Depends(get_search_service),
    case_service: CaseService = Depends(get_case_service)
) -> ApiResponse:
    """Get search query suggestions."""
    log_route_entry(
        request,
        partial_query=partial_query,
        case_id=case_id,
        limit=limit
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "search_suggestions",
            partial_query=partial_query,
            case_id=case_id
        ):
            # Validate case access if specified
            if case_id:
                case = await case_service.get_case(case_id, user_id)
                if not case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": "Case not found"}
                    )
            
            # Get suggestions from service
            suggestions = await search_service.get_search_suggestions(
                partial_query=partial_query,
                user_id=user_id,
                case_id=case_id,
                limit=limit
            )
            
            response_data = ApiResponse(
                success=True,
                message="Search suggestions retrieved",
                data={
                    "partial_query": partial_query,
                    "suggestions": suggestions,
                    "case_id": case_id,
                    "suggestion_count": len(suggestions)
                }
            )
            
            log_route_exit(request, response_data)
            return response_data
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get search suggestions",
            partial_query=partial_query,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get search suggestions"}
        )


@router.get(
    "/history",
    response_model=SearchHistoryResponse,
    summary="Get Search History",
    description="Retrieve search history with filtering and pagination"
)
async def get_search_history(
    request: Request,
    case_id: Optional[str] = Query(None, description="Filter by case ID"),
    search_type: Optional[SearchType] = Query(None, description="Filter by search type"),
    days_back: int = Query(30, description="Number of days to look back", ge=1, le=365),
    limit: int = Query(50, description="Maximum number of entries", ge=1, le=200),
    offset: int = Query(0, description="Number of entries to skip", ge=0),
    search_service: SearchService = Depends(get_search_service)
) -> SearchHistoryResponse:
    """Get search history for the user."""
    log_route_entry(
        request,
        case_id=case_id,
        search_type=search_type,
        days_back=days_back,
        limit=limit,
        offset=offset
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "search_history",
            user_id=user_id,
            days_back=days_back,
            limit=limit
        ):
            # Get search history from service
            history = await search_service.get_search_history(
                user_id=user_id,
                case_id=case_id,
                search_type=search_type,
                days_back=days_back,
                limit=limit,
                offset=offset
            )
            
            log_route_exit(request, history)
            return history
            
    except Exception as exc:
        logger.error(
            "Failed to get search history",
            user_id=user_id,
            case_id=case_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        # Return empty response on error
        return SearchHistoryResponse(
            searches=[],
            total_count=0,
            offset=offset,
            limit=limit,
            has_more=False
        )


@router.get(
    "/analytics",
    response_model=SearchAnalyticsResponse,
    summary="Get Search Analytics",
    description="Get comprehensive search analytics and metrics"
)
async def get_search_analytics(
    request: Request,
    case_id: Optional[str] = Query(None, description="Filter by case ID"),
    timeframe: str = Query(
        "30d",
        description="Analytics timeframe (7d, 30d, 90d, 1y)",
        regex=r"^(7d|30d|90d|1y)$"
    ),
    search_service: SearchService = Depends(get_search_service)
) -> SearchAnalyticsResponse:
    """Get search analytics and metrics."""
    log_route_entry(
        request,
        case_id=case_id,
        timeframe=timeframe
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "search_analytics",
            user_id=user_id,
            case_id=case_id,
            timeframe=timeframe
        ):
            # Get analytics from service
            analytics = await search_service.get_search_analytics(
                user_id=user_id,
                case_id=case_id,
                timeframe=timeframe
            )
            
            log_route_exit(request, analytics)
            return analytics
            
    except Exception as exc:
        logger.error(
            "Failed to get search analytics",
            user_id=user_id,
            case_id=case_id,
            timeframe=timeframe,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get search analytics"}
        )


# Search Configuration and Testing

@router.post(
    "/test",
    response_model=ApiResponse,
    summary="Test Search Configuration",
    description="Test search functionality with different configurations"
)
async def test_search_configuration(
    request: Request,
    test_query: str = Body(..., description="Test query to execute"),
    search_configs: List[Dict[str, Any]] = Body(
        ...,
        description="List of search configurations to test"
    ),
    case_id: Optional[str] = Body(None, description="Case ID for testing"),
    search_service: SearchService = Depends(get_search_service),
    case_service: CaseService = Depends(get_case_service)
) -> ApiResponse:
    """Test different search configurations."""
    log_route_entry(
        request,
        test_query=test_query,
        config_count=len(search_configs),
        case_id=case_id
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "search_configuration_test",
            test_query=test_query,
            config_count=len(search_configs)
        ):
            # Validate case access if specified
            if case_id:
                case = await case_service.get_case(case_id, user_id)
                if not case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": "Case not found"}
                    )
            
            # Test search configurations
            test_results = await search_service.test_search_configurations(
                test_query=test_query,
                search_configs=search_configs,
                case_id=case_id,
                user_id=user_id
            )
            
            log_business_event(
                "search_configuration_tested",
                request,
                test_query=test_query,
                config_count=len(search_configs),
                case_id=case_id
            )
            
            response_data = ApiResponse(
                success=True,
                message="Search configuration test completed",
                data={
                    "test_query": test_query,
                    "configurations_tested": len(search_configs),
                    "results": test_results,
                    "case_id": case_id
                }
            )
            
            log_route_exit(request, response_data)
            return response_data
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Search configuration test failed",
            test_query=test_query,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search configuration test failed"}
        )


@router.get(
    "/popular-queries",
    response_model=ApiResponse,
    summary="Get Popular Queries",
    description="Get most popular search queries based on frequency and success"
)
async def get_popular_queries(
    request: Request,
    case_id: Optional[str] = Query(None, description="Filter by case ID"),
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    limit: int = Query(10, description="Maximum number of queries", ge=1, le=50),
    min_frequency: int = Query(2, description="Minimum query frequency", ge=1),
    search_service: SearchService = Depends(get_search_service)
) -> ApiResponse:
    """Get popular search queries."""
    log_route_entry(
        request,
        case_id=case_id,
        days_back=days_back,
        limit=limit,
        min_frequency=min_frequency
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "popular_queries",
            user_id=user_id,
            days_back=days_back,
            limit=limit
        ):
            # Get popular queries from service
            popular_queries = await search_service.get_popular_queries(
                user_id=user_id,
                case_id=case_id,
                days_back=days_back,
                limit=limit,
                min_frequency=min_frequency
            )
            
            response_data = ApiResponse(
                success=True,
                message="Popular queries retrieved",
                data={
                    "popular_queries": popular_queries,
                    "analysis_period_days": days_back,
                    "query_count": len(popular_queries),
                    "case_id": case_id
                }
            )
            
            log_route_exit(request, response_data)
            return response_data
            
    except Exception as exc:
        logger.error(
            "Failed to get popular queries",
            user_id=user_id,
            case_id=case_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get popular queries"}
        )


# Helper Functions

async def _validate_search_access(
    search_request: SearchRequest,
    user_id: str,
    case_service: CaseService
) -> None:
    """Validate user access to search scope."""
    # Validate case access if searching within a specific case
    if search_request.search_scope == SearchScope.CASE:
        if not search_request.case_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Case ID required for case-scoped search"}
            )
        
        case = await case_service.get_case(search_request.case_id, user_id)
        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Case not found or access denied"}
            )
    
    # Validate document access if searching within specific documents
    if search_request.search_scope == SearchScope.DOCUMENT:
        if not search_request.document_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Document IDs required for document-scoped search"}
            )
        
        # Additional document access validation would go here
        # For now, we assume document access is controlled by case access


async def _track_search_analytics(
    search_service: SearchService,
    search_request: SearchRequest,
    search_response: SearchResponse,
    user_id: str
) -> None:
    """Track search analytics in background."""
    try:
        await search_service.track_search_execution(
            search_request=search_request,
            search_response=search_response,
            user_id=user_id
        )
    except Exception as exc:
        logger.warning(f"Failed to track search analytics: {exc}")


async def _track_similarity_analytics(
    search_service: SearchService,
    similarity_request: SimilaritySearchRequest,
    search_response: SearchResponse,
    user_id: str
) -> None:
    """Track similarity search analytics in background."""
    try:
        await search_service.track_similarity_search(
            similarity_request=similarity_request,
            search_response=search_response,
            user_id=user_id
        )
    except Exception as exc:
        logger.warning(f"Failed to track similarity analytics: {exc}")


# Export router for main application
__all__ = ["router"]