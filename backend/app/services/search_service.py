"""
Search Service - Business Logic Layer

This module provides the business logic layer for search and retrieval operations
in the Patexia Legal AI Chatbot. It orchestrates search workflows, manages query
processing, enforces business rules, and provides service layer abstraction for
semantic and hybrid search across legal documents.

Key Features:
- Multi-type search operations (semantic, keyword, hybrid, citation)
- Query processing and optimization for legal document search
- Result ranking and relevance scoring with legal context awareness
- Search history tracking and analytics for user behavior analysis
- Citation extraction and legal reference identification
- Document highlighting and context generation
- Performance monitoring and query optimization
- Business rule enforcement for search access control

Search Types:
- Semantic search: Vector similarity using embedding models
- Keyword search: Traditional BM25 with legal term weighting
- Hybrid search: Optimal combination of semantic and keyword search
- Citation search: Legal citation pattern matching and extraction
- Similarity search: Find documents similar to a reference document

Business Rules:
- Case-based access control and document isolation
- Search result filtering based on user permissions
- Query validation and sanitization for security
- Rate limiting and performance optimization
- Search analytics and usage tracking
- Result ranking with legal relevance factors
- Citation validation and legal reference verification

Architecture Integration:
- Integrates with VectorRepository for vector search operations
- Uses EmbeddingService for query vectorization
- Coordinates with DocumentRepository for metadata enrichment
- Employs NotificationService for progress tracking
- Implements search history persistence and analytics
- Provides service layer abstraction for API controllers
"""

import asyncio
import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from config.settings import get_settings
from ..models.api.search_schemas import (
    SearchRequest, SearchResponse, SearchResultItem, SearchHighlight,
    CitationSearchRequest, SearchHistoryEntry, SearchType, SearchScope,
    SortBy, SortOrder
)
from ..models.domain.document import DocumentChunk, DocumentType
from ..models.domain.case import LegalCase
from ..repositories.weaviate.vector_repository import VectorRepository
from ..repositories.mongodb.document_repository import DocumentRepository
from ..repositories.mongodb.search_history_repository import SearchHistoryRepository
from ..services.embedding_service import EmbeddingService
from ..services.notification_service import NotificationService
from ..core.exceptions import (
    SearchError, ValidationError, AccessError, PerformanceError,
    ErrorCode, raise_search_error, raise_validation_error, raise_access_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class SearchOperation(str, Enum):
    """Types of search operations for audit tracking."""
    SEARCH = "search"
    CITATION_SEARCH = "citation_search"
    SIMILARITY_SEARCH = "similarity_search"
    SUGGESTION = "suggestion"
    HISTORY_LOOKUP = "history_lookup"


class RankingStrategy(str, Enum):
    """Result ranking strategies."""
    HYBRID_SCORE = "hybrid_score"
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    LEGAL_RELEVANCE = "legal_relevance"
    TEMPORAL = "temporal"
    CITATION_WEIGHT = "citation_weight"


@dataclass
class SearchMetrics:
    """Search performance and analytics metrics."""
    total_searches: int = 0
    total_results: int = 0
    average_execution_time_ms: float = 0.0
    average_results_per_search: float = 0.0
    search_type_distribution: Dict[SearchType, int] = field(default_factory=dict)
    popular_queries: Dict[str, int] = field(default_factory=dict)
    average_relevance_score: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class QueryContext:
    """Context information for query processing."""
    user_id: str
    case_id: Optional[str]
    query_text: str
    search_type: SearchType
    search_scope: SearchScope
    timestamp: datetime
    session_id: Optional[str] = None
    client_ip: Optional[str] = None


@dataclass
class SearchResult:
    """Internal search result representation."""
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    relevance_score: float
    chunk_index: int
    start_char: int
    end_char: int
    section_title: Optional[str]
    page_number: Optional[int]
    legal_citations: List[str]
    file_type: DocumentType
    created_at: datetime
    highlights: List[str] = field(default_factory=list)
    citation_matches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchService:
    """
    Business logic service for search and retrieval operations.
    
    Orchestrates search workflows, manages query processing, enforces
    business rules, and provides service layer abstraction for semantic
    and hybrid search across legal documents.
    """
    
    def __init__(
        self,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository,
        search_history_repository: SearchHistoryRepository,
        embedding_service: EmbeddingService,
        notification_service: Optional[NotificationService] = None
    ):
        """
        Initialize search service with required dependencies.
        
        Args:
            vector_repository: Weaviate repository for vector operations
            document_repository: MongoDB repository for document metadata
            search_history_repository: MongoDB repository for search history
            embedding_service: Service for query vectorization
            notification_service: Optional service for progress notifications
        """
        self.vector_repository = vector_repository
        self.document_repository = document_repository
        self.search_history_repository = search_history_repository
        self.embedding_service = embedding_service
        self.notification_service = notification_service
        
        # Load configuration
        self.settings = get_settings()
        
        # Performance and caching
        self._query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)
        self._max_cache_size = 1000
        
        # Analytics and metrics
        self.metrics = SearchMetrics()
        self._recent_queries: List[Tuple[str, datetime]] = []
        
        # Legal citation patterns
        self._citation_patterns = self._initialize_citation_patterns()
        
        # Rate limiting
        self._rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self._max_searches_per_minute = 60
        
        logger.info(
            "SearchService initialized",
            cache_ttl_minutes=self._cache_ttl.total_seconds() / 60,
            max_cache_size=self._max_cache_size
        )
    
    async def search(
        self,
        request: SearchRequest,
        user_id: str,
        client_ip: Optional[str] = None
    ) -> SearchResponse:
        """
        Perform a search operation with comprehensive result processing.
        
        Args:
            request: Search request with parameters
            user_id: User identifier for access control
            client_ip: Client IP for rate limiting
            
        Returns:
            Search response with ranked results
            
        Raises:
            SearchError: If search execution fails
            ValidationError: If request validation fails
            AccessError: If user lacks access permissions
        """
        async with performance_context("search_service_search", search_type=request.search_type.value):
            # Validate request and check access
            await self._validate_search_request(request, user_id)
            await self._check_rate_limit(user_id, client_ip)
            
            # Create query context
            context = QueryContext(
                user_id=user_id,
                case_id=request.case_id,
                query_text=request.query,
                search_type=request.search_type,
                search_scope=request.search_scope,
                timestamp=datetime.now(timezone.utc),
                client_ip=client_ip
            )
            
            start_time = time.time()
            search_id = str(uuid.uuid4())
            
            try:
                # Check cache first
                cache_key = self._generate_cache_key(request, user_id)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    logger.debug(
                        "Returning cached search result",
                        search_id=search_id,
                        cache_key=cache_key[:16] + "..."
                    )
                    self.metrics.cache_hit_rate += 1
                    return cached_result
                
                # Execute search based on type
                raw_results = await self._execute_search(request, context)
                
                # Process and rank results
                processed_results = await self._process_search_results(
                    raw_results, request, context
                )
                
                # Generate response
                execution_time_ms = (time.time() - start_time) * 1000
                
                response = SearchResponse(
                    search_id=search_id,
                    query=request.query,
                    search_type=request.search_type,
                    results=processed_results,
                    total_results=len(processed_results),
                    execution_time_ms=execution_time_ms,
                    has_more=len(raw_results) > request.limit,
                    metadata={
                        "search_scope": request.search_scope.value,
                        "semantic_weight": request.semantic_weight,
                        "keyword_weight": request.keyword_weight,
                        "similarity_threshold": request.similarity_threshold
                    }
                )
                
                # Cache result
                self._cache_result(cache_key, response)
                
                # Record search history
                await self._record_search_history(
                    search_id, request, context, len(processed_results), execution_time_ms
                )
                
                # Update metrics
                self._update_search_metrics(request, len(processed_results), execution_time_ms)
                
                # Send notification if long-running
                if execution_time_ms > 5000 and self.notification_service:
                    await self.notification_service.send_notification(
                        user_id=user_id,
                        notification_type="search_completed",
                        context={
                            "query": request.query,
                            "results_count": len(processed_results),
                            "execution_time_ms": execution_time_ms
                        }
                    )
                
                logger.info(
                    "Search completed successfully",
                    search_id=search_id,
                    query=request.query[:50],
                    search_type=request.search_type.value,
                    results_count=len(processed_results),
                    execution_time_ms=execution_time_ms
                )
                
                return response
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Record failed search
                await self._record_search_history(
                    search_id, request, context, 0, execution_time_ms, str(e)
                )
                
                self.metrics.error_rate += 1
                
                logger.error(
                    "Search failed",
                    search_id=search_id,
                    query=request.query[:50],
                    error=str(e),
                    execution_time_ms=execution_time_ms
                )
                
                raise_search_error(
                    f"Search execution failed: {str(e)}",
                    ErrorCode.SEARCH_EXECUTION_FAILED,
                    {"search_id": search_id, "query": request.query[:100]}
                )
    
    async def citation_search(
        self,
        request: CitationSearchRequest,
        user_id: str
    ) -> SearchResponse:
        """
        Search for legal citations across documents.
        
        Args:
            request: Citation search request
            user_id: User identifier for access control
            
        Returns:
            Search response with citation matches
        """
        async with performance_context("citation_search", citation=request.citation_query):
            # Validate access to case
            if request.case_id:
                await self._validate_case_access(request.case_id, user_id)
            
            # Extract and normalize citation
            normalized_citation = self._normalize_citation(request.citation_query)
            
            # Build citation search patterns
            search_patterns = self._build_citation_patterns(
                normalized_citation, request.exact_match, request.include_variants
            )
            
            start_time = time.time()
            search_id = str(uuid.uuid4())
            
            try:
                # Search across documents
                raw_results = []
                
                if request.case_id:
                    # Search within specific case
                    raw_results = await self._search_citations_in_case(
                        request.case_id, search_patterns, request.limit
                    )
                else:
                    # Search across all accessible cases
                    raw_results = await self._search_citations_global(
                        user_id, search_patterns, request.limit
                    )
                
                # Process citation results
                processed_results = await self._process_citation_results(
                    raw_results, normalized_citation, request
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                response = SearchResponse(
                    search_id=search_id,
                    query=request.citation_query,
                    search_type=SearchType.CITATION,
                    results=processed_results,
                    total_results=len(processed_results),
                    execution_time_ms=execution_time_ms,
                    has_more=False,
                    metadata={
                        "normalized_citation": normalized_citation,
                        "exact_match": request.exact_match,
                        "include_variants": request.include_variants,
                        "patterns_used": len(search_patterns)
                    }
                )
                
                logger.info(
                    "Citation search completed",
                    search_id=search_id,
                    citation=request.citation_query,
                    results_count=len(processed_results),
                    execution_time_ms=execution_time_ms
                )
                
                return response
                
            except Exception as e:
                logger.error(
                    "Citation search failed",
                    search_id=search_id,
                    citation=request.citation_query,
                    error=str(e)
                )
                raise
    
    async def find_similar_documents(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.8
    ) -> SearchResponse:
        """
        Find documents similar to a reference document.
        
        Args:
            document_id: Reference document ID
            case_id: Case containing the reference document
            user_id: User identifier for access control
            limit: Maximum number of similar documents
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search response with similar documents
        """
        async with performance_context("similarity_search", document_id=document_id):
            # Validate access
            await self._validate_case_access(case_id, user_id)
            
            # Get reference document embeddings
            reference_chunks = await self.document_repository.get_document_chunks(document_id)
            if not reference_chunks:
                raise_search_error(
                    f"Document {document_id} has no processable chunks",
                    ErrorCode.SEARCH_NO_REFERENCE_DATA,
                    {"document_id": document_id}
                )
            
            # Calculate average embedding for document
            embeddings = [chunk.embedding_vector for chunk in reference_chunks if chunk.embedding_vector]
            if not embeddings:
                raise_search_error(
                    f"Document {document_id} has no embeddings",
                    ErrorCode.SEARCH_NO_EMBEDDINGS,
                    {"document_id": document_id}
                )
            
            # Average embeddings to represent document
            import numpy as np
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            
            start_time = time.time()
            search_id = str(uuid.uuid4())
            
            try:
                # Find similar chunks
                similar_chunks = await self.vector_repository.get_similar_chunks(
                    case_id=case_id,
                    reference_embedding=avg_embedding,
                    exclude_chunk_ids=[chunk.chunk_id for chunk in reference_chunks],
                    limit=limit * 3,  # Get more chunks to find diverse documents
                    min_score=similarity_threshold
                )
                
                # Group by document and select best matches
                document_scores = defaultdict(list)
                for chunk in similar_chunks:
                    document_scores[chunk["document_id"]].append(chunk["similarity_score"])
                
                # Calculate document-level scores
                similar_documents = []
                for doc_id, scores in document_scores.items():
                    if doc_id != document_id:  # Exclude reference document
                        avg_score = sum(scores) / len(scores)
                        max_score = max(scores)
                        combined_score = (avg_score * 0.7) + (max_score * 0.3)
                        
                        similar_documents.append({
                            "document_id": doc_id,
                            "similarity_score": combined_score,
                            "chunk_count": len(scores),
                            "max_chunk_score": max_score
                        })
                
                # Sort by similarity and limit
                similar_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
                similar_documents = similar_documents[:limit]
                
                # Convert to search results
                processed_results = await self._convert_similarity_results(
                    similar_documents, case_id
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                response = SearchResponse(
                    search_id=search_id,
                    query=f"Similar to document: {document_id}",
                    search_type=SearchType.SIMILARITY,
                    results=processed_results,
                    total_results=len(processed_results),
                    execution_time_ms=execution_time_ms,
                    has_more=False,
                    metadata={
                        "reference_document_id": document_id,
                        "similarity_threshold": similarity_threshold,
                        "chunks_analyzed": len(reference_chunks)
                    }
                )
                
                logger.info(
                    "Similarity search completed",
                    search_id=search_id,
                    reference_document=document_id,
                    similar_documents=len(processed_results),
                    execution_time_ms=execution_time_ms
                )
                
                return response
                
            except Exception as e:
                logger.error(
                    "Similarity search failed",
                    search_id=search_id,
                    reference_document=document_id,
                    error=str(e)
                )
                raise
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        user_id: str,
        case_id: Optional[str] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Get search suggestions based on partial query and history.
        
        Args:
            partial_query: Partial search query
            user_id: User identifier
            case_id: Optional case context
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search queries
        """
        async with performance_context("search_suggestions", partial_query=partial_query):
            try:
                # Get user search history
                history_suggestions = await self.search_history_repository.get_query_suggestions(
                    user_id=user_id,
                    partial_query=partial_query,
                    case_id=case_id,
                    limit=limit // 2
                )
                
                # Get global popular queries
                popular_suggestions = await self.search_history_repository.get_popular_queries(
                    partial_query=partial_query,
                    case_id=case_id,
                    limit=limit // 2
                )
                
                # Combine and deduplicate
                all_suggestions = []
                seen_queries = set()
                
                # Prioritize user history
                for suggestion in history_suggestions:
                    query = suggestion["query"].lower().strip()
                    if query not in seen_queries and len(query) > len(partial_query):
                        all_suggestions.append(suggestion["query"])
                        seen_queries.add(query)
                
                # Add popular queries
                for suggestion in popular_suggestions:
                    query = suggestion["query"].lower().strip()
                    if query not in seen_queries and len(query) > len(partial_query):
                        all_suggestions.append(suggestion["query"])
                        seen_queries.add(query)
                
                # Add legal-specific suggestions if applicable
                legal_suggestions = self._generate_legal_suggestions(partial_query)
                for suggestion in legal_suggestions:
                    query = suggestion.lower().strip()
                    if query not in seen_queries:
                        all_suggestions.append(suggestion)
                        seen_queries.add(query)
                
                return all_suggestions[:limit]
                
            except Exception as e:
                logger.error(
                    "Failed to get search suggestions",
                    partial_query=partial_query,
                    user_id=user_id,
                    error=str(e)
                )
                return []
    
    async def get_search_history(
        self,
        user_id: str,
        case_id: Optional[str] = None,
        limit: int = 50,
        days: int = 30
    ) -> List[SearchHistoryEntry]:
        """
        Get search history for a user.
        
        Args:
            user_id: User identifier
            case_id: Optional case filter
            limit: Maximum number of entries
            days: Number of days to look back
            
        Returns:
            List of search history entries
        """
        try:
            return await self.search_history_repository.get_search_history(
                user_id=user_id,
                case_id=case_id,
                limit=limit,
                days_back=days
            )
        except Exception as e:
            logger.error(
                "Failed to get search history",
                user_id=user_id,
                case_id=case_id,
                error=str(e)
            )
            return []
    
    def get_search_metrics(self) -> SearchMetrics:
        """Get current search metrics and analytics."""
        return self.metrics
    
    # Private helper methods
    
    async def _validate_search_request(self, request: SearchRequest, user_id: str) -> None:
        """Validate search request and user access."""
        # Validate query
        if not request.query or not request.query.strip():
            raise_validation_error(
                "Search query cannot be empty",
                ErrorCode.SEARCH_EMPTY_QUERY
            )
        
        if len(request.query) > 1000:
            raise_validation_error(
                "Search query too long (max 1000 characters)",
                ErrorCode.SEARCH_QUERY_TOO_LONG,
                {"query_length": len(request.query)}
            )
        
        # Validate case access if specified
        if request.case_id:
            await self._validate_case_access(request.case_id, user_id)
        
        # Validate weight combination for hybrid search
        if request.search_type == SearchType.HYBRID:
            total_weight = request.semantic_weight + request.keyword_weight
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point error
                raise_validation_error(
                    "Semantic and keyword weights must sum to 1.0",
                    ErrorCode.SEARCH_INVALID_WEIGHTS,
                    {"semantic_weight": request.semantic_weight, "keyword_weight": request.keyword_weight}
                )
    
    async def _validate_case_access(self, case_id: str, user_id: str) -> None:
        """Validate user access to a case."""
        # In a full implementation, this would check user permissions
        # For now, we'll just verify the case exists
        case = await self.document_repository.get_case_basic_info(case_id)
        if not case:
            raise_access_error(
                f"Case {case_id} not found or access denied",
                ErrorCode.SEARCH_CASE_ACCESS_DENIED,
                {"case_id": case_id, "user_id": user_id}
            )
    
    async def _check_rate_limit(self, user_id: str, client_ip: Optional[str]) -> None:
        """Check rate limits for search operations."""
        current_time = datetime.now(timezone.utc)
        minute_ago = current_time - timedelta(minutes=1)
        
        # Clean old entries
        user_requests = self._rate_limits[user_id]
        self._rate_limits[user_id] = [
            timestamp for timestamp in user_requests if timestamp > minute_ago
        ]
        
        # Check limit
        if len(self._rate_limits[user_id]) >= self._max_searches_per_minute:
            raise_validation_error(
                f"Rate limit exceeded: {self._max_searches_per_minute} searches per minute",
                ErrorCode.SEARCH_RATE_LIMIT_EXCEEDED,
                {"user_id": user_id, "current_requests": len(self._rate_limits[user_id])}
            )
        
        # Add current request
        self._rate_limits[user_id].append(current_time)
    
    async def _execute_search(
        self,
        request: SearchRequest,
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Execute the appropriate search based on request type."""
        if request.search_type == SearchType.SEMANTIC:
            return await self._execute_semantic_search(request, context)
        elif request.search_type == SearchType.KEYWORD:
            return await self._execute_keyword_search(request, context)
        elif request.search_type == SearchType.HYBRID:
            return await self._execute_hybrid_search(request, context)
        else:
            raise_search_error(
                f"Unsupported search type: {request.search_type.value}",
                ErrorCode.SEARCH_UNSUPPORTED_TYPE,
                {"search_type": request.search_type.value}
            )
    
    async def _execute_semantic_search(
        self,
        request: SearchRequest,
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Execute semantic vector search."""
        # Generate query embedding
        embedding_response = await self.embedding_service.generate_embeddings(
            request.query, model_name=None, normalize=True, use_cache=True
        )
        
        query_embedding = embedding_response.embeddings[0]
        
        # Build filters
        filters = self._build_search_filters(request)
        
        # Execute vector search
        if request.case_id:
            return await self.vector_repository.semantic_search(
                case_id=request.case_id,
                query_embedding=query_embedding,
                limit=request.limit + request.offset,
                min_score=request.similarity_threshold,
                filters=filters
            )
        else:
            # Search across multiple cases (if user has access)
            # For now, require case_id
            raise_validation_error(
                "Global semantic search requires case_id",
                ErrorCode.SEARCH_SCOPE_NOT_SUPPORTED
            )
    
    async def _execute_keyword_search(
        self,
        request: SearchRequest,
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Execute keyword/BM25 search."""
        # Build filters
        filters = self._build_search_filters(request)
        
        # Execute keyword search
        if request.case_id:
            return await self.vector_repository.keyword_search(
                case_id=request.case_id,
                query_text=request.query,
                limit=request.limit + request.offset,
                filters=filters
            )
        else:
            raise_validation_error(
                "Global keyword search requires case_id",
                ErrorCode.SEARCH_SCOPE_NOT_SUPPORTED
            )
    
    async def _execute_hybrid_search(
        self,
        request: SearchRequest,
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Execute hybrid semantic + keyword search."""
        # Generate query embedding
        embedding_response = await self.embedding_service.generate_embeddings(
            request.query, model_name=None, normalize=True, use_cache=True
        )
        
        query_embedding = embedding_response.embeddings[0]
        
        # Build filters
        filters = self._build_search_filters(request)
        
        # Execute hybrid search
        if request.case_id:
            return await self.vector_repository.hybrid_search(
                case_id=request.case_id,
                query_text=request.query,
                query_embedding=query_embedding,
                semantic_weight=request.semantic_weight,
                keyword_weight=request.keyword_weight,
                limit=request.limit + request.offset,
                min_score=request.similarity_threshold,
                filters=filters
            )
        else:
            raise_validation_error(
                "Global hybrid search requires case_id",
                ErrorCode.SEARCH_SCOPE_NOT_SUPPORTED
            )
    
    def _build_search_filters(self, request: SearchRequest) -> Dict[str, Any]:
        """Build search filters from request parameters."""
        filters = {}
        
        if request.document_ids:
            filters["document_ids"] = request.document_ids
        
        if request.file_types:
            filters["file_types"] = [ft.value for ft in request.file_types]
        
        if request.date_from:
            filters["date_from"] = request.date_from
        
        if request.date_to:
            filters["date_to"] = request.date_to
        
        return filters
    
    async def _process_search_results(
        self,
        raw_results: List[Dict[str, Any]],
        request: SearchRequest,
        context: QueryContext
    ) -> List[SearchResultItem]:
        """Process and format search results."""
        if not raw_results:
            return []
        
        # Apply pagination
        start_idx = request.offset
        end_idx = start_idx + request.limit
        paginated_results = raw_results[start_idx:end_idx]
        
        processed_results = []
        
        for result in paginated_results:
            # Generate highlights if requested
            highlights = []
            if request.include_highlights:
                highlights = self._generate_highlights(
                    result.get("content", ""),
                    request.query,
                    request.highlight_context
                )
            
            # Extract citations if requested
            citations = []
            if request.include_citations:
                citations = self._extract_citations(result.get("content", ""))
            
            # Build result item
            result_item = SearchResultItem(
                chunk_id=result.get("chunk_id", ""),
                document_id=result.get("document_id", ""),
                document_name=result.get("document_name", ""),
                content=result.get("content", ""),
                relevance_score=result.get("hybrid_score", result.get("similarity_score", 0.0)),
                chunk_index=result.get("chunk_index", 0),
                start_char=result.get("start_char", 0),
                end_char=result.get("end_char", 0),
                section_title=result.get("section_title"),
                page_number=result.get("page_number"),
                file_type=DocumentType(result.get("file_type", "unknown")),
                created_at=result.get("created_at", datetime.now(timezone.utc)),
                highlights=highlights,
                legal_citations=citations,
                metadata=result.get("metadata", {}) if request.include_metadata else {}
            )
            
            processed_results.append(result_item)
        
        return processed_results
    
    def _generate_highlights(
        self,
        content: str,
        query: str,
        context_sentences: int
    ) -> List[SearchHighlight]:
        """Generate highlighted snippets from content."""
        highlights = []
        
        # Simple highlighting logic (can be enhanced)
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find matches
        matches = []
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                start = 0
                while True:
                    pos = content_lower.find(term, start)
                    if pos == -1:
                        break
                    matches.append((pos, pos + len(term), term))
                    start = pos + 1
        
        # Sort matches by position
        matches.sort(key=lambda x: x[0])
        
        # Create highlights with context
        for start_pos, end_pos, term in matches[:5]:  # Limit to 5 highlights
            # Find sentence boundaries around match
            sentence_start = max(0, content.rfind('.', 0, start_pos) + 1)
            sentence_end = content.find('.', end_pos)
            if sentence_end == -1:
                sentence_end = len(content)
            
            highlight_text = content[sentence_start:sentence_end].strip()
            
            # Mark the matched term
            highlighted = highlight_text.replace(
                content[start_pos:end_pos],
                f"<mark>{content[start_pos:end_pos]}</mark>"
            )
            
            highlights.append(SearchHighlight(
                text=highlighted,
                start_char=sentence_start,
                end_char=sentence_end,
                score=0.8  # Simple fixed score
            ))
        
        return highlights
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract legal citations from content."""
        citations = []
        
        for pattern_name, pattern in self._citation_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                citation = match if isinstance(match, str) else ' '.join(match)
                if citation not in citations:
                    citations.append(citation)
        
        return citations[:10]  # Limit to 10 citations
    
    def _initialize_citation_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize legal citation regex patterns."""
        patterns = {}
        
        # US Code patterns
        patterns["us_code"] = re.compile(
            r'\b\d+\s+U\.S\.C\.?\s*§?\s*\d+(?:\(\w+\))*\b',
            re.IGNORECASE
        )
        
        # CFR patterns
        patterns["cfr"] = re.compile(
            r'\b\d+\s+C\.F\.R\.?\s*§?\s*\d+(?:\.\d+)*\b',
            re.IGNORECASE
        )
        
        # Federal Register patterns
        patterns["fed_reg"] = re.compile(
            r'\b\d+\s+Fed\.?\s+Reg\.?\s+\d+\b',
            re.IGNORECASE
        )
        
        # Case law patterns (simplified)
        patterns["case_law"] = re.compile(
            r'\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+.*?\d+\s+[A-Z\.]+\s+\d+',
            re.IGNORECASE
        )
        
        return patterns
    
    def _normalize_citation(self, citation: str) -> str:
        """Normalize a legal citation for consistent matching."""
        # Basic normalization
        normalized = citation.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'§\s*', '§ ', normalized)  # Normalize section symbol
        return normalized
    
    def _build_citation_patterns(
        self,
        citation: str,
        exact_match: bool,
        include_variants: bool
    ) -> List[str]:
        """Build search patterns for citation matching."""
        patterns = [citation]
        
        if not exact_match:
            # Add fuzzy patterns
            citation_escaped = re.escape(citation)
            patterns.append(citation_escaped.replace(r'\ ', r'\s+'))
        
        if include_variants:
            # Add common variants
            variants = []
            
            # Handle section symbol variants
            if '§' in citation:
                variants.append(citation.replace('§', 'Section'))
                variants.append(citation.replace('§', 'Sec.'))
            
            # Handle U.S.C. variants
            if 'U.S.C.' in citation:
                variants.append(citation.replace('U.S.C.', 'USC'))
                variants.append(citation.replace('U.S.C.', 'United States Code'))
            
            patterns.extend(variants)
        
        return patterns
    
    async def _search_citations_in_case(
        self,
        case_id: str,
        patterns: List[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for citations within a specific case."""
        # This would use the vector repository's text search capabilities
        # For now, implementing a basic version
        all_results = []
        
        for pattern in patterns:
            results = await self.vector_repository.keyword_search(
                case_id=case_id,
                query_text=pattern,
                limit=limit
            )
            all_results.extend(results)
        
        # Deduplicate and limit
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        return unique_results[:limit]
    
    async def _search_citations_global(
        self,
        user_id: str,
        patterns: List[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for citations across all accessible cases."""
        # Get user's accessible cases
        accessible_cases = await self.document_repository.get_user_cases(user_id)
        
        all_results = []
        for case in accessible_cases:
            case_results = await self._search_citations_in_case(
                case["case_id"], patterns, limit // len(accessible_cases) + 1
            )
            all_results.extend(case_results)
        
        # Sort by relevance and limit
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_results[:limit]
    
    async def _process_citation_results(
        self,
        raw_results: List[Dict[str, Any]],
        citation: str,
        request: CitationSearchRequest
    ) -> List[SearchResultItem]:
        """Process citation search results."""
        processed_results = []
        
        for result in raw_results:
            # Highlight citation matches
            content = result.get("content", "")
            highlighted_content = self._highlight_citations(content, citation)
            
            result_item = SearchResultItem(
                chunk_id=result.get("chunk_id", ""),
                document_id=result.get("document_id", ""),
                document_name=result.get("document_name", ""),
                content=highlighted_content,
                relevance_score=result.get("similarity_score", 0.0),
                chunk_index=result.get("chunk_index", 0),
                start_char=result.get("start_char", 0),
                end_char=result.get("end_char", 0),
                section_title=result.get("section_title"),
                page_number=result.get("page_number"),
                file_type=DocumentType(result.get("file_type", "unknown")),
                created_at=result.get("created_at", datetime.now(timezone.utc)),
                highlights=[],
                legal_citations=[citation],
                metadata={}
            )
            
            processed_results.append(result_item)
        
        return processed_results
    
    def _highlight_citations(self, content: str, citation: str) -> str:
        """Highlight citation matches in content."""
        # Simple highlighting
        citation_pattern = re.escape(citation)
        highlighted = re.sub(
            citation_pattern,
            f"<mark>{citation}</mark>",
            content,
            flags=re.IGNORECASE
        )
        return highlighted
    
    async def _convert_similarity_results(
        self,
        similar_documents: List[Dict[str, Any]],
        case_id: str
    ) -> List[SearchResultItem]:
        """Convert similarity search results to search result items."""
        processed_results = []
        
        for doc_info in similar_documents:
            # Get document metadata
            document = await self.document_repository.get_document(doc_info["document_id"])
            if not document:
                continue
            
            result_item = SearchResultItem(
                chunk_id="",  # Not chunk-specific
                document_id=doc_info["document_id"],
                document_name=document.document_name,
                content=f"Similar document with {doc_info['chunk_count']} matching chunks",
                relevance_score=doc_info["similarity_score"],
                chunk_index=0,
                start_char=0,
                end_char=0,
                section_title=None,
                page_number=None,
                file_type=document.file_type,
                created_at=document.created_at,
                highlights=[],
                legal_citations=[],
                metadata={
                    "similarity_type": "document_level",
                    "chunk_count": doc_info["chunk_count"],
                    "max_chunk_score": doc_info["max_chunk_score"]
                }
            )
            
            processed_results.append(result_item)
        
        return processed_results
    
    def _generate_legal_suggestions(self, partial_query: str) -> List[str]:
        """Generate legal-specific search suggestions."""
        legal_terms = [
            "intellectual property",
            "patent application",
            "trademark registration",
            "copyright infringement",
            "contract amendment",
            "legal precedent",
            "statutory interpretation",
            "case law analysis",
            "regulatory compliance",
            "due process"
        ]
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        for term in legal_terms:
            if partial_lower in term.lower() and len(term) > len(partial_query):
                suggestions.append(term)
        
        return suggestions[:5]
    
    def _generate_cache_key(self, request: SearchRequest, user_id: str) -> str:
        """Generate cache key for search request."""
        key_data = {
            "query": request.query,
            "search_type": request.search_type.value,
            "case_id": request.case_id,
            "semantic_weight": request.semantic_weight,
            "keyword_weight": request.keyword_weight,
            "limit": request.limit,
            "offset": request.offset,
            "similarity_threshold": request.similarity_threshold,
            "file_types": [ft.value for ft in request.file_types] if request.file_types else None
        }
        
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[SearchResponse]:
        """Get cached search result if valid."""
        if cache_key in self._query_cache:
            result, timestamp = self._query_cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return result
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: SearchResponse) -> None:
        """Cache search result."""
        # Implement LRU-like behavior
        if len(self._query_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k][1]
            )
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (result, datetime.now(timezone.utc))
    
    async def _record_search_history(
        self,
        search_id: str,
        request: SearchRequest,
        context: QueryContext,
        results_count: int,
        execution_time_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """Record search in history for analytics."""
        try:
            history_entry = SearchHistoryEntry(
                search_id=search_id,
                query=request.query,
                search_type=request.search_type,
                search_scope=request.search_scope,
                case_id=request.case_id,
                timestamp=context.timestamp,
                results_count=results_count,
                execution_time_ms=execution_time_ms,
                user_id=context.user_id
            )
            
            await self.search_history_repository.record_search(
                history_entry, error_message
            )
        except Exception as e:
            logger.warning(
                "Failed to record search history",
                search_id=search_id,
                error=str(e)
            )
    
    def _update_search_metrics(
        self,
        request: SearchRequest,
        results_count: int,
        execution_time_ms: float
    ) -> None:
        """Update search performance metrics."""
        self.metrics.total_searches += 1
        self.metrics.total_results += results_count
        
        # Update average execution time
        current_avg = self.metrics.average_execution_time_ms
        total_searches = self.metrics.total_searches
        
        self.metrics.average_execution_time_ms = (
            (current_avg * (total_searches - 1) + execution_time_ms) / total_searches
        )
        
        # Update average results per search
        self.metrics.average_results_per_search = (
            self.metrics.total_results / self.metrics.total_searches
        )
        
        # Update search type distribution
        if request.search_type not in self.metrics.search_type_distribution:
            self.metrics.search_type_distribution[request.search_type] = 0
        self.metrics.search_type_distribution[request.search_type] += 1
        
        # Track popular queries
        query_key = request.query.lower().strip()
        if query_key not in self.metrics.popular_queries:
            self.metrics.popular_queries[query_key] = 0
        self.metrics.popular_queries[query_key] += 1
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Clear caches
        self._query_cache.clear()
        self._rate_limits.clear()
        
        logger.info("SearchService cleanup completed")