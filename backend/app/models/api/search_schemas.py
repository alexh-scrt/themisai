"""
Pydantic API schemas for search and retrieval in Patexia Legal AI Chatbot.

This module defines the request/response schemas for search-related API endpoints:
- Semantic and keyword search schemas
- Hybrid search configuration schemas
- Search result and ranking schemas
- Search history and analytics schemas
- Citation and highlighting schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from backend.app.models.domain.document import DocumentType


class SearchType(str, Enum):
    """Types of search operations supported."""
    
    SEMANTIC = "semantic"           # Vector similarity search
    KEYWORD = "keyword"             # Traditional keyword/BM25 search
    HYBRID = "hybrid"               # Combination of semantic and keyword
    CITATION = "citation"           # Search for legal citations
    SIMILARITY = "similarity"       # Find similar documents/chunks


class SearchScope(str, Enum):
    """Scope of search operations."""
    
    CASE = "case"                   # Search within a specific case
    DOCUMENT = "document"           # Search within a specific document
    GLOBAL = "global"               # Search across all accessible cases
    RECENT = "recent"               # Search in recently accessed documents


class SortBy(str, Enum):
    """Sort options for search results."""
    
    RELEVANCE = "relevance"         # Sort by relevance score (default)
    DATE = "date"                   # Sort by document date
    DOCUMENT_NAME = "document_name" # Sort by document name
    FILE_SIZE = "file_size"         # Sort by file size
    CHUNK_INDEX = "chunk_index"     # Sort by chunk position in document


class SortOrder(str, Enum):
    """Sort order options."""
    
    DESC = "desc"                   # Descending order
    ASC = "asc"                     # Ascending order


class SearchRequest(BaseModel):
    """Schema for search request with flexible configuration."""
    
    query: str = Field(
        ...,
        description="Search query text",
        min_length=1,
        max_length=1000
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Restrict search to specific case"
    )
    
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict search to specific documents",
        max_items=50
    )
    
    search_type: SearchType = Field(
        SearchType.HYBRID,
        description="Type of search to perform"
    )
    
    search_scope: SearchScope = Field(
        SearchScope.CASE,
        description="Scope of search operation"
    )
    
    # Hybrid search parameters
    semantic_weight: float = Field(
        0.6,
        description="Weight for semantic search in hybrid mode (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    keyword_weight: float = Field(
        0.4,
        description="Weight for keyword search in hybrid mode (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Result configuration
    limit: int = Field(
        15,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    
    offset: int = Field(
        0,
        description="Number of results to skip for pagination",
        ge=0
    )
    
    similarity_threshold: float = Field(
        0.7,
        description="Minimum similarity score for results (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Filtering options
    file_types: Optional[List[DocumentType]] = Field(
        None,
        description="Filter by document types"
    )
    
    date_from: Optional[datetime] = Field(
        None,
        description="Filter documents created after this date"
    )
    
    date_to: Optional[datetime] = Field(
        None,
        description="Filter documents created before this date"
    )
    
    # Result customization
    include_highlights: bool = Field(
        True,
        description="Include highlighted text snippets in results"
    )
    
    highlight_context: int = Field(
        3,
        description="Number of sentences around matches for context",
        ge=1,
        le=10
    )
    
    include_citations: bool = Field(
        True,
        description="Include legal citations found in matching chunks"
    )
    
    include_metadata: bool = Field(
        False,
        description="Include detailed document metadata in results"
    )
    
    # Sorting and ranking
    sort_by: SortBy = Field(
        SortBy.RELEVANCE,
        description="Field to sort results by"
    )
    
    sort_order: SortOrder = Field(
        SortOrder.DESC,
        description="Sort order for results"
    )
    
    enable_reranking: bool = Field(
        True,
        description="Enable advanced reranking of search results"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate and normalize search query."""
        if not v.strip():
            raise ValueError('Search query cannot be empty or whitespace only')
        return v.strip()
    
    @root_validator
    def validate_hybrid_weights(cls, values):
        """Validate hybrid search weights sum to 1.0."""
        search_type = values.get('search_type')
        semantic_weight = values.get('semantic_weight', 0.6)
        keyword_weight = values.get('keyword_weight', 0.4)
        
        if search_type == SearchType.HYBRID:
            total_weight = semantic_weight + keyword_weight
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f'Hybrid search weights must sum to 1.0, got {total_weight}')
        
        return values
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate date range consistency."""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from >= date_to:
            raise ValueError('date_from must be before date_to')
        
        return values
    
    @root_validator
    def validate_search_scope(cls, values):
        """Validate search scope configuration."""
        search_scope = values.get('search_scope')
        case_id = values.get('case_id')
        document_ids = values.get('document_ids')
        
        if search_scope == SearchScope.CASE and not case_id:
            raise ValueError('case_id is required when search_scope is "case"')
        
        if search_scope == SearchScope.DOCUMENT and not document_ids:
            raise ValueError('document_ids is required when search_scope is "document"')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "query": "wireless standard WiFi6 patent applications",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "search_type": "hybrid",
                "search_scope": "case",
                "semantic_weight": 0.6,
                "keyword_weight": 0.4,
                "limit": 15,
                "similarity_threshold": 0.7,
                "include_highlights": True,
                "highlight_context": 3,
                "include_citations": True,
                "sort_by": "relevance",
                "sort_order": "desc"
            }
        }


class SearchResultChunk(BaseModel):
    """Schema for individual search result chunk."""
    
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier"
    )
    
    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )
    
    document_name: str = Field(
        ...,
        description="Document display name"
    )
    
    content: str = Field(
        ...,
        description="Text content of the chunk"
    )
    
    highlighted_content: Optional[str] = Field(
        None,
        description="Content with search terms highlighted"
    )
    
    relevance_score: float = Field(
        ...,
        description="Relevance score for this result (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    semantic_score: Optional[float] = Field(
        None,
        description="Semantic similarity score",
        ge=0.0,
        le=1.0
    )
    
    keyword_score: Optional[float] = Field(
        None,
        description="Keyword matching score",
        ge=0.0,
        le=1.0
    )
    
    chunk_index: int = Field(
        ...,
        description="Position of chunk within document",
        ge=0
    )
    
    start_char: int = Field(
        ...,
        description="Starting character position in document",
        ge=0
    )
    
    end_char: int = Field(
        ...,
        description="Ending character position in document",
        ge=0
    )
    
    # Legal document structure
    section_title: Optional[str] = Field(
        None,
        description="Section header if available"
    )
    
    page_number: Optional[int] = Field(
        None,
        description="Page number if available",
        ge=1
    )
    
    paragraph_number: Optional[int] = Field(
        None,
        description="Paragraph number if available",
        ge=1
    )
    
    legal_citations: List[str] = Field(
        ...,
        description="Legal citations found in this chunk"
    )
    
    # Context and navigation
    context_before: Optional[str] = Field(
        None,
        description="Text context before the matching chunk"
    )
    
    context_after: Optional[str] = Field(
        None,
        description="Text context after the matching chunk"
    )
    
    # Document metadata
    file_type: DocumentType = Field(
        ...,
        description="Document file type"
    )
    
    document_created_at: datetime = Field(
        ...,
        description="Document creation timestamp"
    )
    
    document_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional document metadata if requested"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "DOC_123_chunk_0001",
                "document_id": "DOC_20250115_A1B2C3D4",
                "document_name": "Patent Application WiFi6 2024",
                "content": "The implementation of wireless standard WiFi6 provides enhanced throughput capabilities...",
                "highlighted_content": "The implementation of <mark>wireless standard WiFi6</mark> provides enhanced throughput...",
                "relevance_score": 0.95,
                "semantic_score": 0.92,
                "keyword_score": 0.87,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 512,
                "section_title": "3.2 Technical Implementation Details",
                "page_number": 3,
                "paragraph_number": 1,
                "legal_citations": ["35 U.S.C. § 101", "IEEE 802.11ax"],
                "file_type": "pdf",
                "document_created_at": "2025-01-15T10:30:00Z"
            }
        }


class SearchResponse(BaseModel):
    """Schema for search response with results and metadata."""
    
    query: str = Field(
        ...,
        description="Original search query"
    )
    
    search_type: SearchType = Field(
        ...,
        description="Type of search performed"
    )
    
    search_scope: SearchScope = Field(
        ...,
        description="Scope of search operation"
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Case ID if search was restricted to a case"
    )
    
    results: List[SearchResultChunk] = Field(
        ...,
        description="List of matching chunks ordered by relevance"
    )
    
    total_results: int = Field(
        ...,
        description="Total number of matching results",
        ge=0
    )
    
    offset: int = Field(
        ...,
        description="Number of results skipped",
        ge=0
    )
    
    limit: int = Field(
        ...,
        description="Maximum results per page",
        ge=1
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more results available"
    )
    
    # Search performance metrics
    search_time_ms: float = Field(
        ...,
        description="Search execution time in milliseconds",
        ge=0.0
    )
    
    # Search quality metrics
    max_relevance_score: float = Field(
        ...,
        description="Highest relevance score in results",
        ge=0.0,
        le=1.0
    )
    
    min_relevance_score: float = Field(
        ...,
        description="Lowest relevance score in results",
        ge=0.0,
        le=1.0
    )
    
    average_relevance_score: float = Field(
        ...,
        description="Average relevance score of results",
        ge=0.0,
        le=1.0
    )
    
    # Document coverage
    documents_searched: int = Field(
        ...,
        description="Number of documents searched",
        ge=0
    )
    
    chunks_searched: int = Field(
        ...,
        description="Number of chunks searched",
        ge=0
    )
    
    documents_with_matches: int = Field(
        ...,
        description="Number of documents containing matches",
        ge=0
    )
    
    # Search suggestions
    suggested_queries: List[str] = Field(
        ...,
        description="Suggested alternative or refined queries"
    )
    
    # Faceted search results
    facets: Optional[Dict[str, Dict[str, int]]] = Field(
        None,
        description="Faceted breakdown of results by various dimensions"
    )
    
    # Search metadata
    search_id: str = Field(
        ...,
        description="Unique identifier for this search operation"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Search execution timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "wireless standard WiFi6 patent applications",
                "search_type": "hybrid",
                "search_scope": "case",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "results": [],
                "total_results": 15,
                "offset": 0,
                "limit": 15,
                "has_more": False,
                "search_time_ms": 234.5,
                "max_relevance_score": 0.95,
                "min_relevance_score": 0.71,
                "average_relevance_score": 0.83,
                "documents_searched": 23,
                "chunks_searched": 567,
                "documents_with_matches": 8,
                "suggested_queries": ["WiFi6 implementation", "802.11ax patent"],
                "search_id": "search_20250115_12345",
                "timestamp": "2025-01-15T14:30:00Z"
            }
        }


class SimilaritySearchRequest(BaseModel):
    """Schema for finding similar documents or chunks."""
    
    reference_document_id: Optional[str] = Field(
        None,
        description="Find documents similar to this document"
    )
    
    reference_chunk_id: Optional[str] = Field(
        None,
        description="Find chunks similar to this chunk"
    )
    
    reference_text: Optional[str] = Field(
        None,
        description="Find content similar to this text",
        max_length=5000
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Restrict similarity search to specific case"
    )
    
    exclude_self: bool = Field(
        True,
        description="Exclude the reference document/chunk from results"
    )
    
    similarity_threshold: float = Field(
        0.8,
        description="Minimum similarity score for results",
        ge=0.0,
        le=1.0
    )
    
    limit: int = Field(
        10,
        description="Maximum number of similar items to return",
        ge=1,
        le=50
    )
    
    include_metadata: bool = Field(
        False,
        description="Include detailed metadata in results"
    )
    
    @root_validator
    def validate_reference(cls, values):
        """Validate that exactly one reference is provided."""
        ref_doc = values.get('reference_document_id')
        ref_chunk = values.get('reference_chunk_id')
        ref_text = values.get('reference_text')
        
        references = [ref_doc, ref_chunk, ref_text]
        non_null_refs = [r for r in references if r is not None]
        
        if len(non_null_refs) != 1:
            raise ValueError('Exactly one reference (document_id, chunk_id, or text) must be provided')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "reference_chunk_id": "DOC_123_chunk_0001",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "exclude_self": True,
                "similarity_threshold": 0.8,
                "limit": 10,
                "include_metadata": False
            }
        }


class CitationSearchRequest(BaseModel):
    """Schema for searching legal citations."""
    
    citation_query: str = Field(
        ...,
        description="Legal citation to search for",
        min_length=1,
        max_length=200
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Restrict search to specific case"
    )
    
    exact_match: bool = Field(
        False,
        description="Whether to require exact citation match"
    )
    
    include_variants: bool = Field(
        True,
        description="Include citation variants and abbreviations"
    )
    
    limit: int = Field(
        20,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    
    @validator('citation_query')
    def validate_citation_query(cls, v):
        """Validate citation query format."""
        if not v.strip():
            raise ValueError('Citation query cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "citation_query": "35 U.S.C. § 101",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "exact_match": False,
                "include_variants": True,
                "limit": 20
            }
        }


class SearchHistoryEntry(BaseModel):
    """Schema for search history entry."""
    
    search_id: str = Field(
        ...,
        description="Unique search identifier"
    )
    
    query: str = Field(
        ...,
        description="Search query text"
    )
    
    search_type: SearchType = Field(
        ...,
        description="Type of search performed"
    )
    
    search_scope: SearchScope = Field(
        ...,
        description="Scope of search operation"
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Case ID if search was case-specific"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Search execution timestamp"
    )
    
    results_count: int = Field(
        ...,
        description="Number of results returned",
        ge=0
    )
    
    execution_time_ms: float = Field(
        ...,
        description="Search execution time in milliseconds",
        ge=0.0
    )
    
    user_id: str = Field(
        ...,
        description="User who performed the search"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "search_id": "search_20250115_12345",
                "query": "wireless standard WiFi6",
                "search_type": "hybrid",
                "search_scope": "case",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "timestamp": "2025-01-15T14:30:00Z",
                "results_count": 15,
                "execution_time_ms": 234.5,
                "user_id": "user_123"
            }
        }


class SearchHistoryRequest(BaseModel):
    """Schema for retrieving search history."""
    
    user_id: Optional[str] = Field(
        None,
        description="Filter by user ID (admin only)"
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Filter by case ID"
    )
    
    search_type: Optional[SearchType] = Field(
        None,
        description="Filter by search type"
    )
    
    date_from: Optional[datetime] = Field(
        None,
        description="Filter searches after this date"
    )
    
    date_to: Optional[datetime] = Field(
        None,
        description="Filter searches before this date"
    )
    
    limit: int = Field(
        50,
        description="Maximum number of history entries to return",
        ge=1,
        le=500
    )
    
    offset: int = Field(
        0,
        description="Number of entries to skip for pagination",
        ge=0
    )
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate date range consistency."""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from >= date_to:
            raise ValueError('date_from must be before date_to')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "search_type": "hybrid",
                "limit": 20,
                "offset": 0
            }
        }


class SearchHistoryResponse(BaseModel):
    """Schema for search history response."""
    
    history: List[SearchHistoryEntry] = Field(
        ...,
        description="List of search history entries"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of history entries matching filters",
        ge=0
    )
    
    offset: int = Field(
        ...,
        description="Number of entries skipped",
        ge=0
    )
    
    limit: int = Field(
        ...,
        description="Maximum entries per page",
        ge=1
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more entries available"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "history": [],
                "total_count": 157,
                "offset": 0,
                "limit": 20,
                "has_more": True
            }
        }


class SearchStatsResponse(BaseModel):
    """Schema for search analytics and statistics."""
    
    total_searches: int = Field(
        ...,
        description="Total number of searches performed",
        ge=0
    )
    
    unique_queries: int = Field(
        ...,
        description="Number of unique search queries",
        ge=0
    )
    
    average_results_per_search: float = Field(
        ...,
        description="Average number of results per search",
        ge=0.0
    )
    
    average_search_time_ms: float = Field(
        ...,
        description="Average search execution time in milliseconds",
        ge=0.0
    )
    
    search_type_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown of searches by type"
    )
    
    search_scope_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown of searches by scope"
    )
    
    popular_queries: List[Dict[str, Any]] = Field(
        ...,
        description="Most popular search queries with frequency"
    )
    
    recent_searches: List[SearchHistoryEntry] = Field(
        ...,
        description="Recent search activity"
    )
    
    performance_trends: Dict[str, List[float]] = Field(
        ...,
        description="Search performance trends over time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_searches": 2156,
                "unique_queries": 1247,
                "average_results_per_search": 12.3,
                "average_search_time_ms": 187.5,
                "search_type_breakdown": {
                    "hybrid": 1456,
                    "semantic": 423,
                    "keyword": 277
                },
                "search_scope_breakdown": {
                    "case": 1834,
                    "document": 245,
                    "global": 77
                },
                "popular_queries": [
                    {
                        "query": "patent application",
                        "count": 89
                    }
                ],
                "recent_searches": [],
                "performance_trends": {
                    "daily_avg_time": [180.2, 185.7, 172.4]
                }
            }
        }


class SearchSuggestionRequest(BaseModel):
    """Schema for getting search query suggestions."""
    
    partial_query: str = Field(
        ...,
        description="Partial query text for suggestions",
        min_length=1,
        max_length=200
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Context case for relevant suggestions"
    )
    
    limit: int = Field(
        5,
        description="Maximum number of suggestions to return",
        ge=1,
        le=20
    )
    
    include_popular: bool = Field(
        True,
        description="Include popular queries in suggestions"
    )
    
    include_similar: bool = Field(
        True,
        description="Include semantically similar queries"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "partial_query": "wifi6 pat",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "limit": 5,
                "include_popular": True,
                "include_similar": True
            }
        }


class SearchSuggestionResponse(BaseModel):
    """Schema for search query suggestions response."""
    
    suggestions: List[str] = Field(
        ...,
        description="List of suggested queries"
    )
    
    popular_queries: List[str] = Field(
        ...,
        description="Popular queries related to the input"
    )
    
    similar_queries: List[str] = Field(
        ...,
        description="Semantically similar queries"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "suggestions": [
                    "wifi6 patent application",
                    "wifi6 patent claims",
                    "wifi6 patent prior art"
                ],
                "popular_queries": [
                    "wifi6 patent application",
                    "patent application wireless"
                ],
                "similar_queries": [
                    "802.11ax patent",
                    "wireless patent technology"
                ]
            }
        }