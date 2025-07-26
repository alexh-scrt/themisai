"""
MongoDB Database Models for Search History Management

This module defines the database layer models and schemas for search history tracking
and analytics in the Patexia Legal AI Chatbot. It provides MongoDB document structures,
query optimization, analytics aggregation, and search behavior analysis.

Key Features:
- MongoDB document schemas for search query tracking
- Search analytics and pattern analysis capabilities
- Query suggestion generation based on historical data
- Performance metrics and timing analysis
- User search behavior tracking and insights
- Popular query identification and trending analysis
- Field validation and business rule enforcement
- Index definitions for query optimization and analytics

Database Schema Design:
- Search history collection with comprehensive query tracking
- Analytics aggregation with time-based trend analysis
- Suggestion generation based on frequency and context
- Performance monitoring with execution time analysis
- User behavior analysis with personalization support
- Search quality metrics and success rate tracking

Collections:
- search_history: Primary search query tracking with metadata
- search_analytics: Aggregated analytics and trend data
- search_suggestions: Pre-computed suggestion data for performance
- search_patterns: User search behavior patterns and insights

Analytics Features:
- Real-time search performance monitoring
- Query frequency analysis and trending
- Search success rate tracking and optimization
- User behavior pattern recognition
- Context-aware suggestion generation
- Performance bottleneck identification

Performance Optimizations:
- Compound indexes for user and case-based queries
- Text search indexes for query content analysis
- Time-based indexes for trend analysis
- TTL indexes for automatic data lifecycle management
- Aggregation pipeline optimization for analytics
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
from bson import ObjectId
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pydantic import BaseModel, Field, validator, root_validator

from ...utils.logging import get_logger

logger = get_logger(__name__)


class SearchCollection(str, Enum):
    """Search-related MongoDB collection names."""
    SEARCH_HISTORY = "search_history"
    SEARCH_ANALYTICS = "search_analytics"
    SEARCH_SUGGESTIONS = "search_suggestions"
    SEARCH_PATTERNS = "search_patterns"


class SearchType(str, Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CITATION = "citation"
    FUZZY = "fuzzy"


class SearchScope(str, Enum):
    """Scope of search operations."""
    CASE = "case"
    DOCUMENT = "document"
    GLOBAL = "global"
    CHUNK = "chunk"


class SearchStatus(str, Enum):
    """Search execution status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SearchFilter:
    """Search filter configuration."""
    field: str
    operator: str
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchFilter":
        """Create from dictionary."""
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"]
        )


class SearchResultsDocument(BaseModel):
    """MongoDB subdocument for search results metadata."""
    total_results: int = Field(
        default=0,
        description="Total number of results found",
        ge=0
    )
    
    displayed_results: int = Field(
        default=0,
        description="Number of results displayed to user",
        ge=0
    )
    
    top_result_score: Optional[float] = Field(
        None,
        description="Relevance score of top result",
        ge=0.0,
        le=1.0
    )
    
    average_score: Optional[float] = Field(
        None,
        description="Average relevance score of results",
        ge=0.0,
        le=1.0
    )
    
    result_sources: List[str] = Field(
        default_factory=list,
        description="Sources of search results (document IDs, collections)"
    )
    
    result_types: List[str] = Field(
        default_factory=list,
        description="Types of results returned (document, chunk, citation)"
    )
    
    has_documents: bool = Field(
        default=False,
        description="Whether results include full documents"
    )
    
    has_chunks: bool = Field(
        default=False,
        description="Whether results include text chunks"
    )
    
    has_citations: bool = Field(
        default=False,
        description="Whether results include legal citations"
    )


class SearchPerformanceDocument(BaseModel):
    """MongoDB subdocument for search performance metrics."""
    execution_time_ms: float = Field(
        default=0.0,
        description="Total search execution time in milliseconds",
        ge=0.0
    )
    
    query_parsing_time_ms: Optional[float] = Field(
        None,
        description="Query parsing time in milliseconds",
        ge=0.0
    )
    
    vector_search_time_ms: Optional[float] = Field(
        None,
        description="Vector search time in milliseconds",
        ge=0.0
    )
    
    keyword_search_time_ms: Optional[float] = Field(
        None,
        description="Keyword search time in milliseconds",
        ge=0.0
    )
    
    ranking_time_ms: Optional[float] = Field(
        None,
        description="Result ranking time in milliseconds",
        ge=0.0
    )
    
    database_queries: int = Field(
        default=0,
        description="Number of database queries executed",
        ge=0
    )
    
    cache_hits: int = Field(
        default=0,
        description="Number of cache hits",
        ge=0
    )
    
    cache_misses: int = Field(
        default=0,
        description="Number of cache misses",
        ge=0
    )
    
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Peak memory usage during search in MB",
        ge=0.0
    )
    
    cpu_time_ms: Optional[float] = Field(
        None,
        description="CPU processing time in milliseconds",
        ge=0.0
    )
    
    is_slow_query: bool = Field(
        default=False,
        description="Whether this query exceeded performance thresholds"
    )
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests


class SearchHistoryDocument(BaseModel):
    """MongoDB document model for search query history."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    search_id: str = Field(
        ...,
        description="Unique search identifier",
        min_length=1,
        max_length=100
    )
    
    user_id: str = Field(
        ...,
        description="User who performed the search",
        min_length=1,
        max_length=100
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Case ID if search was case-specific",
        min_length=1,
        max_length=100
    )
    
    query: str = Field(
        ...,
        description="Original search query text",
        min_length=1,
        max_length=1000
    )
    
    query_normalized: str = Field(
        ...,
        description="Normalized query for matching and suggestions",
        min_length=1,
        max_length=1000
    )
    
    search_type: str = Field(
        ...,
        description="Type of search performed"
    )
    
    search_scope: str = Field(
        ...,
        description="Scope of search operation"
    )
    
    search_status: str = Field(
        default=SearchStatus.SUCCESS.value,
        description="Search execution status"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Search execution timestamp"
    )
    
    # Query characteristics
    query_length: int = Field(
        default=0,
        description="Length of query in characters",
        ge=0
    )
    
    query_word_count: int = Field(
        default=0,
        description="Number of words in query",
        ge=0
    )
    
    query_complexity: Optional[str] = Field(
        None,
        description="Query complexity classification (simple, medium, complex)"
    )
    
    # Search filters and parameters
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search filters applied"
    )
    
    search_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search configuration parameters"
    )
    
    # Results information
    results: SearchResultsDocument = Field(
        default_factory=SearchResultsDocument,
        description="Search results metadata"
    )
    
    # Performance metrics
    performance: SearchPerformanceDocument = Field(
        default_factory=SearchPerformanceDocument,
        description="Search performance metrics"
    )
    
    # User interaction
    clicked_results: List[str] = Field(
        default_factory=list,
        description="Result IDs that user clicked on"
    )
    
    result_clicked_rank: Optional[int] = Field(
        None,
        description="Rank of first clicked result",
        ge=1
    )
    
    time_to_first_click_ms: Optional[float] = Field(
        None,
        description="Time to first result click in milliseconds",
        ge=0.0
    )
    
    session_id: Optional[str] = Field(
        None,
        description="User session identifier"
    )
    
    # Context and metadata
    ip_address: Optional[str] = Field(
        None,
        description="Client IP address (hashed for privacy)"
    )
    
    user_agent: Optional[str] = Field(
        None,
        description="Client user agent string"
    )
    
    referer: Optional[str] = Field(
        None,
        description="HTTP referer"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search metadata"
    )
    
    # Derived fields
    has_results: bool = Field(
        default=False,
        description="Whether search returned any results"
    )
    
    is_successful: bool = Field(
        default=True,
        description="Whether search executed successfully"
    )
    
    is_repeat_query: bool = Field(
        default=False,
        description="Whether this is a repeat of a recent query"
    )
    
    query_intent: Optional[str] = Field(
        None,
        description="Inferred search intent (exploratory, specific, comparison)"
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "search_id": "search_20250115_12345",
                "user_id": "user_123",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "query": "wireless standard WiFi6 patent applications",
                "query_normalized": "wireless standard wifi6 patent applications",
                "search_type": "hybrid",
                "search_scope": "case",
                "timestamp": "2025-01-15T14:30:00Z",
                "query_length": 39,
                "query_word_count": 5,
                "results": {
                    "total_results": 15,
                    "displayed_results": 10,
                    "top_result_score": 0.92
                },
                "performance": {
                    "execution_time_ms": 234.5,
                    "database_queries": 3,
                    "cache_hits": 2
                }
            }
        }
    
    @validator('search_type')
    def validate_search_type(cls, v):
        """Validate search type."""
        try:
            SearchType(v)
        except ValueError:
            raise ValueError(f'Invalid search type: {v}')
        return v
    
    @validator('search_scope')
    def validate_search_scope(cls, v):
        """Validate search scope."""
        try:
            SearchScope(v)
        except ValueError:
            raise ValueError(f'Invalid search scope: {v}')
        return v
    
    @validator('search_status')
    def validate_search_status(cls, v):
        """Validate search status."""
        try:
            SearchStatus(v)
        except ValueError:
            raise ValueError(f'Invalid search status: {v}')
        return v
    
    @validator('query_normalized', pre=True, always=True)
    def normalize_query(cls, v, values):
        """Auto-normalize query if not provided."""
        if not v and 'query' in values:
            return cls._normalize_query_text(values['query'])
        return v
    
    @root_validator
    def validate_query_fields(cls, values):
        """Validate query-related field consistency."""
        query = values.get('query', '')
        
        # Calculate derived fields
        if not values.get('query_length'):
            values['query_length'] = len(query)
        
        if not values.get('query_word_count'):
            values['query_word_count'] = len(query.split())
        
        # Determine query complexity
        word_count = values.get('query_word_count', 0)
        if word_count <= 3:
            values['query_complexity'] = 'simple'
        elif word_count <= 8:
            values['query_complexity'] = 'medium'
        else:
            values['query_complexity'] = 'complex'
        
        # Set has_results based on results
        results = values.get('results')
        if results and hasattr(results, 'total_results'):
            values['has_results'] = results.total_results > 0
        
        # Set success status
        status = values.get('search_status', SearchStatus.SUCCESS.value)
        values['is_successful'] = status == SearchStatus.SUCCESS.value
        
        return values
    
    @staticmethod
    def _normalize_query_text(query: str) -> str:
        """Normalize query text for matching and suggestions."""
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common legal stopwords for better matching
        legal_stopwords = {
            'the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'by', 'a', 'an',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'shall', 'must', 'ought'
        }
        
        words = normalized.split()
        # Only remove stopwords if query has more than 3 words
        if len(words) > 3:
            words = [word for word in words if word not in legal_stopwords]
        
        return ' '.join(words)
    
    def update_performance(self, execution_time_ms: float, **kwargs) -> None:
        """Update performance metrics."""
        self.performance.execution_time_ms = execution_time_ms
        
        # Update other performance fields if provided
        for field, value in kwargs.items():
            if hasattr(self.performance, field):
                setattr(self.performance, field, value)
        
        # Mark as slow query if threshold exceeded (configurable)
        slow_query_threshold = 5000  # 5 seconds
        self.performance.is_slow_query = execution_time_ms > slow_query_threshold
    
    def update_results(self, total_results: int, **kwargs) -> None:
        """Update results metadata."""
        self.results.total_results = total_results
        self.has_results = total_results > 0
        
        # Update other result fields if provided
        for field, value in kwargs.items():
            if hasattr(self.results, field):
                setattr(self.results, field, value)
    
    def record_click(self, result_id: str, rank: int, click_time_ms: float) -> None:
        """Record user click on search result."""
        if result_id not in self.clicked_results:
            self.clicked_results.append(result_id)
        
        # Record first click metrics
        if self.result_clicked_rank is None:
            self.result_clicked_rank = rank
            self.time_to_first_click_ms = click_time_ms


class SearchAnalyticsDocument(BaseModel):
    """MongoDB document model for search analytics aggregation."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    analytics_id: str = Field(
        ...,
        description="Unique analytics record identifier"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="User ID for user-specific analytics"
    )
    
    case_id: Optional[str] = Field(
        None,
        description="Case ID for case-specific analytics"
    )
    
    period_start: datetime = Field(
        ...,
        description="Analytics period start time"
    )
    
    period_end: datetime = Field(
        ...,
        description="Analytics period end time"
    )
    
    period_type: str = Field(
        ...,
        description="Period type (hourly, daily, weekly, monthly)"
    )
    
    # Search volume metrics
    total_searches: int = Field(
        default=0,
        description="Total number of searches in period",
        ge=0
    )
    
    unique_queries: int = Field(
        default=0,
        description="Number of unique search queries",
        ge=0
    )
    
    unique_users: int = Field(
        default=0,
        description="Number of unique users who searched",
        ge=0
    )
    
    # Performance metrics
    average_execution_time_ms: float = Field(
        default=0.0,
        description="Average search execution time",
        ge=0.0
    )
    
    median_execution_time_ms: Optional[float] = Field(
        None,
        description="Median search execution time",
        ge=0.0
    )
    
    p95_execution_time_ms: Optional[float] = Field(
        None,
        description="95th percentile execution time",
        ge=0.0
    )
    
    slow_query_count: int = Field(
        default=0,
        description="Number of slow queries",
        ge=0
    )
    
    # Quality metrics
    average_results_per_search: float = Field(
        default=0.0,
        description="Average number of results per search",
        ge=0.0
    )
    
    zero_result_searches: int = Field(
        default=0,
        description="Number of searches with zero results",
        ge=0
    )
    
    successful_searches: int = Field(
        default=0,
        description="Number of successful searches",
        ge=0
    )
    
    failed_searches: int = Field(
        default=0,
        description="Number of failed searches",
        ge=0
    )
    
    # User behavior metrics
    average_clicks_per_search: float = Field(
        default=0.0,
        description="Average number of result clicks per search",
        ge=0.0
    )
    
    click_through_rate: float = Field(
        default=0.0,
        description="Percentage of searches that resulted in clicks",
        ge=0.0,
        le=1.0
    )
    
    average_time_to_click_ms: Optional[float] = Field(
        None,
        description="Average time from search to first click",
        ge=0.0
    )
    
    # Search type breakdown
    search_type_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of searches by type"
    )
    
    search_scope_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of searches by scope"
    )
    
    # Popular queries
    top_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most popular queries in period"
    )
    
    trending_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Trending queries compared to previous period"
    )
    
    # Cache performance
    cache_hit_rate: float = Field(
        default=0.0,
        description="Cache hit rate for searches",
        ge=0.0,
        le=1.0
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analytics record creation time"
    )
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "analytics_id": "analytics_daily_20250115",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "period_start": "2025-01-15T00:00:00Z",
                "period_end": "2025-01-15T23:59:59Z",
                "period_type": "daily",
                "total_searches": 156,
                "unique_queries": 89,
                "unique_users": 12,
                "average_execution_time_ms": 234.5,
                "successful_searches": 148,
                "zero_result_searches": 8
            }
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate search success rate."""
        total = self.successful_searches + self.failed_searches
        if total == 0:
            return 0.0
        return self.successful_searches / total
    
    @property
    def zero_result_rate(self) -> float:
        """Calculate zero result rate."""
        if self.total_searches == 0:
            return 0.0
        return self.zero_result_searches / self.total_searches


class SearchSuggestionDocument(BaseModel):
    """MongoDB document model for search suggestions."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    suggestion_text: str = Field(
        ...,
        description="Suggestion query text",
        min_length=1,
        max_length=200
    )
    
    normalized_text: str = Field(
        ...,
        description="Normalized suggestion text"
    )
    
    frequency: int = Field(
        ...,
        description="Number of times this query was searched",
        ge=1
    )
    
    context_case_ids: List[str] = Field(
        default_factory=list,
        description="Case IDs where this suggestion is relevant"
    )
    
    context_users: List[str] = Field(
        default_factory=list,
        description="Users who have used this query"
    )
    
    last_used: datetime = Field(
        ...,
        description="Last time this query was used"
    )
    
    average_results: float = Field(
        default=0.0,
        description="Average number of results for this query",
        ge=0.0
    )
    
    success_rate: float = Field(
        default=0.0,
        description="Success rate for this query",
        ge=0.0,
        le=1.0
    )
    
    query_type: str = Field(
        default="general",
        description="Type of query (legal, technical, procedural)"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Suggestion record creation time"
    )
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "suggestion_text": "patent application wireless",
                "normalized_text": "patent application wireless",
                "frequency": 23,
                "last_used": "2025-01-15T14:30:00Z",
                "average_results": 12.5,
                "success_rate": 0.89,
                "query_type": "legal"
            }
        }


class SearchPatternDocument(BaseModel):
    """MongoDB document model for search behavior patterns."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    pattern_id: str = Field(
        ...,
        description="Unique pattern identifier"
    )
    
    user_id: str = Field(
        ...,
        description="User ID for this pattern"
    )
    
    pattern_type: str = Field(
        ...,
        description="Type of pattern (temporal, topical, behavioral)"
    )
    
    pattern_data: Dict[str, Any] = Field(
        ...,
        description="Pattern-specific data and metrics"
    )
    
    confidence_score: float = Field(
        ...,
        description="Confidence in pattern accuracy",
        ge=0.0,
        le=1.0
    )
    
    observations_count: int = Field(
        ...,
        description="Number of observations supporting this pattern",
        ge=1
    )
    
    first_observed: datetime = Field(
        ...,
        description="When pattern was first observed"
    )
    
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last pattern update time"
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether pattern is still active"
    )


class SearchHistoryIndexes:
    """MongoDB index definitions for search history collections."""
    
    @classmethod
    def get_search_history_indexes(cls) -> List[IndexModel]:
        """Get index models for search_history collection."""
        return [
            # Unique constraint on search_id
            IndexModel([("search_id", ASCENDING)], unique=True, name="search_id_unique"),
            
            # User search history with timestamp
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="user_history_timestamp"
            ),
            
            # Case search history
            IndexModel(
                [("case_id", ASCENDING), ("timestamp", DESCENDING)],
                name="case_history_timestamp"
            ),
            
            # Search type analysis
            IndexModel(
                [("search_type", ASCENDING), ("timestamp", DESCENDING)],
                name="search_type_timestamp"
            ),
            
            # Search scope analysis
            IndexModel(
                [("search_scope", ASCENDING), ("timestamp", DESCENDING)],
                name="search_scope_timestamp"
            ),
            
            # Performance analysis
            IndexModel(
                [("performance.execution_time_ms", ASCENDING)],
                name="execution_time_performance"
            ),
            
            # Results analysis
            IndexModel(
                [("results.total_results", ASCENDING)],
                name="results_count_analysis"
            ),
            
            # Query text search
            IndexModel([("query", TEXT)], name="query_text_search"),
            
            # Normalized query for suggestions
            IndexModel(
                [("query_normalized", ASCENDING), ("timestamp", DESCENDING)],
                name="normalized_query_suggestions"
            ),
            
            # Success/failure analysis
            IndexModel([("is_successful", ASCENDING)], name="success_status"),
            
            # Slow query identification
            IndexModel([("performance.is_slow_query", ASCENDING)], name="slow_queries"),
            
            # User session tracking
            IndexModel(
                [("session_id", ASCENDING), ("timestamp", ASCENDING)],
                name="session_tracking",
                sparse=True
            ),
            
            # Recent searches
            IndexModel([("timestamp", DESCENDING)], name="recent_searches"),
            
            # User and case compound queries
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("case_id", ASCENDING),
                    ("timestamp", DESCENDING)
                ],
                name="user_case_searches"
            ),
            
            # TTL index for automatic cleanup (optional)
            IndexModel(
                [("timestamp", ASCENDING)],
                name="search_history_ttl",
                expireAfterSeconds=60*60*24*730,  # 2 years retention
                sparse=True
            )
        ]
    
    @classmethod
    def get_analytics_indexes(cls) -> List[IndexModel]:
        """Get index models for search_analytics collection."""
        return [
            # Analytics period queries
            IndexModel(
                [("period_type", ASCENDING), ("period_start", DESCENDING)],
                name="analytics_period"
            ),
            
            # User analytics
            IndexModel(
                [("user_id", ASCENDING), ("period_start", DESCENDING)],
                name="user_analytics",
                sparse=True
            ),
            
            # Case analytics
            IndexModel(
                [("case_id", ASCENDING), ("period_start", DESCENDING)],
                name="case_analytics",
                sparse=True
            ),
            
            # TTL for analytics cleanup
            IndexModel(
                [("created_at", ASCENDING)],
                name="analytics_ttl",
                expireAfterSeconds=60*60*24*365*2  # 2 years retention
            )
        ]
    
    @classmethod
    def get_suggestions_indexes(cls) -> List[IndexModel]:
        """Get index models for search_suggestions collection."""
        return [
            # Suggestion text lookup
            IndexModel([("normalized_text", ASCENDING)], name="suggestion_text"),
            
            # Frequency ranking
            IndexModel([("frequency", DESCENDING)], name="suggestion_frequency"),
            
            # Recent suggestions
            IndexModel([("last_used", DESCENDING)], name="suggestion_recent"),
            
            # Success rate ranking
            IndexModel([("success_rate", DESCENDING)], name="suggestion_quality"),
            
            # Context-based suggestions
            IndexModel([("context_case_ids", ASCENDING)], name="suggestion_context"),
            
            # Query type filtering
            IndexModel([("query_type", ASCENDING)], name="suggestion_type")
        ]
    
    @classmethod
    def get_patterns_indexes(cls) -> List[IndexModel]:
        """Get index models for search_patterns collection."""
        return [
            # User patterns
            IndexModel([("user_id", ASCENDING)], name="user_patterns"),
            
            # Pattern type filtering
            IndexModel([("pattern_type", ASCENDING)], name="pattern_type"),
            
            # Active patterns
            IndexModel([("is_active", ASCENDING)], name="active_patterns"),
            
            # Confidence ranking
            IndexModel([("confidence_score", DESCENDING)], name="pattern_confidence")
        ]


class SearchHistoryAggregations:
    """MongoDB aggregation pipelines for search analytics."""
    
    @classmethod
    def get_user_search_summary(cls, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Aggregation pipeline for user search summary."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        return [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": cutoff_date}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_searches": {"$sum": 1},
                    "unique_queries": {"$addToSet": "$query_normalized"},
                    "successful_searches": {
                        "$sum": {"$cond": ["$is_successful", 1, 0]}
                    },
                    "avg_execution_time": {"$avg": "$performance.execution_time_ms"},
                    "avg_results": {"$avg": "$results.total_results"},
                    "search_types": {"$push": "$search_type"},
                    "search_scopes": {"$push": "$search_scope"}
                }
            },
            {
                "$addFields": {
                    "unique_query_count": {"$size": "$unique_queries"},
                    "success_rate": {
                        "$divide": ["$successful_searches", "$total_searches"]
                    }
                }
            }
        ]
    
    @classmethod
    def get_popular_queries(cls, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Aggregation pipeline for popular queries."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        return [
            {
                "$match": {
                    "timestamp": {"$gte": cutoff_date},
                    "is_successful": True
                }
            },
            {
                "$group": {
                    "_id": "$query_normalized",
                    "query": {"$first": "$query"},
                    "frequency": {"$sum": 1},
                    "avg_results": {"$avg": "$results.total_results"},
                    "avg_execution_time": {"$avg": "$performance.execution_time_ms"},
                    "unique_users": {"$addToSet": "$user_id"},
                    "last_used": {"$max": "$timestamp"}
                }
            },
            {
                "$addFields": {
                    "unique_user_count": {"$size": "$unique_users"}
                }
            },
            {
                "$match": {
                    "frequency": {"$gte": 2}  # Minimum frequency threshold
                }
            },
            {
                "$sort": {
                    "frequency": -1,
                    "avg_results": -1,
                    "last_used": -1
                }
            },
            {"$limit": limit},
            {
                "$project": {
                    "query": 1,
                    "frequency": 1,
                    "avg_results": {"$round": ["$avg_results", 1]},
                    "avg_execution_time": {"$round": ["$avg_execution_time", 1]},
                    "unique_user_count": 1,
                    "last_used": 1
                }
            }
        ]
    
    @classmethod
    def get_performance_trends(cls, days: int = 7) -> List[Dict[str, Any]]:
        """Aggregation pipeline for performance trends."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        return [
            {
                "$match": {
                    "timestamp": {"$gte": cutoff_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "total_searches": {"$sum": 1},
                    "avg_execution_time": {"$avg": "$performance.execution_time_ms"},
                    "slow_queries": {
                        "$sum": {"$cond": ["$performance.is_slow_query", 1, 0]}
                    },
                    "successful_searches": {
                        "$sum": {"$cond": ["$is_successful", 1, 0]}
                    },
                    "zero_result_searches": {
                        "$sum": {"$cond": [{"$eq": ["$results.total_results", 0]}, 1, 0]}
                    }
                }
            },
            {
                "$addFields": {
                    "success_rate": {
                        "$divide": ["$successful_searches", "$total_searches"]
                    },
                    "slow_query_rate": {
                        "$divide": ["$slow_queries", "$total_searches"]
                    }
                }
            },
            {"$sort": {"_id": 1}}
        ]


# Collection configuration
SEARCH_COLLECTIONS = {
    SearchCollection.SEARCH_HISTORY: {
        "document_class": SearchHistoryDocument,
        "indexes": SearchHistoryIndexes.get_search_history_indexes(),
        "validators": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["search_id", "user_id", "query", "search_type", "search_scope"],
                "properties": {
                    "search_id": {"bsonType": "string", "minLength": 1},
                    "user_id": {"bsonType": "string", "minLength": 1},
                    "query": {"bsonType": "string", "minLength": 1, "maxLength": 1000},
                    "search_type": {"bsonType": "string"},
                    "search_scope": {"bsonType": "string"}
                }
            }
        }
    },
    SearchCollection.SEARCH_ANALYTICS: {
        "document_class": SearchAnalyticsDocument,
        "indexes": SearchHistoryIndexes.get_analytics_indexes(),
        "validators": None
    },
    SearchCollection.SEARCH_SUGGESTIONS: {
        "document_class": SearchSuggestionDocument,
        "indexes": SearchHistoryIndexes.get_suggestions_indexes(),
        "validators": None
    },
    SearchCollection.SEARCH_PATTERNS: {
        "document_class": SearchPatternDocument,
        "indexes": SearchHistoryIndexes.get_patterns_indexes(),
        "validators": None
    }
}


def get_collection_config(collection: SearchCollection) -> Dict[str, Any]:
    """Get configuration for a specific collection."""
    return SEARCH_COLLECTIONS.get(collection, {})


def validate_search_history_data(data: Dict[str, Any]) -> bool:
    """Validate search history data against schema."""
    try:
        SearchHistoryDocument(**data)
        return True
    except Exception as e:
        logger.error(f"Search history validation failed: {e}")
        return False


def create_search_id(user_id: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a unique search ID."""
    if not timestamp:
        timestamp = datetime.now(timezone.utc)
    
    date_str = timestamp.strftime('%Y%m%d_%H%M%S')
    unique_suffix = str(uuid.uuid4())[:8]
    
    return f"search_{date_str}_{unique_suffix}"


def normalize_query_for_suggestions(query: str) -> str:
    """Normalize query text for suggestion matching."""
    return SearchHistoryDocument._normalize_query_text(query)


def extract_query_intent(query: str, context: Dict[str, Any] = None) -> str:
    """Extract search intent from query text and context."""
    query_lower = query.lower()
    
    # Intent classification patterns
    exploratory_patterns = ['what', 'how', 'why', 'explain', 'overview', 'about']
    specific_patterns = ['find', 'locate', 'get', 'show', 'list', 'document']
    comparison_patterns = ['compare', 'versus', 'vs', 'difference', 'similar', 'like']
    
    # Check for patterns
    if any(pattern in query_lower for pattern in comparison_patterns):
        return 'comparison'
    elif any(pattern in query_lower for pattern in exploratory_patterns):
        return 'exploratory'
    elif any(pattern in query_lower for pattern in specific_patterns):
        return 'specific'
    
    # Default based on query length
    word_count = len(query.split())
    if word_count <= 2:
        return 'specific'
    elif word_count >= 8:
        return 'exploratory'
    else:
        return 'general'