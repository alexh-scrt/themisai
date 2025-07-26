"""
MongoDB Database Models for Legal Case Management

This module defines the database layer models and schemas for legal case management
in the Patexia Legal AI Chatbot. It provides MongoDB document structures, validation,
indexing specifications, and data transformation utilities.

Key Features:
- MongoDB document schemas for case entities
- Field validation and business rule enforcement
- Index definitions for query optimization
- Data transformation utilities between domain and database models
- Aggregation pipeline definitions for analytics
- Performance optimization configurations
- Data integrity constraints and validation

Database Schema Design:
- Cases collection with comprehensive indexing
- Document tracking and metrics aggregation
- Visual marker and metadata management
- Search optimization with text indexes
- User-based access control integration
- Audit trail and timestamp management

Collections:
- cases: Primary case data with metrics and metadata
- case_analytics: Aggregated analytics and reporting data
- case_history: Audit trail and change tracking
- case_templates: Reusable case templates and configurations

Data Integrity:
- Unique constraints on case identifiers
- Referential integrity with document collections
- Validation rules for business logic enforcement
- Atomic operations for consistency
- Transaction support for complex operations

Performance Optimizations:
- Compound indexes for common query patterns
- Text search indexes for case discovery
- Aggregation pipeline optimization
- Efficient pagination and sorting
- Cache-friendly data structures
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from bson import ObjectId
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pydantic import BaseModel, Field, validator, root_validator

from ..domain.case import CaseStatus, CasePriority, VisualMarker, CaseMetrics, LegalCase
from ...utils.logging import get_logger

logger = get_logger(__name__)


class CaseCollection(str, Enum):
    """Case-related MongoDB collection names."""
    CASES = "cases"
    CASE_ANALYTICS = "case_analytics"
    CASE_HISTORY = "case_history"
    CASE_TEMPLATES = "case_templates"


class IndexType(str, Enum):
    """MongoDB index types for case collections."""
    UNIQUE = "unique"
    COMPOUND = "compound"
    TEXT = "text"
    SPARSE = "sparse"
    TTL = "ttl"


@dataclass
class IndexDefinition:
    """MongoDB index definition for case collections."""
    fields: List[Tuple[str, int]]
    options: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    description: Optional[str] = None


class VisualMarkerDocument(BaseModel):
    """MongoDB document model for visual markers."""
    color: str = Field(
        ...,
        description="Color code for visual identification",
        regex=r"^#[0-9A-Fa-f]{6}$|^[a-z-]+$"
    )
    
    icon: str = Field(
        ...,
        description="Icon identifier for visual marker",
        min_length=1,
        max_length=50
    )
    
    class Config:
        schema_extra = {
            "example": {
                "color": "#2563eb",
                "icon": "briefcase"
            }
        }
    
    @validator('color')
    def validate_color(cls, v):
        """Validate color format."""
        if not (v.startswith('#') and len(v) == 7) and not v.islower():
            raise ValueError('Color must be hex code #RRGGBB or lowercase color name')
        return v
    
    def to_domain(self) -> VisualMarker:
        """Convert to domain model."""
        return VisualMarker(color=self.color, icon=self.icon)
    
    @classmethod
    def from_domain(cls, marker: VisualMarker) -> "VisualMarkerDocument":
        """Create from domain model."""
        return cls(color=marker.color, icon=marker.icon)


class CaseMetricsDocument(BaseModel):
    """MongoDB document model for case metrics."""
    total_documents: int = Field(
        default=0,
        description="Total number of documents in case",
        ge=0
    )
    
    processed_documents: int = Field(
        default=0,
        description="Number of successfully processed documents",
        ge=0
    )
    
    failed_documents: int = Field(
        default=0,
        description="Number of failed document processing attempts",
        ge=0
    )
    
    total_chunks: int = Field(
        default=0,
        description="Total text chunks generated",
        ge=0
    )
    
    total_search_queries: int = Field(
        default=0,
        description="Total search queries performed",
        ge=0
    )
    
    last_activity: Optional[datetime] = Field(
        None,
        description="Timestamp of last case activity"
    )
    
    processing_time_seconds: float = Field(
        default=0.0,
        description="Total processing time in seconds",
        ge=0.0
    )
    
    storage_size_bytes: int = Field(
        default=0,
        description="Total storage size in bytes",
        ge=0
    )
    
    processing_progress: float = Field(
        default=0.0,
        description="Document processing progress percentage",
        ge=0.0,
        le=100.0
    )
    
    average_query_time_ms: Optional[float] = Field(
        None,
        description="Average search query time in milliseconds"
    )
    
    success_rate: Optional[float] = Field(
        None,
        description="Document processing success rate",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_documents": 15,
                "processed_documents": 14,
                "failed_documents": 1,
                "total_chunks": 342,
                "total_search_queries": 28,
                "processing_time_seconds": 45.7,
                "storage_size_bytes": 1048576,
                "processing_progress": 93.3
            }
        }
    
    @validator('processed_documents')
    def validate_processed_documents(cls, v, values):
        """Ensure processed documents don't exceed total."""
        if 'total_documents' in values and v > values['total_documents']:
            raise ValueError('Processed documents cannot exceed total documents')
        return v
    
    @validator('failed_documents')
    def validate_failed_documents(cls, v, values):
        """Ensure failed documents don't exceed total."""
        if 'total_documents' in values and v > values['total_documents']:
            raise ValueError('Failed documents cannot exceed total documents')
        return v
    
    @root_validator
    def validate_document_counts(cls, values):
        """Validate document count relationships."""
        total = values.get('total_documents', 0)
        processed = values.get('processed_documents', 0)
        failed = values.get('failed_documents', 0)
        
        if processed + failed > total:
            raise ValueError('Sum of processed and failed documents cannot exceed total')
        
        # Calculate derived metrics
        if total > 0:
            values['processing_progress'] = (processed / total) * 100.0
            values['success_rate'] = processed / total
        
        return values
    
    def to_domain(self) -> CaseMetrics:
        """Convert to domain model."""
        metrics = CaseMetrics(
            total_documents=self.total_documents,
            processed_documents=self.processed_documents,
            failed_documents=self.failed_documents,
            total_chunks=self.total_chunks,
            total_search_queries=self.total_search_queries,
            processing_time_seconds=self.processing_time_seconds,
            storage_size_bytes=self.storage_size_bytes
        )
        
        if self.last_activity:
            metrics.last_activity = self.last_activity
        
        return metrics
    
    @classmethod
    def from_domain(cls, metrics: CaseMetrics) -> "CaseMetricsDocument":
        """Create from domain model."""
        return cls(
            total_documents=metrics.total_documents,
            processed_documents=metrics.processed_documents,
            failed_documents=metrics.failed_documents,
            total_chunks=metrics.total_chunks,
            total_search_queries=metrics.total_search_queries,
            last_activity=metrics.last_activity,
            processing_time_seconds=metrics.processing_time_seconds,
            storage_size_bytes=metrics.storage_size_bytes
        )


class CaseDocument(BaseModel):
    """MongoDB document model for legal cases."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    case_id: str = Field(
        ...,
        description="Unique case identifier",
        min_length=10,
        max_length=100
    )
    
    user_id: str = Field(
        ...,
        description="Case owner identifier",
        min_length=1,
        max_length=100
    )
    
    case_name: str = Field(
        ...,
        description="Human-readable case name",
        min_length=3,
        max_length=200
    )
    
    initial_summary: str = Field(
        ...,
        description="User-provided case description",
        min_length=10,
        max_length=2000
    )
    
    auto_summary: Optional[str] = Field(
        None,
        description="AI-generated case summary",
        max_length=5000
    )
    
    status: str = Field(
        default=CaseStatus.DRAFT.value,
        description="Current case processing status"
    )
    
    priority: str = Field(
        default=CasePriority.MEDIUM.value,
        description="Case priority level"
    )
    
    visual_marker: VisualMarkerDocument = Field(
        ...,
        description="Visual identification marker"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Case creation timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Case tags for organization",
        max_items=20
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional case metadata"
    )
    
    metrics: CaseMetricsDocument = Field(
        default_factory=CaseMetricsDocument,
        description="Case metrics and statistics"
    )
    
    document_ids: List[str] = Field(
        default_factory=list,
        description="List of document identifiers in this case",
        max_items=50  # Allow for manual override beyond standard 25 limit
    )
    
    # Administrative fields
    version: int = Field(
        default=1,
        description="Document version for optimistic locking"
    )
    
    archived_at: Optional[datetime] = Field(
        None,
        description="Timestamp when case was archived"
    )
    
    last_accessed: Optional[datetime] = Field(
        None,
        description="Timestamp of last case access"
    )
    
    access_count: int = Field(
        default=0,
        description="Number of times case has been accessed"
    )
    
    # Search and indexing optimization
    search_keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords for search optimization"
    )
    
    full_text_content: Optional[str] = Field(
        None,
        description="Combined text content for full-text search"
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "user_id": "user_123",
                "case_name": "Patent-2025-WiFi6-Dispute",
                "initial_summary": "Patent infringement case involving WiFi 6 technology...",
                "status": "active",
                "priority": "high",
                "visual_marker": {
                    "color": "#2563eb",
                    "icon": "briefcase"
                },
                "tags": ["patent", "wifi", "technology"],
                "metadata": {
                    "case_type": "patent_infringement",
                    "jurisdiction": "federal"
                }
            }
        }
    
    @validator('case_id')
    def validate_case_id(cls, v):
        """Validate case ID format."""
        if not v.startswith('CASE_'):
            raise ValueError('Case ID must start with CASE_')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """Validate case status."""
        try:
            CaseStatus(v)
        except ValueError:
            raise ValueError(f'Invalid case status: {v}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate case priority."""
        try:
            CasePriority(v)
        except ValueError:
            raise ValueError(f'Invalid case priority: {v}')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and normalize tags."""
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    @validator('updated_at')
    def validate_updated_at(cls, v, values):
        """Ensure updated_at is not before created_at."""
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('Updated timestamp cannot be before created timestamp')
        return v
    
    @root_validator
    def validate_document_limits(cls, values):
        """Validate document count limits."""
        document_ids = values.get('document_ids', [])
        status = values.get('status')
        
        # Standard limit enforcement
        if len(document_ids) > 25 and status != CaseStatus.ARCHIVED.value:
            # Allow override but log warning
            logger.warning(
                f"Case exceeds standard document limit",
                case_id=values.get('case_id'),
                document_count=len(document_ids)
            )
        
        return values
    
    def to_domain(self) -> LegalCase:
        """Convert to domain model."""
        # Convert visual marker
        visual_marker = self.visual_marker.to_domain()
        
        # Convert metrics
        metrics = self.metrics.to_domain()
        
        # Create domain object
        case = LegalCase(
            case_id=self.case_id,
            user_id=self.user_id,
            case_name=self.case_name,
            initial_summary=self.initial_summary,
            visual_marker=visual_marker,
            priority=CasePriority(self.priority),
            created_at=self.created_at,
            updated_at=self.updated_at,
            status=CaseStatus(self.status),
            auto_summary=self.auto_summary,
            tags=set(self.tags),
            metadata=self.metadata.copy()
        )
        
        # Restore internal state
        case._document_ids = set(self.document_ids)
        case._metrics = metrics
        
        return case
    
    @classmethod
    def from_domain(cls, case: LegalCase) -> "CaseDocument":
        """Create from domain model."""
        return cls(
            case_id=case.case_id,
            user_id=case.user_id,
            case_name=case.case_name,
            initial_summary=case.initial_summary,
            auto_summary=case.auto_summary,
            status=case.status.value,
            priority=case.priority.value,
            visual_marker=VisualMarkerDocument.from_domain(case.visual_marker),
            created_at=case.created_at,
            updated_at=case.updated_at,
            tags=list(case.tags),
            metadata=case.metadata.copy(),
            metrics=CaseMetricsDocument.from_domain(case.metrics),
            document_ids=list(case.document_ids)
        )
    
    def update_search_content(self) -> None:
        """Update search-optimized content fields."""
        # Extract keywords from case name and summary
        text_content = f"{self.case_name} {self.initial_summary}"
        if self.auto_summary:
            text_content += f" {self.auto_summary}"
        
        # Add tags
        text_content += " " + " ".join(self.tags)
        
        # Store full text content for search
        self.full_text_content = text_content
        
        # Extract keywords (simple implementation)
        words = text_content.lower().split()
        self.search_keywords = list(set([
            word.strip('.,!?";()[]{}')
            for word in words
            if len(word) > 3 and word.isalpha()
        ]))
    
    def increment_access(self) -> None:
        """Increment access count and update timestamp."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class CaseAnalyticsDocument(BaseModel):
    """MongoDB document model for case analytics aggregation."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    case_id: str = Field(
        ...,
        description="Associated case identifier"
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
        description="Period type (daily, weekly, monthly)"
    )
    
    # Activity metrics
    search_queries_count: int = Field(
        default=0,
        description="Number of search queries in period"
    )
    
    document_uploads_count: int = Field(
        default=0,
        description="Number of documents uploaded in period"
    )
    
    processing_time_total: float = Field(
        default=0.0,
        description="Total processing time in period (seconds)"
    )
    
    average_response_time: Optional[float] = Field(
        None,
        description="Average query response time (milliseconds)"
    )
    
    # Usage patterns
    peak_usage_hour: Optional[int] = Field(
        None,
        description="Hour of day with peak usage"
    )
    
    most_searched_terms: List[str] = Field(
        default_factory=list,
        description="Most frequently searched terms"
    )
    
    # Performance metrics
    error_count: int = Field(
        default=0,
        description="Number of errors in period"
    )
    
    success_rate: Optional[float] = Field(
        None,
        description="Operation success rate in period"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analytics record creation time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "period_start": "2025-01-15T00:00:00Z",
                "period_end": "2025-01-15T23:59:59Z",
                "period_type": "daily",
                "search_queries_count": 45,
                "processing_time_total": 123.5,
                "average_response_time": 2.8
            }
        }


class CaseHistoryDocument(BaseModel):
    """MongoDB document model for case change history."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    case_id: str = Field(
        ...,
        description="Associated case identifier"
    )
    
    user_id: str = Field(
        ...,
        description="User who made the change"
    )
    
    action: str = Field(
        ...,
        description="Type of action performed"
    )
    
    field_changed: Optional[str] = Field(
        None,
        description="Specific field that was changed"
    )
    
    old_value: Optional[Any] = Field(
        None,
        description="Previous value before change"
    )
    
    new_value: Optional[Any] = Field(
        None,
        description="New value after change"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the change occurred"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional change metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "user_id": "user_123",
                "action": "status_change",
                "field_changed": "status",
                "old_value": "draft",
                "new_value": "active",
                "metadata": {"reason": "case_activated"}
            }
        }


class CaseIndexes:
    """MongoDB index definitions for case collections."""
    
    @classmethod
    def get_case_indexes(cls) -> List[IndexModel]:
        """Get index models for cases collection."""
        return [
            # Unique constraint on case_id
            IndexModel([("case_id", ASCENDING)], unique=True, name="case_id_unique"),
            
            # Unique constraint on user_id + case_id combination
            IndexModel(
                [("user_id", ASCENDING), ("case_id", ASCENDING)],
                unique=True,
                name="user_case_unique"
            ),
            
            # User cases with sorting by update time
            IndexModel(
                [("user_id", ASCENDING), ("updated_at", DESCENDING)],
                name="user_cases_updated"
            ),
            
            # Status filtering with sorting
            IndexModel(
                [("status", ASCENDING), ("updated_at", DESCENDING)],
                name="status_updated"
            ),
            
            # Priority filtering with sorting
            IndexModel(
                [("priority", ASCENDING), ("updated_at", DESCENDING)],
                name="priority_updated"
            ),
            
            # Text search index for case discovery
            IndexModel(
                [
                    ("case_name", TEXT),
                    ("initial_summary", TEXT),
                    ("auto_summary", TEXT),
                    ("full_text_content", TEXT)
                ],
                name="case_text_search",
                weights={
                    "case_name": 10,
                    "initial_summary": 5,
                    "auto_summary": 3,
                    "full_text_content": 1
                }
            ),
            
            # Creation date range queries
            IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
            
            # Tag filtering
            IndexModel([("tags", ASCENDING)], name="tags_filter"),
            
            # Last accessed for analytics
            IndexModel([("last_accessed", DESCENDING)], name="last_accessed_desc"),
            
            # Access count for popular cases
            IndexModel([("access_count", DESCENDING)], name="access_count_desc"),
            
            # Compound index for active cases by user
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("status", ASCENDING),
                    ("updated_at", DESCENDING)
                ],
                name="user_active_cases"
            ),
            
            # Archive status with TTL for cleanup
            IndexModel(
                [("archived_at", ASCENDING)],
                name="archived_cases",
                sparse=True,
                expireAfterSeconds=60*60*24*365*2  # 2 years retention for archived cases
            )
        ]
    
    @classmethod
    def get_analytics_indexes(cls) -> List[IndexModel]:
        """Get index models for case analytics collection."""
        return [
            # Case analytics by period
            IndexModel(
                [("case_id", ASCENDING), ("period_start", DESCENDING)],
                name="case_analytics_period"
            ),
            
            # Analytics by period type
            IndexModel(
                [("period_type", ASCENDING), ("period_start", DESCENDING)],
                name="period_type_analytics"
            ),
            
            # TTL for analytics data cleanup
            IndexModel(
                [("created_at", ASCENDING)],
                name="analytics_ttl",
                expireAfterSeconds=60*60*24*365  # 1 year retention
            )
        ]
    
    @classmethod
    def get_history_indexes(cls) -> List[IndexModel]:
        """Get index models for case history collection."""
        return [
            # Case history chronological
            IndexModel(
                [("case_id", ASCENDING), ("timestamp", DESCENDING)],
                name="case_history_chrono"
            ),
            
            # User activity history
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="user_activity_history"
            ),
            
            # Action type filtering
            IndexModel([("action", ASCENDING)], name="action_filter"),
            
            # TTL for history cleanup
            IndexModel(
                [("timestamp", ASCENDING)],
                name="history_ttl",
                expireAfterSeconds=60*60*24*365*3  # 3 years retention
            )
        ]


class CaseAggregations:
    """MongoDB aggregation pipelines for case analytics."""
    
    @classmethod
    def get_user_case_summary(cls, user_id: str) -> List[Dict[str, Any]]:
        """Aggregation pipeline for user case summary."""
        return [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "total_documents": {"$sum": "$metrics.total_documents"},
                    "total_queries": {"$sum": "$metrics.total_search_queries"},
                    "avg_processing_time": {"$avg": "$metrics.processing_time_seconds"}
                }
            },
            {"$sort": {"count": -1}}
        ]
    
    @classmethod
    def get_case_activity_timeline(cls, case_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Aggregation pipeline for case activity timeline."""
        return [
            {
                "$match": {
                    "case_id": case_id,
                    "period_start": {
                        "$gte": datetime.now(timezone.utc) - timedelta(days=days)
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$period_start"
                        }
                    },
                    "searches": {"$sum": "$search_queries_count"},
                    "uploads": {"$sum": "$document_uploads_count"},
                    "processing_time": {"$sum": "$processing_time_total"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
    
    @classmethod
    def get_popular_search_terms(cls, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Aggregation pipeline for popular search terms."""
        match_stage = {}
        if user_id:
            match_stage["user_id"] = user_id
        
        pipeline = []
        if match_stage:
            pipeline.append({"$match": match_stage})
        
        pipeline.extend([
            {"$unwind": "$most_searched_terms"},
            {
                "$group": {
                    "_id": "$most_searched_terms",
                    "frequency": {"$sum": 1}
                }
            },
            {"$sort": {"frequency": -1}},
            {"$limit": limit}
        ])
        
        return pipeline


# Collection configuration
CASE_COLLECTIONS = {
    CaseCollection.CASES: {
        "document_class": CaseDocument,
        "indexes": CaseIndexes.get_case_indexes(),
        "validators": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["case_id", "user_id", "case_name", "initial_summary"],
                "properties": {
                    "case_id": {"bsonType": "string", "pattern": "^CASE_"},
                    "user_id": {"bsonType": "string", "minLength": 1},
                    "case_name": {"bsonType": "string", "minLength": 3, "maxLength": 200},
                    "initial_summary": {"bsonType": "string", "minLength": 10, "maxLength": 2000}
                }
            }
        }
    },
    CaseCollection.CASE_ANALYTICS: {
        "document_class": CaseAnalyticsDocument,
        "indexes": CaseIndexes.get_analytics_indexes(),
        "validators": None
    },
    CaseCollection.CASE_HISTORY: {
        "document_class": CaseHistoryDocument,
        "indexes": CaseIndexes.get_history_indexes(),
        "validators": None
    }
}


def get_collection_config(collection: CaseCollection) -> Dict[str, Any]:
    """Get configuration for a specific collection."""
    return CASE_COLLECTIONS.get(collection, {})


def validate_case_document(doc_data: Dict[str, Any]) -> bool:
    """Validate case document data against schema."""
    try:
        CaseDocument(**doc_data)
        return True
    except Exception as e:
        logger.error(f"Case document validation failed: {e}")
        return False


def create_case_id(date_prefix: Optional[str] = None) -> str:
    """Generate a unique case ID with optional date prefix."""
    if not date_prefix:
        date_prefix = datetime.now().strftime('%Y_%m_%d')
    
    unique_suffix = str(uuid.uuid4())[:8].upper()
    return f"CASE_{date_prefix}_{unique_suffix}"


def hash_case_content(case_name: str, initial_summary: str) -> str:
    """Generate hash for case content deduplication."""
    content = f"{case_name.lower().strip()}|{initial_summary.lower().strip()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]