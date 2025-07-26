"""
Pydantic API schemas for legal case management in Patexia Legal AI Chatbot.

This module defines the request/response schemas for case-related API endpoints:
- Case creation and update schemas
- Visual marker configuration schemas
- Case listing and filtering schemas
- Case metrics and statistics schemas
- API response wrappers with validation
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from backend.app.models.domain.case import (
    CaseStatus,
    CasePriority,
    VisualMarker,
    CaseMetrics
)


class VisualMarkerSchema(BaseModel):
    """Schema for case visual marker configuration."""
    
    color: str = Field(
        ...,
        description="Hex color code for case identification",
        regex=r"^#[0-9A-Fa-f]{6}$",
        example="#e74c3c"
    )
    
    icon: str = Field(
        ...,
        description="Icon emoji for case identification",
        min_length=1,
        max_length=2,
        example="📄"
    )
    
    @validator('color')
    def validate_color(cls, v):
        """Validate color is in predefined palette."""
        if v not in VisualMarker.COLORS:
            raise ValueError(f'Color must be one of: {", ".join(VisualMarker.COLORS)}')
        return v
    
    @validator('icon')
    def validate_icon(cls, v):
        """Validate icon is in predefined set."""
        if v not in VisualMarker.ICONS:
            raise ValueError(f'Icon must be one of: {", ".join(VisualMarker.ICONS)}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "color": "#3498db",
                "icon": "⚖️"
            }
        }


class CaseCreateRequest(BaseModel):
    """Schema for creating a new legal case."""
    
    case_name: str = Field(
        ...,
        description="Human-readable case name",
        min_length=3,
        max_length=200,
        example="Patent-2025-WiFi6-Dispute"
    )
    
    initial_summary: str = Field(
        ...,
        description="User-provided case description and context",
        min_length=10,
        max_length=2000,
        example="Patent infringement case involving WiFi6 technology patents. "
                "Multiple defendants alleged to be using proprietary wireless "
                "communication methods without proper licensing."
    )
    
    visual_marker: Optional[VisualMarkerSchema] = Field(
        None,
        description="Visual identification marker (auto-generated if not provided)"
    )
    
    priority: CasePriority = Field(
        CasePriority.MEDIUM,
        description="Case priority level for workflow management"
    )
    
    case_type: Optional[str] = Field(
        None,
        description="Type of legal case for automatic visual marker selection",
        max_length=100,
        example="patent"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Case tags for organization and filtering",
        max_items=20
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional case metadata"
    )
    
    @validator('case_name')
    def validate_case_name(cls, v):
        """Validate case name format."""
        if not v.strip():
            raise ValueError('Case name cannot be empty or whitespace only')
        return v.strip()
    
    @validator('initial_summary')
    def validate_initial_summary(cls, v):
        """Validate initial summary content."""
        if not v.strip():
            raise ValueError('Initial summary cannot be empty or whitespace only')
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and normalize tags."""
        if v is None:
            return v
        
        # Normalize tags: strip whitespace, convert to lowercase, remove duplicates
        normalized_tags = []
        seen_tags = set()
        
        for tag in v:
            if isinstance(tag, str) and tag.strip():
                normalized_tag = tag.strip().lower()
                if normalized_tag not in seen_tags and len(normalized_tag) <= 50:
                    normalized_tags.append(normalized_tag)
                    seen_tags.add(normalized_tag)
        
        return normalized_tags if normalized_tags else None
    
    class Config:
        schema_extra = {
            "example": {
                "case_name": "Patent-2025-WiFi6-Dispute",
                "initial_summary": "Patent infringement case involving WiFi6 technology. Multiple defendants using proprietary wireless methods without licensing.",
                "visual_marker": {
                    "color": "#3498db",
                    "icon": "🔍"
                },
                "priority": "high",
                "case_type": "patent",
                "tags": ["patent", "wireless", "technology", "litigation"],
                "metadata": {
                    "jurisdiction": "federal",
                    "court": "District Court for the Eastern District of Texas"
                }
            }
        }


class CaseUpdateRequest(BaseModel):
    """Schema for updating an existing legal case."""
    
    case_name: Optional[str] = Field(
        None,
        description="Updated case name",
        min_length=3,
        max_length=200
    )
    
    auto_summary: Optional[str] = Field(
        None,
        description="AI-generated case summary",
        max_length=5000
    )
    
    visual_marker: Optional[VisualMarkerSchema] = Field(
        None,
        description="Updated visual marker"
    )
    
    priority: Optional[CasePriority] = Field(
        None,
        description="Updated priority level"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Updated case tags",
        max_items=20
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated case metadata"
    )
    
    @validator('case_name')
    def validate_case_name(cls, v):
        """Validate case name if provided."""
        if v is not None and not v.strip():
            raise ValueError('Case name cannot be empty or whitespace only')
        return v.strip() if v else v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate and normalize tags if provided."""
        if v is None:
            return v
        
        # Same normalization logic as create request
        normalized_tags = []
        seen_tags = set()
        
        for tag in v:
            if isinstance(tag, str) and tag.strip():
                normalized_tag = tag.strip().lower()
                if normalized_tag not in seen_tags and len(normalized_tag) <= 50:
                    normalized_tags.append(normalized_tag)
                    seen_tags.add(normalized_tag)
        
        return normalized_tags if normalized_tags else None
    
    class Config:
        schema_extra = {
            "example": {
                "case_name": "Patent-2025-WiFi6-Updated",
                "auto_summary": "AI-generated summary based on uploaded documents...",
                "priority": "urgent",
                "tags": ["patent", "wireless", "urgent", "settlement"]
            }
        }


class CaseMetricsSchema(BaseModel):
    """Schema for case metrics and statistics."""
    
    total_documents: int = Field(
        ...,
        description="Total number of documents in the case",
        ge=0
    )
    
    processed_documents: int = Field(
        ...,
        description="Number of successfully processed documents",
        ge=0
    )
    
    failed_documents: int = Field(
        ...,
        description="Number of documents that failed processing",
        ge=0
    )
    
    total_chunks: int = Field(
        ...,
        description="Total number of text chunks across all documents",
        ge=0
    )
    
    total_search_queries: int = Field(
        ...,
        description="Total number of search queries performed",
        ge=0
    )
    
    last_activity: Optional[datetime] = Field(
        None,
        description="Timestamp of last case activity"
    )
    
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time in seconds",
        ge=0.0
    )
    
    storage_size_bytes: int = Field(
        ...,
        description="Total storage size in bytes",
        ge=0
    )
    
    processing_progress: float = Field(
        ...,
        description="Processing progress as percentage (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    has_failures: bool = Field(
        ...,
        description="Whether the case has any failed documents"
    )
    
    is_processing_complete: bool = Field(
        ...,
        description="Whether all documents have been processed successfully"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_documents": 15,
                "processed_documents": 14,
                "failed_documents": 1,
                "total_chunks": 347,
                "total_search_queries": 23,
                "last_activity": "2025-01-15T10:30:00Z",
                "processing_time_seconds": 125.5,
                "storage_size_bytes": 2457600,
                "processing_progress": 0.93,
                "has_failures": True,
                "is_processing_complete": False
            }
        }


class CaseResponse(BaseModel):
    """Schema for case API response."""
    
    case_id: str = Field(
        ...,
        description="Unique case identifier"
    )
    
    user_id: str = Field(
        ...,
        description="Case owner identifier"
    )
    
    case_name: str = Field(
        ...,
        description="Human-readable case name"
    )
    
    initial_summary: str = Field(
        ...,
        description="User-provided case description"
    )
    
    auto_summary: Optional[str] = Field(
        None,
        description="AI-generated case summary"
    )
    
    status: CaseStatus = Field(
        ...,
        description="Current case processing status"
    )
    
    priority: CasePriority = Field(
        ...,
        description="Case priority level"
    )
    
    visual_marker: VisualMarkerSchema = Field(
        ...,
        description="Visual identification marker"
    )
    
    created_at: datetime = Field(
        ...,
        description="Case creation timestamp"
    )
    
    updated_at: datetime = Field(
        ...,
        description="Last modification timestamp"
    )
    
    tags: List[str] = Field(
        ...,
        description="Case tags for organization"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Case metadata"
    )
    
    metrics: CaseMetricsSchema = Field(
        ...,
        description="Case metrics and statistics"
    )
    
    document_count: int = Field(
        ...,
        description="Current number of documents",
        ge=0
    )
    
    document_ids: List[str] = Field(
        ...,
        description="List of document identifiers in this case"
    )
    
    is_ready_for_search: bool = Field(
        ...,
        description="Whether the case is ready for search operations"
    )
    
    is_active: bool = Field(
        ...,
        description="Whether the case is currently active"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "user_id": "user_123",
                "case_name": "Patent-2025-WiFi6-Dispute",
                "initial_summary": "Patent infringement case involving WiFi6 technology...",
                "auto_summary": "This case involves multiple WiFi6 patents...",
                "status": "active",
                "priority": "high",
                "visual_marker": {
                    "color": "#3498db",
                    "icon": "🔍"
                },
                "created_at": "2025-01-15T09:00:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
                "tags": ["patent", "wireless", "technology"],
                "metadata": {
                    "jurisdiction": "federal"
                },
                "metrics": {
                    "total_documents": 15,
                    "processed_documents": 14,
                    "failed_documents": 1
                },
                "document_count": 15,
                "document_ids": ["DOC_001", "DOC_002"],
                "is_ready_for_search": True,
                "is_active": True
            }
        }


class CaseListRequest(BaseModel):
    """Schema for case listing and filtering request."""
    
    user_id: Optional[str] = Field(
        None,
        description="Filter by user ID (admin only)"
    )
    
    status: Optional[CaseStatus] = Field(
        None,
        description="Filter by case status"
    )
    
    priority: Optional[CasePriority] = Field(
        None,
        description="Filter by case priority"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="Filter by case tags (any match)",
        max_items=10
    )
    
    search_query: Optional[str] = Field(
        None,
        description="Search in case names and summaries",
        max_length=200
    )
    
    created_after: Optional[datetime] = Field(
        None,
        description="Filter cases created after this date"
    )
    
    created_before: Optional[datetime] = Field(
        None,
        description="Filter cases created before this date"
    )
    
    limit: int = Field(
        50,
        description="Maximum number of cases to return",
        ge=1,
        le=500
    )
    
    offset: int = Field(
        0,
        description="Number of cases to skip for pagination",
        ge=0
    )
    
    sort_by: str = Field(
        "updated_at",
        description="Field to sort by",
        regex=r"^(created_at|updated_at|case_name|priority|status)$"
    )
    
    sort_order: str = Field(
        "desc",
        description="Sort order",
        regex=r"^(asc|desc)$"
    )
    
    @validator('tags')
    def validate_tags(cls, v):
        """Normalize tag filters."""
        if v is None:
            return v
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @root_validator
    def validate_date_range(cls, values):
        """Validate date range consistency."""
        created_after = values.get('created_after')
        created_before = values.get('created_before')
        
        if created_after and created_before and created_after >= created_before:
            raise ValueError('created_after must be before created_before')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "status": "active",
                "priority": "high",
                "tags": ["patent", "urgent"],
                "search_query": "WiFi6",
                "limit": 20,
                "offset": 0,
                "sort_by": "updated_at",
                "sort_order": "desc"
            }
        }


class CaseListResponse(BaseModel):
    """Schema for case listing response with pagination."""
    
    cases: List[CaseResponse] = Field(
        ...,
        description="List of cases matching the criteria"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of cases matching filters",
        ge=0
    )
    
    offset: int = Field(
        ...,
        description="Number of cases skipped",
        ge=0
    )
    
    limit: int = Field(
        ...,
        description="Maximum cases per page",
        ge=1
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more cases available"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "cases": [
                    {
                        "case_id": "CASE_2025_01_15_A1B2C3D4",
                        "case_name": "Patent-2025-WiFi6-Dispute",
                        "status": "active"
                    }
                ],
                "total_count": 45,
                "offset": 0,
                "limit": 20,
                "has_more": True
            }
        }


class CaseStatusUpdateRequest(BaseModel):
    """Schema for updating case status."""
    
    status: CaseStatus = Field(
        ...,
        description="New case status"
    )
    
    reason: Optional[str] = Field(
        None,
        description="Reason for status change",
        max_length=500
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "archived",
                "reason": "Case resolved and archived for reference"
            }
        }


class CaseArchiveRequest(BaseModel):
    """Schema for archiving a case."""
    
    reason: Optional[str] = Field(
        None,
        description="Reason for archiving the case",
        max_length=500
    )
    
    preserve_documents: bool = Field(
        True,
        description="Whether to preserve documents when archiving"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "reason": "Case settled out of court",
                "preserve_documents": True
            }
        }


class CaseStatsResponse(BaseModel):
    """Schema for case statistics and analytics."""
    
    total_cases: int = Field(
        ...,
        description="Total number of cases",
        ge=0
    )
    
    active_cases: int = Field(
        ...,
        description="Number of active cases",
        ge=0
    )
    
    completed_cases: int = Field(
        ...,
        description="Number of completed cases",
        ge=0
    )
    
    archived_cases: int = Field(
        ...,
        description="Number of archived cases",
        ge=0
    )
    
    cases_with_errors: int = Field(
        ...,
        description="Number of cases with processing errors",
        ge=0
    )
    
    total_documents: int = Field(
        ...,
        description="Total documents across all cases",
        ge=0
    )
    
    total_processed_documents: int = Field(
        ...,
        description="Total successfully processed documents",
        ge=0
    )
    
    total_failed_documents: int = Field(
        ...,
        description="Total failed document processing attempts",
        ge=0
    )
    
    total_search_queries: int = Field(
        ...,
        description="Total search queries performed",
        ge=0
    )
    
    average_documents_per_case: float = Field(
        ...,
        description="Average number of documents per case",
        ge=0.0
    )
    
    processing_success_rate: float = Field(
        ...,
        description="Document processing success rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    status_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown of cases by status"
    )
    
    priority_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown of cases by priority"
    )
    
    recent_activity: List[Dict[str, Any]] = Field(
        ...,
        description="Recent case activity summary"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_cases": 127,
                "active_cases": 45,
                "completed_cases": 67,
                "archived_cases": 12,
                "cases_with_errors": 3,
                "total_documents": 1842,
                "total_processed_documents": 1789,
                "total_failed_documents": 53,
                "total_search_queries": 2156,
                "average_documents_per_case": 14.5,
                "processing_success_rate": 0.971,
                "status_breakdown": {
                    "active": 45,
                    "complete": 67,
                    "archived": 12,
                    "error": 3
                },
                "priority_breakdown": {
                    "low": 23,
                    "medium": 78,
                    "high": 21,
                    "urgent": 5
                },
                "recent_activity": [
                    {
                        "case_id": "CASE_123",
                        "action": "document_uploaded",
                        "timestamp": "2025-01-15T10:30:00Z"
                    }
                ]
            }
        }


class ApiResponse(BaseModel):
    """Generic API response wrapper."""
    
    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )
    
    message: Optional[str] = Field(
        None,
        description="Human-readable message"
    )
    
    data: Optional[Any] = Field(
        None,
        description="Response payload"
    )
    
    error: Optional[Dict[str, Any]] = Field(
        None,
        description="Error information if request failed"
    )
    
    correlation_id: Optional[str] = Field(
        None,
        description="Request correlation ID for tracking"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Case created successfully",
                "data": {
                    "case_id": "CASE_2025_01_15_A1B2C3D4"
                },
                "correlation_id": "req_123456"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for API error responses."""
    
    success: bool = Field(
        False,
        description="Always false for error responses"
    )
    
    error: Dict[str, Any] = Field(
        ...,
        description="Error details"
    )
    
    correlation_id: Optional[str] = Field(
        None,
        description="Request correlation ID for tracking"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "4001",
                    "message": "Case not found",
                    "details": {
                        "case_id": "CASE_INVALID"
                    }
                },
                "correlation_id": "req_123456"
            }
        }


# Convenience type aliases for API responses
CaseCreateResponse = ApiResponse
CaseUpdateResponse = ApiResponse
CaseDeleteResponse = ApiResponse
CaseArchiveResponse = ApiResponse