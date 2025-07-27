"""
Pydantic API schemas for document management in Patexia Legal AI Chatbot.

This module defines the request/response schemas for document-related API endpoints:
- Document upload and processing schemas
- Document chunk and embedding schemas
- Document status and progress tracking schemas
- Document search and retrieval schemas
- Batch operations and validation schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from backend.app.models.domain.document import (
    DocumentType,
    ProcessingStatus,
    DocumentPriority
)


class DocumentUploadRequest(BaseModel):
    """Schema for document upload request."""
    
    case_id: str = Field(
        ...,
        description="Case identifier to associate the document with",
        min_length=1,
        max_length=100
    )
    
    document_name: str = Field(
        ...,
        description="Display name for the document",
        min_length=1,
        max_length=255
    )
    
    priority: DocumentPriority = Field(
        DocumentPriority.NORMAL,
        description="Processing priority level"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional document metadata"
    )
    
    @validator('document_name')
    @classmethod
    def validate_document_name(cls, v):
        """Validate and normalize document name."""
        if not v.strip():
            raise ValueError('Document name cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "document_name": "Patent Application WiFi6 2024",
                "priority": "high",
                "metadata": {
                    "source": "USPTO filing",
                    "confidentiality": "attorney-client"
                }
            }
        }


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    
    document_id: str = Field(
        ...,
        description="Unique document identifier"
    )
    
    document_name: str = Field(
        ...,
        description="Document display name"
    )
    
    file_type: DocumentType = Field(
        ...,
        description="Detected document type"
    )
    
    file_size: int = Field(
        ...,
        description="File size in bytes",
        ge=0
    )
    
    status: ProcessingStatus = Field(
        ...,
        description="Initial processing status"
    )
    
    upload_timestamp: datetime = Field(
        ...,
        description="Document upload timestamp"
    )
    
    estimated_processing_time: Optional[int] = Field(
        None,
        description="Estimated processing time in seconds",
        ge=0
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "DOC_20250115_A1B2C3D4",
                "document_name": "Patent Application WiFi6 2024",
                "file_type": "pdf",
                "file_size": 2457600,
                "status": "pending",
                "upload_timestamp": "2025-01-15T10:30:00Z",
                "estimated_processing_time": 45
            }
        }


class ProcessingMetadataSchema(BaseModel):
    """Schema for document processing metadata."""
    
    processing_method: str = Field(
        ...,
        description="Method used for document processing"
    )
    
    processing_version: str = Field(
        ...,
        description="Version of processing pipeline"
    )
    
    processing_started_at: Optional[datetime] = Field(
        None,
        description="Processing start timestamp"
    )
    
    processing_completed_at: Optional[datetime] = Field(
        None,
        description="Processing completion timestamp"
    )
    
    extraction_confidence: float = Field(
        ...,
        description="Confidence score for text extraction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    chunk_count: int = Field(
        ...,
        description="Number of text chunks created",
        ge=0
    )
    
    embedding_model: Optional[str] = Field(
        None,
        description="Embedding model used for vectorization"
    )
    
    embedding_dimensions: int = Field(
        ...,
        description="Dimensions of embedding vectors",
        ge=0
    )
    
    index_collection: Optional[str] = Field(
        None,
        description="Vector database collection name"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    
    error_stage: Optional[str] = Field(
        None,
        description="Processing stage where error occurred"
    )
    
    retry_count: int = Field(
        ...,
        description="Number of retry attempts",
        ge=0
    )
    
    last_retry_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last retry attempt"
    )
    
    processing_duration_seconds: Optional[float] = Field(
        None,
        description="Total processing duration in seconds",
        ge=0.0
    )
    
    is_processing_active: bool = Field(
        ...,
        description="Whether processing is currently active"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "processing_method": "LlamaIndex",
                "processing_version": "1.0",
                "processing_started_at": "2025-01-15T10:30:00Z",
                "processing_completed_at": "2025-01-15T10:32:15Z",
                "extraction_confidence": 0.95,
                "chunk_count": 47,
                "embedding_model": "mxbai-embed-large",
                "embedding_dimensions": 1000,
                "index_collection": "LegalDocument_CASE_123",
                "error_message": None,
                "error_stage": None,
                "retry_count": 0,
                "last_retry_at": None,
                "processing_duration_seconds": 135.2,
                "is_processing_active": False
            }
        }


class DocumentChunkSchema(BaseModel):
    """Schema for document text chunks."""
    
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier"
    )
    
    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )
    
    content: str = Field(
        ...,
        description="Text content of the chunk",
        max_length=10000
    )
    
    chunk_index: int = Field(
        ...,
        description="Index of this chunk within the document",
        ge=0
    )
    
    start_char: int = Field(
        ...,
        description="Starting character position in original text",
        ge=0
    )
    
    end_char: int = Field(
        ...,
        description="Ending character position in original text",
        ge=0
    )
    
    chunk_size: int = Field(
        ...,
        description="Size of chunk in characters",
        ge=0
    )
    
    section_title: Optional[str] = Field(
        None,
        description="Section header if available",
        max_length=500
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
    
    has_embedding: bool = Field(
        ...,
        description="Whether chunk has an embedding vector"
    )
    
    embedding_dimensions: int = Field(
        ...,
        description="Dimensions of embedding vector",
        ge=0
    )
    
    created_at: datetime = Field(
        ...,
        description="Chunk creation timestamp"
    )
    
    @root_validator
    @classmethod
    def validate_char_positions(cls, values):
        """Validate character position consistency."""
        start_char = values.get('start_char', 0)
        end_char = values.get('end_char', 0)
        chunk_size = values.get('chunk_size', 0)
        
        if end_char <= start_char:
            raise ValueError('end_char must be greater than start_char')
        
        expected_size = end_char - start_char
        if abs(chunk_size - expected_size) > 10:  # Allow small discrepancies
            raise ValueError(f'chunk_size ({chunk_size}) inconsistent with char positions ({expected_size})')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "DOC_123_chunk_0001",
                "document_id": "DOC_20250115_A1B2C3D4",
                "content": "The present invention relates to wireless communication systems implementing IEEE 802.11ax standard...",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 512,
                "chunk_size": 512,
                "section_title": "1. Technical Field",
                "page_number": 1,
                "paragraph_number": 1,
                "legal_citations": ["35 U.S.C. § 101", "IEEE 802.11ax"],
                "has_embedding": True,
                "embedding_dimensions": 1000,
                "created_at": "2025-01-15T10:31:45Z"
            }
        }


class DocumentResponse(BaseModel):
    """Schema for comprehensive document response."""
    
    document_id: str = Field(
        ...,
        description="Unique document identifier"
    )
    
    user_id: str = Field(
        ...,
        description="Document owner identifier"
    )
    
    case_id: str = Field(
        ...,
        description="Associated case identifier"
    )
    
    document_name: str = Field(
        ...,
        description="Document display name"
    )
    
    original_filename: str = Field(
        ...,
        description="Original uploaded filename"
    )
    
    file_type: DocumentType = Field(
        ...,
        description="Document file type"
    )
    
    file_size: int = Field(
        ...,
        description="File size in bytes",
        ge=0
    )
    
    file_hash: str = Field(
        ...,
        description="File content hash for deduplication"
    )
    
    priority: DocumentPriority = Field(
        ...,
        description="Processing priority level"
    )
    
    status: ProcessingStatus = Field(
        ...,
        description="Current processing status"
    )
    
    text_content: Optional[str] = Field(
        None,
        description="Extracted text content"
    )
    
    created_at: datetime = Field(
        ...,
        description="Document creation timestamp"
    )
    
    updated_at: datetime = Field(
        ...,
        description="Last modification timestamp"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Document metadata"
    )
    
    processing_metadata: ProcessingMetadataSchema = Field(
        ...,
        description="Processing metadata and status"
    )
    
    chunk_count: int = Field(
        ...,
        description="Number of text chunks",
        ge=0
    )
    
    legal_citations: List[str] = Field(
        ...,
        description="Legal citations found in document"
    )
    
    section_headers: List[str] = Field(
        ...,
        description="Document section headers"
    )
    
    page_count: Optional[int] = Field(
        None,
        description="Number of pages in document",
        ge=1
    )
    
    is_processing_complete: bool = Field(
        ...,
        description="Whether processing is completed successfully"
    )
    
    is_processing_failed: bool = Field(
        ...,
        description="Whether processing has failed"
    )
    
    is_ready_for_search: bool = Field(
        ...,
        description="Whether document is ready for search operations"
    )
    
    can_retry: bool = Field(
        ...,
        description="Whether document can be retried after failure"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "DOC_20250115_A1B2C3D4",
                "user_id": "user_123",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "document_name": "Patent Application WiFi6 2024",
                "original_filename": "patent_app_wifi6.pdf",
                "file_type": "pdf",
                "file_size": 2457600,
                "file_hash": "sha256:abc123...",
                "priority": "high",
                "status": "completed",
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-15T10:32:15Z",
                "metadata": {},
                "processing_metadata": {},
                "chunk_count": 47,
                "legal_citations": ["35 U.S.C. § 101"],
                "section_headers": ["Technical Field", "Background"],
                "page_count": 25,
                "is_processing_complete": True,
                "is_processing_failed": False,
                "is_ready_for_search": True,
                "can_retry": False
            }
        }


class DocumentListRequest(BaseModel):
    """Schema for document listing and filtering request."""
    
    case_id: Optional[str] = Field(
        None,
        description="Filter by case identifier"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Filter by user identifier (admin only)"
    )
    
    status: Optional[ProcessingStatus] = Field(
        None,
        description="Filter by processing status"
    )
    
    file_type: Optional[DocumentType] = Field(
        None,
        description="Filter by document type"
    )
    
    priority: Optional[DocumentPriority] = Field(
        None,
        description="Filter by processing priority"
    )
    
    search_query: Optional[str] = Field(
        None,
        description="Search in document names and content",
        max_length=200
    )
    
    has_failures: Optional[bool] = Field(
        None,
        description="Filter documents with/without processing failures"
    )
    
    created_after: Optional[datetime] = Field(
        None,
        description="Filter documents created after this date"
    )
    
    created_before: Optional[datetime] = Field(
        None,
        description="Filter documents created before this date"
    )
    
    min_file_size: Optional[int] = Field(
        None,
        description="Minimum file size in bytes",
        ge=0
    )
    
    max_file_size: Optional[int] = Field(
        None,
        description="Maximum file size in bytes",
        ge=0
    )
    
    limit: int = Field(
        50,
        description="Maximum number of documents to return",
        ge=1,
        le=500
    )
    
    offset: int = Field(
        0,
        description="Number of documents to skip for pagination",
        ge=0
    )
    
    sort_by: str = Field(
        "updated_at",
        description="Field to sort by",
        regex=r"^(created_at|updated_at|document_name|file_size|status|priority)$"
    )
    
    sort_order: str = Field(
        "desc",
        description="Sort order",
        regex=r"^(asc|desc)$"
    )
    
    include_chunks: bool = Field(
        False,
        description="Whether to include chunk information in response"
    )
    
    include_content: bool = Field(
        False,
        description="Whether to include full text content in response"
    )
    
    @root_validator
    @classmethod
    def validate_filters(cls, values):
        """Validate filter consistency."""
        created_after = values.get('created_after')
        created_before = values.get('created_before')
        
        if created_after and created_before and created_after >= created_before:
            raise ValueError('created_after must be before created_before')
        
        min_size = values.get('min_file_size')
        max_size = values.get('max_file_size')
        
        if min_size and max_size and min_size > max_size:
            raise ValueError('min_file_size must be less than or equal to max_file_size')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "status": "completed",
                "file_type": "pdf",
                "search_query": "patent",
                "limit": 20,
                "offset": 0,
                "sort_by": "updated_at",
                "sort_order": "desc",
                "include_chunks": False,
                "include_content": False
            }
        }


class DocumentListResponse(BaseModel):
    """Schema for document listing response with pagination."""
    
    documents: List[DocumentResponse] = Field(
        ...,
        description="List of documents matching the criteria"
    )
    
    total_count: int = Field(
        ...,
        description="Total number of documents matching filters",
        ge=0
    )
    
    offset: int = Field(
        ...,
        description="Number of documents skipped",
        ge=0
    )
    
    limit: int = Field(
        ...,
        description="Maximum documents per page",
        ge=1
    )
    
    has_more: bool = Field(
        ...,
        description="Whether there are more documents available"
    )
    
    processing_summary: Dict[str, int] = Field(
        ...,
        description="Summary of documents by processing status"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "documents": [],
                "total_count": 127,
                "offset": 0,
                "limit": 20,
                "has_more": True,
                "processing_summary": {
                    "completed": 98,
                    "processing": 5,
                    "failed": 3,
                    "pending": 21
                }
            }
        }


class DocumentUpdateRequest(BaseModel):
    """Schema for updating document metadata."""
    
    document_name: Optional[str] = Field(
        None,
        description="Updated document name",
        min_length=1,
        max_length=255
    )
    
    priority: Optional[DocumentPriority] = Field(
        None,
        description="Updated processing priority"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated document metadata"
    )
    
    @validator('document_name')
    @classmethod
    def validate_document_name(cls, v):
        """Validate document name if provided."""
        if v is not None and not v.strip():
            raise ValueError('Document name cannot be empty or whitespace only')
        return v.strip() if v else v
    
    class Config:
        schema_extra = {
            "example": {
                "document_name": "Updated Patent Application WiFi6",
                "priority": "urgent",
                "metadata": {
                    "review_status": "approved",
                    "reviewer": "senior_attorney"
                }
            }
        }


class DocumentRetryRequest(BaseModel):
    """Schema for retrying failed document processing."""
    
    force_retry: bool = Field(
        False,
        description="Force retry even if retry limit exceeded"
    )
    
    reset_processing: bool = Field(
        False,
        description="Reset all processing state before retry"
    )
    
    priority: Optional[DocumentPriority] = Field(
        None,
        description="Override priority for retry processing"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "force_retry": False,
                "reset_processing": True,
                "priority": "urgent"
            }
        }


class DocumentProgressUpdate(BaseModel):
    """Schema for document processing progress updates via WebSocket."""
    
    document_id: str = Field(
        ...,
        description="Document identifier"
    )
    
    status: ProcessingStatus = Field(
        ...,
        description="Current processing status"
    )
    
    progress_percentage: float = Field(
        ...,
        description="Processing progress as percentage (0.0 to 100.0)",
        ge=0.0,
        le=100.0
    )
    
    current_stage: str = Field(
        ...,
        description="Current processing stage description"
    )
    
    estimated_time_remaining: Optional[int] = Field(
        None,
        description="Estimated time remaining in seconds",
        ge=0
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    
    chunks_processed: int = Field(
        ...,
        description="Number of chunks processed so far",
        ge=0
    )
    
    total_chunks: Optional[int] = Field(
        None,
        description="Total number of chunks to process",
        ge=0
    )
    
    timestamp: datetime = Field(
        ...,
        description="Progress update timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "DOC_20250115_A1B2C3D4",
                "status": "embedding",
                "progress_percentage": 75.5,
                "current_stage": "Generating embeddings for text chunks",
                "estimated_time_remaining": 30,
                "error_message": None,
                "chunks_processed": 35,
                "total_chunks": 47,
                "timestamp": "2025-01-15T10:31:45Z"
            }
        }


class BatchDocumentOperation(BaseModel):
    """Schema for batch document operations."""
    
    document_ids: List[str] = Field(
        ...,
        description="List of document identifiers",
        min_items=1,
        max_items=100
    )
    
    operation: str = Field(
        ...,
        description="Batch operation to perform",
        regex=r"^(retry|delete|archive|update_priority|reprocess)$"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Operation-specific parameters"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_ids": ["DOC_001", "DOC_002", "DOC_003"],
                "operation": "retry",
                "parameters": {
                    "force_retry": True,
                    "priority": "high"
                }
            }
        }


class BatchOperationResponse(BaseModel):
    """Schema for batch operation response."""
    
    operation: str = Field(
        ...,
        description="Batch operation performed"
    )
    
    total_documents: int = Field(
        ...,
        description="Total number of documents in batch",
        ge=0
    )
    
    successful_operations: int = Field(
        ...,
        description="Number of successful operations",
        ge=0
    )
    
    failed_operations: int = Field(
        ...,
        description="Number of failed operations",
        ge=0
    )
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Detailed results for each document"
    )
    
    errors: List[Dict[str, Any]] = Field(
        ...,
        description="Error details for failed operations"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "operation": "retry",
                "total_documents": 3,
                "successful_operations": 2,
                "failed_operations": 1,
                "results": [
                    {
                        "document_id": "DOC_001",
                        "success": True,
                        "new_status": "pending"
                    }
                ],
                "errors": [
                    {
                        "document_id": "DOC_003",
                        "error": "Document not found"
                    }
                ]
            }
        }


class DocumentStatsResponse(BaseModel):
    """Schema for document statistics and analytics."""
    
    total_documents: int = Field(
        ...,
        description="Total number of documents",
        ge=0
    )
    
    processed_documents: int = Field(
        ...,
        description="Number of successfully processed documents",
        ge=0
    )
    
    failed_documents: int = Field(
        ...,
        description="Number of failed documents",
        ge=0
    )
    
    pending_documents: int = Field(
        ...,
        description="Number of pending documents",
        ge=0
    )
    
    processing_documents: int = Field(
        ...,
        description="Number of currently processing documents",
        ge=0
    )
    
    total_chunks: int = Field(
        ...,
        description="Total number of text chunks",
        ge=0
    )
    
    total_file_size_bytes: int = Field(
        ...,
        description="Total file size in bytes",
        ge=0
    )
    
    average_processing_time: float = Field(
        ...,
        description="Average processing time in seconds",
        ge=0.0
    )
    
    processing_success_rate: float = Field(
        ...,
        description="Processing success rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    status_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown by processing status"
    )
    
    file_type_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown by file type"
    )
    
    priority_breakdown: Dict[str, int] = Field(
        ...,
        description="Breakdown by priority level"
    )
    
    recent_activity: List[Dict[str, Any]] = Field(
        ...,
        description="Recent document activity"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_documents": 1247,
                "processed_documents": 1189,
                "failed_documents": 23,
                "pending_documents": 35,
                "processing_documents": 0,
                "total_chunks": 34567,
                "total_file_size_bytes": 523458764,
                "average_processing_time": 45.7,
                "processing_success_rate": 0.954,
                "status_breakdown": {
                    "completed": 1189,
                    "failed": 23,
                    "pending": 35
                },
                "file_type_breakdown": {
                    "pdf": 945,
                    "txt": 267,
                    "docx": 35
                },
                "priority_breakdown": {
                    "low": 234,
                    "normal": 876,
                    "high": 125,
                    "urgent": 12
                },
                "recent_activity": []
            }
        }


# Convenience type aliases for API responses
DocumentCreateResponse = DocumentUploadResponse
DocumentUpdateResponse = DocumentResponse
DocumentDeleteResponse = Dict[str, Any]



# Add these missing classes to backend/app/models/api/document_schemas.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

# Import from common schemas for consistency
from .common_schemas import ApiResponse, ErrorResponse, PaginatedResponse, MetadataInfo

# Import domain models
from ..domain.document import DocumentType, ProcessingStatus, DocumentPriority


class DocumentAnalyticsResponse(BaseModel):
    """Response schema for document analytics and metrics."""
    
    # Overall document metrics
    total_documents: int = Field(
        ...,
        description="Total number of documents in the system",
        ge=0
    )
    
    documents_by_status: Dict[str, int] = Field(
        ...,
        description="Document count by processing status"
    )
    
    documents_by_type: Dict[str, int] = Field(
        ...,
        description="Document count by document type"
    )
    
    documents_by_priority: Dict[str, int] = Field(
        ...,
        description="Document count by priority level"
    )
    
    # Processing metrics
    processing_success_rate: float = Field(
        ...,
        description="Overall processing success rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    average_processing_time_ms: float = Field(
        ...,
        description="Average document processing time in milliseconds",
        ge=0.0
    )
    
    failed_documents_count: int = Field(
        ...,
        description="Number of documents that failed processing",
        ge=0
    )
    
    pending_documents_count: int = Field(
        ...,
        description="Number of documents pending processing",
        ge=0
    )
    
    # Storage and size metrics
    total_storage_bytes: int = Field(
        ...,
        description="Total storage used by documents in bytes",
        ge=0
    )
    
    average_document_size_bytes: float = Field(
        ...,
        description="Average document size in bytes",
        ge=0.0
    )
    
    largest_document_size_bytes: int = Field(
        ...,
        description="Size of the largest document in bytes",
        ge=0
    )
    
    # Time-based metrics
    documents_uploaded_today: int = Field(
        ...,
        description="Number of documents uploaded today",
        ge=0
    )
    
    documents_uploaded_this_week: int = Field(
        ...,
        description="Number of documents uploaded this week",
        ge=0
    )
    
    documents_uploaded_this_month: int = Field(
        ...,
        description="Number of documents uploaded this month",
        ge=0
    )
    
    # Performance trends
    recent_upload_trends: List[Dict[str, Any]] = Field(
        ...,
        description="Recent upload trends (daily/weekly data)"
    )
    
    processing_performance_trends: List[Dict[str, Any]] = Field(
        ...,
        description="Processing performance trends over time"
    )
    
    # Top performing metrics
    most_common_document_types: List[Dict[str, Any]] = Field(
        ...,
        description="Most common document types with counts"
    )
    
    slowest_processing_documents: List[Dict[str, Any]] = Field(
        ...,
        description="Documents that took longest to process"
    )
    
    # Case-related metrics
    documents_per_case_average: float = Field(
        ...,
        description="Average number of documents per case",
        ge=0.0
    )
    
    cases_with_most_documents: List[Dict[str, Any]] = Field(
        ...,
        description="Cases with the highest document counts"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_documents": 1542,
                "documents_by_status": {
                    "completed": 1489,
                    "processing": 12,
                    "failed": 23,
                    "pending": 18
                },
                "documents_by_type": {
                    "pdf": 1203,
                    "docx": 267,
                    "txt": 72
                },
                "documents_by_priority": {
                    "normal": 1234,
                    "high": 234,
                    "urgent": 74
                },
                "processing_success_rate": 0.965,
                "average_processing_time_ms": 2340.5,
                "failed_documents_count": 23,
                "pending_documents_count": 18,
                "total_storage_bytes": 15678234567,
                "average_document_size_bytes": 10167456.7,
                "largest_document_size_bytes": 67234567,
                "documents_uploaded_today": 23,
                "documents_uploaded_this_week": 156,
                "documents_uploaded_this_month": 689
            }
        }


class DocumentBatchRequest(BaseModel):
    """Request schema for batch document operations."""
    
    operation_type: str = Field(
        ...,
        description="Type of batch operation (reprocess, delete, move, update_metadata, etc.)"
    )
    
    document_ids: List[str] = Field(
        ...,
        description="List of document IDs to process",
        min_items=1,
        max_items=100  # Reasonable batch size limit
    )
    
    target_case_id: Optional[str] = Field(
        None,
        description="Target case ID for move operations"
    )
    
    new_priority: Optional[DocumentPriority] = Field(
        None,
        description="New priority for update operations"
    )
    
    metadata_updates: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata updates to apply"
    )
    
    processing_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Processing options for reprocess operations"
    )
    
    force_operation: bool = Field(
        default=False,
        description="Force operation even if documents are in use"
    )
    
    notify_on_completion: bool = Field(
        default=True,
        description="Send notification when batch operation completes"
    )
    
    reason: Optional[str] = Field(
        None,
        description="Reason for the batch operation (for audit trail)",
        max_length=500
    )
    
    @validator('operation_type')
    @classmethod
    def validate_operation_type(cls, v):
        """Validate operation type."""
        valid_operations = {
            'reprocess', 'delete', 'move', 'update_metadata', 
            'change_priority', 'archive', 'restore', 'export'
        }
        if v not in valid_operations:
            raise ValueError(f'Operation type must be one of: {", ".join(valid_operations)}')
        return v
    
    @validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):
        """Validate document IDs."""
        if len(v) != len(set(v)):
            raise ValueError('Document IDs must be unique')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "operation_type": "reprocess",
                "document_ids": [
                    "doc_123456789",
                    "doc_987654321",
                    "doc_456789123"
                ],
                "processing_options": {
                    "force_ocr": True,
                    "skip_embedding": False
                },
                "notify_on_completion": True,
                "reason": "Reprocessing with updated OCR settings"
            }
        }


class DocumentBatchOperationResult(BaseModel):
    """Result for a single document in a batch operation."""
    
    document_id: str = Field(
        ...,
        description="Document ID"
    )
    
    success: bool = Field(
        ...,
        description="Whether the operation succeeded for this document"
    )
    
    message: Optional[str] = Field(
        None,
        description="Success or error message for this document"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="Error code if operation failed"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process this document",
        ge=0.0
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the operation result"
    )


class DocumentBatchResponse(BaseModel):
    """Response schema for batch document operations."""
    
    batch_id: str = Field(
        ...,
        description="Unique identifier for this batch operation"
    )
    
    operation_type: str = Field(
        ...,
        description="Type of batch operation performed"
    )
    
    total_documents: int = Field(
        ...,
        description="Total number of documents in the batch",
        ge=0
    )
    
    successful_operations: int = Field(
        ...,
        description="Number of successful operations",
        ge=0
    )
    
    failed_operations: int = Field(
        ...,
        description="Number of failed operations",
        ge=0
    )
    
    skipped_operations: int = Field(
        default=0,
        description="Number of skipped operations",
        ge=0
    )
    
    overall_success: bool = Field(
        ...,
        description="Whether the overall batch operation was successful"
    )
    
    started_at: datetime = Field(
        ...,
        description="When the batch operation started"
    )
    
    completed_at: Optional[datetime] = Field(
        None,
        description="When the batch operation completed"
    )
    
    total_processing_time_ms: Optional[float] = Field(
        None,
        description="Total time for the batch operation",
        ge=0.0
    )
    
    results: List[DocumentBatchOperationResult] = Field(
        ...,
        description="Individual results for each document"
    )
    
    summary_statistics: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary statistics for the batch operation"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during the batch operation"
    )
    
    @validator('failed_operations', always=True)
    @classmethod
    def validate_failed_operations(cls, v, values):
        """Validate failed operations count."""
        total = values.get('total_documents', 0)
        successful = values.get('successful_operations', 0)
        skipped = values.get('skipped_operations', 0)
        
        if total > 0 and v != (total - successful - skipped):
            raise ValueError("Failed operations count must equal total minus successful minus skipped")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_abc123def456",
                "operation_type": "reprocess",
                "total_documents": 5,
                "successful_operations": 4,
                "failed_operations": 1,
                "skipped_operations": 0,
                "overall_success": False,
                "started_at": "2025-01-15T10:30:00Z",
                "completed_at": "2025-01-15T10:35:30Z",
                "total_processing_time_ms": 330000,
                "results": [
                    {
                        "document_id": "doc_123",
                        "success": True,
                        "message": "Document reprocessed successfully",
                        "processing_time_ms": 2340.5
                    },
                    {
                        "document_id": "doc_456",
                        "success": False,
                        "message": "Document not found",
                        "error_code": "DOCUMENT_NOT_FOUND"
                    }
                ],
                "warnings": [
                    "Some documents took longer than expected to process"
                ]
            }
        }


# Re-export common schemas for convenience
# This allows imports to work from document_schemas module
__all__ = [
    "DocumentAnalyticsResponse",
    "DocumentBatchRequest", 
    "DocumentBatchResponse",
    "DocumentBatchOperationResult",
    "ApiResponse",  # Re-exported from common_schemas
    "ErrorResponse"  # Re-exported from common_schemas
]
