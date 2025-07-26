"""
MongoDB Database Models for Legal Document Management

This module defines the database layer models and schemas for legal document management
in the Patexia Legal AI Chatbot. It provides MongoDB document structures, GridFS file 
handling, validation, indexing specifications, and data transformation utilities.

Key Features:
- MongoDB document schemas for legal document entities
- GridFS integration for large file storage and retrieval
- Document chunk management with semantic boundaries
- Processing metadata tracking throughout the pipeline
- Field validation and business rule enforcement
- Index definitions for complex query optimization
- Data transformation utilities between domain and database models
- Performance monitoring and analytics aggregation

Database Schema Design:
- Documents collection with file references and metadata
- GridFS storage for binary file content with metadata
- Document chunks with embedding vector support
- Processing status tracking with retry mechanisms
- Legal citation and structure preservation
- User and case-based access control integration

Collections:
- documents: Primary document metadata without file content
- document_files (GridFS): Binary file storage with metadata
- document_chunks: Semantic text chunks with embeddings
- document_analytics: Processing performance and usage metrics
- document_history: Processing and modification audit trail

GridFS Integration:
- Automatic file chunking for large documents
- Metadata preservation in GridFS file records
- Efficient streaming for file upload/download
- Deduplication support via file hash comparison
- Concurrent access handling for multi-user scenarios

Performance Optimizations:
- Compound indexes for case-based document queries
- Text search indexes for content discovery
- Processing status indexes for pipeline monitoring
- File size and type indexes for filtering
- GridFS metadata indexes for file operations
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
from gridfs import GridFS

from ..domain.document import (
    DocumentType, ProcessingStatus, DocumentPriority,
    ProcessingMetadata, DocumentChunk, LegalDocument
)
from ...utils.logging import get_logger

logger = get_logger(__name__)


class DocumentCollection(str, Enum):
    """Document-related MongoDB collection names."""
    DOCUMENTS = "documents"
    DOCUMENT_CHUNKS = "document_chunks"
    DOCUMENT_ANALYTICS = "document_analytics"
    DOCUMENT_HISTORY = "document_history"
    DOCUMENT_FILES = "fs.files"  # GridFS files collection
    DOCUMENT_CHUNKS_FS = "fs.chunks"  # GridFS chunks collection


class GridFSMetadata(BaseModel):
    """GridFS file metadata model."""
    document_id: str = Field(
        ...,
        description="Associated document identifier"
    )
    
    case_id: str = Field(
        ...,
        description="Associated case identifier"
    )
    
    user_id: str = Field(
        ...,
        description="Document owner identifier"
    )
    
    file_type: str = Field(
        ...,
        description="Document file type"
    )
    
    original_filename: str = Field(
        ...,
        description="Original uploaded filename"
    )
    
    file_hash: str = Field(
        ...,
        description="SHA-256 hash of file content"
    )
    
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="File upload timestamp"
    )
    
    upload_method: str = Field(
        default="direct",
        description="Upload method (direct, chunked, stream)"
    )
    
    compression: Optional[str] = Field(
        None,
        description="Compression method if applied"
    )
    
    virus_scan_status: Optional[str] = Field(
        None,
        description="Virus scan status (future security feature)"
    )
    
    access_count: int = Field(
        default=0,
        description="Number of times file has been accessed"
    )
    
    last_accessed: Optional[datetime] = Field(
        None,
        description="Last file access timestamp"
    )


class ProcessingMetadataDocument(BaseModel):
    """MongoDB document model for processing metadata."""
    processing_method: str = Field(
        default="unknown",
        description="Method used for document processing"
    )
    
    processing_version: str = Field(
        default="1.0",
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
        default=1.0,
        description="Confidence score for text extraction",
        ge=0.0,
        le=1.0
    )
    
    chunk_count: int = Field(
        default=0,
        description="Number of text chunks created",
        ge=0
    )
    
    embedding_model: Optional[str] = Field(
        None,
        description="Embedding model used for vectorization"
    )
    
    embedding_dimensions: int = Field(
        default=0,
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
        default=0,
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
    
    # Additional processing statistics
    extraction_method: Optional[str] = Field(
        None,
        description="Text extraction method used"
    )
    
    quality_score: Optional[float] = Field(
        None,
        description="Overall processing quality score",
        ge=0.0,
        le=1.0
    )
    
    tokens_processed: Optional[int] = Field(
        None,
        description="Number of tokens processed",
        ge=0
    )
    
    memory_usage_mb: Optional[float] = Field(
        None,
        description="Peak memory usage during processing",
        ge=0.0
    )
    
    gpu_time_seconds: Optional[float] = Field(
        None,
        description="GPU processing time for embeddings",
        ge=0.0
    )
    
    @property
    def is_processing_active(self) -> bool:
        """Check if processing is currently active."""
        return (
            self.processing_started_at is not None and
            self.processing_completed_at is None
        )
    
    @property
    def has_errors(self) -> bool:
        """Check if processing has encountered errors."""
        return self.error_message is not None
    
    def to_domain(self) -> ProcessingMetadata:
        """Convert to domain model."""
        metadata = ProcessingMetadata(
            processing_method=self.processing_method,
            processing_version=self.processing_version,
            processing_started_at=self.processing_started_at,
            processing_completed_at=self.processing_completed_at,
            extraction_confidence=self.extraction_confidence,
            chunk_count=self.chunk_count,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            index_collection=self.index_collection,
            error_message=self.error_message,
            error_stage=self.error_stage,
            retry_count=self.retry_count,
            last_retry_at=self.last_retry_at
        )
        return metadata
    
    @classmethod
    def from_domain(cls, metadata: ProcessingMetadata) -> "ProcessingMetadataDocument":
        """Create from domain model."""
        return cls(
            processing_method=metadata.processing_method,
            processing_version=metadata.processing_version,
            processing_started_at=metadata.processing_started_at,
            processing_completed_at=metadata.processing_completed_at,
            extraction_confidence=metadata.extraction_confidence,
            chunk_count=metadata.chunk_count,
            embedding_model=metadata.embedding_model,
            embedding_dimensions=metadata.embedding_dimensions,
            index_collection=metadata.index_collection,
            error_message=metadata.error_message,
            error_stage=metadata.error_stage,
            retry_count=metadata.retry_count,
            last_retry_at=metadata.last_retry_at
        )


class DocumentChunkDocument(BaseModel):
    """MongoDB document model for document text chunks."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier"
    )
    
    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )
    
    case_id: str = Field(
        ...,
        description="Associated case identifier"
    )
    
    content: str = Field(
        ...,
        description="Text content of the chunk",
        min_length=1,
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
    
    # Legal document structure metadata
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
        default_factory=list,
        description="Legal citations found in this chunk"
    )
    
    # Embedding information (reference to vector store)
    has_embedding: bool = Field(
        default=False,
        description="Whether chunk has embedding vector"
    )
    
    embedding_model: Optional[str] = Field(
        None,
        description="Model used for embedding generation"
    )
    
    embedding_dimensions: int = Field(
        default=0,
        description="Dimensions of embedding vector",
        ge=0
    )
    
    vector_id: Optional[str] = Field(
        None,
        description="Vector store reference ID"
    )
    
    # Processing metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chunk creation timestamp"
    )
    
    processed_at: Optional[datetime] = Field(
        None,
        description="Embedding processing timestamp"
    )
    
    quality_score: Optional[float] = Field(
        None,
        description="Chunk quality score",
        ge=0.0,
        le=1.0
    )
    
    # Search optimization
    word_count: int = Field(
        default=0,
        description="Number of words in chunk",
        ge=0
    )
    
    sentence_count: int = Field(
        default=0,
        description="Number of sentences in chunk",
        ge=0
    )
    
    keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords for search optimization"
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "chunk_id": "DOC_123_chunk_0001",
                "document_id": "DOC_20250115_A1B2C3D4",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "content": "This invention relates to wireless communication systems...",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 512,
                "chunk_size": 512,
                "section_title": "Technical Field",
                "page_number": 1,
                "legal_citations": ["35 U.S.C. § 101"],
                "has_embedding": True,
                "embedding_model": "mxbai-embed-large",
                "word_count": 85,
                "sentence_count": 3
            }
        }
    
    @validator('end_char')
    def validate_char_positions(cls, v, values):
        """Ensure end_char is after start_char."""
        if 'start_char' in values and v <= values['start_char']:
            raise ValueError('end_char must be greater than start_char')
        return v
    
    @root_validator
    def validate_chunk_consistency(cls, values):
        """Validate chunk data consistency."""
        content = values.get('content', '')
        chunk_size = values.get('chunk_size', 0)
        start_char = values.get('start_char', 0)
        end_char = values.get('end_char', 0)
        
        # Validate chunk size matches content length
        if chunk_size != len(content):
            values['chunk_size'] = len(content)
        
        # Validate character positions
        expected_size = end_char - start_char
        if abs(expected_size - len(content)) > 5:  # Allow small discrepancies
            logger.warning(f"Chunk size mismatch: expected {expected_size}, got {len(content)}")
        
        # Calculate word and sentence counts if not provided
        if not values.get('word_count'):
            values['word_count'] = len(content.split())
        
        if not values.get('sentence_count'):
            values['sentence_count'] = len([s for s in content.split('.') if s.strip()])
        
        return values
    
    def to_domain(self) -> DocumentChunk:
        """Convert to domain model."""
        chunk = DocumentChunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            chunk_index=self.chunk_index,
            start_char=self.start_char,
            end_char=self.end_char,
            section_title=self.section_title,
            page_number=self.page_number,
            paragraph_number=self.paragraph_number,
            legal_citations=self.legal_citations.copy()
        )
        
        # Set creation timestamp
        object.__setattr__(chunk, 'created_at', self.created_at)
        
        return chunk
    
    @classmethod
    def from_domain(cls, chunk: DocumentChunk, case_id: str) -> "DocumentChunkDocument":
        """Create from domain model."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            case_id=case_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            chunk_size=len(chunk.content),
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            paragraph_number=chunk.paragraph_number,
            legal_citations=chunk.legal_citations.copy(),
            created_at=chunk.created_at,
            word_count=len(chunk.content.split()),
            sentence_count=len([s for s in chunk.content.split('.') if s.strip()])
        )


class DocumentDocument(BaseModel):
    """MongoDB document model for legal documents (metadata only, files in GridFS)."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    document_id: str = Field(
        ...,
        description="Unique document identifier",
        min_length=10,
        max_length=100
    )
    
    user_id: str = Field(
        ...,
        description="Document owner identifier",
        min_length=1,
        max_length=100
    )
    
    case_id: str = Field(
        ...,
        description="Associated case identifier",
        min_length=10,
        max_length=100
    )
    
    document_name: str = Field(
        ...,
        description="Document display name",
        min_length=1,
        max_length=255
    )
    
    original_filename: str = Field(
        ...,
        description="Original uploaded filename",
        min_length=1,
        max_length=255
    )
    
    file_type: str = Field(
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
        description="SHA-256 hash of file content"
    )
    
    # GridFS reference
    file_id: Optional[ObjectId] = Field(
        None,
        description="GridFS file identifier for binary content"
    )
    
    # Processing information
    priority: str = Field(
        default=DocumentPriority.NORMAL.value,
        description="Processing priority level"
    )
    
    status: str = Field(
        default=ProcessingStatus.PENDING.value,
        description="Current processing status"
    )
    
    text_content: Optional[str] = Field(
        None,
        description="Extracted text content (for small documents)"
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Document creation timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata"
    )
    
    processing_metadata: ProcessingMetadataDocument = Field(
        default_factory=ProcessingMetadataDocument,
        description="Processing metadata and status"
    )
    
    # Legal document specific fields
    legal_citations: List[str] = Field(
        default_factory=list,
        description="Legal citations found in document"
    )
    
    section_headers: List[str] = Field(
        default_factory=list,
        description="Document section headers"
    )
    
    page_count: Optional[int] = Field(
        None,
        description="Number of pages in document",
        ge=1
    )
    
    # Administrative fields
    version: int = Field(
        default=1,
        description="Document version for optimistic locking"
    )
    
    archived_at: Optional[datetime] = Field(
        None,
        description="Timestamp when document was archived"
    )
    
    last_accessed: Optional[datetime] = Field(
        None,
        description="Timestamp of last document access"
    )
    
    access_count: int = Field(
        default=0,
        description="Number of times document has been accessed"
    )
    
    # Search optimization
    full_text_content: Optional[str] = Field(
        None,
        description="Combined text content for full-text search"
    )
    
    search_keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords for search optimization"
    )
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "document_id": "DOC_20250115_A1B2C3D4",
                "user_id": "user_123",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "document_name": "Patent Application - WiFi 6 Enhancement",
                "original_filename": "wifi6_patent_app.pdf",
                "file_type": "pdf",
                "file_size": 1048576,
                "file_hash": "abc123def456...",
                "priority": "normal",
                "status": "completed",
                "legal_citations": ["35 U.S.C. § 101", "IEEE 802.11ax"],
                "section_headers": ["Technical Field", "Background", "Claims"],
                "page_count": 25
            }
        }
    
    @validator('document_id')
    def validate_document_id(cls, v):
        """Validate document ID format."""
        if not v.startswith('DOC_'):
            raise ValueError('Document ID must start with DOC_')
        return v
    
    @validator('file_type')
    def validate_file_type(cls, v):
        """Validate file type."""
        try:
            DocumentType(v)
        except ValueError:
            raise ValueError(f'Invalid file type: {v}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """Validate processing status."""
        try:
            ProcessingStatus(v)
        except ValueError:
            raise ValueError(f'Invalid processing status: {v}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority level."""
        try:
            DocumentPriority(v)
        except ValueError:
            raise ValueError(f'Invalid priority: {v}')
        return v
    
    @validator('updated_at')
    def validate_updated_at(cls, v, values):
        """Ensure updated_at is not before created_at."""
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('Updated timestamp cannot be before created timestamp')
        return v
    
    @root_validator
    def validate_file_consistency(cls, values):
        """Validate file-related field consistency."""
        file_size = values.get('file_size', 0)
        text_content = values.get('text_content')
        
        # For small text files, store content directly
        if (file_size < 1000 and 
            values.get('file_type') == DocumentType.TEXT.value and 
            text_content):
            # Text content available for small files
            pass
        
        # Ensure file hash is present
        if not values.get('file_hash'):
            logger.warning("Document missing file hash")
        
        return values
    
    def to_domain(self, file_content: bytes = b"") -> LegalDocument:
        """Convert to domain model."""
        # Convert processing metadata
        processing_metadata = self.processing_metadata.to_domain()
        
        # Create domain object
        document = LegalDocument(
            document_id=self.document_id,
            user_id=self.user_id,
            case_id=self.case_id,
            document_name=self.document_name,
            original_filename=self.original_filename,
            file_content=file_content,
            file_type=DocumentType(self.file_type),
            priority=DocumentPriority(self.priority),
            created_at=self.created_at,
            updated_at=self.updated_at,
            status=ProcessingStatus(self.status),
            text_content=self.text_content,
            metadata=self.metadata.copy()
        )
        
        # Restore internal state
        document._file_size = self.file_size
        document._file_hash = self.file_hash
        document._processing_metadata = processing_metadata
        document._legal_citations = set(self.legal_citations)
        document._section_headers = self.section_headers.copy()
        document._page_count = self.page_count
        
        return document
    
    @classmethod
    def from_domain(cls, document: LegalDocument, file_id: Optional[ObjectId] = None) -> "DocumentDocument":
        """Create from domain model."""
        return cls(
            document_id=document.document_id,
            user_id=document.user_id,
            case_id=document.case_id,
            document_name=document.document_name,
            original_filename=document.original_filename,
            file_type=document.file_type.value,
            file_size=document.file_size,
            file_hash=document.file_hash,
            file_id=file_id,
            priority=document.priority.value,
            status=document.status.value,
            text_content=document.text_content,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=document.metadata.copy(),
            processing_metadata=ProcessingMetadataDocument.from_domain(document.processing_metadata),
            legal_citations=list(document.legal_citations),
            section_headers=document.section_headers.copy(),
            page_count=document.page_count
        )
    
    def update_search_content(self) -> None:
        """Update search-optimized content fields."""
        # Combine text content for search
        content_parts = [self.document_name]
        
        if self.text_content:
            content_parts.append(self.text_content)
        
        # Add legal citations
        if self.legal_citations:
            content_parts.extend(self.legal_citations)
        
        # Add section headers
        if self.section_headers:
            content_parts.extend(self.section_headers)
        
        # Store combined content
        self.full_text_content = " ".join(content_parts)
        
        # Extract keywords
        all_text = self.full_text_content.lower()
        words = all_text.split()
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


class DocumentAnalyticsDocument(BaseModel):
    """MongoDB document model for document analytics aggregation."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    document_id: str = Field(
        ...,
        description="Associated document identifier"
    )
    
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
    
    # Access metrics
    access_count: int = Field(
        default=0,
        description="Number of document accesses in period"
    )
    
    unique_users: int = Field(
        default=0,
        description="Number of unique users who accessed document"
    )
    
    search_matches: int = Field(
        default=0,
        description="Number of times document appeared in search results"
    )
    
    # Processing metrics
    chunk_retrievals: int = Field(
        default=0,
        description="Number of chunk retrievals for this document"
    )
    
    average_relevance_score: Optional[float] = Field(
        None,
        description="Average relevance score in search results"
    )
    
    # Performance metrics
    load_time_ms: Optional[float] = Field(
        None,
        description="Average document load time in milliseconds"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analytics record creation time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "DOC_20250115_A1B2C3D4",
                "case_id": "CASE_2025_01_15_A1B2C3D4",
                "period_start": "2025-01-15T00:00:00Z",
                "period_end": "2025-01-15T23:59:59Z",
                "period_type": "daily",
                "access_count": 25,
                "unique_users": 3,
                "search_matches": 12,
                "chunk_retrievals": 45
            }
        }


class DocumentHistoryDocument(BaseModel):
    """MongoDB document model for document change history."""
    _id: Optional[ObjectId] = Field(None, alias="_id")
    
    document_id: str = Field(
        ...,
        description="Associated document identifier"
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
                "document_id": "DOC_20250115_A1B2C3D4",
                "user_id": "user_123",
                "action": "status_change",
                "field_changed": "status",
                "old_value": "extracting",
                "new_value": "completed",
                "metadata": {"processing_time": 45.7}
            }
        }


class DocumentIndexes:
    """MongoDB index definitions for document collections."""
    
    @classmethod
    def get_document_indexes(cls) -> List[IndexModel]:
        """Get index models for documents collection."""
        return [
            # Unique constraint on document_id
            IndexModel([("document_id", ASCENDING)], unique=True, name="document_id_unique"),
            
            # Case documents with status and sorting
            IndexModel(
                [("case_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)],
                name="case_status_updated"
            ),
            
            # User documents with sorting
            IndexModel(
                [("user_id", ASCENDING), ("updated_at", DESCENDING)],
                name="user_documents_updated"
            ),
            
            # Processing status filtering
            IndexModel(
                [("status", ASCENDING), ("updated_at", DESCENDING)],
                name="status_updated"
            ),
            
            # File type filtering
            IndexModel([("file_type", ASCENDING)], name="file_type_filter"),
            
            # Priority filtering
            IndexModel(
                [("priority", ASCENDING), ("updated_at", DESCENDING)],
                name="priority_updated"
            ),
            
            # Text search index
            IndexModel(
                [
                    ("document_name", TEXT),
                    ("text_content", TEXT),
                    ("full_text_content", TEXT),
                    ("legal_citations", TEXT)
                ],
                name="document_text_search",
                weights={
                    "document_name": 10,
                    "legal_citations": 5,
                    "text_content": 3,
                    "full_text_content": 1
                }
            ),
            
            # Creation date range queries
            IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
            
            # File size filtering
            IndexModel([("file_size", ASCENDING)], name="file_size_filter"),
            
            # Case and document name for uniqueness
            IndexModel(
                [("case_id", ASCENDING), ("document_name", ASCENDING)],
                name="case_document_name"
            ),
            
            # GridFS file reference
            IndexModel([("file_id", ASCENDING)], name="file_id_ref", sparse=True),
            
            # Last accessed for analytics
            IndexModel([("last_accessed", DESCENDING)], name="last_accessed_desc"),
            
            # Processing metadata queries
            IndexModel(
                [("processing_metadata.embedding_model", ASCENDING)],
                name="embedding_model_filter"
            ),
            
            # Archive status with TTL
            IndexModel(
                [("archived_at", ASCENDING)],
                name="archived_documents",
                sparse=True,
                expireAfterSeconds=60*60*24*365*5  # 5 years retention for archived documents
            )
        ]
    
    @classmethod
    def get_chunk_indexes(cls) -> List[IndexModel]:
        """Get index models for document chunks collection."""
        return [
            # Unique constraint on chunk_id
            IndexModel([("chunk_id", ASCENDING)], unique=True, name="chunk_id_unique"),
            
            # Document chunks with index ordering
            IndexModel(
                [("document_id", ASCENDING), ("chunk_index", ASCENDING)],
                name="document_chunks_ordered"
            ),
            
            # Case chunks
            IndexModel([("case_id", ASCENDING)], name="case_chunks"),
            
            # Embedding status filtering
            IndexModel([("has_embedding", ASCENDING)], name="embedding_status"),
            
            # Page number filtering
            IndexModel([("page_number", ASCENDING)], name="page_filter", sparse=True),
            
            # Text search on chunk content
            IndexModel([("content", TEXT)], name="chunk_text_search"),
            
            # Legal citations
            IndexModel([("legal_citations", ASCENDING)], name="citations_filter"),
            
            # Processing timestamp
            IndexModel([("processed_at", DESCENDING)], name="processed_at_desc"),
            
            # Vector store reference
            IndexModel([("vector_id", ASCENDING)], name="vector_id_ref", sparse=True),
            
            # Quality score for ranking
            IndexModel([("quality_score", DESCENDING)], name="quality_ranking", sparse=True)
        ]
    
    @classmethod
    def get_analytics_indexes(cls) -> List[IndexModel]:
        """Get index models for document analytics collection."""
        return [
            # Document analytics by period
            IndexModel(
                [("document_id", ASCENDING), ("period_start", DESCENDING)],
                name="document_analytics_period"
            ),
            
            # Case analytics aggregation
            IndexModel(
                [("case_id", ASCENDING), ("period_type", ASCENDING), ("period_start", DESCENDING)],
                name="case_analytics_aggregation"
            ),
            
            # TTL for analytics cleanup
            IndexModel(
                [("created_at", ASCENDING)],
                name="analytics_ttl",
                expireAfterSeconds=60*60*24*365  # 1 year retention
            )
        ]
    
    @classmethod
    def get_history_indexes(cls) -> List[IndexModel]:
        """Get index models for document history collection."""
        return [
            # Document history chronological
            IndexModel(
                [("document_id", ASCENDING), ("timestamp", DESCENDING)],
                name="document_history_chrono"
            ),
            
            # User activity history
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="user_document_activity"
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
    
    @classmethod
    def get_gridfs_indexes(cls) -> List[IndexModel]:
        """Get additional index models for GridFS collections."""
        return [
            # GridFS metadata queries
            IndexModel(
                [("metadata.document_id", ASCENDING)],
                name="gridfs_document_id"
            ),
            
            # GridFS case files
            IndexModel(
                [("metadata.case_id", ASCENDING)],
                name="gridfs_case_id"
            ),
            
            # GridFS file hash for deduplication
            IndexModel(
                [("metadata.file_hash", ASCENDING)],
                name="gridfs_file_hash"
            ),
            
            # GridFS upload timestamp
            IndexModel(
                [("metadata.upload_timestamp", DESCENDING)],
                name="gridfs_upload_time"
            )
        ]


class DocumentAggregations:
    """MongoDB aggregation pipelines for document analytics."""
    
    @classmethod
    def get_case_document_summary(cls, case_id: str) -> List[Dict[str, Any]]:
        """Aggregation pipeline for case document summary."""
        return [
            {"$match": {"case_id": case_id}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$file_size"},
                    "total_pages": {"$sum": "$page_count"},
                    "avg_processing_time": {"$avg": "$processing_metadata.processing_duration_seconds"}
                }
            },
            {"$sort": {"count": -1}}
        ]
    
    @classmethod
    def get_processing_performance(cls, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Aggregation pipeline for processing performance metrics."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)
        
        return [
            {
                "$match": {
                    "status": "completed",
                    "processing_metadata.processing_completed_at": {"$gte": cutoff}
                }
            },
            {
                "$group": {
                    "_id": "$processing_metadata.embedding_model",
                    "documents_processed": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_metadata.processing_duration_seconds"},
                    "total_chunks": {"$sum": "$processing_metadata.chunk_count"},
                    "avg_confidence": {"$avg": "$processing_metadata.extraction_confidence"}
                }
            },
            {"$sort": {"documents_processed": -1}}
        ]
    
    @classmethod
    def get_popular_documents(cls, case_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Aggregation pipeline for most accessed documents."""
        match_stage = {"access_count": {"$gt": 0}}
        if case_id:
            match_stage["case_id"] = case_id
        
        return [
            {"$match": match_stage},
            {
                "$project": {
                    "document_id": 1,
                    "document_name": 1,
                    "case_id": 1,
                    "access_count": 1,
                    "last_accessed": 1,
                    "file_size": 1,
                    "status": 1
                }
            },
            {"$sort": {"access_count": -1}},
            {"$limit": limit}
        ]


# Collection configuration
DOCUMENT_COLLECTIONS = {
    DocumentCollection.DOCUMENTS: {
        "document_class": DocumentDocument,
        "indexes": DocumentIndexes.get_document_indexes(),
        "validators": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["document_id", "user_id", "case_id", "document_name", "file_type"],
                "properties": {
                    "document_id": {"bsonType": "string", "pattern": "^DOC_"},
                    "user_id": {"bsonType": "string", "minLength": 1},
                    "case_id": {"bsonType": "string", "pattern": "^CASE_"},
                    "document_name": {"bsonType": "string", "minLength": 1, "maxLength": 255},
                    "file_size": {"bsonType": "int", "minimum": 0}
                }
            }
        }
    },
    DocumentCollection.DOCUMENT_CHUNKS: {
        "document_class": DocumentChunkDocument,
        "indexes": DocumentIndexes.get_chunk_indexes(),
        "validators": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["chunk_id", "document_id", "case_id", "content"],
                "properties": {
                    "chunk_id": {"bsonType": "string"},
                    "document_id": {"bsonType": "string", "pattern": "^DOC_"},
                    "case_id": {"bsonType": "string", "pattern": "^CASE_"},
                    "content": {"bsonType": "string", "minLength": 1}
                }
            }
        }
    },
    DocumentCollection.DOCUMENT_ANALYTICS: {
        "document_class": DocumentAnalyticsDocument,
        "indexes": DocumentIndexes.get_analytics_indexes(),
        "validators": None
    },
    DocumentCollection.DOCUMENT_HISTORY: {
        "document_class": DocumentHistoryDocument,
        "indexes": DocumentIndexes.get_history_indexes(),
        "validators": None
    }
}


def get_collection_config(collection: DocumentCollection) -> Dict[str, Any]:
    """Get configuration for a specific collection."""
    return DOCUMENT_COLLECTIONS.get(collection, {})


def validate_document_data(doc_data: Dict[str, Any]) -> bool:
    """Validate document data against schema."""
    try:
        DocumentDocument(**doc_data)
        return True
    except Exception as e:
        logger.error(f"Document validation failed: {e}")
        return False


def create_document_id(case_id: str, filename: str) -> str:
    """Generate a unique document ID."""
    date_str = datetime.now().strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8].upper()
    
    # Clean filename for ID generation
    clean_name = ''.join(c for c in filename if c.isalnum())[:10]
    
    return f"DOC_{date_str}_{clean_name}_{unique_suffix}"


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def extract_file_metadata(filename: str, content: bytes) -> Dict[str, Any]:
    """Extract metadata from file for GridFS storage."""
    return {
        "original_filename": filename,
        "file_hash": calculate_file_hash(content),
        "file_size": len(content),
        "upload_timestamp": datetime.now(timezone.utc),
        "content_type": "application/octet-stream"  # Default, can be enhanced
    }