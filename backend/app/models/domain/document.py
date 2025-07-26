"""
Domain model for legal documents in Patexia Legal AI Chatbot.

This module defines the core business logic and domain entities for legal documents:
- Document entity with processing lifecycle management
- Text chunk management for semantic search
- Document hierarchy and citation tracking
- Processing status and error handling
- Legal document-specific metadata and validation
"""

import hashlib
import mimetypes
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from backend.app.core.exceptions import (
    DocumentProcessingError,
    ErrorCode,
    ValidationError,
    raise_document_error
)
from backend.app.utils.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)


class DocumentType(str, Enum):
    """Supported legal document types."""
    
    PDF = "pdf"
    TEXT = "txt"
    WORD = "docx"        # Future support
    HTML = "html"        # Future support
    EMAIL = "eml"        # Future support
    IMAGE = "image"      # Future OCR support
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing lifecycle status."""
    
    PENDING = "pending"                    # Document uploaded, awaiting processing
    EXTRACTING = "extracting"              # Text extraction in progress
    CHUNKING = "chunking"                  # Semantic chunking in progress
    EMBEDDING = "embedding"                # Embedding generation in progress
    INDEXING = "indexing"                  # Vector database indexing in progress
    COMPLETED = "completed"                # Successfully processed and indexed
    FAILED = "failed"                      # Processing failed with errors
    RETRY_PENDING = "retry_pending"        # Queued for retry after failure


class DocumentPriority(str, Enum):
    """Document processing priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ProcessingMetadata:
    """
    Metadata about document processing operations.
    
    Tracks the complete processing pipeline with timing, errors,
    and technical details for debugging and optimization.
    """
    
    processing_method: str = "unknown"
    processing_version: str = "1.0"
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    extraction_confidence: float = 1.0
    chunk_count: int = 0
    embedding_model: Optional[str] = None
    embedding_dimensions: int = 0
    index_collection: Optional[str] = None
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    
    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds."""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None
    
    @property
    def is_processing_active(self) -> bool:
        """Check if processing is currently active."""
        return (
            self.processing_started_at is not None and 
            self.processing_completed_at is None
        )
    
    def start_processing(self, method: str = "LlamaIndex") -> None:
        """Mark processing as started."""
        self.processing_method = method
        self.processing_started_at = datetime.now(timezone.utc)
        self.processing_completed_at = None
        self.error_message = None
        self.error_stage = None
    
    def complete_processing(self) -> None:
        """Mark processing as completed successfully."""
        self.processing_completed_at = datetime.now(timezone.utc)
        self.error_message = None
        self.error_stage = None
    
    def fail_processing(self, error_message: str, stage: str) -> None:
        """Mark processing as failed with error details."""
        self.processing_completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        self.error_stage = stage
    
    def start_retry(self) -> None:
        """Increment retry count and reset for retry attempt."""
        self.retry_count += 1
        self.last_retry_at = datetime.now(timezone.utc)
        self.processing_started_at = None
        self.processing_completed_at = None
        self.error_message = None
        self.error_stage = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processing_method": self.processing_method,
            "processing_version": self.processing_version,
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "extraction_confidence": self.extraction_confidence,
            "chunk_count": self.chunk_count,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "index_collection": self.index_collection,
            "error_message": self.error_message,
            "error_stage": self.error_stage,
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "processing_duration_seconds": self.processing_duration_seconds,
            "is_processing_active": self.is_processing_active,
        }


@dataclass(frozen=True)
class DocumentChunk:
    """
    Immutable text chunk extracted from a legal document.
    
    Represents a semantically meaningful segment of text with
    metadata for search, retrieval, and citation purposes.
    """
    
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding_vector: Optional[List[float]] = None
    
    # Legal document structure metadata
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    legal_citations: List[str] = field(default_factory=list)
    
    # Processing metadata
    chunk_size: int = field(init=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        object.__setattr__(self, 'chunk_size', len(self.content))
    
    @property
    def has_embedding(self) -> bool:
        """Check if chunk has an embedding vector."""
        return self.embedding_vector is not None and len(self.embedding_vector) > 0
    
    @property
    def embedding_dimensions(self) -> int:
        """Get embedding vector dimensions."""
        return len(self.embedding_vector) if self.embedding_vector else 0
    
    def get_citation_context(self, context_chars: int = 200) -> str:
        """
        Get citation-ready context around this chunk.
        
        Args:
            context_chars: Number of characters to include for context
            
        Returns:
            Formatted citation context
        """
        # Truncate content for citation if too long
        if len(self.content) <= context_chars:
            return self.content
        
        # Find sentence boundaries for clean truncation
        truncated = self.content[:context_chars]
        last_sentence = truncated.rfind('.')
        if last_sentence > context_chars // 2:
            truncated = truncated[:last_sentence + 1]
        else:
            truncated += "..."
        
        return truncated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_size": self.chunk_size,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "legal_citations": self.legal_citations,
            "has_embedding": self.has_embedding,
            "embedding_dimensions": self.embedding_dimensions,
            "created_at": self.created_at.isoformat(),
        }


class LegalDocument:
    """
    Core domain entity representing a legal document.
    
    Encapsulates all business logic for legal document management including:
    - Document lifecycle and processing status management
    - Text chunking and semantic organization
    - Legal metadata extraction and validation
    - Processing error handling and retry logic
    - Integration with vector search systems
    """
    
    def __init__(
        self,
        document_id: str,
        user_id: str,
        case_id: str,
        document_name: str,
        original_filename: str,
        file_content: bytes,
        file_type: Optional[DocumentType] = None,
        priority: DocumentPriority = DocumentPriority.NORMAL,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        status: ProcessingStatus = ProcessingStatus.PENDING,
        text_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a legal document with validation.
        
        Args:
            document_id: Unique document identifier
            user_id: Owner of the document
            case_id: Case this document belongs to
            document_name: Display name for the document
            original_filename: Original uploaded filename
            file_content: Raw file content as bytes
            file_type: Document type (auto-detected if None)
            priority: Processing priority level
            created_at: Document creation timestamp
            updated_at: Last modification timestamp
            status: Current processing status
            text_content: Extracted text content
            metadata: Additional document metadata
        """
        # Validate required fields
        self._validate_document_name(document_name)
        self._validate_file_content(file_content)
        self._validate_identifiers(document_id, user_id, case_id)
        
        # Core identifiers
        self._document_id = document_id
        self._user_id = user_id
        self._case_id = case_id
        
        # Document information
        self._document_name = document_name.strip()
        self._original_filename = original_filename.strip()
        self._file_content = file_content
        self._file_size = len(file_content)
        self._file_hash = self._calculate_file_hash(file_content)
        
        # Document type detection
        self._file_type = file_type or self._detect_file_type(original_filename)
        
        # Processing information
        self._priority = priority
        self._status = status
        self._text_content = text_content
        
        # Timestamps
        now = datetime.now(timezone.utc)
        self._created_at = created_at or now
        self._updated_at = updated_at or now
        
        # Metadata and processing tracking
        self._metadata = metadata or {}
        self._processing_metadata = ProcessingMetadata()
        
        # Text chunks
        self._chunks: List[DocumentChunk] = []
        
        # Legal document specific fields
        self._legal_citations: Set[str] = set()
        self._section_headers: List[str] = []
        self._page_count: Optional[int] = None
        
        logger.info(
            "Legal document created",
            document_id=self._document_id,
            document_name=self._document_name,
            case_id=self._case_id,
            file_type=self._file_type.value,
            file_size=self._file_size
        )
    
    @staticmethod
    def _validate_document_name(document_name: str) -> None:
        """Validate document name according to business rules."""
        if not document_name or not document_name.strip():
            raise DocumentProcessingError(
                "Document name cannot be empty",
                error_code=ErrorCode.DOCUMENT_INVALID_FORMAT
            )
        
        if len(document_name.strip()) > 255:
            raise DocumentProcessingError(
                "Document name must be less than 255 characters",
                error_code=ErrorCode.DOCUMENT_INVALID_FORMAT
            )
    
    @staticmethod
    def _validate_file_content(file_content: bytes) -> None:
        """Validate file content."""
        if not file_content:
            raise DocumentProcessingError(
                "File content cannot be empty",
                error_code=ErrorCode.DOCUMENT_INVALID_FORMAT
            )
        
        settings = get_settings()
        max_size = 50 * 1024 * 1024  # 50MB limit for POC
        
        if len(file_content) > max_size:
            raise DocumentProcessingError(
                f"File size {len(file_content)} bytes exceeds maximum of {max_size} bytes",
                error_code=ErrorCode.DOCUMENT_TOO_LARGE
            )
    
    @staticmethod
    def _validate_identifiers(document_id: str, user_id: str, case_id: str) -> None:
        """Validate document identifiers."""
        for field_name, field_value in [("document_id", document_id), ("user_id", user_id), ("case_id", case_id)]:
            if not field_value or not field_value.strip():
                raise DocumentProcessingError(
                    f"{field_name} cannot be empty",
                    error_code=ErrorCode.DOCUMENT_INVALID_FORMAT
                )
    
    @staticmethod
    def _calculate_file_hash(file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def _detect_file_type(filename: str) -> DocumentType:
        """Detect file type from filename extension."""
        if not filename:
            return DocumentType.UNKNOWN
        
        extension = Path(filename).suffix.lower()
        
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.txt': DocumentType.TEXT,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.eml': DocumentType.EMAIL,
            '.msg': DocumentType.EMAIL,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.tiff': DocumentType.IMAGE,
        }
        
        return type_mapping.get(extension, DocumentType.UNKNOWN)
    
    @classmethod
    def create_new(
        cls,
        user_id: str,
        case_id: str,
        document_name: str,
        original_filename: str,
        file_content: bytes,
        priority: DocumentPriority = DocumentPriority.NORMAL
    ) -> "LegalDocument":
        """
        Factory method to create a new legal document with generated ID.
        
        Args:
            user_id: Document owner identifier
            case_id: Case identifier
            document_name: Display name for document
            original_filename: Original uploaded filename
            file_content: Raw file bytes
            priority: Processing priority
            
        Returns:
            New LegalDocument instance
        """
        # Generate unique document ID
        document_id = f"DOC_{datetime.now().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"
        
        return cls(
            document_id=document_id,
            user_id=user_id,
            case_id=case_id,
            document_name=document_name,
            original_filename=original_filename,
            file_content=file_content,
            priority=priority
        )
    
    # Properties (read-only access to core attributes)
    
    @property
    def document_id(self) -> str:
        """Get document identifier."""
        return self._document_id
    
    @property
    def user_id(self) -> str:
        """Get document owner identifier."""
        return self._user_id
    
    @property
    def case_id(self) -> str:
        """Get case identifier."""
        return self._case_id
    
    @property
    def document_name(self) -> str:
        """Get document display name."""
        return self._document_name
    
    @property
    def original_filename(self) -> str:
        """Get original filename."""
        return self._original_filename
    
    @property
    def file_type(self) -> DocumentType:
        """Get document file type."""
        return self._file_type
    
    @property
    def file_size(self) -> int:
        """Get file size in bytes."""
        return self._file_size
    
    @property
    def file_hash(self) -> str:
        """Get file content hash for deduplication."""
        return self._file_hash
    
    @property
    def priority(self) -> DocumentPriority:
        """Get processing priority."""
        return self._priority
    
    @property
    def status(self) -> ProcessingStatus:
        """Get current processing status."""
        return self._status
    
    @property
    def text_content(self) -> Optional[str]:
        """Get extracted text content."""
        return self._text_content
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        return self._metadata.copy()
    
    @property
    def processing_metadata(self) -> ProcessingMetadata:
        """Get processing metadata."""
        return self._processing_metadata
    
    @property
    def chunks(self) -> List[DocumentChunk]:
        """Get document text chunks."""
        return self._chunks.copy()
    
    @property
    def chunk_count(self) -> int:
        """Get number of text chunks."""
        return len(self._chunks)
    
    @property
    def legal_citations(self) -> Set[str]:
        """Get extracted legal citations."""
        return self._legal_citations.copy()
    
    @property
    def section_headers(self) -> List[str]:
        """Get document section headers."""
        return self._section_headers.copy()
    
    @property
    def page_count(self) -> Optional[int]:
        """Get document page count (if available)."""
        return self._page_count
    
    @property
    def file_content(self) -> bytes:
        """Get raw file content."""
        return self._file_content
    
    # Business operations
    
    def update_document_name(self, new_name: str) -> None:
        """
        Update document name with validation.
        
        Args:
            new_name: New document name
        """
        self._validate_document_name(new_name)
        old_name = self._document_name
        self._document_name = new_name.strip()
        self._touch()
        
        logger.info(
            "Document name updated",
            document_id=self._document_id,
            old_name=old_name,
            new_name=self._document_name
        )
    
    def update_priority(self, priority: DocumentPriority) -> None:
        """
        Update processing priority.
        
        Args:
            priority: New priority level
        """
        old_priority = self._priority
        self._priority = priority
        self._touch()
        
        logger.info(
            "Document priority updated",
            document_id=self._document_id,
            old_priority=old_priority.value,
            new_priority=priority.value
        )
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update document metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        self._touch()
    
    def set_text_content(self, text_content: str) -> None:
        """
        Set extracted text content.
        
        Args:
            text_content: Extracted text from document
        """
        self._text_content = text_content
        self._touch()
        
        logger.info(
            "Text content set",
            document_id=self._document_id,
            text_length=len(text_content) if text_content else 0
        )
    
    def set_page_count(self, page_count: int) -> None:
        """
        Set document page count.
        
        Args:
            page_count: Number of pages in document
        """
        if page_count < 0:
            raise DocumentProcessingError(
                "Page count cannot be negative",
                error_code=ErrorCode.DOCUMENT_INVALID_FORMAT,
                document_id=self._document_id
            )
        
        self._page_count = page_count
        self._touch()
    
    def add_legal_citation(self, citation: str) -> None:
        """
        Add a legal citation found in the document.
        
        Args:
            citation: Legal citation text
        """
        if citation and citation.strip():
            self._legal_citations.add(citation.strip())
            self._touch()
    
    def add_section_header(self, header: str) -> None:
        """
        Add a section header from the document structure.
        
        Args:
            header: Section header text
        """
        if header and header.strip():
            self._section_headers.append(header.strip())
            self._touch()
    
    def transition_to_status(self, new_status: ProcessingStatus) -> None:
        """
        Transition document to a new processing status.
        
        Args:
            new_status: Target processing status
        """
        if not self._is_valid_status_transition(self._status, new_status):
            raise DocumentProcessingError(
                f"Invalid status transition from {self._status.value} to {new_status.value}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=self._document_id
            )
        
        old_status = self._status
        self._status = new_status
        self._touch()
        
        logger.info(
            "Document status transition",
            document_id=self._document_id,
            old_status=old_status.value,
            new_status=new_status.value
        )
    
    def _is_valid_status_transition(
        self,
        from_status: ProcessingStatus,
        to_status: ProcessingStatus
    ) -> bool:
        """Validate status transition according to business rules."""
        valid_transitions = {
            ProcessingStatus.PENDING: {
                ProcessingStatus.EXTRACTING,
                ProcessingStatus.FAILED,
                ProcessingStatus.RETRY_PENDING
            },
            ProcessingStatus.EXTRACTING: {
                ProcessingStatus.CHUNKING,
                ProcessingStatus.FAILED,
                ProcessingStatus.RETRY_PENDING
            },
            ProcessingStatus.CHUNKING: {
                ProcessingStatus.EMBEDDING,
                ProcessingStatus.FAILED,
                ProcessingStatus.RETRY_PENDING
            },
            ProcessingStatus.EMBEDDING: {
                ProcessingStatus.INDEXING,
                ProcessingStatus.FAILED,
                ProcessingStatus.RETRY_PENDING
            },
            ProcessingStatus.INDEXING: {
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
                ProcessingStatus.RETRY_PENDING
            },
            ProcessingStatus.FAILED: {
                ProcessingStatus.RETRY_PENDING,
                ProcessingStatus.EXTRACTING
            },
            ProcessingStatus.RETRY_PENDING: {
                ProcessingStatus.EXTRACTING,
                ProcessingStatus.FAILED
            },
            ProcessingStatus.COMPLETED: set()  # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, set())
    
    def start_processing(self, method: str = "LlamaIndex") -> None:
        """
        Start document processing pipeline.
        
        Args:
            method: Processing method identifier
        """
        self.transition_to_status(ProcessingStatus.EXTRACTING)
        self._processing_metadata.start_processing(method)
        
        logger.info(
            "Document processing started",
            document_id=self._document_id,
            method=method
        )
    
    def complete_processing(self, embedding_model: str, index_collection: str) -> None:
        """
        Mark document processing as completed.
        
        Args:
            embedding_model: Model used for embeddings
            index_collection: Vector database collection name
        """
        self.transition_to_status(ProcessingStatus.COMPLETED)
        self._processing_metadata.complete_processing()
        self._processing_metadata.embedding_model = embedding_model
        self._processing_metadata.index_collection = index_collection
        self._processing_metadata.chunk_count = len(self._chunks)
        
        logger.info(
            "Document processing completed",
            document_id=self._document_id,
            chunk_count=len(self._chunks),
            embedding_model=embedding_model
        )
    
    def fail_processing(self, error_message: str, stage: str) -> None:
        """
        Mark document processing as failed.
        
        Args:
            error_message: Error description
            stage: Processing stage where failure occurred
        """
        self.transition_to_status(ProcessingStatus.FAILED)
        self._processing_metadata.fail_processing(error_message, stage)
        
        logger.error(
            "Document processing failed",
            document_id=self._document_id,
            error_message=error_message,
            stage=stage
        )
    
    def prepare_for_retry(self) -> None:
        """Prepare document for retry after failure."""
        if self._status != ProcessingStatus.FAILED:
            raise DocumentProcessingError(
                "Can only retry failed documents",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=self._document_id
            )
        
        # Check retry limits
        max_retries = 3
        if self._processing_metadata.retry_count >= max_retries:
            raise DocumentProcessingError(
                f"Document has exceeded maximum retry attempts ({max_retries})",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=self._document_id
            )
        
        self.transition_to_status(ProcessingStatus.RETRY_PENDING)
        self._processing_metadata.start_retry()
        
        logger.info(
            "Document prepared for retry",
            document_id=self._document_id,
            retry_count=self._processing_metadata.retry_count
        )
    
    def add_chunk(
        self,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        paragraph_number: Optional[int] = None,
        legal_citations: Optional[List[str]] = None
    ) -> DocumentChunk:
        """
        Add a text chunk to the document.
        
        Args:
            content: Chunk text content
            chunk_index: Index of this chunk within the document
            start_char: Starting character position in original text
            end_char: Ending character position in original text
            section_title: Section header if available
            page_number: Page number if available
            paragraph_number: Paragraph number if available
            legal_citations: Legal citations found in this chunk
            
        Returns:
            Created DocumentChunk instance
        """
        chunk_id = f"{self._document_id}_chunk_{chunk_index:04d}"
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=self._document_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
            page_number=page_number,
            paragraph_number=paragraph_number,
            legal_citations=legal_citations or []
        )
        
        self._chunks.append(chunk)
        self._touch()
        
        # Add any legal citations to document-level tracking
        if legal_citations:
            for citation in legal_citations:
                self.add_legal_citation(citation)
        
        logger.debug(
            "Text chunk added",
            document_id=self._document_id,
            chunk_id=chunk_id,
            chunk_size=len(content),
            total_chunks=len(self._chunks)
        )
        
        return chunk
    
    def clear_chunks(self) -> None:
        """Clear all text chunks (for reprocessing)."""
        old_count = len(self._chunks)
        self._chunks.clear()
        self._touch()
        
        logger.info(
            "Document chunks cleared",
            document_id=self._document_id,
            cleared_count=old_count
        )
    
    def is_processing_complete(self) -> bool:
        """Check if document processing is completed successfully."""
        return self._status == ProcessingStatus.COMPLETED
    
    def is_processing_failed(self) -> bool:
        """Check if document processing has failed."""
        return self._status == ProcessingStatus.FAILED
    
    def is_ready_for_search(self) -> bool:
        """Check if document is ready for search operations."""
        return (
            self._status == ProcessingStatus.COMPLETED and
            len(self._chunks) > 0 and
            self._text_content is not None
        )
    
    def can_retry(self) -> bool:
        """Check if document can be retried after failure."""
        return (
            self._status == ProcessingStatus.FAILED and
            self._processing_metadata.retry_count < 3
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_page(self, page_number: int) -> List[DocumentChunk]:
        """
        Get all chunks from a specific page.
        
        Args:
            page_number: Page number to filter by
            
        Returns:
            List of chunks from the specified page
        """
        return [
            chunk for chunk in self._chunks
            if chunk.page_number == page_number
        ]
    
    def get_chunks_with_citations(self) -> List[DocumentChunk]:
        """
        Get all chunks that contain legal citations.
        
        Returns:
            List of chunks containing citations
        """
        return [
            chunk for chunk in self._chunks
            if chunk.legal_citations
        ]
    
    def search_chunks_by_content(self, query: str) -> List[DocumentChunk]:
        """
        Simple text search within document chunks.
        
        Args:
            query: Search query text
            
        Returns:
            List of chunks containing the query text
        """
        query_lower = query.lower()
        return [
            chunk for chunk in self._chunks
            if query_lower in chunk.content.lower()
        ]
    
    def get_chunk_context(self, chunk_id: str, context_chunks: int = 1) -> List[DocumentChunk]:
        """
        Get a chunk with surrounding context chunks.
        
        Args:
            chunk_id: Target chunk identifier
            context_chunks: Number of chunks before and after to include
            
        Returns:
            List of chunks including target and context
        """
        target_chunk = self.get_chunk_by_id(chunk_id)
        if not target_chunk:
            return []
        
        target_index = target_chunk.chunk_index
        start_index = max(0, target_index - context_chunks)
        end_index = min(len(self._chunks), target_index + context_chunks + 1)
        
        return [
            chunk for chunk in self._chunks
            if start_index <= chunk.chunk_index < end_index
        ]
    
    def calculate_processing_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive processing metrics.
        
        Returns:
            Dictionary of processing metrics and statistics
        """
        total_chars = len(self._text_content) if self._text_content else 0
        chunks_with_embeddings = sum(1 for chunk in self._chunks if chunk.has_embedding)
        
        return {
            "file_size_bytes": self._file_size,
            "text_length_chars": total_chars,
            "text_length_words": len(self._text_content.split()) if self._text_content else 0,
            "chunk_count": len(self._chunks),
            "chunks_with_embeddings": chunks_with_embeddings,
            "embedding_coverage": chunks_with_embeddings / len(self._chunks) if self._chunks else 0,
            "average_chunk_size": sum(chunk.chunk_size for chunk in self._chunks) / len(self._chunks) if self._chunks else 0,
            "legal_citations_count": len(self._legal_citations),
            "section_headers_count": len(self._section_headers),
            "page_count": self._page_count,
            "processing_duration": self._processing_metadata.processing_duration_seconds,
            "retry_count": self._processing_metadata.retry_count,
            "extraction_confidence": self._processing_metadata.extraction_confidence,
        }
    
    def _touch(self) -> None:
        """Update the last modified timestamp."""
        self._updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for serialization.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "document_id": self._document_id,
            "user_id": self._user_id,
            "case_id": self._case_id,
            "document_name": self._document_name,
            "original_filename": self._original_filename,
            "file_type": self._file_type.value,
            "file_size": self._file_size,
            "file_hash": self._file_hash,
            "priority": self._priority.value,
            "status": self._status.value,
            "text_content": self._text_content,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "metadata": self._metadata,
            "processing_metadata": self._processing_metadata.to_dict(),
            "chunk_count": len(self._chunks),
            "chunks": [chunk.to_dict() for chunk in self._chunks],
            "legal_citations": list(self._legal_citations),
            "section_headers": self._section_headers,
            "page_count": self._page_count,
            "is_processing_complete": self.is_processing_complete(),
            "is_processing_failed": self.is_processing_failed(),
            "is_ready_for_search": self.is_ready_for_search(),
            "can_retry": self.can_retry(),
            "processing_metrics": self.calculate_processing_metrics(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], file_content: bytes) -> "LegalDocument":
        """
        Create document instance from dictionary.
        
        Args:
            data: Dictionary representation of document
            file_content: Raw file content bytes
            
        Returns:
            LegalDocument instance
        """
        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        
        # Create document instance
        document = cls(
            document_id=data["document_id"],
            user_id=data["user_id"],
            case_id=data["case_id"],
            document_name=data["document_name"],
            original_filename=data["original_filename"],
            file_content=file_content,
            file_type=DocumentType(data["file_type"]),
            priority=DocumentPriority(data.get("priority", DocumentPriority.NORMAL.value)),
            created_at=created_at,
            updated_at=updated_at,
            status=ProcessingStatus(data.get("status", ProcessingStatus.PENDING.value)),
            text_content=data.get("text_content"),
            metadata=data.get("metadata", {})
        )
        
        # Restore processing metadata
        proc_meta_data = data.get("processing_metadata", {})
        document._processing_metadata = ProcessingMetadata(
            processing_method=proc_meta_data.get("processing_method", "unknown"),
            processing_version=proc_meta_data.get("processing_version", "1.0"),
            extraction_confidence=proc_meta_data.get("extraction_confidence", 1.0),
            chunk_count=proc_meta_data.get("chunk_count", 0),
            embedding_model=proc_meta_data.get("embedding_model"),
            embedding_dimensions=proc_meta_data.get("embedding_dimensions", 0),
            index_collection=proc_meta_data.get("index_collection"),
            error_message=proc_meta_data.get("error_message"),
            error_stage=proc_meta_data.get("error_stage"),
            retry_count=proc_meta_data.get("retry_count", 0),
        )
        
        # Restore timestamps in processing metadata
        if proc_meta_data.get("processing_started_at"):
            document._processing_metadata.processing_started_at = datetime.fromisoformat(
                proc_meta_data["processing_started_at"].replace("Z", "+00:00")
            )
        
        if proc_meta_data.get("processing_completed_at"):
            document._processing_metadata.processing_completed_at = datetime.fromisoformat(
                proc_meta_data["processing_completed_at"].replace("Z", "+00:00")
            )
        
        if proc_meta_data.get("last_retry_at"):
            document._processing_metadata.last_retry_at = datetime.fromisoformat(
                proc_meta_data["last_retry_at"].replace("Z", "+00:00")
            )
        
        # Restore chunks
        for chunk_data in data.get("chunks", []):
            chunk = DocumentChunk(
                chunk_id=chunk_data["chunk_id"],
                document_id=chunk_data["document_id"],
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                section_title=chunk_data.get("section_title"),
                page_number=chunk_data.get("page_number"),
                paragraph_number=chunk_data.get("paragraph_number"),
                legal_citations=chunk_data.get("legal_citations", [])
            )
            
            # Restore chunk timestamp
            if "created_at" in chunk_data:
                object.__setattr__(
                    chunk, 
                    'created_at', 
                    datetime.fromisoformat(chunk_data["created_at"].replace("Z", "+00:00"))
                )
            
            document._chunks.append(chunk)
        
        # Restore legal document specific fields
        document._legal_citations = set(data.get("legal_citations", []))
        document._section_headers = data.get("section_headers", [])
        document._page_count = data.get("page_count")
        
        return document
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"LegalDocument(id={self._document_id}, name='{self._document_name}', status={self._status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the document."""
        return (
            f"LegalDocument(document_id='{self._document_id}', case_id='{self._case_id}', "
            f"document_name='{self._document_name}', file_type={self._file_type.value}, "
            f"status={self._status.value}, chunks={len(self._chunks)}, "
            f"file_size={self._file_size}, created_at={self._created_at})"
        )


class DocumentCollection:
    """
    Collection of legal documents with batch operations and analytics.
    
    Provides high-level operations for managing multiple documents
    within a case context, including batch processing, search,
    and analytics across the document collection.
    """
    
    def __init__(self, case_id: str):
        """
        Initialize document collection for a case.
        
        Args:
            case_id: Case identifier for this collection
        """
        self._case_id = case_id
        self._documents: Dict[str, LegalDocument] = {}
        self._created_at = datetime.now(timezone.utc)
    
    @property
    def case_id(self) -> str:
        """Get case identifier."""
        return self._case_id
    
    @property
    def document_count(self) -> int:
        """Get total number of documents."""
        return len(self._documents)
    
    @property
    def documents(self) -> List[LegalDocument]:
        """Get list of all documents."""
        return list(self._documents.values())
    
    def add_document(self, document: LegalDocument) -> None:
        """
        Add a document to the collection.
        
        Args:
            document: Document to add
            
        Raises:
            DocumentProcessingError: If document case_id doesn't match collection
        """
        if document.case_id != self._case_id:
            raise DocumentProcessingError(
                f"Document case_id {document.case_id} doesn't match collection case_id {self._case_id}",
                error_code=ErrorCode.DOCUMENT_INVALID_FORMAT,
                document_id=document.document_id
            )
        
        self._documents[document.document_id] = document
        
        logger.info(
            "Document added to collection",
            case_id=self._case_id,
            document_id=document.document_id,
            total_documents=len(self._documents)
        )
    
    def remove_document(self, document_id: str) -> Optional[LegalDocument]:
        """
        Remove a document from the collection.
        
        Args:
            document_id: Document identifier to remove
            
        Returns:
            Removed document if found, None otherwise
        """
        removed = self._documents.pop(document_id, None)
        if removed:
            logger.info(
                "Document removed from collection",
                case_id=self._case_id,
                document_id=document_id,
                remaining_documents=len(self._documents)
            )
        return removed
    
    def get_document(self, document_id: str) -> Optional[LegalDocument]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        return self._documents.get(document_id)
    
    def get_documents_by_status(self, status: ProcessingStatus) -> List[LegalDocument]:
        """
        Get all documents with a specific processing status.
        
        Args:
            status: Processing status to filter by
            
        Returns:
            List of documents with the specified status
        """
        return [
            doc for doc in self._documents.values()
            if doc.status == status
        ]
    
    def get_documents_by_type(self, file_type: DocumentType) -> List[LegalDocument]:
        """
        Get all documents of a specific type.
        
        Args:
            file_type: Document type to filter by
            
        Returns:
            List of documents of the specified type
        """
        return [
            doc for doc in self._documents.values()
            if doc.file_type == file_type
        ]
    
    def get_failed_documents(self) -> List[LegalDocument]:
        """
        Get all documents that failed processing.
        
        Returns:
            List of failed documents
        """
        return self.get_documents_by_status(ProcessingStatus.FAILED)
    
    def get_retryable_documents(self) -> List[LegalDocument]:
        """
        Get all documents that can be retried.
        
        Returns:
            List of documents eligible for retry
        """
        return [
            doc for doc in self._documents.values()
            if doc.can_retry()
        ]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Calculate processing statistics for the collection.
        
        Returns:
            Dictionary of processing statistics
        """
        if not self._documents:
            return {
                "total_documents": 0,
                "processing_complete": 0,
                "processing_failed": 0,
                "processing_pending": 0,
                "total_chunks": 0,
                "total_size_bytes": 0,
                "completion_rate": 0.0,
                "failure_rate": 0.0,
            }
        
        status_counts = {}
        for status in ProcessingStatus:
            status_counts[status.value] = len(self.get_documents_by_status(status))
        
        total_docs = len(self._documents)
        completed_docs = status_counts.get(ProcessingStatus.COMPLETED.value, 0)
        failed_docs = status_counts.get(ProcessingStatus.FAILED.value, 0)
        
        return {
            "total_documents": total_docs,
            "processing_complete": completed_docs,
            "processing_failed": failed_docs,
            "processing_pending": total_docs - completed_docs - failed_docs,
            "total_chunks": sum(doc.chunk_count for doc in self._documents.values()),
            "total_size_bytes": sum(doc.file_size for doc in self._documents.values()),
            "completion_rate": completed_docs / total_docs if total_docs > 0 else 0.0,
            "failure_rate": failed_docs / total_docs if total_docs > 0 else 0.0,
            "status_breakdown": status_counts,
            "file_type_breakdown": self._get_file_type_breakdown(),
        }
    
    def _get_file_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of documents by file type."""
        breakdown = {}
        for file_type in DocumentType:
            breakdown[file_type.value] = len(self.get_documents_by_type(file_type))
        return breakdown
    
    def search_documents(self, query: str) -> List[Tuple[LegalDocument, List[DocumentChunk]]]:
        """
        Search for documents containing the query text.
        
        Args:
            query: Search query text
            
        Returns:
            List of tuples (document, matching_chunks)
        """
        results = []
        for document in self._documents.values():
            matching_chunks = document.search_chunks_by_content(query)
            if matching_chunks:
                results.append((document, matching_chunks))
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary for serialization."""
        return {
            "case_id": self._case_id,
            "document_count": self.document_count,
            "created_at": self._created_at.isoformat(),
            "documents": [doc.to_dict() for doc in self._documents.values()],
            "processing_statistics": self.get_processing_statistics(),
        }