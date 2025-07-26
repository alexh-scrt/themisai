"""
Document Processing Service - Business Logic Layer

This module provides the business logic layer for legal document management in the
Patexia Legal AI Chatbot. It orchestrates document operations, coordinates processing
pipelines, manages business rules, and provides service layer abstraction.

Key Features:
- Document lifecycle management with status transitions
- LlamaIndex processing pipeline orchestration
- Real-time progress tracking via WebSocket notifications
- Business rule validation and enforcement
- Error handling with retry mechanisms and fallback strategies
- File validation and security checks
- Processing capacity management and queuing
- Document deduplication and conflict resolution
- Integration with vector storage and metadata management

Business Rules:
- File size limits (100MB max, configurable)
- Supported file types (PDF, TXT with future DOCX support)
- Processing queue management and priority handling
- Document naming and metadata validation
- Case capacity enforcement integration
- Processing timeout and retry logic
- Content extraction and validation requirements

Processing Pipeline:
1. File validation and security checks
2. Document metadata extraction and storage
3. LlamaIndex processing coordination
4. Vector embedding generation and indexing
5. Real-time progress updates and completion notification
6. Error handling and cleanup procedures

Architecture Integration:
- Coordinates with DocumentProcessor for LlamaIndex operations
- Integrates with DocumentRepository for metadata storage
- Uses VectorRepository for vector storage management
- Employs WebSocketManager for real-time notifications
- Implements business rules and validation logic
- Provides service layer abstraction for API controllers
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import tempfile
import aiofiles
import mimetypes

from ..core.config import get_settings
from ..core.websocket_manager import WebSocketManager
from ..models.domain.document import (
    LegalDocument, DocumentType, ProcessingStatus, DocumentPriority,
    DocumentChunk, DocumentCollection, ProcessingMetadata
)
from ..models.api.document_schemas import (
    DocumentUploadRequest, DocumentUpdateRequest, DocumentListRequest,
    DocumentResponse, DocumentListResponse, DocumentProgressUpdate,
    ProcessingStatusUpdate
)
from ..processors.document_processor import DocumentProcessor
from ..repositories.mongodb.document_repository import DocumentRepository
from ..repositories.weaviate.vector_repository import VectorRepository
from ..services.embedding_service import EmbeddingService
from ..exceptions import (
    DocumentProcessingError, ValidationError, ResourceError, StorageError,
    ErrorCode, raise_document_error, raise_validation_error, raise_resource_error
)
from ..utils.logging import get_logger, performance_context
from ..utils.validators import (
    validate_file_type, validate_file_size, validate_document_name,
    calculate_file_hash, extract_file_metadata
)

logger = get_logger(__name__)


class DocumentOperation(str, Enum):
    """Types of document operations for audit tracking."""
    UPLOAD = "upload"
    UPDATE = "update"
    DELETE = "delete"
    PROCESS = "process"
    RETRY = "retry"
    CANCEL = "cancel"


class ProcessingQueue(str, Enum):
    """Processing queue priorities."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class DocumentProcessingTask:
    """Represents a document processing task in the queue."""
    document_id: str
    case_id: str
    user_id: str
    priority: ProcessingQueue
    created_at: datetime
    attempts: int = 0
    max_attempts: int = 3
    
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.attempts < self.max_attempts
    
    def increment_attempts(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1


class DocumentService:
    """
    Business logic service for legal document management.
    
    Orchestrates document operations, enforces business rules, manages the
    processing pipeline, and coordinates between data access and API layers.
    """
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_repository: VectorRepository,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        websocket_manager: WebSocketManager
    ):
        """
        Initialize document service with required dependencies.
        
        Args:
            document_repository: MongoDB repository for document metadata
            vector_repository: Weaviate repository for vector storage
            document_processor: LlamaIndex processing pipeline
            embedding_service: Ollama embedding service
            websocket_manager: WebSocket manager for real-time updates
        """
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.websocket_manager = websocket_manager
        
        # Load configuration
        self.settings = get_settings()
        self.max_file_size = self.settings.legal_documents.max_file_size_mb * 1024 * 1024
        self.supported_types = self.settings.legal_documents.supported_file_types
        self.processing_timeout = self.settings.legal_documents.processing_timeout_minutes * 60
        
        # Processing state management
        self._processing_queue: Dict[ProcessingQueue, List[DocumentProcessingTask]] = {
            ProcessingQueue.HIGH: [],
            ProcessingQueue.NORMAL: [],
            ProcessingQueue.LOW: [],
            ProcessingQueue.BACKGROUND: []
        }
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._processing_stats = {
            "total_uploaded": 0,
            "total_processed": 0,
            "total_failed": 0,
            "current_active": 0,
            "queue_size": 0
        }
        
        # Start background processing
        self._processing_task = asyncio.create_task(self._process_queue())
        
        logger.info(
            "DocumentService initialized",
            max_file_size_mb=self.max_file_size // (1024 * 1024),
            supported_types=self.supported_types,
            processing_timeout_minutes=self.processing_timeout // 60
        )
    
    async def upload_document(
        self,
        upload_request: DocumentUploadRequest,
        file_content: bytes,
        user_id: str
    ) -> DocumentResponse:
        """
        Upload and queue a document for processing.
        
        Args:
            upload_request: Document upload request with metadata
            file_content: Raw file content bytes
            user_id: User identifier for ownership
            
        Returns:
            Document response with upload confirmation
            
        Raises:
            ValidationError: If file validation fails
            ResourceError: If case capacity exceeded
            DocumentProcessingError: If upload fails
        """
        async with performance_context("document_upload", document_name=upload_request.document_name):
            try:
                # Validate file and request
                await self._validate_upload_request(upload_request, file_content, user_id)
                
                # Check case capacity
                await self._check_case_capacity(upload_request.case_id, user_id)
                
                # Calculate file hash for deduplication
                file_hash = calculate_file_hash(file_content)
                
                # Check for duplicates
                existing_doc = await self.document_repository.find_by_hash(
                    file_hash, upload_request.case_id
                )
                if existing_doc:
                    logger.warning(
                        "Duplicate document detected",
                        existing_id=existing_doc.document_id,
                        case_id=upload_request.case_id,
                        file_hash=file_hash
                    )
                    raise_validation_error(
                        "Document with identical content already exists in this case",
                        ErrorCode.DOCUMENT_DUPLICATE,
                        {"existing_document_id": existing_doc.document_id}
                    )
                
                # Extract file metadata
                file_metadata = extract_file_metadata(
                    file_content, 
                    upload_request.original_filename
                )
                
                # Determine file type
                file_type = self._determine_file_type(upload_request.original_filename)
                
                # Create document entity
                document = LegalDocument(
                    user_id=user_id,
                    case_id=upload_request.case_id,
                    document_name=upload_request.document_name,
                    original_filename=upload_request.original_filename,
                    file_content=file_content,
                    file_type=file_type,
                    priority=upload_request.priority or DocumentPriority.NORMAL,
                    file_hash=file_hash,
                    metadata=file_metadata
                )
                
                # Store document in repository
                stored_document = await self.document_repository.create_document(document)
                
                # Queue for processing
                await self._queue_for_processing(
                    stored_document.document_id,
                    upload_request.case_id,
                    user_id,
                    upload_request.priority or DocumentPriority.NORMAL
                )
                
                # Update statistics
                self._processing_stats["total_uploaded"] += 1
                self._processing_stats["queue_size"] += 1
                
                # Send upload confirmation
                await self._send_progress_update(
                    stored_document.document_id,
                    ProcessingStatus.PENDING,
                    "Document uploaded successfully - queued for processing",
                    user_id
                )
                
                logger.info(
                    "Document uploaded successfully",
                    document_id=stored_document.document_id,
                    case_id=upload_request.case_id,
                    file_size=len(file_content),
                    file_type=file_type.value
                )
                
                return self._convert_to_response(stored_document)
                
            except Exception as e:
                logger.error(
                    "Document upload failed",
                    error=str(e),
                    case_id=upload_request.case_id,
                    filename=upload_request.original_filename
                )
                raise
    
    async def get_document(
        self,
        document_id: str,
        user_id: str,
        include_content: bool = False
    ) -> Optional[DocumentResponse]:
        """
        Retrieve a document by ID with access control.
        
        Args:
            document_id: Document identifier
            user_id: User identifier for access control
            include_content: Whether to include full text content
            
        Returns:
            Document response if found and accessible, None otherwise
        """
        document = await self.document_repository.get_document(document_id)
        
        if not document:
            return None
        
        # Check access control
        if document.user_id != user_id:
            logger.warning(
                "Unauthorized document access attempt",
                document_id=document_id,
                owner_id=document.user_id,
                requester_id=user_id
            )
            return None
        
        response = self._convert_to_response(document)
        
        # Include content if requested and available
        if include_content and document.text_content:
            response.text_content = document.text_content
        
        return response
    
    async def list_documents(
        self,
        list_request: DocumentListRequest,
        user_id: str
    ) -> DocumentListResponse:
        """
        List documents with filtering and pagination.
        
        Args:
            list_request: List request with filters and pagination
            user_id: User identifier for access control
            
        Returns:
            Paginated list of documents matching criteria
        """
        # Apply user access control
        if not list_request.user_id:
            list_request.user_id = user_id
        elif list_request.user_id != user_id:
            # Only allow admin users to list other users' documents
            # This would require admin role checking in a full implementation
            logger.warning(
                "Cross-user document listing attempt",
                requester_id=user_id,
                target_user_id=list_request.user_id
            )
            list_request.user_id = user_id
        
        # Retrieve documents from repository
        documents, total_count = await self.document_repository.list_documents(
            filters=self._build_repository_filters(list_request),
            limit=list_request.limit,
            offset=list_request.offset,
            sort_by=list_request.sort_by,
            sort_order=list_request.sort_order
        )
        
        # Convert to response format
        document_responses = [
            self._convert_to_response(doc, include_chunks=list_request.include_chunks)
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total_count=total_count,
            limit=list_request.limit,
            offset=list_request.offset,
            has_more=total_count > (list_request.offset + len(document_responses))
        )
    
    async def update_document(
        self,
        document_id: str,
        update_request: DocumentUpdateRequest,
        user_id: str
    ) -> Optional[DocumentResponse]:
        """
        Update document metadata and settings.
        
        Args:
            document_id: Document identifier
            update_request: Update request with new values
            user_id: User identifier for access control
            
        Returns:
            Updated document response if successful, None if not found
        """
        document = await self.document_repository.get_document(document_id)
        
        if not document or document.user_id != user_id:
            return None
        
        # Apply updates
        updates = {}
        if update_request.document_name is not None:
            validate_document_name(update_request.document_name)
            updates["document_name"] = update_request.document_name
        
        if update_request.priority is not None:
            updates["priority"] = update_request.priority
            
            # Update processing queue if still pending/processing
            if document.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
                await self._update_processing_priority(document_id, update_request.priority)
        
        if updates:
            updated_document = await self.document_repository.update_document(
                document_id, updates
            )
            
            logger.info(
                "Document updated successfully",
                document_id=document_id,
                updates=list(updates.keys())
            )
            
            return self._convert_to_response(updated_document)
        
        return self._convert_to_response(document)
    
    async def delete_document(
        self,
        document_id: str,
        user_id: str,
        force: bool = False
    ) -> bool:
        """
        Delete a document and cleanup associated data.
        
        Args:
            document_id: Document identifier
            user_id: User identifier for access control
            force: Whether to force deletion of processing documents
            
        Returns:
            True if deleted successfully, False if not found
        """
        document = await self.document_repository.get_document(document_id)
        
        if not document or document.user_id != user_id:
            return False
        
        # Check if deletion is allowed
        if not force and document.status == ProcessingStatus.PROCESSING:
            raise_validation_error(
                "Cannot delete document while processing. Use force=True to override.",
                ErrorCode.DOCUMENT_PROCESSING_IN_PROGRESS,
                {"document_id": document_id, "status": document.status.value}
            )
        
        try:
            # Cancel processing if active
            if document_id in self._active_tasks:
                self._active_tasks[document_id].cancel()
                del self._active_tasks[document_id]
            
            # Remove from processing queue
            await self._remove_from_queue(document_id)
            
            # Delete vector embeddings if they exist
            if document.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                try:
                    collection_name = f"LegalDocument_CASE_{document.case_id}"
                    await self.vector_repository.delete_document_vectors(
                        collection_name, document_id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to delete vector embeddings",
                        document_id=document_id,
                        error=str(e)
                    )
            
            # Delete document from repository
            await self.document_repository.delete_document(document_id)
            
            logger.info(
                "Document deleted successfully",
                document_id=document_id,
                case_id=document.case_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Document deletion failed",
                document_id=document_id,
                error=str(e)
            )
            raise_document_error(
                f"Failed to delete document: {str(e)}",
                ErrorCode.DOCUMENT_DELETE_FAILED,
                {"document_id": document_id}
            )
    
    async def retry_processing(
        self,
        document_id: str,
        user_id: str
    ) -> bool:
        """
        Retry processing for a failed document.
        
        Args:
            document_id: Document identifier
            user_id: User identifier for access control
            
        Returns:
            True if retry queued successfully, False if not applicable
        """
        document = await self.document_repository.get_document(document_id)
        
        if not document or document.user_id != user_id:
            return False
        
        if document.status != ProcessingStatus.FAILED:
            raise_validation_error(
                "Document is not in failed state",
                ErrorCode.DOCUMENT_INVALID_STATUS,
                {"document_id": document_id, "current_status": document.status.value}
            )
        
        # Reset document status
        await self.document_repository.update_document(
            document_id,
            {
                "status": ProcessingStatus.PENDING,
                "processing_metadata.error_message": None,
                "processing_metadata.retry_count": document.processing_metadata.retry_count + 1,
                "updated_at": datetime.now(timezone.utc)
            }
        )
        
        # Queue for processing with higher priority
        await self._queue_for_processing(
            document_id,
            document.case_id,
            user_id,
            DocumentPriority.HIGH  # Give retries higher priority
        )
        
        logger.info(
            "Document retry queued",
            document_id=document_id,
            retry_count=document.processing_metadata.retry_count + 1
        )
        
        return True
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        queue_sizes = {
            queue_type.value: len(tasks)
            for queue_type, tasks in self._processing_queue.items()
        }
        
        total_queue_size = sum(queue_sizes.values())
        self._processing_stats["queue_size"] = total_queue_size
        
        return {
            **self._processing_stats,
            "queue_sizes": queue_sizes,
            "active_tasks": len(self._active_tasks),
            "processing_capacity": self.settings.legal_documents.max_concurrent_processing
        }
    
    # Private helper methods
    
    async def _validate_upload_request(
        self,
        upload_request: DocumentUploadRequest,
        file_content: bytes,
        user_id: str
    ) -> None:
        """Validate upload request and file content."""
        # Validate file size
        if not validate_file_size(len(file_content), self.max_file_size):
            raise_validation_error(
                f"File size exceeds maximum limit of {self.max_file_size // (1024*1024)}MB",
                ErrorCode.FILE_SIZE_EXCEEDED,
                {"file_size": len(file_content), "max_size": self.max_file_size}
            )
        
        # Validate file type
        if not validate_file_type(upload_request.original_filename, self.supported_types):
            raise_validation_error(
                f"Unsupported file type. Supported types: {', '.join(self.supported_types)}",
                ErrorCode.FILE_TYPE_UNSUPPORTED,
                {"filename": upload_request.original_filename, "supported_types": self.supported_types}
            )
        
        # Validate document name
        validate_document_name(upload_request.document_name)
        
        # Validate file content
        if len(file_content) == 0:
            raise_validation_error(
                "File content is empty",
                ErrorCode.FILE_EMPTY,
                {"filename": upload_request.original_filename}
            )
    
    async def _check_case_capacity(self, case_id: str, user_id: str) -> None:
        """Check if case has capacity for additional documents."""
        document_count = await self.document_repository.count_case_documents(case_id)
        max_documents = self.settings.legal_documents.max_documents_per_case
        
        if document_count >= max_documents:
            raise_resource_error(
                f"Case has reached maximum document limit of {max_documents}",
                ErrorCode.CASE_CAPACITY_EXCEEDED,
                {"case_id": case_id, "current_count": document_count, "max_allowed": max_documents}
            )
    
    def _determine_file_type(self, filename: str) -> DocumentType:
        """Determine document type from filename."""
        suffix = Path(filename).suffix.lower()
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TEXT,
            ".docx": DocumentType.DOCX,  # Future support
            ".doc": DocumentType.DOC    # Future support
        }
        return type_mapping.get(suffix, DocumentType.UNKNOWN)
    
    async def _queue_for_processing(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        priority: DocumentPriority
    ) -> None:
        """Add document to processing queue."""
        queue_priority = self._map_to_queue_priority(priority)
        
        task = DocumentProcessingTask(
            document_id=document_id,
            case_id=case_id,
            user_id=user_id,
            priority=queue_priority,
            created_at=datetime.now(timezone.utc)
        )
        
        self._processing_queue[queue_priority].append(task)
        
        logger.debug(
            "Document queued for processing",
            document_id=document_id,
            queue=queue_priority.value,
            queue_size=len(self._processing_queue[queue_priority])
        )
    
    def _map_to_queue_priority(self, doc_priority: DocumentPriority) -> ProcessingQueue:
        """Map document priority to processing queue priority."""
        mapping = {
            DocumentPriority.LOW: ProcessingQueue.LOW,
            DocumentPriority.NORMAL: ProcessingQueue.NORMAL,
            DocumentPriority.HIGH: ProcessingQueue.HIGH,
            DocumentPriority.URGENT: ProcessingQueue.HIGH
        }
        return mapping.get(doc_priority, ProcessingQueue.NORMAL)
    
    async def _process_queue(self) -> None:
        """Background task to process document queue."""
        while True:
            try:
                # Check if we have capacity for more processing
                max_concurrent = self.settings.legal_documents.max_concurrent_processing
                if len(self._active_tasks) >= max_concurrent:
                    await asyncio.sleep(5)  # Wait before checking again
                    continue
                
                # Find next task to process (priority order)
                task = None
                for queue_priority in [ProcessingQueue.HIGH, ProcessingQueue.NORMAL, 
                                     ProcessingQueue.LOW, ProcessingQueue.BACKGROUND]:
                    if self._processing_queue[queue_priority]:
                        task = self._processing_queue[queue_priority].pop(0)
                        break
                
                if not task:
                    await asyncio.sleep(2)  # No tasks available
                    continue
                
                # Start processing task
                processing_task = asyncio.create_task(
                    self._process_document_task(task)
                )
                self._active_tasks[task.document_id] = processing_task
                
            except Exception as e:
                logger.error("Error in queue processing", error=str(e))
                await asyncio.sleep(5)
    
    async def _process_document_task(self, task: DocumentProcessingTask) -> None:
        """Process a single document task."""
        try:
            # Update statistics
            self._processing_stats["current_active"] += 1
            
            # Process document using document processor
            success = await self.document_processor.process_document(
                task.document_id,
                task.case_id,
                task.user_id
            )
            
            if success:
                self._processing_stats["total_processed"] += 1
            else:
                self._processing_stats["total_failed"] += 1
                
                # Handle retry logic
                task.increment_attempts()
                if task.should_retry():
                    # Re-queue for retry with lower priority
                    await asyncio.sleep(30)  # Wait before retry
                    self._processing_queue[ProcessingQueue.BACKGROUND].append(task)
                    
        except Exception as e:
            logger.error(
                "Document processing task failed",
                document_id=task.document_id,
                error=str(e)
            )
            self._processing_stats["total_failed"] += 1
            
        finally:
            # Cleanup
            if task.document_id in self._active_tasks:
                del self._active_tasks[task.document_id]
            self._processing_stats["current_active"] -= 1
    
    async def _send_progress_update(
        self,
        document_id: str,
        status: ProcessingStatus,
        message: str,
        user_id: str
    ) -> None:
        """Send progress update via WebSocket."""
        update = DocumentProgressUpdate(
            document_id=document_id,
            status=status,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.websocket_manager.send_to_user(
            user_id,
            {
                "type": "document_progress",
                "data": update.dict()
            }
        )
    
    def _convert_to_response(
        self,
        document: LegalDocument,
        include_chunks: bool = False
    ) -> DocumentResponse:
        """Convert domain model to API response."""
        response_data = {
            "document_id": document.document_id,
            "user_id": document.user_id,
            "case_id": document.case_id,
            "document_name": document.document_name,
            "original_filename": document.original_filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "file_hash": document.file_hash,
            "priority": document.priority,
            "status": document.status,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "metadata": document.metadata,
            "processing_metadata": document.processing_metadata,
            "chunk_count": document.chunk_count,
            "legal_citations": document.legal_citations,
            "section_headers": document.section_headers,
            "page_count": document.page_count,
            "is_processing_complete": document.is_processing_complete,
            "is_processing_failed": document.is_processing_failed,
            "can_retry": document.can_retry
        }
        
        return DocumentResponse(**response_data)
    
    def _build_repository_filters(self, list_request: DocumentListRequest) -> Dict[str, Any]:
        """Build repository filters from list request."""
        filters = {}
        
        if list_request.user_id:
            filters["user_id"] = list_request.user_id
        if list_request.case_id:
            filters["case_id"] = list_request.case_id
        if list_request.status:
            filters["status"] = list_request.status
        if list_request.file_type:
            filters["file_type"] = list_request.file_type
        if list_request.priority:
            filters["priority"] = list_request.priority
        if list_request.search_query:
            filters["search_query"] = list_request.search_query
        if list_request.has_failures is not None:
            filters["has_failures"] = list_request.has_failures
        if list_request.created_after:
            filters["created_after"] = list_request.created_after
        if list_request.created_before:
            filters["created_before"] = list_request.created_before
        if list_request.min_file_size:
            filters["min_file_size"] = list_request.min_file_size
        if list_request.max_file_size:
            filters["max_file_size"] = list_request.max_file_size
        
        return filters
    
    async def _update_processing_priority(
        self,
        document_id: str,
        new_priority: DocumentPriority
    ) -> None:
        """Update priority for document in processing queue."""
        new_queue_priority = self._map_to_queue_priority(new_priority)
        
        # Find and move task in queue
        for queue_priority, tasks in self._processing_queue.items():
            for i, task in enumerate(tasks):
                if task.document_id == document_id:
                    # Remove from current queue
                    removed_task = tasks.pop(i)
                    # Update priority and add to new queue
                    removed_task.priority = new_queue_priority
                    self._processing_queue[new_queue_priority].append(removed_task)
                    return
    
    async def _remove_from_queue(self, document_id: str) -> bool:
        """Remove document from processing queue."""
        for queue_priority, tasks in self._processing_queue.items():
            for i, task in enumerate(tasks):
                if task.document_id == document_id:
                    tasks.pop(i)
                    return True
        return False
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Cancel processing task
        if hasattr(self, '_processing_task'):
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active tasks
        for task in self._active_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        
        logger.info("DocumentService cleanup completed")