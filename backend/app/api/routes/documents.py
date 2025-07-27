"""
Document Management API Routes for LegalAI Document Processing System

This module provides REST API endpoints for legal document management in the Patexia 
Legal AI Chatbot. It enables document upload, processing, tracking, and retrieval with
real-time progress updates and comprehensive error handling.

Key Features:
- Document upload with file validation and metadata extraction
- Real-time processing progress tracking via WebSocket updates
- Document CRUD operations with access control
- Batch document operations and bulk processing
- Document search and filtering within cases
- Processing status monitoring and error recovery
- File type validation and size limits
- Document analytics and metrics aggregation

Document Operations:
- Upload: Multi-format file upload with validation
- Process: Background document processing with progress tracking
- Retrieve: Document content and metadata access
- Update: Document metadata and configuration changes
- Delete: Document removal with cleanup
- Batch: Multiple document operations

Processing Pipeline:
- File validation and type detection
- Text extraction from PDF and text files
- Semantic chunking with legal document structure awareness
- Vector embedding generation via Ollama models
- Storage in Weaviate vector database per case
- Real-time progress updates via WebSocket

Business Rules:
- 25 documents per case limit (configurable with override)
- Supported file types: PDF, TXT (future: DOCX)
- Maximum file size: 50MB per document
- Document name uniqueness within case
- User access control and ownership validation

Architecture Integration:
- Uses DocumentService for business logic and processing pipeline
- Integrates with WebSocketManager for real-time progress updates
- Connects to MongoDB via DocumentRepository for metadata persistence
- Coordinates with VectorRepository for embedding storage
- Implements comprehensive error handling and logging
"""

import asyncio
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Path as FastAPIPath,
    Body,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    Request,
    Response,
    status
)
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config.settings import get_settings
from ...core.websocket_manager import WebSocketManager, get_websocket_manager
from backend.app.services.document_service import DocumentService
from ...services.case_service import CaseService
from backend.app.models.api.document_schemas import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentProgressUpdate,
    DocumentAnalyticsResponse,
    DocumentBatchRequest,
    DocumentBatchResponse,
    DocumentUpdateRequest,
    ApiResponse,
    ErrorResponse
)
from backend.app.models.api.common_schemas import (
    ApiResponse,
    ErrorResponse
)

from ...models.domain.document import (
    DocumentType,
    ProcessingStatus,
    DocumentPriority,
    LegalDocument
)
from ...utils.logging import (
    get_logger,
    performance_context,
    log_business_event,
    log_route_entry,
    log_route_exit,
    log_document_processing
)
from ...core.exceptions import (
    DocumentProcessingError,
    CaseManagementError,
    ValidationError,
    ResourceError,
    ErrorCode,
    get_exception_response_data
)
from ..deps import get_document_service, get_case_service


logger = get_logger(__name__)
router = APIRouter()


# Document Upload and Processing

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload a document file for processing with real-time progress tracking"
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    case_id: str = Form(..., description="Case ID to associate document with"),
    document_name: str = Form(..., description="Display name for the document"),
    priority: DocumentPriority = Form(
        DocumentPriority.NORMAL,
        description="Processing priority level"
    ),
    metadata: Optional[str] = Form(
        None,
        description="Additional metadata as JSON string"
    ),
    document_service: DocumentService = Depends(get_document_service),
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> DocumentUploadResponse:
    """Upload and process a document file."""
    log_route_entry(
        request,
        filename=file.filename,
        case_id=case_id,
        document_name=document_name,
        file_size=file.size
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "document_upload",
            filename=file.filename,
            case_id=case_id,
            file_size=file.size
        ):
            # Validate file upload
            await _validate_file_upload(file, case_id, document_name)
            
            # Check case exists and user has access
            case = await case_service.get_case(case_id, user_id)
            if not case:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Case not found"}
                )
            
            # Check document capacity for case
            if case.document_count >= 25:  # Business rule: 25 docs per case
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Case document limit exceeded",
                        "limit": 25,
                        "current_count": case.document_count
                    }
                )
            
            # Read file content
            file_content = await file.read()
            
            # Parse metadata if provided
            parsed_metadata = {}
            if metadata:
                try:
                    import json
                    parsed_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"error": "Invalid metadata JSON format"}
                    )
            
            # Create upload request
            upload_request = DocumentUploadRequest(
                case_id=case_id,
                document_name=document_name,
                priority=priority,
                metadata=parsed_metadata
            )
            
            # Upload document via service
            result = await document_service.upload_document(
                upload_request=upload_request,
                file_content=file_content,
                original_filename=file.filename or "unknown",
                user_id=user_id
            )
            
            if result.success:
                # Start background processing
                background_tasks.add_task(
                    _process_document_background,
                    document_service,
                    result.document_id,
                    user_id,
                    websocket_manager
                )
                
                # Send initial WebSocket notification
                background_tasks.add_task(
                    _notify_document_event,
                    websocket_manager,
                    user_id,
                    "document_uploaded",
                    {
                        "document_id": result.document_id,
                        "document_name": document_name,
                        "case_id": case_id,
                        "file_size": len(file_content),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "document_uploaded",
                    request,
                    document_id=result.document_id,
                    case_id=case_id,
                    file_size=len(file_content)
                )
                
                response_data = DocumentUploadResponse(
                    document_id=result.document_id,
                    document_name=document_name,
                    file_type=DocumentType.PDF if file.filename.lower().endswith('.pdf') else DocumentType.TEXT,
                    file_size=len(file_content),
                    status=ProcessingStatus.PENDING,
                    upload_timestamp=datetime.now(timezone.utc),
                    estimated_processing_time=_estimate_processing_time(len(file_content))
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise DocumentProcessingError(
                    message=result.error or "Failed to upload document",
                    error_code=ErrorCode.DOCUMENT_UPLOAD_FAILED
                )
                
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to upload document",
            filename=file.filename,
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (DocumentProcessingError, ValidationError, ResourceError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to upload document"}
        )


@router.post(
    "/batch-upload",
    response_model=DocumentBatchResponse,
    summary="Batch Upload Documents",
    description="Upload multiple documents for processing in batch"
)
async def batch_upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    files: List[UploadFile] = File(..., description="Document files to upload"),
    case_id: str = Form(..., description="Case ID to associate documents with"),
    priority: DocumentPriority = Form(
        DocumentPriority.NORMAL,
        description="Processing priority level for all documents"
    )
) -> DocumentBatchResponse:
    """Upload multiple documents in batch."""
    log_route_entry(
        request,
        case_id=case_id,
        file_count=len(files),
        total_size=sum(f.size or 0 for f in files)
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "document_batch_upload",
            case_id=case_id,
            file_count=len(files)
        ):
            # Validate batch size
            if len(files) > 10:  # Limit batch size
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Batch size limited to 10 files"}
                )
            
            # Check case exists and capacity
            case = await case_service.get_case(case_id, user_id)
            if not case:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Case not found"}
                )
            
            if case.document_count + len(files) > 25:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Batch would exceed case document limit",
                        "limit": 25,
                        "current_count": case.document_count,
                        "batch_size": len(files)
                    }
                )
            
            # Process each file
            upload_results = []
            for file in files:
                try:
                    # Basic validation
                    await _validate_file_upload(file, case_id, file.filename or "unknown")
                    
                    # Read file content
                    file_content = await file.read()
                    
                    # Create upload request
                    upload_request = DocumentUploadRequest(
                        case_id=case_id,
                        document_name=file.filename or "unknown",
                        priority=priority,
                        metadata={"batch_upload": True}
                    )
                    
                    # Upload document
                    result = await document_service.upload_document(
                        upload_request=upload_request,
                        file_content=file_content,
                        original_filename=file.filename or "unknown",
                        user_id=user_id
                    )
                    
                    if result.success:
                        upload_results.append({
                            "filename": file.filename,
                            "document_id": result.document_id,
                            "status": "uploaded",
                            "file_size": len(file_content)
                        })
                        
                        # Start background processing
                        background_tasks.add_task(
                            _process_document_background,
                            document_service,
                            result.document_id,
                            user_id,
                            websocket_manager
                        )
                    else:
                        upload_results.append({
                            "filename": file.filename,
                            "status": "failed",
                            "error": result.error
                        })
                        
                except Exception as exc:
                    upload_results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(exc)
                    })
            
            # Send batch notification
            background_tasks.add_task(
                _notify_document_event,
                websocket_manager,
                user_id,
                "documents_batch_uploaded",
                {
                    "case_id": case_id,
                    "file_count": len(files),
                    "successful_uploads": len([r for r in upload_results if r["status"] == "uploaded"]),
                    "failed_uploads": len([r for r in upload_results if r["status"] == "failed"]),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            log_business_event(
                "documents_batch_uploaded",
                request,
                case_id=case_id,
                file_count=len(files),
                results=upload_results
            )
            
            response_data = DocumentBatchResponse(
                total_files=len(files),
                successful_uploads=len([r for r in upload_results if r["status"] == "uploaded"]),
                failed_uploads=len([r for r in upload_results if r["status"] == "failed"]),
                results=upload_results,
                processing_started=True
            )
            
            log_route_exit(request, response_data)
            return response_data
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to batch upload documents",
            case_id=case_id,
            file_count=len(files),
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to batch upload documents"}
        )


# Document Retrieval and Management

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Details",
    description="Retrieve detailed information about a specific document"
)
async def get_document(
    request: Request,
    document_id: str = FastAPIPath(..., description="Document identifier"),
    include_content: bool = Query(
        False,
        description="Include full text content in response"
    ),
    include_chunks: bool = Query(
        False,
        description="Include document chunks information"
    ),
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """Get detailed document information."""
    log_route_entry(request, document_id=document_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("document_get", document_id=document_id):
            document = await document_service.get_document(
                document_id,
                user_id,
                include_content=include_content,
                include_chunks=include_chunks
            )
            
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Document not found"}
                )
            
            log_route_exit(request, document)
            return document
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get document",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (DocumentProcessingError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get document"}
        )


@router.get(
    "/case/{case_id}",
    response_model=DocumentListResponse,
    summary="List Documents in Case",
    description="List all documents in a case with filtering and pagination"
)
async def list_documents_in_case(
    request: Request,
    case_id: str = FastAPIPath(..., description="Case identifier"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    file_type: Optional[DocumentType] = Query(None, description="Filter by file type"),
    priority: Optional[DocumentPriority] = Query(None, description="Filter by priority"),
    search_query: Optional[str] = Query(None, description="Search in document names"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(50, description="Number of documents to return", ge=1, le=200),
    offset: int = Query(0, description="Number of documents to skip", ge=0),
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentListResponse:
    """List documents in a case."""
    log_route_entry(
        request,
        case_id=case_id,
        status=status,
        limit=limit,
        offset=offset
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "document_list",
            case_id=case_id,
            limit=limit,
            offset=offset
        ):
            documents = await document_service.list_documents_in_case(
                case_id=case_id,
                user_id=user_id,
                status=status,
                file_type=file_type,
                priority=priority,
                search_query=search_query,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset
            )
            
            log_route_exit(request, documents)
            return documents
            
    except Exception as exc:
        logger.error(
            "Failed to list documents in case",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        # Return empty response on error
        return DocumentListResponse(
            documents=[],
            total_count=0,
            offset=offset,
            limit=limit,
            has_more=False
        )


@router.put(
    "/{document_id}",
    response_model=ApiResponse,
    summary="Update Document",
    description="Update document metadata and configuration"
)
async def update_document(
    request: Request,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    document_id: str = FastAPIPath(..., description="Document identifier"),
    update_request: DocumentUpdateRequest = Body(...),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Update document metadata."""
    log_route_entry(request, document_id=document_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("document_update", document_id=document_id):
            result = await document_service.update_document(
                document_id,
                update_request,
                user_id
            )
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_document_event,
                    websocket_manager,
                    user_id,
                    "document_updated",
                    {
                        "document_id": document_id,
                        "changes": update_request.dict(exclude_unset=True),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "document_updated",
                    request,
                    document_id=document_id,
                    changes=update_request.dict(exclude_unset=True)
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Document updated successfully",
                    data={"document_id": document_id}
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise DocumentProcessingError(
                    message=result.error or "Failed to update document",
                    error_code=ErrorCode.DOCUMENT_UPDATE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to update document",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (DocumentProcessingError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to update document"}
        )


@router.delete(
    "/{document_id}",
    response_model=ApiResponse,
    summary="Delete Document",
    description="Delete a document and all associated data"
)
async def delete_document(
    request: Request,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    document_id: str = FastAPIPath(..., description="Document identifier"),
    confirm: bool = Query(
        False,
        description="Confirmation flag required for deletion"
    ),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Delete a document with confirmation."""
    log_route_entry(request, document_id=document_id, confirm=confirm)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    # Require confirmation for deletion
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Deletion requires confirmation flag"}
        )
    
    try:
        with performance_context("document_delete", document_id=document_id):
            result = await document_service.delete_document(document_id, user_id)
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_document_event,
                    websocket_manager,
                    user_id,
                    "document_deleted",
                    {
                        "document_id": document_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event("document_deleted", request, document_id=document_id)
                
                response_data = ApiResponse(
                    success=True,
                    message="Document deleted successfully",
                    data={"document_id": document_id}
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise DocumentProcessingError(
                    message=result.error or "Failed to delete document",
                    error_code=ErrorCode.DOCUMENT_DELETE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to delete document",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (DocumentProcessingError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to delete document"}
        )


# Document Processing and Status

@router.post(
    "/{document_id}/reprocess",
    response_model=ApiResponse,
    summary="Reprocess Document",
    description="Reprocess a document through the pipeline"
)
async def reprocess_document(
    request: Request,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    document_id: str = FastAPIPath(..., description="Document identifier"),
    force: bool = Query(
        False,
        description="Force reprocessing even if already completed"
    ),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Reprocess a document."""
    log_route_entry(request, document_id=document_id, force=force)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("document_reprocess", document_id=document_id):
            result = await document_service.reprocess_document(
                document_id,
                user_id,
                force=force
            )
            
            if result.success:
                # Start background processing
                background_tasks.add_task(
                    _process_document_background,
                    document_service,
                    document_id,
                    user_id,
                    websocket_manager
                )
                
                log_business_event(
                    "document_reprocessing_started",
                    request,
                    document_id=document_id,
                    force=force
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Document reprocessing started",
                    data={
                        "document_id": document_id,
                        "processing_started": True
                    }
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise DocumentProcessingError(
                    message=result.error or "Failed to start reprocessing",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to reprocess document",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (DocumentProcessingError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to reprocess document"}
        )


@router.get(
    "/{document_id}/status",
    response_model=DocumentProgressUpdate,
    summary="Get Processing Status",
    description="Get current processing status and progress for a document"
)
async def get_processing_status(
    request: Request,
    document_id: str = FastAPIPath(..., description="Document identifier"),
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentProgressUpdate:
    """Get document processing status."""
    log_route_entry(request, document_id=document_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("document_status", document_id=document_id):
            status_info = await document_service.get_processing_status(
                document_id,
                user_id
            )
            
            if not status_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Document not found"}
                )
            
            log_route_exit(request, status_info)
            return status_info
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get processing status",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get processing status"}
        )


# Document Analytics and Metrics

@router.get(
    "/analytics/summary",
    response_model=DocumentAnalyticsResponse,
    summary="Get Document Analytics",
    description="Get aggregated document analytics and metrics"
)
async def get_document_analytics(
    request: Request,
    case_id: Optional[str] = Query(None, description="Filter by case ID"),
    timeframe: str = Query(
        "30d",
        description="Analytics timeframe (7d, 30d, 90d, 1y)",
        regex=r"^(7d|30d|90d|1y)$"
    ),
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentAnalyticsResponse:
    """Get document analytics summary."""
    log_route_entry(request, case_id=case_id, timeframe=timeframe)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "document_analytics",
            user_id=user_id,
            case_id=case_id,
            timeframe=timeframe
        ):
            analytics = await document_service.get_document_analytics(
                user_id,
                case_id=case_id,
                timeframe=timeframe
            )
            
            log_route_exit(request, analytics)
            return analytics
            
    except Exception as exc:
        logger.error(
            "Failed to get document analytics",
            user_id=user_id,
            case_id=case_id,
            timeframe=timeframe,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get document analytics"}
        )


# File Download

@router.get(
    "/{document_id}/download",
    summary="Download Document File",
    description="Download the original document file"
)
async def download_document(
    request: Request,
    document_id: str = FastAPIPath(..., description="Document identifier"),
    document_service: DocumentService = Depends(get_document_service)
) -> StreamingResponse:
    """Download original document file."""
    log_route_entry(request, document_id=document_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("document_download", document_id=document_id):
            # Get document metadata
            document = await document_service.get_document(document_id, user_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Document not found"}
                )
            
            # Get file content
            file_content = await document_service.get_document_file_content(
                document_id,
                user_id
            )
            
            if not file_content:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Document file not found"}
                )
            
            # Determine content type
            content_type = "application/pdf" if document.file_type == DocumentType.PDF else "text/plain"
            
            # Create streaming response
            import io
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=\"{document.original_filename}\""
                }
            )
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to download document",
            document_id=document_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to download document"}
        )


# Helper Functions

async def _validate_file_upload(
    file: UploadFile,
    case_id: str,
    document_name: str
) -> None:
    """Validate file upload requirements."""
    # Check file is provided
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No file provided"}
        )
    
    # Check file type
    allowed_extensions = {".pdf", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Unsupported file type",
                "allowed_types": list(allowed_extensions),
                "provided_type": file_ext
            }
        )
    
    # Check file size
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "File too large",
                "max_size": "50MB",
                "file_size": file.size
            }
        )
    
    # Validate document name
    if not document_name or not document_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Document name is required"}
        )


def _estimate_processing_time(file_size: int) -> int:
    """Estimate processing time based on file size."""
    # Simple estimation: ~1 second per 100KB
    base_time = 10  # Base processing time in seconds
    size_factor = file_size / (100 * 1024)  # Convert to 100KB chunks
    return int(base_time + size_factor)


async def _process_document_background(
    document_service: DocumentService,
    document_id: str,
    user_id: str,
    websocket_manager: WebSocketManager
) -> None:
    """Process document in background with progress updates."""
    try:
        await document_service.process_document(document_id, user_id)
        
        # Send completion notification
        await _notify_document_event(
            websocket_manager,
            user_id,
            "document_processing_completed",
            {
                "document_id": document_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        log_document_processing(
            document_id=document_id,
            document_name="Unknown",  # Would need to fetch from service
            stage="completed",
            status="success"
        )
        
    except Exception as exc:
        logger.error(
            "Background document processing failed",
            document_id=document_id,
            user_id=user_id,
            error=str(exc)
        )
        
        # Send error notification
        await _notify_document_event(
            websocket_manager,
            user_id,
            "document_processing_failed",
            {
                "document_id": document_id,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


async def _notify_document_event(
    websocket_manager: WebSocketManager,
    user_id: str,
    event_type: str,
    data: Dict[str, Any]
) -> None:
    """Send document event notification via WebSocket."""
    try:
        await websocket_manager.broadcast_to_user(
            user_id,
            {
                "event": event_type,
                "data": data
            }
        )
    except Exception as exc:
        logger.warning(f"Failed to send document notification: {exc}")


# Export router for main application
__all__ = ["router"]