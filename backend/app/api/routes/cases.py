"""
Case Management API Routes for LegalAI Document Processing System

This module provides REST API endpoints for legal case management in the Patexia 
Legal AI Chatbot. It enables case creation, organization, visual marker management,
and multi-user case operations with real-time updates.

Key Features:
- Case CRUD operations with business rule enforcement
- Visual marker assignment and uniqueness validation
- Case listing with filtering, sorting, and pagination
- Document capacity management and override handling
- Real-time WebSocket notifications for case events
- Case analytics and metrics aggregation
- Multi-user support with access control
- Case status lifecycle management

Case Operations:
- Create: New case creation with visual marker assignment
- Read: Individual case details and list operations
- Update: Case metadata, status, and configuration changes
- Delete: Case removal with confirmation and cleanup
- Archive: Case archival with data preservation
- Search: Case discovery with text and metadata search

Business Rules:
- 25 documents per case limit (configurable with override)
- Visual marker uniqueness within user context
- Valid status transitions based on case state
- Case name uniqueness validation
- User access control and ownership validation

Architecture Integration:
- Uses CaseService for business logic and validation
- Integrates with WebSocketManager for real-time notifications
- Connects to MongoDB via CaseRepository for persistence
- Coordinates with VectorRepository for collection management
- Implements comprehensive error handling and logging
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Path,
    Body,
    BackgroundTasks,
    Request,
    Response,
    status
)
from fastapi.responses import JSONResponse

from ...core.config import get_settings
from ...core.websocket_manager import WebSocketManager, get_websocket_manager
from ...services.case_service import CaseService, CaseOperation, CaseOperationResult
from ...models.api.case_schemas import (
    CaseCreateRequest,
    CaseUpdateRequest,
    CaseListRequest,
    CaseStatusUpdateRequest,
    CaseResponse,
    CaseListResponse,
    CaseMetricsResponse,
    CaseAnalyticsResponse,
    VisualMarkerSchema,
    ApiResponse,
    ErrorResponse
)
from ...models.domain.case import CaseStatus, CasePriority, VisualMarker
from ...utils.logging import (
    get_logger,
    performance_context,
    log_business_event,
    log_route_entry,
    log_route_exit
)
from ...exceptions import (
    CaseManagementError,
    ValidationError,
    ResourceError,
    ErrorCode,
    get_exception_response_data
)
from ..deps import get_case_service


logger = get_logger(__name__)
router = APIRouter()


# Case CRUD Operations

@router.post(
    "/",
    response_model=ApiResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Case",
    description="Create a new legal case with validation and visual marker assignment"
)
async def create_case(
    request: Request,
    case_request: CaseCreateRequest,
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Create a new legal case."""
    log_route_entry(
        request,
        case_name=case_request.case_name,
        priority=case_request.priority
    )
    
    # Extract user ID from request context (set by auth middleware)
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "case_create",
            case_name=case_request.case_name,
            user_id=user_id
        ):
            # Create case via service
            result = await case_service.create_case(case_request, user_id)
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_created",
                    {
                        "case_id": result.case_id,
                        "case_name": case_request.case_name,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "case_created",
                    request,
                    case_id=result.case_id,
                    case_name=case_request.case_name
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Case created successfully",
                    data={
                        "case_id": result.case_id,
                        "case_name": case_request.case_name,
                        "status": CaseStatus.DRAFT.value
                    }
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to create case",
                    error_code=ErrorCode.CASE_CREATION_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to create case",
            case_name=case_request.case_name,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError, ResourceError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to create case"}
        )


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Get Case Details",
    description="Retrieve detailed information about a specific case"
)
async def get_case(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    include_analytics: bool = Query(
        False,
        description="Include detailed analytics and metrics"
    ),
    case_service: CaseService = Depends(get_case_service)
) -> CaseResponse:
    """Get detailed case information."""
    log_route_entry(request, case_id=case_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_get", case_id=case_id, user_id=user_id):
            case = await case_service.get_case(case_id, user_id)
            
            if not case:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Case not found"}
                )
            
            # Update case metrics if analytics requested
            if include_analytics:
                await case_service.update_case_metrics(case_id, user_id)
                # Refresh case data
                case = await case_service.get_case(case_id, user_id)
            
            log_route_exit(request, case)
            return case
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get case",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get case"}
        )


@router.get(
    "/",
    response_model=CaseListResponse,
    summary="List Cases",
    description="List cases with filtering, sorting, and pagination options"
)
async def list_cases(
    request: Request,
    status: Optional[CaseStatus] = Query(None, description="Filter by case status"),
    priority: Optional[CasePriority] = Query(None, description="Filter by case priority"),
    tags: Optional[List[str]] = Query(None, description="Filter by case tags"),
    search_query: Optional[str] = Query(None, description="Search in case names and summaries"),
    created_after: Optional[datetime] = Query(None, description="Filter cases created after date"),
    created_before: Optional[datetime] = Query(None, description="Filter cases created before date"),
    sort_by: str = Query("updated_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(50, description="Number of cases to return", ge=1, le=200),
    offset: int = Query(0, description="Number of cases to skip", ge=0),
    case_service: CaseService = Depends(get_case_service)
) -> CaseListResponse:
    """List cases with filtering and pagination."""
    log_route_entry(
        request,
        status=status,
        priority=priority,
        limit=limit,
        offset=offset
    )
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "case_list",
            user_id=user_id,
            limit=limit,
            offset=offset
        ):
            # Create list request
            list_request = CaseListRequest(
                user_id=user_id,
                status=status,
                priority=priority,
                tags=tags or [],
                search_query=search_query,
                created_after=created_after,
                created_before=created_before,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset
            )
            
            # Get cases from service
            response = await case_service.list_cases(list_request, user_id)
            
            log_route_exit(request, response)
            return response
            
    except Exception as exc:
        logger.error(
            "Failed to list cases",
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        # Return empty response on error
        return CaseListResponse(
            cases=[],
            total_count=0,
            offset=offset,
            limit=limit,
            has_more=False
        )


@router.put(
    "/{case_id}",
    response_model=ApiResponse,
    summary="Update Case",
    description="Update case metadata, status, or configuration"
)
async def update_case(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    update_request: CaseUpdateRequest = Body(...),
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Update case information."""
    log_route_entry(request, case_id=case_id)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_update", case_id=case_id, user_id=user_id):
            # Update case via service
            result = await case_service.update_case(case_id, update_request, user_id)
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_updated",
                    {
                        "case_id": case_id,
                        "changes": update_request.dict(exclude_unset=True),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "case_updated",
                    request,
                    case_id=case_id,
                    changes=update_request.dict(exclude_unset=True)
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Case updated successfully",
                    data={"case_id": case_id}
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to update case",
                    error_code=ErrorCode.CASE_UPDATE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to update case",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to update case"}
        )


@router.delete(
    "/{case_id}",
    response_model=ApiResponse,
    summary="Delete Case",
    description="Delete a case and all associated data"
)
async def delete_case(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    confirm: bool = Query(
        False,
        description="Confirmation flag required for deletion"
    ),
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Delete a case with confirmation."""
    log_route_entry(request, case_id=case_id, confirm=confirm)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    # Require confirmation for deletion
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Deletion requires confirmation flag"}
        )
    
    try:
        with performance_context("case_delete", case_id=case_id, user_id=user_id):
            # Delete case via service
            result = await case_service.delete_case(case_id, user_id)
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_deleted",
                    {
                        "case_id": case_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event("case_deleted", request, case_id=case_id)
                
                response_data = ApiResponse(
                    success=True,
                    message="Case deleted successfully",
                    data={"case_id": case_id}
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to delete case",
                    error_code=ErrorCode.CASE_DELETE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to delete case",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to delete case"}
        )


# Case Status Management

@router.put(
    "/{case_id}/status",
    response_model=ApiResponse,
    summary="Update Case Status",
    description="Update case status with validation and side effects"
)
async def update_case_status(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    status_request: CaseStatusUpdateRequest = Body(...),
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Update case status."""
    log_route_entry(request, case_id=case_id, new_status=status_request.status)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context(
            "case_status_update",
            case_id=case_id,
            status=status_request.status
        ):
            # Update status via service
            result = await case_service.update_case_status(
                case_id,
                status_request,
                user_id
            )
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_status_updated",
                    {
                        "case_id": case_id,
                        "status": status_request.status.value,
                        "reason": status_request.reason,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "case_status_updated",
                    request,
                    case_id=case_id,
                    new_status=status_request.status,
                    reason=status_request.reason
                )
                
                response_data = ApiResponse(
                    success=True,
                    message=f"Case status updated to {status_request.status.value}",
                    data={
                        "case_id": case_id,
                        "status": status_request.status.value
                    }
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to update case status",
                    error_code=ErrorCode.CASE_STATUS_UPDATE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to update case status",
            case_id=case_id,
            status=status_request.status,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to update case status"}
        )


@router.post(
    "/{case_id}/archive",
    response_model=ApiResponse,
    summary="Archive Case",
    description="Archive a case while preserving data for future access"
)
async def archive_case(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    reason: Optional[str] = Body(
        None,
        description="Reason for archiving the case",
        max_length=500
    ),
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Archive a case."""
    log_route_entry(request, case_id=case_id, reason=reason)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_archive", case_id=case_id, user_id=user_id):
            # Archive case via service
            result = await case_service.archive_case(case_id, user_id, reason)
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_archived",
                    {
                        "case_id": case_id,
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "case_archived",
                    request,
                    case_id=case_id,
                    reason=reason
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Case archived successfully",
                    data={
                        "case_id": case_id,
                        "status": CaseStatus.ARCHIVED.value
                    }
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to archive case",
                    error_code=ErrorCode.CASE_ARCHIVE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to archive case",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to archive case"}
        )


# Case Analytics and Metrics

@router.get(
    "/{case_id}/metrics",
    response_model=CaseMetricsResponse,
    summary="Get Case Metrics",
    description="Retrieve detailed metrics and analytics for a case"
)
async def get_case_metrics(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    refresh: bool = Query(
        False,
        description="Force refresh of metrics from source data"
    ),
    case_service: CaseService = Depends(get_case_service)
) -> CaseMetricsResponse:
    """Get detailed case metrics."""
    log_route_entry(request, case_id=case_id, refresh=refresh)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_metrics", case_id=case_id, refresh=refresh):
            metrics = await case_service.get_case_metrics(
                case_id,
                user_id,
                refresh=refresh
            )
            
            if not metrics:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Case not found or metrics unavailable"}
                )
            
            log_route_exit(request, metrics)
            return metrics
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get case metrics",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get case metrics"}
        )


@router.get(
    "/analytics/summary",
    response_model=CaseAnalyticsResponse,
    summary="Get Case Analytics Summary",
    description="Get aggregated analytics across all user cases"
)
async def get_case_analytics(
    request: Request,
    timeframe: str = Query(
        "30d",
        description="Analytics timeframe (7d, 30d, 90d, 1y)",
        regex=r"^(7d|30d|90d|1y)$"
    ),
    case_service: CaseService = Depends(get_case_service)
) -> CaseAnalyticsResponse:
    """Get case analytics summary."""
    log_route_entry(request, timeframe=timeframe)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_analytics", user_id=user_id, timeframe=timeframe):
            analytics = await case_service.get_case_analytics(user_id, timeframe)
            
            log_route_exit(request, analytics)
            return analytics
            
    except Exception as exc:
        logger.error(
            "Failed to get case analytics",
            user_id=user_id,
            timeframe=timeframe,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get case analytics"}
        )


# Visual Marker Management

@router.get(
    "/visual-markers/available",
    response_model=ApiResponse,
    summary="Get Available Visual Markers",
    description="Get list of available visual marker combinations for the user"
)
async def get_available_visual_markers(
    request: Request,
    case_service: CaseService = Depends(get_case_service)
) -> ApiResponse:
    """Get available visual markers."""
    log_route_entry(request)
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("visual_markers", user_id=user_id):
            available_markers = await case_service.get_available_visual_markers(user_id)
            
            response_data = ApiResponse(
                success=True,
                message="Available visual markers retrieved",
                data={
                    "available_markers": [
                        {
                            "color": marker.color,
                            "icon": marker.icon
                        }
                        for marker in available_markers
                    ],
                    "total_combinations": len(VisualMarker.COLORS) * len(VisualMarker.ICONS),
                    "available_count": len(available_markers)
                }
            )
            
            log_route_exit(request, response_data)
            return response_data
            
    except Exception as exc:
        logger.error(
            "Failed to get available visual markers",
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get available visual markers"}
        )


@router.put(
    "/{case_id}/visual-marker",
    response_model=ApiResponse,
    summary="Update Case Visual Marker",
    description="Update the visual marker for a case"
)
async def update_case_visual_marker(
    request: Request,
    case_id: str = Path(..., description="Case identifier"),
    visual_marker: VisualMarkerSchema = Body(...),
    background_tasks: BackgroundTasks,
    case_service: CaseService = Depends(get_case_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> ApiResponse:
    """Update case visual marker."""
    log_route_entry(request, case_id=case_id, visual_marker=visual_marker.dict())
    
    user_id = getattr(request.state, "user_id", "default_user")
    
    try:
        with performance_context("case_visual_marker_update", case_id=case_id):
            # Update visual marker via service
            result = await case_service.update_case_visual_marker(
                case_id,
                visual_marker,
                user_id
            )
            
            if result.success:
                # Send WebSocket notification
                background_tasks.add_task(
                    _notify_case_event,
                    websocket_manager,
                    user_id,
                    "case_visual_marker_updated",
                    {
                        "case_id": case_id,
                        "visual_marker": visual_marker.dict(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                log_business_event(
                    "case_visual_marker_updated",
                    request,
                    case_id=case_id,
                    visual_marker=visual_marker.dict()
                )
                
                response_data = ApiResponse(
                    success=True,
                    message="Visual marker updated successfully",
                    data={
                        "case_id": case_id,
                        "visual_marker": visual_marker.dict()
                    }
                )
                
                log_route_exit(request, response_data)
                return response_data
            else:
                raise CaseManagementError(
                    message=result.error or "Failed to update visual marker",
                    error_code=ErrorCode.CASE_UPDATE_FAILED
                )
                
    except Exception as exc:
        logger.error(
            "Failed to update case visual marker",
            case_id=case_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=getattr(request.state, "correlation_id", "unknown")
        )
        
        if isinstance(exc, (CaseManagementError, ValidationError)):
            raise HTTPException(
                status_code=exc.http_status_code,
                detail=get_exception_response_data(exc)
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to update visual marker"}
        )


# Helper Functions

async def _notify_case_event(
    websocket_manager: WebSocketManager,
    user_id: str,
    event_type: str,
    data: Dict[str, Any]
) -> None:
    """Send case event notification via WebSocket."""
    try:
        await websocket_manager.broadcast_to_user(
            user_id,
            {
                "event": event_type,
                "data": data
            }
        )
    except Exception as exc:
        logger.warning(f"Failed to send case notification: {exc}")


# Export router for main application
__all__ = ["router"]