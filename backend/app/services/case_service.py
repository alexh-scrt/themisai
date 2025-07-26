"""
Case Management Service - Business Logic Layer

This module provides the business logic layer for legal case management in the
Patexia Legal AI Chatbot. It orchestrates case operations, enforces business rules,
manages transactions, and coordinates between data access and API layers.

Key Features:
- Case lifecycle management with status transitions
- Document capacity enforcement and override handling
- Visual marker assignment and management
- Business rule validation and enforcement
- Transaction coordination across multiple data stores
- Real-time progress tracking via WebSocket notifications
- Case analytics and metrics aggregation
- User access control and authorization
- Audit trail and change tracking

Business Rules:
- 25 documents per case limit (configurable with manual override)
- Valid status transitions based on case state
- Visual marker uniqueness within user context
- Case name uniqueness validation
- Document capacity checks before adding documents
- Auto-summary generation triggers
- Search readiness validation

Data Orchestration:
- MongoDB for case metadata and business data
- Weaviate collection management for vector storage
- WebSocket notifications for real-time updates
- Cache invalidation for performance optimization
- Cross-service coordination for document operations

Architecture Integration:
- Coordinates with DocumentService for document operations
- Integrates with VectorRepository for collection management
- Uses WebSocketManager for real-time notifications
- Implements repository pattern for data access
- Provides service layer abstraction for API controllers
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..core.config import get_settings
from ..core.websocket_manager import WebSocketManager
from ..core.database import get_database_manager
from ..models.domain.case import (
    LegalCase, CaseStatus, CasePriority, VisualMarker, CaseMetrics
)
from ..models.api.case_schemas import (
    CaseCreateRequest, CaseUpdateRequest, CaseListRequest,
    CaseResponse, CaseListResponse, CaseStatusUpdateRequest
)
from ..repositories.mongodb.case_repository import CaseRepository
from ..repositories.weaviate.vector_repository import VectorRepository
from ..exceptions import (
    CaseManagementError, ValidationError, ResourceError,
    ErrorCode, raise_case_error, raise_validation_error, raise_resource_error
)
from ..utils.logging import get_logger, performance_context
from ..utils.validators import validate_case_name, validate_user_access

logger = get_logger(__name__)


class CaseOperation(str, Enum):
    """Types of case operations for audit tracking."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ARCHIVE = "archive"
    ACTIVATE = "activate"
    STATUS_CHANGE = "status_change"
    DOCUMENT_ADD = "document_add"
    DOCUMENT_REMOVE = "document_remove"


@dataclass
class CaseOperationResult:
    """Result of a case operation with metadata."""
    success: bool
    case_id: Optional[str] = None
    operation: Optional[CaseOperation] = None
    message: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CaseValidationContext:
    """Context for case validation operations."""
    user_id: str
    operation: CaseOperation
    existing_case: Optional[LegalCase] = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


class CaseService:
    """
    Business logic service for legal case management.
    
    Provides high-level case operations with business rule enforcement,
    transaction coordination, and real-time notifications.
    """
    
    def __init__(
        self,
        case_repository: Optional[CaseRepository] = None,
        vector_repository: Optional[VectorRepository] = None,
        websocket_manager: Optional[WebSocketManager] = None
    ):
        """
        Initialize case service with dependencies.
        
        Args:
            case_repository: MongoDB case repository
            vector_repository: Weaviate vector repository
            websocket_manager: WebSocket notification manager
        """
        self.case_repository = case_repository or CaseRepository()
        self.vector_repository = vector_repository or VectorRepository()
        self.websocket_manager = websocket_manager
        self.settings = get_settings()
        
        # Visual marker management
        self._used_markers_cache: Dict[str, Set[Tuple[str, str]]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        logger.info("CaseService initialized")
    
    async def create_case(
        self,
        request: CaseCreateRequest,
        user_id: str
    ) -> CaseOperationResult:
        """
        Create a new legal case with validation and setup.
        
        Args:
            request: Case creation request data
            user_id: User creating the case
            
        Returns:
            CaseOperationResult with case creation details
        """
        try:
            with performance_context("case_service_create", user_id=user_id):
                # Validate request
                validation_context = CaseValidationContext(
                    user_id=user_id,
                    operation=CaseOperation.CREATE,
                    additional_data={"request": request}
                )
                await self._validate_case_operation(validation_context)
                
                # Generate visual marker if not provided
                visual_marker = request.visual_marker
                if not visual_marker:
                    visual_marker = await self._generate_visual_marker(user_id, request.case_type)
                
                # Create domain object
                case = LegalCase.create_new(
                    user_id=user_id,
                    case_name=request.case_name,
                    initial_summary=request.initial_summary,
                    visual_marker=visual_marker,
                    priority=request.priority,
                    case_type=request.case_type
                )
                
                # Add tags if provided
                if request.tags:
                    for tag in request.tags:
                        case.add_tag(tag)
                
                # Add metadata if provided
                if request.metadata:
                    for key, value in request.metadata.items():
                        case.update_metadata(key, value)
                
                # Perform database transaction
                result = await self._execute_case_creation_transaction(case)
                
                if result.success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "case_created", case.to_dict()
                    )
                    
                    logger.info(
                        "Case created successfully",
                        case_id=case.case_id,
                        user_id=user_id,
                        case_name=case.case_name
                    )
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to create case: {e}", exc_info=True)
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.CREATE,
                error=str(e)
            )
    
    async def get_case_by_id(
        self,
        case_id: str,
        user_id: str,
        include_metrics: bool = True
    ) -> Optional[LegalCase]:
        """
        Retrieve a case by ID with user access validation.
        
        Args:
            case_id: Case identifier
            user_id: Requesting user ID
            include_metrics: Whether to include detailed metrics
            
        Returns:
            LegalCase object if found and accessible, None otherwise
        """
        try:
            with performance_context("case_service_get", case_id=case_id):
                # Get case from repository
                case = await self.case_repository.get_case_by_id(case_id, user_id)
                
                if not case:
                    return None
                
                # Validate user access
                if not await self._validate_user_access(user_id, case):
                    logger.warning(
                        "User access denied for case",
                        user_id=user_id,
                        case_id=case_id
                    )
                    return None
                
                # Update metrics if requested
                if include_metrics:
                    await self._update_case_metrics(case)
                
                return case
                
        except Exception as e:
            logger.error(f"Failed to get case {case_id}: {e}")
            return None
    
    async def update_case(
        self,
        case_id: str,
        request: CaseUpdateRequest,
        user_id: str
    ) -> CaseOperationResult:
        """
        Update an existing case with validation.
        
        Args:
            case_id: Case identifier
            request: Update request data
            user_id: User updating the case
            
        Returns:
            CaseOperationResult with update details
        """
        try:
            with performance_context("case_service_update", case_id=case_id):
                # Get existing case
                existing_case = await self.get_case_by_id(case_id, user_id)
                if not existing_case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.UPDATE,
                        error="Case not found or access denied"
                    )
                
                # Validate update operation
                validation_context = CaseValidationContext(
                    user_id=user_id,
                    operation=CaseOperation.UPDATE,
                    existing_case=existing_case,
                    additional_data={"request": request}
                )
                await self._validate_case_operation(validation_context)
                
                # Apply updates
                updated_case = await self._apply_case_updates(existing_case, request, user_id)
                
                # Save to repository
                success = await self.case_repository.update_case(updated_case)
                
                if success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "case_updated", updated_case.to_dict()
                    )
                    
                    return CaseOperationResult(
                        success=True,
                        case_id=case_id,
                        operation=CaseOperation.UPDATE,
                        message="Case updated successfully"
                    )
                else:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.UPDATE,
                        error="Failed to save case updates"
                    )
                
        except Exception as e:
            logger.error(f"Failed to update case {case_id}: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.UPDATE,
                error=str(e)
            )
    
    async def list_cases(
        self,
        request: CaseListRequest,
        user_id: str
    ) -> CaseListResponse:
        """
        List cases with filtering, sorting, and pagination.
        
        Args:
            request: List request with filters and pagination
            user_id: Requesting user ID
            
        Returns:
            CaseListResponse with cases and pagination info
        """
        try:
            with performance_context("case_service_list", user_id=user_id):
                # Apply user context to request
                effective_user_id = request.user_id if self._is_admin_user(user_id) else user_id
                
                # Get cases from repository
                cases, total_count = await self.case_repository.list_cases(
                    user_id=effective_user_id,
                    status=request.status,
                    priority=request.priority,
                    tags=request.tags,
                    search_query=request.search_query,
                    created_after=request.created_after,
                    created_before=request.created_before,
                    limit=request.limit,
                    offset=request.offset,
                    sort_by=request.sort_by,
                    sort_order=request.sort_order
                )
                
                # Convert to response format
                case_responses = []
                for case in cases:
                    # Update metrics for each case
                    await self._update_case_metrics(case)
                    case_responses.append(self._case_to_response(case))
                
                return CaseListResponse(
                    cases=case_responses,
                    total_count=total_count,
                    offset=request.offset,
                    limit=request.limit,
                    has_more=(request.offset + len(cases)) < total_count
                )
                
        except Exception as e:
            logger.error(f"Failed to list cases: {e}")
            return CaseListResponse(
                cases=[],
                total_count=0,
                offset=request.offset,
                limit=request.limit,
                has_more=False
            )
    
    async def update_case_status(
        self,
        case_id: str,
        request: CaseStatusUpdateRequest,
        user_id: str
    ) -> CaseOperationResult:
        """
        Update case status with validation and side effects.
        
        Args:
            case_id: Case identifier
            request: Status update request
            user_id: User updating the status
            
        Returns:
            CaseOperationResult with status update details
        """
        try:
            with performance_context("case_service_status_update", case_id=case_id):
                # Get existing case
                case = await self.get_case_by_id(case_id, user_id)
                if not case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.STATUS_CHANGE,
                        error="Case not found or access denied"
                    )
                
                old_status = case.status
                
                # Validate status transition
                try:
                    case.transition_to_status(request.status)
                except CaseManagementError as e:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.STATUS_CHANGE,
                        error=str(e)
                    )
                
                # Handle special status transitions
                await self._handle_status_transition_side_effects(
                    case, old_status, request.status, user_id
                )
                
                # Save to repository
                success = await self.case_repository.update_case(case)
                
                if success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "case_status_updated", {
                            "case_id": case_id,
                            "old_status": old_status.value,
                            "new_status": request.status.value,
                            "reason": request.reason
                        }
                    )
                    
                    return CaseOperationResult(
                        success=True,
                        case_id=case_id,
                        operation=CaseOperation.STATUS_CHANGE,
                        message=f"Status updated from {old_status.value} to {request.status.value}",
                        metadata={
                            "old_status": old_status.value,
                            "new_status": request.status.value,
                            "reason": request.reason
                        }
                    )
                else:
                    # Revert status change
                    case.transition_to_status(old_status)
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.STATUS_CHANGE,
                        error="Failed to save status update"
                    )
                
        except Exception as e:
            logger.error(f"Failed to update case status: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.STATUS_CHANGE,
                error=str(e)
            )
    
    async def archive_case(
        self,
        case_id: str,
        user_id: str,
        reason: Optional[str] = None
    ) -> CaseOperationResult:
        """
        Archive a case with proper cleanup.
        
        Args:
            case_id: Case identifier
            user_id: User archiving the case
            reason: Optional reason for archiving
            
        Returns:
            CaseOperationResult with archive details
        """
        try:
            with performance_context("case_service_archive", case_id=case_id):
                # Get existing case
                case = await self.get_case_by_id(case_id, user_id)
                if not case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.ARCHIVE,
                        error="Case not found or access denied"
                    )
                
                # Validate archiving is allowed
                if case.status == CaseStatus.PROCESSING:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.ARCHIVE,
                        error="Cannot archive case while documents are being processed"
                    )
                
                # Archive the case
                case.archive()
                
                # Add archive metadata
                case.update_metadata("archived_by", user_id)
                case.update_metadata("archived_at", datetime.now(timezone.utc).isoformat())
                if reason:
                    case.update_metadata("archive_reason", reason)
                
                # Save to repository
                success = await self.case_repository.update_case(case)
                
                if success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "case_archived", {
                            "case_id": case_id,
                            "reason": reason
                        }
                    )
                    
                    return CaseOperationResult(
                        success=True,
                        case_id=case_id,
                        operation=CaseOperation.ARCHIVE,
                        message="Case archived successfully",
                        metadata={"reason": reason}
                    )
                else:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.ARCHIVE,
                        error="Failed to save archive status"
                    )
                
        except Exception as e:
            logger.error(f"Failed to archive case {case_id}: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.ARCHIVE,
                error=str(e)
            )
    
    async def delete_case(
        self,
        case_id: str,
        user_id: str,
        force: bool = False
    ) -> CaseOperationResult:
        """
        Delete a case with proper cleanup of associated resources.
        
        Args:
            case_id: Case identifier
            user_id: User deleting the case
            force: Whether to force deletion even with documents
            
        Returns:
            CaseOperationResult with deletion details
        """
        try:
            with performance_context("case_service_delete", case_id=case_id):
                # Get existing case
                case = await self.get_case_by_id(case_id, user_id)
                if not case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DELETE,
                        error="Case not found or access denied"
                    )
                
                # Validate deletion is allowed
                if case.document_count > 0 and not force:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DELETE,
                        error=f"Case has {case.document_count} documents. Use force=true to delete anyway."
                    )
                
                # Perform deletion transaction
                result = await self._execute_case_deletion_transaction(case_id, user_id)
                
                if result.success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "case_deleted", {"case_id": case_id}
                    )
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to delete case {case_id}: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.DELETE,
                error=str(e)
            )
    
    async def add_document_to_case(
        self,
        case_id: str,
        document_id: str,
        user_id: str
    ) -> CaseOperationResult:
        """
        Add a document to a case with capacity validation.
        
        Args:
            case_id: Case identifier
            document_id: Document identifier to add
            user_id: User adding the document
            
        Returns:
            CaseOperationResult with operation details
        """
        try:
            with performance_context("case_service_add_document", case_id=case_id):
                # Get existing case
                case = await self.get_case_by_id(case_id, user_id)
                if not case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DOCUMENT_ADD,
                        error="Case not found or access denied"
                    )
                
                # Add document with capacity checking
                try:
                    case.add_document(document_id)
                except (CaseManagementError, ResourceError) as e:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DOCUMENT_ADD,
                        error=str(e)
                    )
                
                # Save to repository
                success = await self.case_repository.update_case(case)
                
                if success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "document_added_to_case", {
                            "case_id": case_id,
                            "document_id": document_id,
                            "total_documents": case.document_count
                        }
                    )
                    
                    return CaseOperationResult(
                        success=True,
                        case_id=case_id,
                        operation=CaseOperation.DOCUMENT_ADD,
                        message=f"Document added to case (total: {case.document_count})",
                        metadata={
                            "document_id": document_id,
                            "total_documents": case.document_count
                        }
                    )
                else:
                    # Revert document addition
                    case.remove_document(document_id)
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DOCUMENT_ADD,
                        error="Failed to save document addition"
                    )
                
        except Exception as e:
            logger.error(f"Failed to add document to case: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.DOCUMENT_ADD,
                error=str(e)
            )
    
    async def remove_document_from_case(
        self,
        case_id: str,
        document_id: str,
        user_id: str
    ) -> CaseOperationResult:
        """
        Remove a document from a case.
        
        Args:
            case_id: Case identifier
            document_id: Document identifier to remove
            user_id: User removing the document
            
        Returns:
            CaseOperationResult with operation details
        """
        try:
            with performance_context("case_service_remove_document", case_id=case_id):
                # Get existing case
                case = await self.get_case_by_id(case_id, user_id)
                if not case:
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DOCUMENT_REMOVE,
                        error="Case not found or access denied"
                    )
                
                # Remove document
                case.remove_document(document_id)
                
                # Save to repository
                success = await self.case_repository.update_case(case)
                
                if success:
                    # Send WebSocket notification
                    await self._send_case_notification(
                        user_id, "document_removed_from_case", {
                            "case_id": case_id,
                            "document_id": document_id,
                            "total_documents": case.document_count
                        }
                    )
                    
                    return CaseOperationResult(
                        success=True,
                        case_id=case_id,
                        operation=CaseOperation.DOCUMENT_REMOVE,
                        message=f"Document removed from case (remaining: {case.document_count})",
                        metadata={
                            "document_id": document_id,
                            "total_documents": case.document_count
                        }
                    )
                else:
                    # Revert document removal
                    case.add_document(document_id)
                    return CaseOperationResult(
                        success=False,
                        operation=CaseOperation.DOCUMENT_REMOVE,
                        error="Failed to save document removal"
                    )
                
        except Exception as e:
            logger.error(f"Failed to remove document from case: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.DOCUMENT_REMOVE,
                error=str(e)
            )
    
    async def get_case_analytics(
        self,
        user_id: str,
        case_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get case analytics and metrics.
        
        Args:
            user_id: User requesting analytics
            case_id: Specific case ID or None for all cases
            days_back: Number of days to include in analysis
            
        Returns:
            Dictionary containing analytics data
        """
        try:
            with performance_context("case_service_analytics", user_id=user_id):
                # Get cases for analysis
                if case_id:
                    case = await self.get_case_by_id(case_id, user_id)
                    cases = [case] if case else []
                else:
                    cases, _ = await self.case_repository.list_cases(
                        user_id=user_id,
                        limit=1000  # Large limit for analytics
                    )
                
                # Calculate analytics
                analytics = await self._calculate_case_analytics(cases, days_back)
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get case analytics: {e}")
            return {
                "error": str(e),
                "total_cases": 0,
                "active_cases": 0,
                "total_documents": 0
            }
    
    # Private helper methods
    
    async def _validate_case_operation(self, context: CaseValidationContext) -> None:
        """Validate case operation according to business rules."""
        if context.operation == CaseOperation.CREATE:
            request = context.additional_data.get("request")
            if request:
                # Validate case name uniqueness
                await self._validate_case_name_uniqueness(
                    context.user_id, request.case_name
                )
                
                # Validate visual marker availability
                if request.visual_marker:
                    await self._validate_visual_marker_availability(
                        context.user_id, request.visual_marker
                    )
    
    async def _validate_case_name_uniqueness(
        self,
        user_id: str,
        case_name: str
    ) -> None:
        """Validate that case name is unique for user."""
        # Check for existing cases with same name
        cases, _ = await self.case_repository.list_cases(
            user_id=user_id,
            search_query=case_name,
            limit=1
        )
        
        # Check for exact match
        for case in cases:
            if case.case_name.lower() == case_name.lower():
                raise_validation_error(
                    f"A case named '{case_name}' already exists",
                    field="case_name"
                )
    
    async def _validate_visual_marker_availability(
        self,
        user_id: str,
        visual_marker: VisualMarker
    ) -> None:
        """Validate that visual marker is not already in use."""
        used_markers = await self._get_used_visual_markers(user_id)
        marker_tuple = (visual_marker.color, visual_marker.icon)
        
        if marker_tuple in used_markers:
            raise_validation_error(
                f"Visual marker combination already in use",
                field="visual_marker"
            )
    
    async def _get_used_visual_markers(self, user_id: str) -> Set[Tuple[str, str]]:
        """Get set of visual markers already in use by user."""
        # Check cache first
        cache_key = user_id
        if (cache_key in self._used_markers_cache and 
            cache_key in self._cache_expiry and
            self._cache_expiry[cache_key] > datetime.now(timezone.utc)):
            return self._used_markers_cache[cache_key]
        
        # Get from database
        cases, _ = await self.case_repository.list_cases(
            user_id=user_id,
            limit=1000  # Should be enough for most users
        )
        
        used_markers = set()
        for case in cases:
            if case.status != CaseStatus.ARCHIVED:  # Exclude archived cases
                marker_tuple = (case.visual_marker.color, case.visual_marker.icon)
                used_markers.add(marker_tuple)
        
        # Cache the result
        self._used_markers_cache[cache_key] = used_markers
        self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        return used_markers
    
    async def _generate_visual_marker(
        self,
        user_id: str,
        case_type: Optional[str] = None
    ) -> VisualMarker:
        """Generate an available visual marker for the user."""
        used_markers = await self._get_used_visual_markers(user_id)
        
        # Try to assign based on case type first
        if case_type:
            suggested_marker = VisualMarker.create_for_case_type(case_type)
            marker_tuple = (suggested_marker.color, suggested_marker.icon)
            if marker_tuple not in used_markers:
                return suggested_marker
        
        # Find first available combination
        for color in VisualMarker.COLORS:
            for icon in VisualMarker.ICONS:
                marker_tuple = (color, icon)
                if marker_tuple not in used_markers:
                    return VisualMarker(color=color, icon=icon)
        
        # Fallback to default if all combinations are used
        return VisualMarker.create_default()
    
    async def _execute_case_creation_transaction(
        self,
        case: LegalCase
    ) -> CaseOperationResult:
        """Execute case creation with transaction coordination."""
        try:
            # Create case in MongoDB
            case_id = await self.case_repository.create_case(case)
            
            # Create Weaviate collection for the case
            await self.vector_repository.create_case_collection(case_id)
            
            # Invalidate visual marker cache
            cache_key = case.user_id
            self._used_markers_cache.pop(cache_key, None)
            self._cache_expiry.pop(cache_key, None)
            
            return CaseOperationResult(
                success=True,
                case_id=case_id,
                operation=CaseOperation.CREATE,
                message="Case created successfully"
            )
            
        except Exception as e:
            # Cleanup on failure
            try:
                await self.case_repository.delete_case(case.case_id, case.user_id)
                await self.vector_repository.delete_case_collection(case.case_id)
            except:
                pass  # Best effort cleanup
            
            raise e
    
    async def _execute_case_deletion_transaction(
        self,
        case_id: str,
        user_id: str
    ) -> CaseOperationResult:
        """Execute case deletion with proper cleanup."""
        try:
            # Delete from MongoDB
            deleted = await self.case_repository.delete_case(case_id, user_id)
            
            if not deleted:
                return CaseOperationResult(
                    success=False,
                    operation=CaseOperation.DELETE,
                    error="Case not found or already deleted"
                )
            
            # Delete Weaviate collection
            await self.vector_repository.delete_case_collection(case_id)
            
            # Invalidate visual marker cache
            cache_key = user_id
            self._used_markers_cache.pop(cache_key, None)
            self._cache_expiry.pop(cache_key, None)
            
            return CaseOperationResult(
                success=True,
                case_id=case_id,
                operation=CaseOperation.DELETE,
                message="Case deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Case deletion transaction failed: {e}")
            return CaseOperationResult(
                success=False,
                operation=CaseOperation.DELETE,
                error=str(e)
            )
    
    async def _apply_case_updates(
        self,
        case: LegalCase,
        request: CaseUpdateRequest,
        user_id: str
    ) -> LegalCase:
        """Apply updates to a case object."""
        # Update case name if provided
        if request.case_name is not None:
            case.update_case_name(request.case_name)
        
        # Update initial summary if provided
        if request.initial_summary is not None:
            case.update_initial_summary(request.initial_summary)
        
        # Update auto summary if provided
        if request.auto_summary is not None:
            case.update_auto_summary(request.auto_summary)
        
        # Update priority if provided
        if request.priority is not None:
            case.update_priority(request.priority)
        
        # Update visual marker if provided
        if request.visual_marker is not None:
            # Validate new visual marker
            await self._validate_visual_marker_availability(user_id, request.visual_marker)
            case.update_visual_marker(request.visual_marker)
        
        # Update tags if provided
        if request.tags is not None:
            # Clear existing tags and add new ones
            current_tags = case.tags.copy()
            for tag in current_tags:
                case.remove_tag(tag)
            for tag in request.tags:
                case.add_tag(tag)
        
        # Update metadata if provided
        if request.metadata is not None:
            for key, value in request.metadata.items():
                case.update_metadata(key, value)
        
        return case
    
    async def _handle_status_transition_side_effects(
        self,
        case: LegalCase,
        old_status: CaseStatus,
        new_status: CaseStatus,
        user_id: str
    ) -> None:
        """Handle side effects of status transitions."""
        # Archive transition - update metadata
        if new_status == CaseStatus.ARCHIVED:
            case.update_metadata("archived_by", user_id)
            case.update_metadata("archived_at", datetime.now(timezone.utc).isoformat())
        
        # Activation from archive - clear archive metadata
        if old_status == CaseStatus.ARCHIVED and new_status == CaseStatus.ACTIVE:
            case.update_metadata("reactivated_by", user_id)
            case.update_metadata("reactivated_at", datetime.now(timezone.utc).isoformat())
    
    async def _update_case_metrics(self, case: LegalCase) -> None:
        """Update case metrics from various sources."""
        try:
            # Get document count from case
            # Note: In a full implementation, this would query document repository
            # for accurate counts of processed/failed documents
            
            # Get vector collection stats
            try:
                stats = await self.vector_repository.get_collection_stats(case.case_id)
                case.metrics.total_chunks = stats.get("total_chunks", 0)
            except:
                pass  # Non-critical operation
                
        except Exception as e:
            logger.warning(f"Failed to update metrics for case {case.case_id}: {e}")
    
    async def _calculate_case_analytics(
        self,
        cases: List[LegalCase],
        days_back: int
    ) -> Dict[str, Any]:
        """Calculate analytics from case data."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        total_cases = len(cases)
        active_cases = sum(1 for case in cases if case.is_active())
        
        # Status breakdown
        status_counts = {}
        for status in CaseStatus:
            status_counts[status.value] = sum(
                1 for case in cases if case.status == status
            )
        
        # Priority breakdown
        priority_counts = {}
        for priority in CasePriority:
            priority_counts[priority.value] = sum(
                1 for case in cases if case.priority == priority
            )
        
        # Recent activity
        recent_cases = [
            case for case in cases 
            if case.updated_at >= cutoff_date
        ]
        
        # Document metrics
        total_documents = sum(case.document_count for case in cases)
        total_processed = sum(case.metrics.processed_documents for case in cases)
        total_failed = sum(case.metrics.failed_documents for case in cases)
        
        return {
            "total_cases": total_cases,
            "active_cases": active_cases,
            "status_breakdown": status_counts,
            "priority_breakdown": priority_counts,
            "recent_activity_count": len(recent_cases),
            "total_documents": total_documents,
            "total_processed_documents": total_processed,
            "total_failed_documents": total_failed,
            "processing_success_rate": (
                total_processed / max(total_documents, 1)
            ) if total_documents > 0 else 0.0,
            "period_days": days_back
        }
    
    async def _validate_user_access(self, user_id: str, case: LegalCase) -> bool:
        """Validate user has access to the case."""
        # Basic access control - user owns the case
        return case.user_id == user_id or self._is_admin_user(user_id)
    
    def _is_admin_user(self, user_id: str) -> bool:
        """Check if user has admin privileges."""
        # Basic implementation - in production this would check roles/permissions
        admin_users = self.settings.admin_users if hasattr(self.settings, 'admin_users') else []
        return user_id in admin_users
    
    async def _send_case_notification(
        self,
        user_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Send WebSocket notification for case events."""
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast_to_user(
                    user_id, event_type, data
                )
            except Exception as e:
                logger.warning(f"Failed to send case notification: {e}")
    
    def _case_to_response(self, case: LegalCase) -> CaseResponse:
        """Convert domain case to API response format."""
        return CaseResponse(
            case_id=case.case_id,
            user_id=case.user_id,
            case_name=case.case_name,
            initial_summary=case.initial_summary,
            auto_summary=case.auto_summary,
            status=case.status,
            priority=case.priority,
            visual_marker=case.visual_marker,
            created_at=case.created_at,
            updated_at=case.updated_at,
            tags=list(case.tags),
            metadata=case.metadata,
            metrics=case.metrics,
            document_count=case.document_count,
            document_ids=list(case.document_ids),
            is_ready_for_search=case.is_ready_for_search(),
            is_active=case.is_active()
        )


# Factory function for dependency injection
def create_case_service(
    websocket_manager: Optional[WebSocketManager] = None
) -> CaseService:
    """
    Factory function to create CaseService with dependencies.
    
    Args:
        websocket_manager: Optional WebSocket manager for notifications
        
    Returns:
        Configured CaseService instance
    """
    return CaseService(websocket_manager=websocket_manager)