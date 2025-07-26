"""
Domain model for legal cases in Patexia Legal AI Chatbot.

This module defines the core business logic and domain entities for legal cases:
- Case entity with business rules and validation
- Visual marker management for case identification
- Case capacity and document management
- Status tracking and lifecycle management
- Business rule enforcement and validation
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from backend.app.core.exceptions import (
    CaseManagementError,
    ErrorCode,
    raise_case_error,
    raise_capacity_error
)
from backend.app.utils.logging import get_logger
from backend.config.settings import get_settings

logger = get_logger(__name__)


class CaseStatus(str, Enum):
    """Legal case processing status enumeration."""
    
    DRAFT = "draft"                    # Case created but no documents uploaded
    ACTIVE = "active"                  # Documents being processed or ready for search
    PROCESSING = "processing"          # Documents currently being processed
    COMPLETE = "complete"              # All documents processed and indexed
    ARCHIVED = "archived"              # Case archived but accessible
    ERROR = "error"                    # Case has processing errors requiring attention


class CasePriority(str, Enum):
    """Case priority levels for workflow management."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass(frozen=True)
class VisualMarker:
    """
    Immutable visual marker for case identification in the UI.
    
    Provides predefined color and icon combinations for easy case recognition
    in the sidebar navigation and case switching interface.
    """
    
    color: str
    icon: str
    
    # Predefined color palette for legal cases
    COLORS = [
        "#e74c3c",  # Red - Urgent/Litigation
        "#27ae60",  # Green - Contract/Completed
        "#3498db",  # Blue - IP/Patent
        "#f39c12",  # Orange - Corporate/M&A
        "#9b59b6",  # Purple - Regulatory/Compliance
        "#1abc9c",  # Teal - Discovery/Investigation
        "#e67e22",  # Dark Orange - Appeals/Motion
        "#34495e",  # Dark Gray - Archived/Reference
    ]
    
    # Predefined icon set for legal document types
    ICONS = [
        "📄",  # Document - General legal documents
        "⚖️",  # Legal - Court filings and litigation
        "🏢",  # Corporate - Business and corporate law
        "💼",  # Business - Commercial transactions
        "📋",  # Contract - Agreements and contracts
        "🔍",  # Investigation - Discovery and research
        "⚡",  # Urgent - High priority cases
        "🎯",  # Priority - Focused cases
        "📊",  # Analytics - Data-driven cases
        "🔒",  # Confidential - Sensitive matters
    ]
    
    def __post_init__(self):
        """Validate visual marker after creation."""
        if self.color not in self.COLORS:
            raise ValueError(f"Invalid color: {self.color}. Must be one of {self.COLORS}")
        
        if self.icon not in self.ICONS:
            raise ValueError(f"Invalid icon: {self.icon}. Must be one of {self.ICONS}")
    
    @classmethod
    def create_default(cls) -> "VisualMarker":
        """Create a default visual marker."""
        return cls(color=cls.COLORS[0], icon=cls.ICONS[0])
    
    @classmethod
    def create_for_case_type(cls, case_type: str) -> "VisualMarker":
        """
        Create a visual marker based on case type.
        
        Args:
            case_type: Type of legal case (patent, contract, litigation, etc.)
            
        Returns:
            Appropriate visual marker for the case type
        """
        case_type_lower = case_type.lower()
        
        if any(keyword in case_type_lower for keyword in ["patent", "ip", "intellectual"]):
            return cls(color="#3498db", icon="🔍")  # Blue + Investigation
        elif any(keyword in case_type_lower for keyword in ["contract", "agreement"]):
            return cls(color="#27ae60", icon="📋")  # Green + Contract
        elif any(keyword in case_type_lower for keyword in ["litigation", "court", "lawsuit"]):
            return cls(color="#e74c3c", icon="⚖️")  # Red + Legal
        elif any(keyword in case_type_lower for keyword in ["corporate", "merger", "acquisition"]):
            return cls(color="#f39c12", icon="🏢")  # Orange + Corporate
        elif any(keyword in case_type_lower for keyword in ["urgent", "emergency"]):
            return cls(color="#e74c3c", icon="⚡")  # Red + Urgent
        else:
            return cls.create_default()
    
    def to_dict(self) -> Dict[str, str]:
        """Convert visual marker to dictionary for serialization."""
        return {
            "color": self.color,
            "icon": self.icon
        }


@dataclass
class CaseMetrics:
    """
    Case-level metrics and statistics for monitoring and analytics.
    
    Tracks document processing progress, search activity, and performance metrics
    for case management and optimization insights.
    """
    
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_search_queries: int = 0
    last_activity: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    storage_size_bytes: int = 0
    
    @property
    def processing_progress(self) -> float:
        """Calculate processing progress as percentage (0.0 to 1.0)."""
        if self.total_documents == 0:
            return 0.0
        return self.processed_documents / self.total_documents
    
    @property
    def has_failures(self) -> bool:
        """Check if case has any failed document processing."""
        return self.failed_documents > 0
    
    @property
    def is_processing_complete(self) -> bool:
        """Check if all documents have been processed successfully."""
        return (
            self.total_documents > 0 and 
            self.processed_documents == self.total_documents and 
            self.failed_documents == 0
        )
    
    def update_document_metrics(
        self,
        total_docs: int,
        processed_docs: int,
        failed_docs: int
    ) -> None:
        """Update document processing metrics."""
        self.total_documents = total_docs
        self.processed_documents = processed_docs
        self.failed_documents = failed_docs
        self.last_activity = datetime.now(timezone.utc)
    
    def increment_search_count(self) -> None:
        """Increment search query count."""
        self.total_search_queries += 1
        self.last_activity = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "total_chunks": self.total_chunks,
            "total_search_queries": self.total_search_queries,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "processing_time_seconds": self.processing_time_seconds,
            "storage_size_bytes": self.storage_size_bytes,
            "processing_progress": self.processing_progress,
            "has_failures": self.has_failures,
            "is_processing_complete": self.is_processing_complete,
        }


class LegalCase:
    """
    Core domain entity representing a legal case.
    
    Encapsulates all business logic for legal case management including:
    - Case lifecycle and status management
    - Document capacity enforcement
    - Visual identification and organization
    - Business rule validation
    - Audit trail and metrics tracking
    """
    
    def __init__(
        self,
        case_id: str,
        user_id: str,
        case_name: str,
        initial_summary: str,
        visual_marker: Optional[VisualMarker] = None,
        priority: CasePriority = CasePriority.MEDIUM,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        status: CaseStatus = CaseStatus.DRAFT,
        auto_summary: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a legal case with validation.
        
        Args:
            case_id: Unique case identifier
            user_id: Owner/creator of the case
            case_name: Human-readable case name
            initial_summary: User-provided case description
            visual_marker: Visual identification marker
            priority: Case priority level
            created_at: Case creation timestamp
            updated_at: Last modification timestamp
            status: Current case status
            auto_summary: AI-generated case summary
            tags: Set of case tags for organization
            metadata: Additional case metadata
        """
        # Validate required fields
        self._validate_case_name(case_name)
        self._validate_user_id(user_id)
        self._validate_initial_summary(initial_summary)
        
        # Core identifiers
        self._case_id = case_id
        self._user_id = user_id
        
        # Case information
        self._case_name = case_name.strip()
        self._initial_summary = initial_summary.strip()
        self._auto_summary = auto_summary
        self._priority = priority
        self._status = status
        
        # Visual identification
        self._visual_marker = visual_marker or VisualMarker.create_default()
        
        # Timestamps
        now = datetime.now(timezone.utc)
        self._created_at = created_at or now
        self._updated_at = updated_at or now
        
        # Organization and metadata
        self._tags = tags or set()
        self._metadata = metadata or {}
        
        # Metrics and tracking
        self._metrics = CaseMetrics()
        
        # Document tracking
        self._document_ids: Set[str] = set()
        
        logger.info(
            "Legal case created",
            case_id=self._case_id,
            case_name=self._case_name,
            user_id=self._user_id,
            status=self._status.value
        )
    
    @staticmethod
    def _validate_case_name(case_name: str) -> None:
        """Validate case name according to business rules."""
        if not case_name or not case_name.strip():
            raise CaseManagementError(
                "Case name cannot be empty",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
        
        if len(case_name.strip()) < 3:
            raise CaseManagementError(
                "Case name must be at least 3 characters long",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
        
        if len(case_name.strip()) > 200:
            raise CaseManagementError(
                "Case name must be less than 200 characters",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
    
    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """Validate user ID format."""
        if not user_id or not user_id.strip():
            raise CaseManagementError(
                "User ID cannot be empty",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
    
    @staticmethod
    def _validate_initial_summary(initial_summary: str) -> None:
        """Validate initial summary content."""
        if not initial_summary or not initial_summary.strip():
            raise CaseManagementError(
                "Initial case summary cannot be empty",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
        
        if len(initial_summary.strip()) > 2000:
            raise CaseManagementError(
                "Initial summary must be less than 2000 characters",
                error_code=ErrorCode.CASE_INVALID_STATE
            )
    
    @classmethod
    def create_new(
        cls,
        user_id: str,
        case_name: str,
        initial_summary: str,
        visual_marker: Optional[VisualMarker] = None,
        priority: CasePriority = CasePriority.MEDIUM,
        case_type: Optional[str] = None
    ) -> "LegalCase":
        """
        Factory method to create a new legal case with generated ID.
        
        Args:
            user_id: Case owner identifier
            case_name: Human-readable case name
            initial_summary: User-provided description
            visual_marker: Optional visual marker (auto-generated if None)
            priority: Case priority level
            case_type: Type of case for visual marker selection
            
        Returns:
            New LegalCase instance
        """
        # Generate unique case ID
        case_id = f"CASE_{datetime.now().strftime('%Y_%m_%d')}_{str(uuid.uuid4())[:8].upper()}"
        
        # Auto-generate visual marker based on case type or name
        if visual_marker is None:
            case_type_or_name = case_type or case_name
            visual_marker = VisualMarker.create_for_case_type(case_type_or_name)
        
        return cls(
            case_id=case_id,
            user_id=user_id,
            case_name=case_name,
            initial_summary=initial_summary,
            visual_marker=visual_marker,
            priority=priority
        )
    
    # Properties (read-only access to core attributes)
    
    @property
    def case_id(self) -> str:
        """Get case identifier."""
        return self._case_id
    
    @property
    def user_id(self) -> str:
        """Get case owner identifier."""
        return self._user_id
    
    @property
    def case_name(self) -> str:
        """Get case name."""
        return self._case_name
    
    @property
    def initial_summary(self) -> str:
        """Get initial case summary."""
        return self._initial_summary
    
    @property
    def auto_summary(self) -> Optional[str]:
        """Get AI-generated case summary."""
        return self._auto_summary
    
    @property
    def status(self) -> CaseStatus:
        """Get current case status."""
        return self._status
    
    @property
    def priority(self) -> CasePriority:
        """Get case priority."""
        return self._priority
    
    @property
    def visual_marker(self) -> VisualMarker:
        """Get visual marker."""
        return self._visual_marker
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def tags(self) -> Set[str]:
        """Get case tags."""
        return self._tags.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get case metadata."""
        return self._metadata.copy()
    
    @property
    def metrics(self) -> CaseMetrics:
        """Get case metrics."""
        return self._metrics
    
    @property
    def document_count(self) -> int:
        """Get current document count."""
        return len(self._document_ids)
    
    @property
    def document_ids(self) -> Set[str]:
        """Get set of document IDs associated with this case."""
        return self._document_ids.copy()
    
    # Business operations
    
    def update_case_name(self, new_name: str) -> None:
        """
        Update case name with validation.
        
        Args:
            new_name: New case name
            
        Raises:
            CaseManagementError: If name is invalid
        """
        self._validate_case_name(new_name)
        old_name = self._case_name
        self._case_name = new_name.strip()
        self._touch()
        
        logger.info(
            "Case name updated",
            case_id=self._case_id,
            old_name=old_name,
            new_name=self._case_name
        )
    
    def update_auto_summary(self, summary: str) -> None:
        """
        Update AI-generated case summary.
        
        Args:
            summary: New auto-generated summary
        """
        self._auto_summary = summary.strip() if summary else None
        self._touch()
        
        logger.info(
            "Auto-summary updated",
            case_id=self._case_id,
            summary_length=len(summary) if summary else 0
        )
    
    def update_priority(self, priority: CasePriority) -> None:
        """
        Update case priority.
        
        Args:
            priority: New priority level
        """
        old_priority = self._priority
        self._priority = priority
        self._touch()
        
        logger.info(
            "Case priority updated",
            case_id=self._case_id,
            old_priority=old_priority.value,
            new_priority=priority.value
        )
    
    def update_visual_marker(self, visual_marker: VisualMarker) -> None:
        """
        Update case visual marker.
        
        Args:
            visual_marker: New visual marker
        """
        old_marker = self._visual_marker
        self._visual_marker = visual_marker
        self._touch()
        
        logger.info(
            "Visual marker updated",
            case_id=self._case_id,
            old_marker=old_marker.to_dict(),
            new_marker=visual_marker.to_dict()
        )
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the case.
        
        Args:
            tag: Tag to add
        """
        if tag and tag.strip():
            self._tags.add(tag.strip().lower())
            self._touch()
    
    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the case.
        
        Args:
            tag: Tag to remove
        """
        self._tags.discard(tag.strip().lower())
        self._touch()
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update case metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        self._touch()
    
    def check_document_capacity(self, additional_docs: int = 1) -> None:
        """
        Check if adding documents would exceed capacity limits.
        
        Args:
            additional_docs: Number of documents to add
            
        Raises:
            ResourceError: If capacity would be exceeded
        """
        settings = get_settings()
        current_count = self.document_count
        new_count = current_count + additional_docs
        limit = settings.capacity_limits.documents_per_case
        
        if new_count > limit and not settings.capacity_limits.manual_override_enabled:
            raise_capacity_error(
                f"Adding {additional_docs} document(s) would exceed the limit of {limit} documents per case",
                resource_type="documents_per_case",
                current_usage=current_count,
                limit=limit
            )
        
        if new_count > limit:
            logger.warning(
                "Document capacity exceeded but manual override enabled",
                case_id=self._case_id,
                current_count=current_count,
                new_count=new_count,
                limit=limit
            )
    
    def add_document(self, document_id: str) -> None:
        """
        Add a document to the case with capacity checking.
        
        Args:
            document_id: Document identifier to add
            
        Raises:
            ResourceError: If capacity exceeded
            CaseManagementError: If document already exists
        """
        if document_id in self._document_ids:
            raise CaseManagementError(
                f"Document {document_id} already exists in case {self._case_id}",
                error_code=ErrorCode.CASE_INVALID_STATE,
                case_id=self._case_id
            )
        
        # Check capacity before adding
        self.check_document_capacity(1)
        
        self._document_ids.add(document_id)
        self._metrics.total_documents = len(self._document_ids)
        self._touch()
        
        # Update status to active if this is the first document
        if self._status == CaseStatus.DRAFT and len(self._document_ids) == 1:
            self.transition_to_status(CaseStatus.ACTIVE)
        
        logger.info(
            "Document added to case",
            case_id=self._case_id,
            document_id=document_id,
            total_documents=len(self._document_ids)
        )
    
    def remove_document(self, document_id: str) -> None:
        """
        Remove a document from the case.
        
        Args:
            document_id: Document identifier to remove
        """
        if document_id not in self._document_ids:
            logger.warning(
                "Attempted to remove non-existent document",
                case_id=self._case_id,
                document_id=document_id
            )
            return
        
        self._document_ids.remove(document_id)
        self._metrics.total_documents = len(self._document_ids)
        self._touch()
        
        # Update status to draft if no documents remain
        if len(self._document_ids) == 0:
            self.transition_to_status(CaseStatus.DRAFT)
        
        logger.info(
            "Document removed from case",
            case_id=self._case_id,
            document_id=document_id,
            remaining_documents=len(self._document_ids)
        )
    
    def transition_to_status(self, new_status: CaseStatus) -> None:
        """
        Transition case to a new status with validation.
        
        Args:
            new_status: Target status
            
        Raises:
            CaseManagementError: If transition is invalid
        """
        if not self._is_valid_status_transition(self._status, new_status):
            raise CaseManagementError(
                f"Invalid status transition from {self._status.value} to {new_status.value}",
                error_code=ErrorCode.CASE_INVALID_STATE,
                case_id=self._case_id
            )
        
        old_status = self._status
        self._status = new_status
        self._touch()
        
        logger.info(
            "Case status transition",
            case_id=self._case_id,
            old_status=old_status.value,
            new_status=new_status.value
        )
    
    def _is_valid_status_transition(
        self,
        from_status: CaseStatus,
        to_status: CaseStatus
    ) -> bool:
        """Validate status transition according to business rules."""
        # Define valid transitions
        valid_transitions = {
            CaseStatus.DRAFT: {CaseStatus.ACTIVE, CaseStatus.ARCHIVED},
            CaseStatus.ACTIVE: {CaseStatus.PROCESSING, CaseStatus.COMPLETE, CaseStatus.ARCHIVED, CaseStatus.ERROR},
            CaseStatus.PROCESSING: {CaseStatus.ACTIVE, CaseStatus.COMPLETE, CaseStatus.ERROR},
            CaseStatus.COMPLETE: {CaseStatus.ACTIVE, CaseStatus.ARCHIVED},
            CaseStatus.ERROR: {CaseStatus.ACTIVE, CaseStatus.PROCESSING},
            CaseStatus.ARCHIVED: {CaseStatus.ACTIVE},
        }
        
        return to_status in valid_transitions.get(from_status, set())
    
    def archive(self) -> None:
        """Archive the case (soft delete)."""
        if self._status == CaseStatus.PROCESSING:
            raise CaseManagementError(
                "Cannot archive case while documents are being processed",
                error_code=ErrorCode.CASE_INVALID_STATE,
                case_id=self._case_id
            )
        
        self.transition_to_status(CaseStatus.ARCHIVED)
    
    def is_ready_for_search(self) -> bool:
        """Check if case is ready for search operations."""
        return (
            self._status in {CaseStatus.ACTIVE, CaseStatus.COMPLETE} and
            len(self._document_ids) > 0 and
            self._metrics.processed_documents > 0
        )
    
    def is_active(self) -> bool:
        """Check if case is currently active."""
        return self._status not in {CaseStatus.ARCHIVED, CaseStatus.ERROR}
    
    def _touch(self) -> None:
        """Update the last modified timestamp."""
        self._updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert case to dictionary for serialization.
        
        Returns:
            Dictionary representation of the case
        """
        return {
            "case_id": self._case_id,
            "user_id": self._user_id,
            "case_name": self._case_name,
            "initial_summary": self._initial_summary,
            "auto_summary": self._auto_summary,
            "status": self._status.value,
            "priority": self._priority.value,
            "visual_marker": self._visual_marker.to_dict(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "tags": list(self._tags),
            "metadata": self._metadata,
            "metrics": self._metrics.to_dict(),
            "document_count": self.document_count,
            "document_ids": list(self._document_ids),
            "is_ready_for_search": self.is_ready_for_search(),
            "is_active": self.is_active(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalCase":
        """
        Create case instance from dictionary.
        
        Args:
            data: Dictionary representation of case
            
        Returns:
            LegalCase instance
        """
        # Parse visual marker
        visual_marker_data = data.get("visual_marker", {})
        visual_marker = VisualMarker(
            color=visual_marker_data.get("color", VisualMarker.COLORS[0]),
            icon=visual_marker_data.get("icon", VisualMarker.ICONS[0])
        )
        
        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        
        # Create case instance
        case = cls(
            case_id=data["case_id"],
            user_id=data["user_id"],
            case_name=data["case_name"],
            initial_summary=data["initial_summary"],
            visual_marker=visual_marker,
            priority=CasePriority(data.get("priority", CasePriority.MEDIUM.value)),
            created_at=created_at,
            updated_at=updated_at,
            status=CaseStatus(data.get("status", CaseStatus.DRAFT.value)),
            auto_summary=data.get("auto_summary"),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {})
        )
        
        # Restore document IDs
        case._document_ids = set(data.get("document_ids", []))
        
        # Restore metrics
        metrics_data = data.get("metrics", {})
        case._metrics = CaseMetrics(
            total_documents=metrics_data.get("total_documents", 0),
            processed_documents=metrics_data.get("processed_documents", 0),
            failed_documents=metrics_data.get("failed_documents", 0),
            total_chunks=metrics_data.get("total_chunks", 0),
            total_search_queries=metrics_data.get("total_search_queries", 0),
            processing_time_seconds=metrics_data.get("processing_time_seconds", 0.0),
            storage_size_bytes=metrics_data.get("storage_size_bytes", 0)
        )
        
        # Restore last activity if present
        if "last_activity" in metrics_data and metrics_data["last_activity"]:
            case._metrics.last_activity = datetime.fromisoformat(
                metrics_data["last_activity"].replace("Z", "+00:00")
            )
        
        return case
    
    def __str__(self) -> str:
        """String representation of the case."""
        return f"LegalCase(id={self._case_id}, name='{self._case_name}', status={self._status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the case."""
        return (
            f"LegalCase(case_id='{self._case_id}', user_id='{self._user_id}', "
            f"case_name='{self._case_name}', status={self._status.value}, "
            f"documents={len(self._document_ids)}, created_at={self._created_at})"
        )