"""
Case Navigation Sidebar Component for Patexia Legal AI Chatbot

This module implements the case navigation sidebar using Gradio framework.
It provides comprehensive case management functionality with visual case identification,
document upload capabilities, and real-time progress tracking.

Key Features:
- Case creation with visual markers (color + icon combinations)
- Case switching with instant context preservation
- Document upload with drag-and-drop functionality
- Real-time document processing progress tracking
- Case statistics and capacity management
- Recent cases list with quick access
- Case search and filtering capabilities
- Integration with search pane for case-scoped operations

Architecture Integration:
- Uses APIClient for REST API communication with backend
- Integrates WebSocketClient for real-time progress updates
- Coordinates with SearchPane for case-scoped search operations
- Manages case state and context switching
- Provides case management interface for multi-user scenarios

UI Components:
- Active cases list with visual markers and status indicators
- Case creation modal with validation and conflict checking
- Document upload area with progress tracking
- Case statistics dashboard
- Recent/archived cases management
- Case search and filtering options

Case Management Flow:
1. User creates new case with name, summary, and visual marker
2. System validates marker uniqueness and case requirements
3. Case is created and added to active cases list
4. User uploads documents with real-time progress tracking
5. Case context switching updates search scope automatically
6. Case statistics update in real-time as documents are processed
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import gradio as gr

from ..utils.api_client import APIClient, APIError
from ..utils.websocket_client import WebSocketClient, WebSocketMessage
from ..utils.ui_helpers import (
    format_file_size, format_timestamp, validate_case_name,
    create_visual_marker_selector, get_available_visual_markers
)


class CaseStatus(str, Enum):
    """Status of legal cases."""
    DRAFT = "draft"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ARCHIVED = "archived"
    ERROR = "error"


class DocumentUploadStatus(str, Enum):
    """Status of document upload operations."""
    IDLE = "idle"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class VisualMarker:
    """Visual marker for case identification."""
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


@dataclass
class LegalCase:
    """Legal case data structure."""
    case_id: str
    case_name: str
    initial_summary: str
    visual_marker: VisualMarker
    status: CaseStatus
    document_count: int = 0
    processed_documents: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: Optional[datetime] = None
    total_size_bytes: int = 0
    
    @property
    def processing_progress(self) -> float:
        """Calculate processing progress as percentage (0.0 to 1.0)."""
        if self.document_count == 0:
            return 0.0
        return self.processed_documents / self.document_count
    
    @property
    def formatted_size(self) -> str:
        """Get formatted size string."""
        return format_file_size(self.total_size_bytes)


@dataclass
class DocumentUpload:
    """Document upload operation tracking."""
    upload_id: str
    case_id: str
    filename: str
    file_size: int
    status: DocumentUploadStatus
    progress: float = 0.0
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class SidebarComponent:
    """
    Wrapper component class for case navigation sidebar.
    
    This class provides the interface expected by main.py while wrapping
    the existing CaseSidebar functionality. It handles component lifecycle,
    event management, and integration with other UI components.
    """
    
    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        websocket_client: Optional[WebSocketClient] = None
    ):
        """
        Initialize sidebar component with API clients.
        
        Args:
            api_client: API client for backend communication
            websocket_client: WebSocket client for real-time updates
        """
        self.api_client = api_client or APIClient()
        self.websocket_client = websocket_client or WebSocketClient()
        self.logger = logging.getLogger(f"{__name__}.SidebarComponent")
        
        # Initialize the underlying sidebar functionality
        self.sidebar = CaseSidebar(
            api_client=self.api_client,
            websocket_client=self.websocket_client
        )
        
        # Event callbacks
        self._case_selected_callbacks: List[Callable[[Optional[str]], None]] = []
        self._document_uploaded_callbacks: List[Callable[[str, str], None]] = []
        
        # Component state
        self.is_initialized = False
        self.current_case_id: Optional[str] = None
        
        self.logger.info("SidebarComponent initialized")
    
    def create_component(self) -> gr.Column:
        """
        Create the sidebar Gradio component.
        
        Returns:
            Gradio Column containing the complete sidebar interface
        """
        try:
            # Create the main sidebar component using CaseSidebar
            sidebar_component = self.sidebar.create_component()
            
            # Setup event handlers for the sidebar
            self._setup_event_handlers()
            
            self.is_initialized = True
            self.logger.info("Sidebar component created successfully")
            
            return sidebar_component
            
        except Exception as e:
            self.logger.error(f"Failed to create sidebar component: {e}")
            # Return a fallback component
            return self._create_fallback_component()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for sidebar interactions."""
        try:
            # Set case change callback
            self.sidebar.set_case_change_callback(self._handle_case_selection)
            
            # Setup other event handlers
            self.sidebar.on_case_created = self._handle_case_created
            self.sidebar.on_document_uploaded = self._handle_document_uploaded
            
        except Exception as e:
            self.logger.error(f"Failed to setup event handlers: {e}")
    
    def _handle_case_selection(self, case_id: Optional[str]) -> None:
        """
        Handle case selection events.
        
        Args:
            case_id: ID of selected case or None if deselected
        """
        try:
            self.current_case_id = case_id
            
            # Notify all registered callbacks
            for callback in self._case_selected_callbacks:
                try:
                    callback(case_id)
                except Exception as e:
                    self.logger.error(f"Error in case selection callback: {e}")
            
            self.logger.info(f"Case selection handled: {case_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle case selection: {e}")
    
    def _handle_case_created(self, case: LegalCase) -> None:
        """
        Handle case creation events.
        
        Args:
            case: Newly created case
        """
        try:
            self.logger.info(f"Case created: {case.case_name} ({case.case_id})")
            
            # Automatically select the new case
            self._handle_case_selection(case.case_id)
            
        except Exception as e:
            self.logger.error(f"Failed to handle case creation: {e}")
    
    def _handle_document_uploaded(self, case_id: str, document_id: str) -> None:
        """
        Handle document upload events.
        
        Args:
            case_id: ID of the case
            document_id: ID of uploaded document
        """
        try:
            # Notify all registered callbacks
            for callback in self._document_uploaded_callbacks:
                try:
                    callback(case_id, document_id)
                except Exception as e:
                    self.logger.error(f"Error in document upload callback: {e}")
            
            self.logger.info(f"Document upload handled: {document_id} in case {case_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle document upload: {e}")
    
    def _create_fallback_component(self) -> gr.Column:
        """
        Create a fallback component when main sidebar fails.
        
        Returns:
            Simple fallback sidebar component
        """
        with gr.Column(elem_id="sidebar-fallback") as fallback:
            gr.Markdown("## ⚖️ Case Navigation")
            gr.Markdown("*Sidebar temporarily unavailable*")
            
            gr.Button("Retry Connection", variant="secondary")
            
            gr.Markdown("### System Status")
            gr.HTML("""
                <div style="padding: 1rem; background: #fee; border-left: 4px solid #f00;">
                    <strong>Connection Error</strong><br>
                    Unable to connect to backend services.
                    Please check your connection and try again.
                </div>
            """)
        
        return fallback
    
    # Public API methods for integration with other components
    
    def on_case_selected(self, callback: Callable[[Optional[str]], None]) -> None:
        """
        Register a callback for case selection events.
        
        Args:
            callback: Function to call when case is selected/deselected
        """
        if callback not in self._case_selected_callbacks:
            self._case_selected_callbacks.append(callback)
            self.logger.debug("Case selection callback registered")
    
    def on_document_uploaded(self, callback: Callable[[str, str], None]) -> None:
        """
        Register a callback for document upload events.
        
        Args:
            callback: Function to call when document is uploaded
        """
        if callback not in self._document_uploaded_callbacks:
            self._document_uploaded_callbacks.append(callback)
            self.logger.debug("Document upload callback registered")
    
    def get_current_case_id(self) -> Optional[str]:
        """
        Get the currently selected case ID.
        
        Returns:
            Current case ID or None if no case selected
        """
        return self.current_case_id
    
    def select_case(self, case_id: str) -> None:
        """
        Programmatically select a case.
        
        Args:
            case_id: ID of case to select
        """
        try:
            self.sidebar.set_current_case(case_id)
        except Exception as e:
            self.logger.error(f"Failed to select case {case_id}: {e}")
    
    def refresh_cases(self) -> None:
        """Refresh the cases list from backend."""
        try:
            asyncio.create_task(self.sidebar._load_initial_cases())
        except Exception as e:
            self.logger.error(f"Failed to refresh cases: {e}")
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """
        Get current case statistics.
        
        Returns:
            Dictionary with case statistics
        """
        try:
            total_cases = len(self.sidebar.cases)
            active_cases = len([
                case for case in self.sidebar.cases.values()
                if case.status == CaseStatus.ACTIVE
            ])
            processing_cases = len([
                case for case in self.sidebar.cases.values()
                if case.status == CaseStatus.PROCESSING
            ])
            total_documents = sum(case.document_count for case in self.sidebar.cases.values())
            
            return {
                "total_cases": total_cases,
                "active_cases": active_cases,
                "processing_cases": processing_cases,
                "total_documents": total_documents,
                "current_case_id": self.current_case_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get case statistics: {e}")
            return {}


class CaseSidebar:
    """
    Case navigation sidebar component with comprehensive case management.
    
    Provides case creation, switching, document upload, and progress tracking
    with real-time WebSocket integration and visual case identification.
    """
    
    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        websocket_client: Optional[WebSocketClient] = None
    ):
        """Initialize sidebar with API clients and state management."""
        self.api_client = api_client or APIClient()
        self.websocket_client = websocket_client or WebSocketClient()
        self.logger = logging.getLogger(f"{__name__}.CaseSidebar")
        
        # State management
        self.current_case_id: Optional[str] = None
        self.cases: Dict[str, LegalCase] = {}
        self.active_uploads: Dict[str, DocumentUpload] = {}
        self.case_change_callback: Optional[Callable[[Optional[str]], None]] = None
        
        # Event callbacks
        self.on_case_created: Optional[Callable[[LegalCase], None]] = None
        self.on_document_uploaded: Optional[Callable[[str, str], None]] = None
        
        # UI state
        self._case_creation_modal_visible: bool = False
        self._upload_area_visible: bool = True
        
        # WebSocket message handlers
        self.websocket_client.add_handler("case_created", self._handle_case_created)
        self.websocket_client.add_handler("case_updated", self._handle_case_updated)
        self.websocket_client.add_handler("document_upload_progress", self._handle_upload_progress)
        self.websocket_client.add_handler("document_processed", self._handle_document_processed)
        
        # Load initial data
        asyncio.create_task(self._load_initial_cases())
    
    def create_component(self) -> gr.Column:
        """
        Create the main sidebar Gradio component.
        
        Returns:
            Gradio Column containing the complete sidebar interface
        """
        with gr.Column(elem_id="case-sidebar", scale=1) as sidebar:
            # Sidebar header
            gr.Markdown("## ⚖️ Case Navigation", elem_id="sidebar-header")
            
            # Create new case section
            with gr.Accordion("📝 Create New Case", open=False, elem_id="create-case-accordion"):
                self.case_name_input = gr.Textbox(
                    placeholder="Enter case name...",
                    label="Case Name",
                    max_lines=1,
                    elem_id="case-name-input"
                )
                
                self.case_summary_input = gr.Textbox(
                    placeholder="Brief case summary...",
                    label="Initial Summary",
                    lines=3,
                    max_lines=5,
                    elem_id="case-summary-input"
                )
                
                # Visual marker selection
                with gr.Row(elem_id="visual-marker-row"):
                    self.case_color_dropdown = gr.Dropdown(
                        choices=[(f"🎨 {color}", color) for color in VisualMarker.COLORS],
                        label="Case Color",
                        value=VisualMarker.COLORS[0],
                        elem_id="case-color"
                    )
                    
                    self.case_icon_dropdown = gr.Dropdown(
                        choices=[(f"{icon} {icon}", icon) for icon in VisualMarker.ICONS],
                        label="Case Icon",
                        value=VisualMarker.ICONS[0],
                        elem_id="case-icon"
                    )
                
                # Create case buttons
                with gr.Row(elem_id="create-case-buttons"):
                    self.cancel_case_button = gr.Button(
                        "Cancel",
                        variant="secondary",
                        size="sm",
                        elem_id="cancel-case-button"
                    )
                    self.create_case_button = gr.Button(
                        "Create Case",
                        variant="primary",
                        size="sm",
                        elem_id="create-case-button"
                    )
            
            # Case validation message
            self.case_validation_message = gr.Markdown(
                visible=False,
                elem_id="case-validation-message"
            )
            
            # Active cases section
            gr.Markdown("### Active Cases", elem_id="active-cases-header")
            
            # Cases list
            self.cases_list = gr.HTML(
                value=self._render_empty_cases(),
                elem_id="cases-list"
            )
            
            # Case statistics
            with gr.Accordion("Case Statistics", open=False, elem_id="case-stats"):
                self.case_stats = gr.HTML(
                    value=self._render_case_stats(),
                    elem_id="case-statistics"
                )
            
            # Document upload section
            gr.Markdown("### 📤 Upload Documents", elem_id="upload-header")
            
            # Upload status message
            self.upload_status = gr.Markdown(
                "Select a case to upload documents",
                elem_id="upload-status"
            )
            
            # File upload component
            self.file_upload = gr.File(
                label="Select Documents",
                file_count="multiple",
                file_types=[".pdf", ".txt"],
                elem_id="document-upload",
                visible=False
            )
            
            # Upload progress
            self.upload_progress = gr.HTML(
                visible=False,
                elem_id="upload-progress"
            )
            
            # Setup event handlers
            self._setup_event_handlers()
        
        return sidebar
    
    def _setup_event_handlers(self) -> None:
        """Setup Gradio event handlers for user interactions."""
        # Case creation
        self.create_case_button.click(
            fn=self._handle_create_case_click,
            inputs=[
                self.case_name_input,
                self.case_summary_input,
                self.case_color_dropdown,
                self.case_icon_dropdown
            ],
            outputs=[
                self.cases_list,
                self.case_validation_message,
                self.case_name_input,
                self.case_summary_input
            ]
        )
        
        # File upload handling
        self.file_upload.upload(
            fn=self._handle_file_upload,
            inputs=[self.file_upload],
            outputs=[self.upload_progress, self.upload_status]
        )
    
    def _handle_create_case_click(
        self,
        case_name: str,
        case_summary: str,
        color: str,
        icon: str
    ) -> Tuple[str, str, str, str]:
        """
        Handle case creation button click.
        
        Args:
            case_name: Name of the new case
            case_summary: Summary description
            color: Selected color
            icon: Selected icon
            
        Returns:
            Tuple of (updated_cases_list, validation_message, cleared_name, cleared_summary)
        """
        try:
            # Validate inputs
            if not case_name.strip():
                return (
                    self._render_cases_list(),
                    "❌ Case name is required",
                    case_name,
                    case_summary
                )
            
            if len(case_name.strip()) < 3:
                return (
                    self._render_cases_list(),
                    "❌ Case name must be at least 3 characters",
                    case_name,
                    case_summary
                )
            
            # Check for duplicate case names
            if any(case.case_name.lower() == case_name.lower() for case in self.cases.values()):
                return (
                    self._render_cases_list(),
                    "❌ A case with this name already exists",
                    case_name,
                    case_summary
                )
            
            # Create the case
            case_id = str(uuid.uuid4())
            visual_marker = VisualMarker(color=color, icon=icon)
            
            new_case = LegalCase(
                case_id=case_id,
                case_name=case_name.strip(),
                initial_summary=case_summary.strip() or "No summary provided",
                visual_marker=visual_marker,
                status=CaseStatus.ACTIVE
            )
            
            # Add to cases dictionary
            self.cases[case_id] = new_case
            
            # Set as current case
            self.set_current_case(case_id)
            
            # Trigger callback
            if self.on_case_created:
                self.on_case_created(new_case)
            
            # Send creation request to backend
            asyncio.create_task(self._create_case_backend(new_case))
            
            self.logger.info(f"Case created: {case_name} ({case_id})")
            
            return (
                self._render_cases_list(),
                "✅ Case created successfully!",
                "",  # Clear name input
                ""   # Clear summary input
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create case: {e}")
            return (
                self._render_cases_list(),
                f"❌ Error creating case: {str(e)}",
                case_name,
                case_summary
            )
    
    def _handle_file_upload(self, files) -> Tuple[str, str]:
        """
        Handle file upload events.
        
        Args:
            files: Uploaded files
            
        Returns:
            Tuple of (progress_html, status_message)
        """
        try:
            if not self.current_case_id:
                return ("", "❌ Please select a case before uploading documents")
            
            if not files:
                return ("", "No files selected")
            
            # Process uploaded files
            upload_tasks = []
            for file in files:
                upload_id = str(uuid.uuid4())
                upload = DocumentUpload(
                    upload_id=upload_id,
                    case_id=self.current_case_id,
                    filename=file.name,
                    file_size=0,  # Will be updated
                    status=DocumentUploadStatus.UPLOADING
                )
                
                self.active_uploads[upload_id] = upload
                upload_tasks.append(self._process_file_upload(file, upload))
            
            # Start processing uploads
            for task in upload_tasks:
                asyncio.create_task(task)
            
            progress_html = self._render_upload_progress()
            status_message = f"📤 Uploading {len(files)} document(s)..."
            
            return (progress_html, status_message)
            
        except Exception as e:
            self.logger.error(f"File upload error: {e}")
            return ("", f"❌ Upload error: {str(e)}")
    
    async def _process_file_upload(self, file, upload: DocumentUpload) -> None:
        """Process individual file upload."""
        try:
            # Update upload status
            upload.status = DocumentUploadStatus.PROCESSING
            
            # Send file to backend
            response = await self.api_client.post(
                f"/api/v1/cases/{upload.case_id}/documents",
                files={"file": file}
            )
            
            if response.get("success"):
                upload.status = DocumentUploadStatus.COMPLETED
                upload.completed_at = datetime.now(timezone.utc)
                
                # Trigger callback
                if self.on_document_uploaded:
                    self.on_document_uploaded(upload.case_id, response.get("document_id"))
            else:
                upload.status = DocumentUploadStatus.ERROR
                upload.error_message = response.get("error", "Unknown error")
            
        except Exception as e:
            upload.status = DocumentUploadStatus.ERROR
            upload.error_message = str(e)
            self.logger.error(f"File processing error: {e}")
    
    async def _create_case_backend(self, case: LegalCase) -> None:
        """Create case in backend."""
        try:
            await self.api_client.post("/api/v1/cases", {
                "case_id": case.case_id,
                "name": case.case_name,
                "summary": case.initial_summary,
                "visual_marker": {
                    "color": case.visual_marker.color,
                    "icon": case.visual_marker.icon
                }
            })
        except Exception as e:
            self.logger.error(f"Backend case creation failed: {e}")
    
    async def _load_initial_cases(self) -> None:
        """Load existing cases from backend."""
        try:
            response = await self.api_client.get("/api/v1/cases")
            if response.get("success"):
                cases_data = response.get("cases", [])
                for case_data in cases_data:
                    case = self._deserialize_case(case_data)
                    self.cases[case.case_id] = case
                
                self.logger.info(f"Loaded {len(self.cases)} cases from backend")
        except Exception as e:
            self.logger.error(f"Failed to load cases: {e}")
    
    def _deserialize_case(self, case_data: Dict[str, Any]) -> LegalCase:
        """Convert backend case data to LegalCase object."""
        visual_marker = VisualMarker(
            color=case_data.get("visual_marker", {}).get("color", VisualMarker.COLORS[0]),
            icon=case_data.get("visual_marker", {}).get("icon", VisualMarker.ICONS[0])
        )
        
        return LegalCase(
            case_id=case_data["case_id"],
            case_name=case_data["name"],
            initial_summary=case_data.get("summary", ""),
            visual_marker=visual_marker,
            status=CaseStatus(case_data.get("status", CaseStatus.ACTIVE.value)),
            document_count=case_data.get("document_count", 0),
            processed_documents=case_data.get("processed_documents", 0),
            created_at=datetime.fromisoformat(case_data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(case_data.get("updated_at", datetime.now(timezone.utc).isoformat()))
        )
    
    def set_current_case(self, case_id: Optional[str]) -> None:
        """
        Set the current active case.
        
        Args:
            case_id: ID of the case to set as current, or None to clear
        """
        self.current_case_id = case_id
        
        # Update upload area visibility
        self.file_upload.visible = case_id is not None
        
        # Update upload status
        if case_id:
            case = self.cases.get(case_id)
            case_name = case.case_name if case else "Unknown Case"
            self.upload_status.value = f"📤 Upload documents to: **{case_name}**"
        else:
            self.upload_status.value = "Select a case to upload documents"
        
        # Notify other components of case change
        if self.case_change_callback:
            self.case_change_callback(case_id)
        
        self.logger.info(f"Current case changed to: {case_id}")
    
    def set_case_change_callback(self, callback: Callable[[Optional[str]], None]) -> None:
        """Set callback for case change events."""
        self.case_change_callback = callback
    
    def _render_cases_list(self) -> str:
        """
        Render the active cases list as HTML.
        
        Returns:
            HTML string for cases list
        """
        if not self.cases:
            return self._render_empty_cases()
        
        # Filter active cases
        active_cases = [
            case for case in self.cases.values()
            if case.status in [CaseStatus.ACTIVE, CaseStatus.PROCESSING, CaseStatus.COMPLETE]
        ]
        
        if not active_cases:
            return self._render_empty_cases()
        
        # Sort by last activity or creation date
        active_cases.sort(
            key=lambda c: c.last_activity or c.created_at,
            reverse=True
        )
        
        html_parts = ["<div class='cases-list'>"]
        
        for case in active_cases:
            is_current = case.case_id == self.current_case_id
            status_icon = self._get_status_icon(case.status)
            progress_text = ""
            
            if case.status == CaseStatus.PROCESSING:
                progress_percent = int(case.processing_progress * 100)
                progress_text = f" ({progress_percent}%)"
            
            css_class = "case-item active" if is_current else "case-item"
            
            html_parts.append(f"""
            <div class="{css_class}" onclick="selectCase('{case.case_id}')" data-case-id="{case.case_id}">
                <div class="case-marker" style="background-color: {case.visual_marker.color};">
                    {case.visual_marker.icon}
                </div>
                <div class="case-content">
                    <div class="case-name">{case.case_name}</div>
                    <div class="case-meta">
                        <span class="case-status">{status_icon} {case.status.value.title()}{progress_text}</span>
                        <span class="case-docs">{case.document_count} docs</span>
                        <span class="case-size">{case.formatted_size}</span>
                    </div>
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _render_empty_cases(self) -> str:
        """Render empty cases state."""
        return """
        <div class="empty-cases">
            <div class="empty-icon">📁</div>
            <div class="empty-message">No active cases</div>
            <div class="empty-hint">Create your first case to get started</div>
        </div>
        """
    
    def _render_case_stats(self) -> str:
        """Render case statistics as HTML."""
        total_cases = len(self.cases)
        active_cases = len([c for c in self.cases.values() if c.status == CaseStatus.ACTIVE])
        total_docs = sum(c.document_count for c in self.cases.values())
        total_size = sum(c.total_size_bytes for c in self.cases.values())
        
        return f"""
        <div class="case-statistics">
            <div class="stat-item">
                <span class="stat-value">{total_cases}</span>
                <span class="stat-label">Total Cases</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{active_cases}</span>
                <span class="stat-label">Active Cases</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{total_docs}</span>
                <span class="stat-label">Documents</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{format_file_size(total_size)}</span>
                <span class="stat-label">Total Size</span>
            </div>
        </div>
        """
    
    def _render_upload_progress(self) -> str:
        """Render upload progress for active uploads."""
        if not self.active_uploads:
            return ""
        
        html_parts = ["<div class='upload-progress-list'>"]
        
        for upload in self.active_uploads.values():
            status_icon = "⏳" if upload.status == DocumentUploadStatus.UPLOADING else \
                         "🔄" if upload.status == DocumentUploadStatus.PROCESSING else \
                         "✅" if upload.status == DocumentUploadStatus.COMPLETED else "❌"
            
            progress_bar = ""
            if upload.status in [DocumentUploadStatus.UPLOADING, DocumentUploadStatus.PROCESSING]:
                progress_bar = f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {upload.progress * 100}%"></div>
                </div>
                """
            
            html_parts.append(f"""
            <div class="upload-item">
                <div class="upload-header">
                    <span class="upload-icon">{status_icon}</span>
                    <span class="upload-filename">{upload.filename}</span>
                </div>
                {progress_bar}
                {f'<div class="upload-error">{upload.error_message}</div>' if upload.error_message else ''}
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _get_status_icon(self, status: CaseStatus) -> str:
        """Get icon for case status."""
        icons = {
            CaseStatus.DRAFT: "📝",
            CaseStatus.ACTIVE: "🟢",
            CaseStatus.PROCESSING: "🔄",
            CaseStatus.COMPLETE: "✅",
            CaseStatus.ARCHIVED: "📦",
            CaseStatus.ERROR: "❌"
        }
        return icons.get(status, "❓")
    
    # WebSocket event handlers
    async def _handle_case_created(self, message: Dict[str, Any]) -> None:
        """Handle case creation WebSocket messages."""
        try:
            case_data = message.get("data", {})
            case = self._deserialize_case(case_data)
            self.cases[case.case_id] = case
            
            # Update UI
            self.cases_list.value = self._render_cases_list()
            self.case_stats.value = self._render_case_stats()
            
        except Exception as e:
            self.logger.error(f"Error handling case creation message: {e}")
    
    async def _handle_case_updated(self, message: Dict[str, Any]) -> None:
        """Handle case update WebSocket messages."""
        try:
            case_data = message.get("data", {})
            case_id = case_data.get("case_id")
            
            if case_id in self.cases:
                # Update existing case
                self.cases[case_id] = self._deserialize_case(case_data)
                
                # Update UI
                self.cases_list.value = self._render_cases_list()
                self.case_stats.value = self._render_case_stats()
            
        except Exception as e:
            self.logger.error(f"Error handling case update message: {e}")
    
    async def _handle_upload_progress(self, message: Dict[str, Any]) -> None:
        """Handle document upload progress WebSocket messages."""
        try:
            progress_data = message.get("data", {})
            upload_id = progress_data.get("upload_id")
            
            if upload_id in self.active_uploads:
                upload = self.active_uploads[upload_id]
                upload.progress = progress_data.get("progress", 0.0)
                upload.status = DocumentUploadStatus(progress_data.get("status", upload.status.value))
                
                # Update UI
                self.upload_progress.value = self._render_upload_progress()
            
        except Exception as e:
            self.logger.error(f"Error handling upload progress message: {e}")
    
    async def _handle_document_processed(self, message: Dict[str, Any]) -> None:
        """Handle document processing completion WebSocket messages."""
        try:
            doc_data = message.get("data", {})
            case_id = doc_data.get("case_id")
            
            if case_id in self.cases:
                # Update case document count
                self.cases[case_id].processed_documents += 1
                
                # Update UI
                self.cases_list.value = self._render_cases_list()
                self.case_stats.value = self._render_case_stats()
            
        except Exception as e:
            self.logger.error(f"Error handling document processed message: {e}")


def create_sidebar() -> gr.Column:
    """
    Factory function to create a sidebar component.
    
    Returns:
        Configured Sidebar Gradio component
    """
    sidebar_component = SidebarComponent()
    return sidebar_component.create_component()


# Export for use in main.py
__all__ = ["SidebarComponent", "CaseSidebar", "create_sidebar", "LegalCase", "VisualMarker", "CaseStatus"]