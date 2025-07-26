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


class CaseSidebar:
    """
    Case navigation sidebar component with comprehensive case management.
    
    Provides case creation, switching, document upload, and progress tracking
    with real-time WebSocket integration and visual case identification.
    """
    
    def __init__(self):
        """Initialize sidebar with API clients and state management."""
        self.api_client = APIClient()
        self.websocket_client = WebSocketClient()
        self.logger = logging.getLogger(f"{__name__}.CaseSidebar")
        
        # State management
        self.current_case_id: Optional[str] = None
        self.cases: Dict[str, LegalCase] = {}
        self.active_uploads: Dict[str, DocumentUpload] = {}
        self.case_change_callback: Optional[Callable[[Optional[str]], None]] = None
        
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
        with gr.Column(elem_id="case-sidebar", scale=1, min_width=280) as sidebar:
            # Sidebar header
            gr.Markdown("## 📁 Case Management", elem_id="sidebar-header")
            
            # New case button
            self.new_case_button = gr.Button(
                "➕ Create New Case",
                variant="primary",
                elem_id="new-case-button",
                size="sm"
            )
            
            # Case creation modal (initially hidden)
            with gr.Column(visible=False, elem_id="case-creation-modal") as self.case_modal:
                gr.Markdown("### Create New Legal Case")
                
                self.case_name_input = gr.Textbox(
                    label="Case Name",
                    placeholder="e.g., Patent-2025-WiFi6-Dispute",
                    elem_id="case-name-input"
                )
                
                self.case_summary_input = gr.Textbox(
                    label="Initial Summary",
                    placeholder="Brief description of the case, key parties, and main legal issues...",
                    lines=3,
                    max_lines=5,
                    elem_id="case-summary-input"
                )
                
                # Visual marker selection
                with gr.Row(elem_id="visual-marker-row"):
                    self.marker_color_radio = gr.Radio(
                        choices=[
                            ("🔴 Red (Urgent/Litigation)", "#e74c3c"),
                            ("🟢 Green (Contract/Complete)", "#27ae60"),
                            ("🔵 Blue (IP/Patent)", "#3498db"),
                            ("🟠 Orange (Corporate/M&A)", "#f39c12"),
                            ("🟣 Purple (Regulatory)", "#9b59b6"),
                            ("🟡 Teal (Investigation)", "#1abc9c"),
                            ("🟤 Dark Orange (Appeals)", "#e67e22"),
                            ("⚫ Gray (Archive/Reference)", "#34495e")
                        ],
                        label="Color",
                        value="#3498db",
                        elem_id="marker-color"
                    )
                    
                    self.marker_icon_dropdown = gr.Dropdown(
                        choices=[
                            ("📄 Document", "📄"),
                            ("⚖️ Legal", "⚖️"),
                            ("🏢 Corporate", "🏢"),
                            ("💼 Business", "💼"),
                            ("📋 Contract", "📋"),
                            ("🔍 Investigation", "🔍"),
                            ("⚡ Urgent", "⚡"),
                            ("🎯 Priority", "🎯"),
                            ("📊 Analytics", "📊"),
                            ("🔒 Confidential", "🔒")
                        ],
                        label="Icon",
                        value="📄",
                        elem_id="marker-icon"
                    )
                
                # Modal buttons
                with gr.Row(elem_id="modal-buttons"):
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
                interactive=False,
                elem_id="file-upload"
            )
            
            # Upload progress
            self.upload_progress = gr.Progress(
                track_tqdm=True,
                visible=False,
                elem_id="upload-progress"
            )
            
            # Upload progress display
            self.upload_progress_display = gr.HTML(
                visible=False,
                elem_id="upload-progress-display"
            )
            
            # Recent cases section
            with gr.Accordion("Recent Cases", open=False, elem_id="recent-cases"):
                self.recent_cases_list = gr.HTML(
                    value="<div class='no-recent-cases'>No recent cases</div>",
                    elem_id="recent-cases-list"
                )
                
                with gr.Row():
                    self.refresh_cases_button = gr.Button(
                        "🔄 Refresh",
                        size="sm",
                        elem_id="refresh-cases-button"
                    )
                    self.archive_case_button = gr.Button(
                        "📦 Archive Current",
                        size="sm",
                        elem_id="archive-case-button",
                        interactive=False
                    )
        
        # Setup event handlers
        self._setup_event_handlers()
        
        return sidebar
    
    def _setup_event_handlers(self) -> None:
        """Setup all event handlers for the sidebar interface."""
        
        # Case creation modal
        self.new_case_button.click(
            fn=self._show_case_creation_modal,
            inputs=[],
            outputs=[self.case_modal, self.case_validation_message]
        )
        
        self.cancel_case_button.click(
            fn=self._hide_case_creation_modal,
            inputs=[],
            outputs=[self.case_modal, self.case_validation_message]
        )
        
        self.create_case_button.click(
            fn=self._handle_case_creation,
            inputs=[
                self.case_name_input,
                self.case_summary_input,
                self.marker_color_radio,
                self.marker_icon_dropdown
            ],
            outputs=[
                self.case_modal,
                self.case_validation_message,
                self.cases_list,
                self.case_stats,
                self.upload_status,
                self.file_upload
            ]
        )
        
        # File upload handling
        self.file_upload.upload(
            fn=self._handle_file_upload,
            inputs=[self.file_upload],
            outputs=[
                self.upload_status,
                self.upload_progress_display,
                self.cases_list,
                self.case_stats
            ]
        )
        
        # Case management
        self.refresh_cases_button.click(
            fn=self._refresh_cases,
            inputs=[],
            outputs=[self.cases_list, self.recent_cases_list, self.case_stats]
        )
        
        self.archive_case_button.click(
            fn=self._archive_current_case,
            inputs=[],
            outputs=[
                self.cases_list,
                self.recent_cases_list,
                self.case_stats,
                self.archive_case_button
            ]
        )
    
    def _show_case_creation_modal(self) -> Tuple[gr.Column, gr.Markdown]:
        """
        Show the case creation modal.
        
        Returns:
            Updated modal visibility and cleared validation message
        """
        return gr.Column(visible=True), gr.Markdown(visible=False)
    
    def _hide_case_creation_modal(self) -> Tuple[gr.Column, gr.Markdown]:
        """
        Hide the case creation modal.
        
        Returns:
            Updated modal visibility and cleared validation message
        """
        return gr.Column(visible=False), gr.Markdown(visible=False)
    
    async def _handle_case_creation(
        self,
        case_name: str,
        case_summary: str,
        marker_color: str,
        marker_icon: str
    ) -> Tuple[gr.Column, gr.Markdown, str, str, str, gr.File]:
        """
        Handle case creation with validation and API call.
        
        Args:
            case_name: Name of the new case
            case_summary: Initial case summary
            marker_color: Selected color for visual marker
            marker_icon: Selected icon for visual marker
            
        Returns:
            Tuple of updated UI components
        """
        try:
            # Validate inputs
            validation_result = validate_case_name(case_name)
            if not validation_result.is_valid:
                return (
                    gr.Column(visible=True),
                    gr.Markdown(
                        f"❌ **Validation Error:** {validation_result.error_message}",
                        visible=True
                    ),
                    self._render_cases_list(),
                    self._render_case_stats(),
                    self._get_upload_status(),
                    gr.File(interactive=False)
                )
            
            if not case_summary.strip():
                return (
                    gr.Column(visible=True),
                    gr.Markdown(
                        "❌ **Validation Error:** Case summary is required",
                        visible=True
                    ),
                    self._render_cases_list(),
                    self._render_case_stats(),
                    self._get_upload_status(),
                    gr.File(interactive=False)
                )
            
            # Check visual marker availability
            if not await self._is_visual_marker_available(marker_color, marker_icon):
                return (
                    gr.Column(visible=True),
                    gr.Markdown(
                        "❌ **Validation Error:** This color and icon combination is already in use",
                        visible=True
                    ),
                    self._render_cases_list(),
                    self._render_case_stats(),
                    self._get_upload_status(),
                    gr.File(interactive=False)
                )
            
            # Create case via API
            case_data = {
                "case_name": case_name.strip(),
                "initial_summary": case_summary.strip(),
                "visual_marker": {
                    "color": marker_color,
                    "icon": marker_icon
                }
            }
            
            response = await self.api_client.post("/api/v1/cases/", case_data)
            
            if response.get("success", False):
                # Create case object
                case_info = response["case"]
                new_case = LegalCase(
                    case_id=case_info["case_id"],
                    case_name=case_info["case_name"],
                    initial_summary=case_info["initial_summary"],
                    visual_marker=VisualMarker(
                        color=case_info["visual_marker"]["color"],
                        icon=case_info["visual_marker"]["icon"]
                    ),
                    status=CaseStatus(case_info["status"]),
                    created_at=datetime.fromisoformat(case_info["created_at"])
                )
                
                # Add to local state
                self.cases[new_case.case_id] = new_case
                
                # Set as current case
                await self._set_current_case(new_case.case_id)
                
                self.logger.info(f"Case created successfully: {new_case.case_id}")
                
                return (
                    gr.Column(visible=False),  # Hide modal
                    gr.Markdown(visible=False),  # Clear validation message
                    self._render_cases_list(),
                    self._render_case_stats(),
                    self._get_upload_status(),
                    gr.File(interactive=True)  # Enable file upload
                )
            else:
                error_message = response.get("error", "Unknown error occurred")
                return (
                    gr.Column(visible=True),
                    gr.Markdown(
                        f"❌ **Creation Error:** {error_message}",
                        visible=True
                    ),
                    self._render_cases_list(),
                    self._render_case_stats(),
                    self._get_upload_status(),
                    gr.File(interactive=False)
                )
                
        except APIError as e:
            self.logger.error(f"API error during case creation: {str(e)}")
            return (
                gr.Column(visible=True),
                gr.Markdown(
                    f"❌ **API Error:** {str(e)}",
                    visible=True
                ),
                self._render_cases_list(),
                self._render_case_stats(),
                self._get_upload_status(),
                gr.File(interactive=False)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during case creation: {str(e)}")
            return (
                gr.Column(visible=True),
                gr.Markdown(
                    f"❌ **Error:** {str(e)}",
                    visible=True
                ),
                self._render_cases_list(),
                self._render_case_stats(),
                self._get_upload_status(),
                gr.File(interactive=False)
            )
    
    async def _handle_file_upload(
        self,
        files: List[Any]
    ) -> Tuple[str, str, str, str]:
        """
        Handle file upload for the current case.
        
        Args:
            files: List of uploaded files
            
        Returns:
            Tuple of updated UI components
        """
        if not self.current_case_id:
            return (
                "❌ No case selected for upload",
                "",
                self._render_cases_list(),
                self._render_case_stats()
            )
        
        if not files:
            return (
                "❌ No files selected",
                "",
                self._render_cases_list(),
                self._render_case_stats()
            )
        
        try:
            # Validate files
            validated_files = []
            total_size = 0
            
            for file in files:
                if hasattr(file, 'name') and hasattr(file, 'size'):
                    file_size = getattr(file, 'size', 0)
                    file_name = getattr(file, 'name', 'unknown')
                    
                    # Validate file type
                    if not (file_name.endswith('.pdf') or file_name.endswith('.txt')):
                        return (
                            f"❌ Invalid file type: {file_name}. Only PDF and TXT files are supported.",
                            "",
                            self._render_cases_list(),
                            self._render_case_stats()
                        )
                    
                    # Validate file size (max 50MB per file)
                    if file_size > 50 * 1024 * 1024:
                        return (
                            f"❌ File too large: {file_name}. Maximum file size is 50MB.",
                            "",
                            self._render_cases_list(),
                            self._render_case_stats()
                        )
                    
                    validated_files.append(file)
                    total_size += file_size
            
            # Check total upload size (max 500MB total)
            if total_size > 500 * 1024 * 1024:
                return (
                    f"❌ Total upload size too large: {format_file_size(total_size)}. Maximum total size is 500MB.",
                    "",
                    self._render_cases_list(),
                    self._render_case_stats()
                )
            
            # Start upload process
            upload_id = str(uuid.uuid4())
            upload_status = f"📤 Uploading {len(validated_files)} files ({format_file_size(total_size)})..."
            
            # Track upload
            for file in validated_files:
                file_upload = DocumentUpload(
                    upload_id=f"{upload_id}_{file.name}",
                    case_id=self.current_case_id,
                    filename=file.name,
                    file_size=getattr(file, 'size', 0),
                    status=DocumentUploadStatus.UPLOADING
                )
                self.active_uploads[file_upload.upload_id] = file_upload
            
            # Start async upload
            asyncio.create_task(self._execute_file_upload(validated_files, upload_id))
            
            progress_html = self._render_upload_progress()
            
            return (
                upload_status,
                progress_html,
                self._render_cases_list(),
                self._render_case_stats()
            )
            
        except Exception as e:
            self.logger.error(f"Error handling file upload: {str(e)}")
            return (
                f"❌ Upload error: {str(e)}",
                "",
                self._render_cases_list(),
                self._render_case_stats()
            )
    
    async def _execute_file_upload(self, files: List[Any], upload_id: str) -> None:
        """
        Execute the actual file upload process asynchronously.
        
        Args:
            files: List of files to upload
            upload_id: Unique upload operation ID
        """
        try:
            for file in files:
                file_upload_id = f"{upload_id}_{file.name}"
                
                if file_upload_id in self.active_uploads:
                    # Update status to uploading
                    self.active_uploads[file_upload_id].status = DocumentUploadStatus.UPLOADING
                    
                    # Read file content
                    file_content = file.read() if hasattr(file, 'read') else None
                    if not file_content:
                        # Handle file path case
                        with open(file, 'rb') as f:
                            file_content = f.read()
                    
                    # Prepare upload data
                    upload_data = {
                        "case_id": self.current_case_id,
                        "filename": file.name,
                        "file_size": len(file_content),
                        "upload_id": file_upload_id
                    }
                    
                    # Upload via API
                    response = await self.api_client.post_file(
                        "/api/v1/documents/upload",
                        files={"file": (file.name, file_content)},
                        data=upload_data
                    )
                    
                    if response.get("success", False):
                        # Update upload status
                        self.active_uploads[file_upload_id].status = DocumentUploadStatus.PROCESSING
                        self.active_uploads[file_upload_id].progress = 0.5
                        
                        self.logger.info(f"File uploaded successfully: {file.name}")
                    else:
                        # Handle upload failure
                        error_message = response.get("error", "Upload failed")
                        self.active_uploads[file_upload_id].status = DocumentUploadStatus.ERROR
                        self.active_uploads[file_upload_id].error_message = error_message
                        
                        self.logger.error(f"File upload failed: {file.name} - {error_message}")
                        
        except Exception as e:
            # Handle upload errors
            for file in files:
                file_upload_id = f"{upload_id}_{file.name}"
                if file_upload_id in self.active_uploads:
                    self.active_uploads[file_upload_id].status = DocumentUploadStatus.ERROR
                    self.active_uploads[file_upload_id].error_message = str(e)
            
            self.logger.error(f"Upload execution error: {str(e)}")
    
    async def _refresh_cases(self) -> Tuple[str, str, str]:
        """
        Refresh the cases list from the API.
        
        Returns:
            Tuple of updated UI components
        """
        try:
            await self._load_initial_cases()
            return (
                self._render_cases_list(),
                self._render_recent_cases_list(),
                self._render_case_stats()
            )
        except Exception as e:
            self.logger.error(f"Error refreshing cases: {str(e)}")
            return (
                self._render_cases_list(),
                self._render_recent_cases_list(),
                self._render_case_stats()
            )
    
    async def _archive_current_case(self) -> Tuple[str, str, str, gr.Button]:
        """
        Archive the current case.
        
        Returns:
            Tuple of updated UI components
        """
        if not self.current_case_id:
            return (
                self._render_cases_list(),
                self._render_recent_cases_list(),
                self._render_case_stats(),
                gr.Button(interactive=False)
            )
        
        try:
            response = await self.api_client.patch(
                f"/api/v1/cases/{self.current_case_id}/archive",
                {}
            )
            
            if response.get("success", False):
                # Update local state
                if self.current_case_id in self.cases:
                    self.cases[self.current_case_id].status = CaseStatus.ARCHIVED
                
                # Clear current case
                await self._set_current_case(None)
                
                self.logger.info(f"Case archived: {self.current_case_id}")
                
                return (
                    self._render_cases_list(),
                    self._render_recent_cases_list(),
                    self._render_case_stats(),
                    gr.Button(interactive=False)
                )
            else:
                self.logger.error(f"Failed to archive case: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error archiving case: {str(e)}")
        
        return (
            self._render_cases_list(),
            self._render_recent_cases_list(),
            self._render_case_stats(),
            gr.Button(interactive=bool(self.current_case_id))
        )
    
    async def _load_initial_cases(self) -> None:
        """Load initial cases from the API."""
        try:
            response = await self.api_client.get("/api/v1/cases/")
            
            if response.get("success", False):
                cases_data = response.get("cases", [])
                
                # Convert to case objects
                self.cases.clear()
                for case_data in cases_data:
                    case = LegalCase(
                        case_id=case_data["case_id"],
                        case_name=case_data["case_name"],
                        initial_summary=case_data["initial_summary"],
                        visual_marker=VisualMarker(
                            color=case_data["visual_marker"]["color"],
                            icon=case_data["visual_marker"]["icon"]
                        ),
                        status=CaseStatus(case_data["status"]),
                        document_count=case_data.get("document_count", 0),
                        processed_documents=case_data.get("processed_documents", 0),
                        created_at=datetime.fromisoformat(case_data["created_at"]),
                        updated_at=datetime.fromisoformat(case_data["updated_at"]),
                        total_size_bytes=case_data.get("total_size_bytes", 0)
                    )
                    
                    if case_data.get("last_activity"):
                        case.last_activity = datetime.fromisoformat(case_data["last_activity"])
                    
                    self.cases[case.case_id] = case
                
                self.logger.info(f"Loaded {len(self.cases)} cases")
                
        except Exception as e:
            self.logger.error(f"Error loading initial cases: {str(e)}")
    
    async def _is_visual_marker_available(self, color: str, icon: str) -> bool:
        """
        Check if a visual marker combination is available.
        
        Args:
            color: Color code
            icon: Icon identifier
            
        Returns:
            True if the combination is available
        """
        for case in self.cases.values():
            if (case.visual_marker.color == color and 
                case.visual_marker.icon == icon and 
                case.status != CaseStatus.ARCHIVED):
                return False
        return True
    
    async def _set_current_case(self, case_id: Optional[str]) -> None:
        """
        Set the current active case.
        
        Args:
            case_id: ID of the case to set as current, or None to clear
        """
        self.current_case_id = case_id
        
        # Notify other components of case change
        if self.case_change_callback:
            self.case_change_callback(case_id)
        
        self.logger.info(f"Current case changed to: {case_id}")
    
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
        """
        Render case statistics as HTML.
        
        Returns:
            HTML string for case statistics
        """
        if not self.cases:
            return "<div class='no-stats'>No statistics available</div>"
        
        total_cases = len(self.cases)
        active_cases = len([c for c in self.cases.values() if c.status == CaseStatus.ACTIVE])
        processing_cases = len([c for c in self.cases.values() if c.status == CaseStatus.PROCESSING])
        total_documents = sum(c.document_count for c in self.cases.values())
        total_size = sum(c.total_size_bytes for c in self.cases.values())
        
        return f"""
        <div class="case-statistics">
            <div class="stat-item">
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{active_cases}</div>
                <div class="stat-label">Active</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{processing_cases}</div>
                <div class="stat-label">Processing</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_documents}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{format_file_size(total_size)}</div>
                <div class="stat-label">Storage</div>
            </div>
        </div>
        """
    
    def _render_recent_cases_list(self) -> str:
        """
        Render recent cases list as HTML.
        
        Returns:
            HTML string for recent cases
        """
        # Get cases sorted by last activity
        recent_cases = sorted(
            self.cases.values(),
            key=lambda c: c.last_activity or c.created_at,
            reverse=True
        )[:10]  # Last 10 cases
        
        if not recent_cases:
            return "<div class='no-recent-cases'>No recent cases</div>"
        
        html_parts = ["<div class='recent-cases-list'>"]
        
        for case in recent_cases:
            last_activity = format_timestamp(case.last_activity or case.created_at)
            
            html_parts.append(f"""
            <div class="recent-case-item" onclick="selectCase('{case.case_id}')">
                <div class="recent-case-marker" style="background-color: {case.visual_marker.color};">
                    {case.visual_marker.icon}
                </div>
                <div class="recent-case-info">
                    <div class="recent-case-name">{case.case_name}</div>
                    <div class="recent-case-activity">{last_activity}</div>
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _render_upload_progress(self) -> str:
        """
        Render upload progress display.
        
        Returns:
            HTML string for upload progress
        """
        if not self.active_uploads:
            return ""
        
        html_parts = ["<div class='upload-progress-list'>"]
        
        for upload in self.active_uploads.values():
            status_icon = "📤" if upload.status == DocumentUploadStatus.UPLOADING else "⚙️"
            if upload.status == DocumentUploadStatus.COMPLETED:
                status_icon = "✅"
            elif upload.status == DocumentUploadStatus.ERROR:
                status_icon = "❌"
            
            progress_percent = int(upload.progress * 100)
            
            html_parts.append(f"""
            <div class="upload-item">
                <div class="upload-icon">{status_icon}</div>
                <div class="upload-info">
                    <div class="upload-filename">{upload.filename}</div>
                    <div class="upload-status">{upload.status.value.title()} ({progress_percent}%)</div>
                    {f'<div class="upload-error">{upload.error_message}</div>' if upload.error_message else ''}
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _get_status_icon(self, status: CaseStatus) -> str:
        """Get icon for case status."""
        icons = {
            CaseStatus.DRAFT: "📝",
            CaseStatus.ACTIVE: "✅",
            CaseStatus.PROCESSING: "⚙️",
            CaseStatus.COMPLETE: "🎯",
            CaseStatus.ARCHIVED: "📦",
            CaseStatus.ERROR: "❌"
        }
        return icons.get(status, "❓")
    
    def _get_upload_status(self) -> str:
        """Get current upload status message."""
        if not self.current_case_id:
            return "Select a case to upload documents"
        
        current_case = self.cases.get(self.current_case_id)
        if not current_case:
            return "Selected case not found"
        
        return f"Ready to upload documents to: {current_case.case_name}"
    
    # WebSocket event handlers
    async def _handle_case_created(self, message: WebSocketMessage) -> None:
        """Handle case creation notification."""
        try:
            case_data = message.data.get("case", {})
            # Refresh cases list
            await self._load_initial_cases()
        except Exception as e:
            self.logger.error(f"Error handling case created event: {str(e)}")
    
    async def _handle_case_updated(self, message: WebSocketMessage) -> None:
        """Handle case update notification."""
        try:
            case_data = message.data.get("case", {})
            case_id = case_data.get("case_id")
            
            if case_id in self.cases:
                # Update case in local state
                self.cases[case_id].status = CaseStatus(case_data.get("status", "active"))
                self.cases[case_id].document_count = case_data.get("document_count", 0)
                self.cases[case_id].processed_documents = case_data.get("processed_documents", 0)
                self.cases[case_id].total_size_bytes = case_data.get("total_size_bytes", 0)
                
                if case_data.get("last_activity"):
                    self.cases[case_id].last_activity = datetime.fromisoformat(case_data["last_activity"])
                
        except Exception as e:
            self.logger.error(f"Error handling case updated event: {str(e)}")
    
    async def _handle_upload_progress(self, message: WebSocketMessage) -> None:
        """Handle document upload progress."""
        try:
            upload_id = message.data.get("upload_id")
            progress = message.data.get("progress", 0.0)
            status = message.data.get("status", "uploading")
            
            if upload_id in self.active_uploads:
                self.active_uploads[upload_id].progress = progress
                self.active_uploads[upload_id].status = DocumentUploadStatus(status)
                
        except Exception as e:
            self.logger.error(f"Error handling upload progress: {str(e)}")
    
    async def _handle_document_processed(self, message: WebSocketMessage) -> None:
        """Handle document processing completion."""
        try:
            upload_id = message.data.get("upload_id")
            success = message.data.get("success", False)
            
            if upload_id in self.active_uploads:
                if success:
                    self.active_uploads[upload_id].status = DocumentUploadStatus.COMPLETED
                    self.active_uploads[upload_id].progress = 1.0
                    self.active_uploads[upload_id].completed_at = datetime.now(timezone.utc)
                else:
                    self.active_uploads[upload_id].status = DocumentUploadStatus.ERROR
                    self.active_uploads[upload_id].error_message = message.data.get("error", "Processing failed")
                
                # Update case statistics
                case_id = self.active_uploads[upload_id].case_id
                if case_id in self.cases:
                    await self._refresh_case_stats(case_id)
                
        except Exception as e:
            self.logger.error(f"Error handling document processed event: {str(e)}")
    
    async def _refresh_case_stats(self, case_id: str) -> None:
        """Refresh statistics for a specific case."""
        try:
            response = await self.api_client.get(f"/api/v1/cases/{case_id}")
            
            if response.get("success", False) and case_id in self.cases:
                case_data = response["case"]
                self.cases[case_id].document_count = case_data.get("document_count", 0)
                self.cases[case_id].processed_documents = case_data.get("processed_documents", 0)
                self.cases[case_id].total_size_bytes = case_data.get("total_size_bytes", 0)
                
        except Exception as e:
            self.logger.error(f"Error refreshing case stats: {str(e)}")
    
    def set_case_change_callback(self, callback: Callable[[Optional[str]], None]) -> None:
        """
        Set callback function for case change events.
        
        Args:
            callback: Function to call when case changes
        """
        self.case_change_callback = callback
    
    def get_current_case_id(self) -> Optional[str]:
        """
        Get the current case ID.
        
        Returns:
            Current case ID or None
        """
        return self.current_case_id


def create_sidebar() -> gr.Column:
    """
    Factory function to create a sidebar component.
    
    Returns:
        Configured CaseSidebar Gradio component
    """
    sidebar = CaseSidebar()
    return sidebar.create_component()


# Export for use in main.py
__all__ = ["CaseSidebar", "create_sidebar", "CaseStatus", "VisualMarker", "LegalCase"]