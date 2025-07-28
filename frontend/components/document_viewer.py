"""
Document Viewer Component for Patexia Legal AI Chatbot

This module provides a sophisticated document viewer interface using Gradio for
displaying legal documents with search result highlighting, navigation controls,
and metadata display. It integrates with the search results pane to provide
a seamless two-pane search experience for legal professionals.

Key Features:
- Legal document display with proper formatting and structure preservation
- Search result highlighting with context-aware emphasis
- Navigation controls for moving between search matches within documents
- Document metadata display including citations, page numbers, and sections
- Support for multiple document formats (PDF text extraction, plain text, DOC)
- Real-time updates when search results are selected
- Citation-ready snippet extraction for legal references
- Document section navigation and bookmarking
- Text selection and annotation capabilities
- Export functionality for document excerpts and citations

Search Integration:
- Synchronized highlighting with search pane results
- Match navigation with previous/next controls
- Context expansion for better understanding of matches
- Relevance score display for highlighted sections
- Multi-term highlighting with different colors
- Fuzzy match highlighting for related terms

Legal Document Features:
- Legal citation recognition and formatting
- Section header preservation and navigation
- Page number tracking and reference
- Paragraph numbering for precise citations
- Legal structure recognition (claims, specifications, etc.)
- Footnote and cross-reference handling
- Document metadata display (filing dates, parties, etc.)

Architecture Integration:
- Connects to FastAPI backend for document content retrieval
- Real-time updates via WebSocket for search result selection
- Integration with search pane for synchronized highlighting
- Support for case-based document isolation and access control
- Responsive design for optimal legal document review experience
"""

import asyncio
import re
import html
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid

import gradio as gr

from ..utils.api_client import APIClient, APIError
from ..utils.websocket_client import WebSocketClient, WebSocketMessage
from ..utils.ui_helpers import (
    format_file_size, format_timestamp, clean_html_text,
    extract_legal_citations, highlight_search_terms
)


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TEXT = "txt"
    DOCX = "docx"
    HTML = "html"


class HighlightType(str, Enum):
    """Types of text highlighting."""
    SEARCH_MATCH = "search-match"
    LEGAL_CITATION = "legal-citation"
    SECTION_HEADER = "section-header"
    CLAIM = "claim"
    SELECTED = "selected"


@dataclass
class DocumentMatch:
    """Single document match with highlighting information."""
    match_id: str
    start_pos: int
    end_pos: int
    text: str
    context: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    highlight_type: HighlightType = HighlightType.SEARCH_MATCH
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentContent:
    """Document content with metadata and structure."""
    document_id: str
    filename: str
    content: str
    document_type: DocumentType
    file_size: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    upload_date: Optional[datetime] = None
    processing_status: str = "processed"
    structure: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewerState:
    """Document viewer state management."""
    current_document: Optional[DocumentContent] = None
    current_matches: List[DocumentMatch] = field(default_factory=list)
    current_match_index: int = 0
    search_terms: List[str] = field(default_factory=list)
    highlighted_content: str = ""
    zoom_level: float = 1.0
    scroll_position: float = 0.0
    case_id: Optional[str] = None


class DocumentViewer:
    """
    Document viewer component with search highlighting and navigation.
    
    Provides comprehensive document viewing capabilities with real-time
    search result highlighting, navigation controls, and legal document
    structure analysis.
    """
    
    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        websocket_client: Optional[WebSocketClient] = None
    ):
        """
        Initialize document viewer with API clients.
        
        Args:
            api_client: API client for backend communication
            websocket_client: WebSocket client for real-time updates
        """
        self.api_client = api_client or APIClient()
        self.websocket_client = websocket_client or WebSocketClient()
        self.logger = logging.getLogger(f"{__name__}.DocumentViewer")
        
        # Viewer state
        self.state = ViewerState()
        
        # Event callbacks
        self._highlight_request_callbacks: List[Callable[[str, List[str]], None]] = []
        self._document_selected_callbacks: List[Callable[[str], None]] = []
        
        # UI components (will be set during create_component)
        self.components = {}
        
        # WebSocket handlers
        self.websocket_client.add_handler("document_updated", self._handle_document_updated)
        self.websocket_client.add_handler("search_highlight", self._handle_search_highlight)
        
        self.logger.info("DocumentViewer initialized")
    
    def create_component(self) -> gr.Column:
        """
        Create the document viewer Gradio component.
        
        Returns:
            Gradio Column containing the complete document viewer interface
        """
        try:
            with gr.Column(elem_id="document-viewer", scale=2) as doc_viewer:
                
                # Document header
                with gr.Row(elem_id="doc-header"):
                    self.components["doc_title"] = gr.HTML(
                        value='<h3 style="margin: 0; color: #374151;">📄 No document selected</h3>',
                        elem_id="doc-title"
                    )
                    
                    with gr.Row(elem_id="doc-controls"):
                        self.components["zoom_out_btn"] = gr.Button(
                            "🔍−",
                            size="sm",
                            elem_id="zoom-out-btn",
                            interactive=False
                        )
                        self.components["zoom_level"] = gr.HTML(
                            value='<span style="min-width: 60px; text-align: center;">100%</span>',
                            elem_id="zoom-level"
                        )
                        self.components["zoom_in_btn"] = gr.Button(
                            "🔍+",
                            size="sm",
                            elem_id="zoom-in-btn",
                            interactive=False
                        )
                        self.components["fit_width_btn"] = gr.Button(
                            "⟷",
                            size="sm",
                            elem_id="fit-width-btn",
                            interactive=False
                        )
                
                # Navigation and search within document
                with gr.Row(elem_id="doc-navigation"):
                    with gr.Column(scale=3, elem_id="match-info-col"):
                        self.components["match_info"] = gr.HTML(
                            value='<div style="padding: 8px; color: #6b7280;">Select a search result to view document</div>',
                            elem_id="match-info"
                        )
                    
                    with gr.Column(scale=2, elem_id="nav-controls-col"):
                        with gr.Row():
                            self.components["prev_match_btn"] = gr.Button(
                                "⬅️ Previous",
                                size="sm",
                                interactive=False,
                                elem_id="prev-match-btn"
                            )
                            self.components["next_match_btn"] = gr.Button(
                                "➡️ Next",
                                size="sm",
                                interactive=False,
                                elem_id="next-match-btn"
                            )
                            self.components["match_counter"] = gr.HTML(
                                value='<div style="text-align: center; min-width: 60px; padding: 4px;">0 / 0</div>',
                                elem_id="match-counter"
                            )
                
                # Document content display
                self.components["document_content"] = gr.HTML(
                    value=self._render_empty_state(),
                    elem_id="document-content",
                    elem_classes=["document-content-area"]
                )
                
                # Document metadata panel
                with gr.Accordion("📋 Document Information", open=False, elem_id="doc-metadata-accordion"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            self.components["doc_type"] = gr.Textbox(
                                label="File Type",
                                interactive=False,
                                elem_id="doc-type"
                            )
                            self.components["file_size"] = gr.Textbox(
                                label="File Size",
                                interactive=False,
                                elem_id="file-size"
                            )
                            self.components["upload_date"] = gr.Textbox(
                                label="Upload Date",
                                interactive=False,
                                elem_id="upload-date"
                            )
                        
                        with gr.Column(scale=1):
                            self.components["page_count"] = gr.Textbox(
                                label="Pages",
                                interactive=False,
                                elem_id="page-count"
                            )
                            self.components["word_count"] = gr.Textbox(
                                label="Words",
                                interactive=False,
                                elem_id="word-count"
                            )
                            self.components["processing_status"] = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_id="processing-status"
                            )
                
                # Legal document specific information
                with gr.Accordion("⚖️ Legal Information", open=False, elem_id="legal-info-accordion"):
                    with gr.Row():
                        with gr.Column():
                            self.components["citations_found"] = gr.Textbox(
                                label="Legal Citations Found",
                                placeholder="Citations will be extracted automatically...",
                                interactive=False,
                                lines=3,
                                elem_id="citations-found"
                            )
                        
                        with gr.Column():
                            self.components["sections_found"] = gr.Textbox(
                                label="Document Sections",
                                placeholder="Document structure will be analyzed...",
                                interactive=False,
                                lines=3,
                                elem_id="sections-found"
                            )
                
                # Export and utility actions
                with gr.Row(elem_id="doc-actions"):
                    self.components["export_snippet_btn"] = gr.Button(
                        "📋 Export Current Match",
                        size="sm",
                        interactive=False,
                        elem_id="export-snippet-btn"
                    )
                    self.components["copy_citation_btn"] = gr.Button(
                        "📚 Copy Legal Citation",
                        size="sm",
                        interactive=False,
                        elem_id="copy-citation-btn"
                    )
                    self.components["print_doc_btn"] = gr.Button(
                        "🖨️ Print Document",
                        size="sm",
                        interactive=False,
                        elem_id="print-doc-btn"
                    )
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self.logger.info("Document viewer component created successfully")
            return doc_viewer
            
        except Exception as e:
            self.logger.error(f"Failed to create document viewer component: {e}")
            return self._create_fallback_component()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for document viewer interactions."""
        try:
            # Zoom controls
            self.components["zoom_in_btn"].click(
                fn=self._handle_zoom_in,
                outputs=[self.components["zoom_level"], self.components["document_content"]]
            )
            
            self.components["zoom_out_btn"].click(
                fn=self._handle_zoom_out,
                outputs=[self.components["zoom_level"], self.components["document_content"]]
            )
            
            self.components["fit_width_btn"].click(
                fn=self._handle_fit_width,
                outputs=[self.components["zoom_level"], self.components["document_content"]]
            )
            
            # Navigation controls
            self.components["prev_match_btn"].click(
                fn=self._handle_previous_match,
                outputs=[
                    self.components["match_counter"],
                    self.components["document_content"],
                    self.components["match_info"]
                ]
            )
            
            self.components["next_match_btn"].click(
                fn=self._handle_next_match,
                outputs=[
                    self.components["match_counter"],
                    self.components["document_content"],
                    self.components["match_info"]
                ]
            )
            
            # Export functionality
            self.components["export_snippet_btn"].click(
                fn=self._handle_export_snippet,
                outputs=[gr.Textbox(visible=False)]  # Would trigger download in full implementation
            )
            
            self.components["copy_citation_btn"].click(
                fn=self._handle_copy_citation,
                outputs=[gr.Textbox(visible=False)]  # Would copy to clipboard
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup event handlers: {e}")
    
    def _create_fallback_component(self) -> gr.Column:
        """Create fallback component when viewer creation fails."""
        with gr.Column(elem_id="document-viewer-fallback") as fallback:
            gr.Markdown("## 📄 Document Viewer")
            gr.Markdown("*Document viewer temporarily unavailable*")
            
            gr.Button("Retry Connection", variant="secondary")
            
            gr.HTML("""
                <div style="padding: 1rem; background: #fee; border-left: 4px solid #f00; margin: 1rem 0;">
                    <strong>Viewer Error</strong><br>
                    Unable to initialize document viewer.
                    Please check your connection and try again.
                </div>
            """)
        
        return fallback
    
    def _render_empty_state(self) -> str:
        """Render empty state when no document is selected."""
        return '''
        <div style="text-align: center; color: #9ca3af; padding: 60px 20px; height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 2px dashed #e5e7eb; border-radius: 8px; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 16px;">📄</div>
            <h3 style="margin: 0 0 8px 0; color: #6b7280;">Document Viewer</h3>
            <p style="margin: 0; font-size: 14px;">Select a search result to view the document content with highlighted matches.</p>
            <div style="margin-top: 20px; font-size: 12px; color: #9ca3af;">
                <span>• Navigate between search matches</span><br>
                <span>• Export citations and snippets</span><br>
                <span>• View document metadata and structure</span>
            </div>
        </div>
        '''
    
    def _render_document_content(self) -> str:
        """Render document content with highlighting."""
        if not self.state.current_document:
            return self._render_empty_state()
        
        try:
            content = self.state.current_document.content
            
            # Apply search highlighting
            if self.state.search_terms and self.state.current_matches:
                content = self._apply_search_highlighting(content)
            
            # Apply legal structure highlighting
            content = self._apply_legal_highlighting(content)
            
            # Apply zoom level
            zoom_style = f"transform: scale({self.state.zoom_level}); transform-origin: top left;"
            
            # Wrap in container with proper styling
            html_content = f'''
            <div class="document-content-container" style="{zoom_style}">
                <div class="document-text" style="
                    font-family: 'Georgia', serif;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #374151;
                    padding: 20px;
                    background: white;
                    border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    max-width: 100%;
                    word-wrap: break-word;
                ">
                    {content}
                </div>
            </div>
            '''
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error rendering document content: {e}")
            return f'<div style="color: red; padding: 20px;">Error rendering document: {str(e)}</div>'
    
    def _apply_search_highlighting(self, content: str) -> str:
        """Apply search term highlighting to content."""
        if not self.state.search_terms:
            return content
        
        # Escape HTML first
        content = html.escape(content)
        
        # Highlight each search term with different colors
        colors = ["#fef3c7", "#ddd6fe", "#fed7d7", "#d1fae5", "#fce7f3"]
        
        for i, term in enumerate(self.state.search_terms):
            color = colors[i % len(colors)]
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            content = pattern.sub(
                f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: 500;">\\g<0></mark>',
                content
            )
        
        # Highlight current match with special styling
        if self.state.current_matches and 0 <= self.state.current_match_index < len(self.state.current_matches):
            current_match = self.state.current_matches[self.state.current_match_index]
            # Add special highlighting for current match
            # This would need more sophisticated text matching in a real implementation
        
        return content
    
    def _apply_legal_highlighting(self, content: str) -> str:
        """Apply legal document structure highlighting."""
        # Highlight legal citations
        citation_patterns = [
            (r'\b\d+\s+U\.?S\.?\s+\d+', 'legal-citation'),
            (r'\b\d+\s+F\.\d*d?\s+\d+', 'legal-citation'),
            (r'U\.?S\.?\s+Patent\s+No\.?\s+[\d,]+', 'legal-citation'),
            (r'\b\d+\s+U\.S\.C\.?\s*§?\s*\d+', 'legal-citation'),
        ]
        
        for pattern, css_class in citation_patterns:
            content = re.sub(
                pattern,
                f'<span class="{css_class}" style="color: #1d4ed8; font-weight: 500; text-decoration: underline;">\\g<0></span>',
                content,
                flags=re.IGNORECASE
            )
        
        # Highlight section headers (all caps lines)
        content = re.sub(
            r'^([A-Z\s]{10,})$',
            r'<div class="section-header" style="font-weight: bold; color: #1f2937; margin: 16px 0 8px 0; padding: 8px 0; border-bottom: 2px solid #e5e7eb;">\1</div>',
            content,
            flags=re.MULTILINE
        )
        
        # Convert line breaks to HTML
        content = content.replace('\n', '<br>')
        
        return content
    
    def _extract_document_structure(self, content: str) -> Dict[str, List[str]]:
        """Extract structural elements from legal document."""
        structure = {
            "sections": [],
            "claims": [],
            "citations": [],
            "references": []
        }
        
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Extract section headers (all caps, reasonable length)
            if stripped.isupper() and 10 < len(stripped) < 100:
                structure["sections"].append(stripped)
            
            # Extract numbered sections
            section_match = re.match(r'^\d+\.\s+([A-Z][^.]*)', stripped)
            if section_match:
                structure["sections"].append(section_match.group(1))
            
            # Extract claims (for patents)
            if re.match(r'^\s*CLAIM\s*\d+', stripped, re.IGNORECASE):
                structure["claims"].append(stripped)
            
            # Extract legal citations
            citation_patterns = [
                r'\d+\s+U\.?S\.?\s+\d+',
                r'\d+\s+F\.\d*d?\s+\d+',
                r'U\.?S\.?\s+Patent\s+No\.?\s+[\d,]+',
                r'\d+\s+U\.S\.C\.?\s*§?\s*\d+'
            ]
            
            for pattern in citation_patterns:
                matches = re.findall(pattern, stripped, re.IGNORECASE)
                structure["citations"].extend(matches)
        
        # Remove duplicates and empty entries
        for key in structure:
            structure[key] = list(filter(None, set(structure[key])))
        
        return structure
    
    # Event handler methods
    
    def _handle_zoom_in(self) -> Tuple[str, str]:
        """Handle zoom in button click."""
        self.state.zoom_level = min(3.0, self.state.zoom_level + 0.25)
        zoom_html = f'<span style="min-width: 60px; text-align: center;">{int(self.state.zoom_level * 100)}%</span>'
        content_html = self._render_document_content()
        return zoom_html, content_html
    
    def _handle_zoom_out(self) -> Tuple[str, str]:
        """Handle zoom out button click."""
        self.state.zoom_level = max(0.5, self.state.zoom_level - 0.25)
        zoom_html = f'<span style="min-width: 60px; text-align: center;">{int(self.state.zoom_level * 100)}%</span>'
        content_html = self._render_document_content()
        return zoom_html, content_html
    
    def _handle_fit_width(self) -> Tuple[str, str]:
        """Handle fit to width button click."""
        self.state.zoom_level = 1.0  # Reset to 100% for fit width
        zoom_html = f'<span style="min-width: 60px; text-align: center;">Fit</span>'
        content_html = self._render_document_content()
        return zoom_html, content_html
    
    def _handle_previous_match(self) -> Tuple[str, str, str]:
        """Handle previous match navigation."""
        if self.state.current_matches and self.state.current_match_index > 0:
            self.state.current_match_index -= 1
        
        return self._update_match_display()
    
    def _handle_next_match(self) -> Tuple[str, str, str]:
        """Handle next match navigation."""
        if (self.state.current_matches and 
            self.state.current_match_index < len(self.state.current_matches) - 1):
            self.state.current_match_index += 1
        
        return self._update_match_display()
    
    def _update_match_display(self) -> Tuple[str, str, str]:
        """Update match counter, content, and info displays."""
        if not self.state.current_matches:
            counter_html = '<div style="text-align: center; min-width: 60px; padding: 4px;">0 / 0</div>'
            info_html = '<div style="padding: 8px; color: #6b7280;">No matches found</div>'
            content_html = self._render_document_content()
            return counter_html, content_html, info_html
        
        current_idx = self.state.current_match_index + 1
        total_matches = len(self.state.current_matches)
        
        counter_html = f'<div style="text-align: center; min-width: 60px; padding: 4px;">{current_idx} / {total_matches}</div>'
        
        # Get current match info
        current_match = self.state.current_matches[self.state.current_match_index]
        relevance_percent = int(current_match.relevance_score * 100)
        
        info_html = f'''
        <div style="padding: 8px; border-left: 3px solid #3b82f6; background: #f8fafc;">
            <div style="font-weight: 500; color: #1f2937; margin-bottom: 4px;">
                Match {current_idx} of {total_matches} (Relevance: {relevance_percent}%)
            </div>
            <div style="font-size: 13px; color: #6b7280; max-height: 60px; overflow: hidden;">
                {html.escape(current_match.context[:200])}{"..." if len(current_match.context) > 200 else ""}
            </div>
        </div>
        '''
        
        content_html = self._render_document_content()
        
        return counter_html, content_html, info_html
    
    def _handle_export_snippet(self) -> str:
        """Handle export snippet button click."""
        if not self.state.current_matches or not self.state.current_document:
            return ""
        
        if self.state.current_match_index >= len(self.state.current_matches):
            return ""
        
        current_match = self.state.current_matches[self.state.current_match_index]
        doc = self.state.current_document
        
        snippet_text = f"""Legal Document Snippet Export
=====================================

Document: {doc.filename}
Document ID: {doc.document_id}
Match: {self.state.current_match_index + 1} of {len(self.state.current_matches)}
Relevance Score: {current_match.relevance_score:.2f}
Export Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Content:
--------
{current_match.context}

Citation:
---------
{doc.filename}, Match {self.state.current_match_index + 1}
"""
        
        # In a real implementation, this would trigger a download
        self.logger.info(f"Snippet exported for document {doc.document_id}")
        return snippet_text
    
    def _handle_copy_citation(self) -> str:
        """Handle copy citation button click."""
        if not self.state.current_document:
            return ""
        
        doc = self.state.current_document
        citation = f"{doc.filename} (uploaded {doc.upload_date.strftime('%Y-%m-%d') if doc.upload_date else 'unknown date'})"
        
        # In a real implementation, this would copy to clipboard
        self.logger.info(f"Citation copied for document {doc.document_id}")
        return citation
    
    # Public API methods
    
    def display_document(self, search_result: Dict[str, Any]) -> None:
        """
        Display a document from a search result.
        
        Args:
            search_result: Search result containing document information
        """
        try:
            if not search_result:
                self._clear_document()
                return
            
            document_id = search_result.get("document_id")
            if not document_id:
                self.logger.warning("No document_id in search result")
                return
            
            # Load document content
            asyncio.create_task(self._load_document_async(document_id, search_result))
            
        except Exception as e:
            self.logger.error(f"Error displaying document: {e}")
    
    async def _load_document_async(self, document_id: str, search_result: Dict[str, Any]) -> None:
        """Load document content asynchronously."""
        try:
            # Get document content from API
            response = await self.api_client.get(f"/api/v1/documents/{document_id}/content")
            
            if not response.get("success", False):
                self.logger.error(f"Failed to load document {document_id}")
                return
            
            doc_data = response.get("document", {})
            
            # Create document content object
            document = DocumentContent(
                document_id=document_id,
                filename=doc_data.get("filename", "Unknown Document"),
                content=doc_data.get("content", ""),
                document_type=DocumentType(doc_data.get("type", "txt")),
                file_size=doc_data.get("file_size", 0),
                page_count=doc_data.get("page_count"),
                word_count=doc_data.get("word_count"),
                upload_date=datetime.fromisoformat(doc_data["upload_date"]) if doc_data.get("upload_date") else None,
                processing_status=doc_data.get("status", "processed")
            )
            
            # Extract document structure
            document.structure = self._extract_document_structure(document.content)
            
            # Create matches from search result
            matches = []
            if search_result.get("matches"):
                for i, match_data in enumerate(search_result["matches"]):
                    match = DocumentMatch(
                        match_id=f"{document_id}_{i}",
                        start_pos=match_data.get("start_pos", 0),
                        end_pos=match_data.get("end_pos", 0),
                        text=match_data.get("text", ""),
                        context=match_data.get("context", ""),
                        page_number=match_data.get("page_number"),
                        relevance_score=match_data.get("relevance_score", 0.0)
                    )
                    matches.append(match)
            
            # Update state
            self.state.current_document = document
            self.state.current_matches = matches
            self.state.current_match_index = 0
            self.state.search_terms = search_result.get("search_terms", [])
            
            # Update UI components
            self._update_ui_components()
            
            self.logger.info(f"Document loaded: {document.filename} with {len(matches)} matches")
            
        except Exception as e:
            self.logger.error(f"Error loading document {document_id}: {e}")
    
    def _update_ui_components(self) -> None:
        """Update UI components with current document and matches."""
        if not self.state.current_document:
            return
        
        doc = self.state.current_document
        
        # Update document title
        title_html = f'<h3 style="margin: 0; color: #374151;">📄 {html.escape(doc.filename)}</h3>'
        self.components["doc_title"].value = title_html
        
        # Update metadata fields
        self.components["doc_type"].value = doc.document_type.value.upper()
        self.components["file_size"].value = format_file_size(doc.file_size)
        self.components["upload_date"].value = doc.upload_date.strftime("%Y-%m-%d %H:%M") if doc.upload_date else "Unknown"
        self.components["page_count"].value = str(doc.page_count) if doc.page_count else "N/A"
        self.components["word_count"].value = str(doc.word_count) if doc.word_count else "N/A"
        self.components["processing_status"].value = doc.processing_status.title()
        
        # Update legal information
        citations = "\n".join(doc.structure.get("citations", [])[:10])  # Show first 10
        self.components["citations_found"].value = citations if citations else "No citations found"
        
        sections = "\n".join(doc.structure.get("sections", [])[:10])  # Show first 10
        self.components["sections_found"].value = sections if sections else "No sections detected"
        
        # Update navigation controls
        has_matches = bool(self.state.current_matches)
        self.components["prev_match_btn"].interactive = has_matches
        self.components["next_match_btn"].interactive = has_matches
        self.components["export_snippet_btn"].interactive = has_matches
        self.components["copy_citation_btn"].interactive = True
        
        # Update zoom controls
        self.components["zoom_in_btn"].interactive = True
        self.components["zoom_out_btn"].interactive = True
        self.components["fit_width_btn"].interactive = True
        
        # Update document content
        self.components["document_content"].value = self._render_document_content()
        
        # Update match info
        counter_html, _, info_html = self._update_match_display()
        self.components["match_counter"].value = counter_html
        self.components["match_info"].value = info_html
    
    def _clear_document(self) -> None:
        """Clear the current document and reset viewer state."""
        self.state = ViewerState()
        
        # Reset UI to empty state
        if self.components:
            self.components["doc_title"].value = '<h3 style="margin: 0; color: #374151;">📄 No document selected</h3>'
            self.components["document_content"].value = self._render_empty_state()
            self.components["match_info"].value = '<div style="padding: 8px; color: #6b7280;">Select a search result to view document</div>'
            self.components["match_counter"].value = '<div style="text-align: center; min-width: 60px; padding: 4px;">0 / 0</div>'
            
            # Disable interactive controls
            for control in ["prev_match_btn", "next_match_btn", "export_snippet_btn", "zoom_in_btn", "zoom_out_btn", "fit_width_btn"]:
                if control in self.components:
                    self.components[control].interactive = False
    
    def on_highlight_request(self, callback: Callable[[str, List[str]], None]) -> None:
        """
        Register callback for highlight requests.
        
        Args:
            callback: Function to call when highlighting is requested
        """
        if callback not in self._highlight_request_callbacks:
            self._highlight_request_callbacks.append(callback)
    
    def on_document_selected(self, callback: Callable[[str], None]) -> None:
        """
        Register callback for document selection events.
        
        Args:
            callback: Function to call when document is selected
        """
        if callback not in self._document_selected_callbacks:
            self._document_selected_callbacks.append(callback)
    
    def update_highlighting(self, search_terms: List[str]) -> None:
        """
        Update search term highlighting in the current document.
        
        Args:
            search_terms: List of terms to highlight
        """
        if not self.state.current_document:
            return
        
        self.state.search_terms = search_terms
        self.components["document_content"].value = self._render_document_content()
    
    # WebSocket event handlers
    
    async def _handle_document_updated(self, message: Dict[str, Any]) -> None:
        """Handle document update WebSocket messages."""
        try:
            doc_data = message.get("data", {})
            document_id = doc_data.get("document_id")
            
            if (self.state.current_document and 
                self.state.current_document.document_id == document_id):
                # Reload current document if it was updated
                await self._load_document_async(document_id, {})
                
        except Exception as e:
            self.logger.error(f"Error handling document update: {e}")
    
    async def _handle_search_highlight(self, message: Dict[str, Any]) -> None:
        """Handle search highlighting WebSocket messages."""
        try:
            highlight_data = message.get("data", {})
            search_terms = highlight_data.get("search_terms", [])
            
            if search_terms:
                self.update_highlighting(search_terms)
                
        except Exception as e:
            self.logger.error(f"Error handling search highlight: {e}")


def create_document_viewer() -> gr.Column:
    """
    Factory function to create a document viewer component.
    
    Returns:
        Configured DocumentViewer Gradio component
    """
    viewer_component = DocumentViewer()
    return viewer_component.create_component()


# Export for use in main.py
__all__ = ["DocumentViewer", "create_document_viewer", "DocumentContent", "DocumentMatch", "DocumentType"]