"""
Search Pane Component for Patexia Legal AI Chatbot

This module implements the main search interface component using Gradio framework.
It provides a comprehensive search experience with real-time WebSocket updates,
hybrid search capabilities, and legal document processing integration.

Key Features:
- Hybrid semantic and keyword search interface
- Real-time search progress tracking via WebSocket
- Search history with replay functionality
- Advanced search options and filters
- Case-specific search scope management
- Search results with relevance scoring
- Integration with document viewer for highlighting
- Search analytics and performance monitoring

Architecture Integration:
- Uses APIClient for REST API communication with backend
- Integrates WebSocketClient for real-time progress updates
- Coordinates with DocumentViewer for result highlighting
- Manages search state and session persistence
- Provides search interface for case-based document management

UI Components:
- Search input with query suggestions
- Search type selection (semantic, keyword, hybrid)
- Advanced filters (date range, document type, relevance threshold)
- Search results list with metadata and scoring
- Search history panel with replay functionality
- Progress tracking for long-running searches
- Search analytics dashboard integration

Search Flow:
1. User enters search query with optional filters
2. Real-time query validation and suggestion
3. Search request sent to backend via API
4. Progress updates received via WebSocket
5. Results displayed with relevance scoring
6. Search history updated and persisted
7. Integration with document viewer for highlighting
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
    format_timestamp, validate_search_query,
    get_search_suggestions
)


class SearchType(str, Enum):
    """Types of search supported by the system."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchStatus(str, Enum):
    """Status of search operations."""
    IDLE = "idle"
    SEARCHING = "searching"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SearchResult:
    """Single search result with metadata."""
    document_id: str
    document_name: str
    document_type: str
    relevance_score: float
    snippet: str
    highlighted_text: str
    chunk_id: str
    page_number: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Search query with parameters and filters."""
    query_text: str
    search_type: SearchType
    case_id: Optional[str] = None
    document_types: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[datetime, datetime]] = None
    relevance_threshold: float = 0.7
    max_results: int = 20
    include_snippets: bool = True
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchSession:
    """Search session state and history."""
    session_id: str
    current_case_id: Optional[str] = None
    search_history: List[SearchQuery] = field(default_factory=list)
    current_query: Optional[SearchQuery] = None
    current_results: List[SearchResult] = field(default_factory=list)
    search_status: SearchStatus = SearchStatus.IDLE
    last_search_time: Optional[datetime] = None
    total_searches: int = 0
    
    def add_search(self, query: SearchQuery) -> None:
        """Add search query to history."""
        self.search_history.append(query)
        self.current_query = query
        self.total_searches += 1
        self.last_search_time = datetime.now(timezone.utc)
        
        # Keep only last 50 searches
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]


class SearchPane:
    """
    Main search interface component with real-time updates.
    
    Provides comprehensive search functionality with WebSocket integration,
    search history management, and advanced filtering capabilities.
    """
    
    def __init__(self):
        """Initialize search pane with API and WebSocket clients."""
        self.api_client = APIClient()
        self.websocket_client = WebSocketClient()
        self.session = SearchSession(session_id=str(uuid.uuid4()))
        self.logger = logging.getLogger(f"{__name__}.SearchPane")
        
        # UI state management
        self._search_suggestions: List[str] = []
        self._search_filters_visible: bool = False
        self._search_in_progress: bool = False
        
        # WebSocket message handlers
        self.websocket_client.add_handler("search_progress", self._handle_search_progress)
        self.websocket_client.add_handler("search_completed", self._handle_search_completed)
        self.websocket_client.add_handler("search_error", self._handle_search_error)
    
    def create_component(self) -> gr.Column:
        """
        Create the main search pane Gradio component.
        
        Returns:
            Gradio Column containing the complete search interface
        """
        with gr.Column(elem_id="search-pane", scale=2) as search_pane:
            # Search header
            gr.Markdown("## 🔍 Legal Document Search", elem_id="search-header")
            
            # Main search input
            with gr.Row(elem_id="search-input-row"):
                self.search_textbox = gr.Textbox(
                    placeholder="Enter your legal search query...",
                    label="Search Query",
                    lines=1,
                    max_lines=3,
                    elem_id="search-textbox",
                    scale=4
                )
                self.search_button = gr.Button(
                    "🔍 Search",
                    variant="primary",
                    elem_id="search-button",
                    scale=1
                )
            # Search type and options
            with gr.Row(elem_id="search-options-row"):
                self.search_type = gr.Radio(
                    choices=[
                        ("🧠 Semantic", SearchType.SEMANTIC.value),
                        ("📝 Keyword", SearchType.KEYWORD.value),
                        ("🔗 Hybrid", SearchType.HYBRID.value)
                    ],
                    value=SearchType.HYBRID.value,
                    label="Search Type",
                    elem_id="search-type",
                    scale=2
                )
                
                self.max_results_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Max Results",
                    elem_id="max-results-slider",
                    scale=1
                )
            
            # Advanced filters (collapsible)
            with gr.Accordion("Advanced Filters", open=False, elem_id="advanced-filters"):
                with gr.Row():
                    self.document_types = gr.CheckboxGroup(
                        choices=[
                            ("📄 PDF Documents", "pdf"),
                            ("📝 Text Files", "txt"),
                            ("📊 Legal Briefs", "legal_brief"),
                            ("⚖️ Case Files", "case_file"),
                            ("📋 Patents", "patent")
                        ],
                        label="Document Types",
                        elem_id="document-types-filter"
                    )
                    
                    with gr.Column():
                        self.relevance_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Relevance Threshold",
                            elem_id="relevance-threshold"
                        )
                        
                        self.include_snippets = gr.Checkbox(
                            value=True,
                            label="Include Text Snippets",
                            elem_id="include-snippets"
                        )
            
            # Search progress indicator
            self.search_progress = gr.Progress(
                track_tqdm=True,
            )
            
            # Search status and results count
            with gr.Row(elem_id="search-status-row", visible=False) as self.status_row:
                self.search_status = gr.Markdown(
                    "Ready to search",
                    elem_id="search-status"
                )
                self.results_count = gr.Markdown(
                    "",
                    elem_id="results-count"
                )
            
            # Search results display
            with gr.Column(elem_id="search-results-container"):
                self.search_results = gr.HTML(
                    value="<div class='no-results'>Enter a search query to begin</div>",
                    elem_id="search-results",
                    visible=True
                )
            
            # Search history panel
            with gr.Accordion("Search History", open=False, elem_id="search-history"):
                self.search_history_list = gr.HTML(
                    value="<div class='no-history'>No searches yet</div>",
                    elem_id="search-history-list"
                )
                
                with gr.Row():
                    self.clear_history_button = gr.Button(
                        "🗑️ Clear History",
                        size="sm",
                        elem_id="clear-history-button"
                    )
                    self.export_history_button = gr.Button(
                        "📤 Export History",
                        size="sm",
                        elem_id="export-history-button"
                    )
        
        # Event handlers
        self._setup_event_handlers()
        
        return search_pane
    
    def _setup_event_handlers(self) -> None:
        """Setup all event handlers for the search interface."""
        
        # Main search functionality
        self.search_button.click(
            fn=self._handle_search_click,
            inputs=[
                self.search_textbox,
                self.search_type,
                self.max_results_slider,
                self.document_types,
                self.relevance_threshold,
                self.include_snippets
            ],
            outputs=[
                self.search_results,
                self.search_status,
                self.results_count,
                self.status_row,
                self.search_history_list
            ]
        )
        
        # Enter key search
        self.search_textbox.submit(
            fn=self._handle_search_click,
            inputs=[
                self.search_textbox,
                self.search_type,
                self.max_results_slider,
                self.document_types,
                self.relevance_threshold,
                self.include_snippets
            ],
            outputs=[
                self.search_results,
                self.search_status,
                self.results_count,
                self.status_row,
                self.search_history_list
            ]
        )
        
        # Search suggestions on input
        self.search_textbox.change(
            fn=self._handle_query_change,
            inputs=[self.search_textbox],
            outputs=[]  # Updates handled internally
        )
        
        # History management
        self.clear_history_button.click(
            fn=self._clear_search_history,
            inputs=[],
            outputs=[self.search_history_list]
        )
        
        self.export_history_button.click(
            fn=self._export_search_history,
            inputs=[],
            outputs=[]
        )
    
    async def _handle_search_click(
        self,
        query_text: str,
        search_type: str,
        max_results: int,
        document_types: List[str],
        relevance_threshold: float,
        include_snippets: bool
    ) -> Tuple[str, str, str, gr.Row, str]:
        """
        Handle search button click or Enter key press.
        
        Args:
            query_text: The search query text
            search_type: Type of search (semantic, keyword, hybrid)
            max_results: Maximum number of results to return
            document_types: List of document types to filter
            relevance_threshold: Minimum relevance score threshold
            include_snippets: Whether to include text snippets
            
        Returns:
            Tuple of updated UI components
        """
        # Validate query
        if not query_text.strip():
            return (
                "<div class='search-error'>Please enter a search query</div>",
                "❌ Invalid query",
                "",
                gr.Row(visible=False),
                self._render_search_history()
            )
        
        # Validate query format
        validation_result = validate_search_query(query_text)
        if not validation_result.is_valid:
            return (
                f"<div class='search-error'>Query validation failed: {validation_result.error_message}</div>",
                "❌ Invalid query format",
                "",
                gr.Row(visible=False),
                self._render_search_history()
            )
        
        try:
            # Create search query
            search_query = SearchQuery(
                query_text=query_text.strip(),
                search_type=SearchType(search_type),
                case_id=self.session.current_case_id,
                document_types=document_types or [],
                relevance_threshold=relevance_threshold,
                max_results=max_results,
                include_snippets=include_snippets
            )
            
            # Add to session
            self.session.add_search(search_query)
            self.session.search_status = SearchStatus.SEARCHING
            
            # Update UI to show searching state
            searching_html = self._render_searching_state()
            status_text = "🔍 Searching..."
            status_row = gr.Row(visible=True)
            
            # Start async search
            asyncio.create_task(self._execute_search(search_query))
            
            return (
                searching_html,
                status_text,
                "",
                status_row,
                self._render_search_history()
            )
            
        except Exception as e:
            self.logger.error(f"Search click error: {str(e)}")
            self.session.search_status = SearchStatus.ERROR
            
            return (
                f"<div class='search-error'>Search failed: {str(e)}</div>",
                "❌ Search error",
                "",
                gr.Row(visible=False),
                self._render_search_history()
            )
    
    async def _execute_search(self, search_query: SearchQuery) -> None:
        """
        Execute the actual search operation asynchronously.
        
        Args:
            search_query: The search query to execute
        """
        try:
            # Prepare search request
            search_request = {
                "query": search_query.query_text,
                "search_type": search_query.search_type.value,
                "case_id": search_query.case_id,
                "filters": {
                    "document_types": search_query.document_types,
                    "relevance_threshold": search_query.relevance_threshold,
                    "max_results": search_query.max_results,
                    "include_snippets": search_query.include_snippets
                },
                "query_id": search_query.query_id
            }
            
            # Execute search via API
            start_time = time.time()
            response = await self.api_client.post("/api/v1/search/query", search_request)
            search_time = time.time() - start_time
            
            if response.get("success", False):
                # Process search results
                raw_results = response.get("results", [])
                search_results = [
                    SearchResult(
                        document_id=result["document_id"],
                        document_name=result["document_name"],
                        document_type=result["document_type"],
                        relevance_score=result["relevance_score"],
                        snippet=result.get("snippet", ""),
                        highlighted_text=result.get("highlighted_text", ""),
                        chunk_id=result["chunk_id"],
                        page_number=result.get("page_number"),
                        metadata=result.get("metadata", {})
                    )
                    for result in raw_results
                ]
                
                # Update session state
                self.session.current_results = search_results
                self.session.search_status = SearchStatus.COMPLETED
                
                # Log search analytics
                self.logger.info(
                    f"Search completed: query='{search_query.query_text}', "
                    f"results={len(search_results)}, time={search_time:.2f}s"
                )
                
                # Trigger UI update via WebSocket
                await self.websocket_client.send_message({
                    "type": "search_completed",
                    "data": {
                        "query_id": search_query.query_id,
                        "results_count": len(search_results),
                        "search_time": search_time,
                        "results": [self._serialize_search_result(r) for r in search_results]
                    }
                })
                
            else:
                # Handle search failure
                error_message = response.get("error", "Unknown search error")
                self.session.search_status = SearchStatus.ERROR
                
                await self.websocket_client.send_message({
                    "type": "search_error",
                    "data": {
                        "query_id": search_query.query_id,
                        "error": error_message
                    }
                })
                
        except APIError as e:
            self.logger.error(f"API error during search: {str(e)}")
            self.session.search_status = SearchStatus.ERROR
            
            await self.websocket_client.send_message({
                "type": "search_error",
                "data": {
                    "query_id": search_query.query_id,
                    "error": f"API error: {str(e)}"
                }
            })
            
        except Exception as e:
            self.logger.error(f"Unexpected error during search: {str(e)}")
            self.session.search_status = SearchStatus.ERROR
            
            await self.websocket_client.send_message({
                "type": "search_error",
                "data": {
                    "query_id": search_query.query_id,
                    "error": f"Search error: {str(e)}"
                }
            })
    
    def _handle_query_change(self, query_text: str) -> None:
        """
        Handle search query text changes for suggestions.
        
        Args:
            query_text: Current query text
        """
        if len(query_text.strip()) >= 3:
            # Get search suggestions asynchronously
            asyncio.create_task(self._get_search_suggestions(query_text))
    
    async def _get_search_suggestions(self, query_text: str) -> None:
        """
        Get search suggestions for the current query.
        
        Args:
            query_text: Query text to get suggestions for
        """
        try:
            suggestions = await get_search_suggestions(
                query_text, 
                case_id=self.session.current_case_id
            )
            self._search_suggestions = suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            self.logger.warning(f"Failed to get search suggestions: {str(e)}")
            self._search_suggestions = []
    
    def _clear_search_history(self) -> str:
        """
        Clear the search history.
        
        Returns:
            Updated search history HTML
        """
        self.session.search_history.clear()
        self.session.total_searches = 0
        return "<div class='no-history'>Search history cleared</div>"
    
    def _export_search_history(self) -> None:
        """Export search history to file."""
        try:
            # Create export data
            export_data = {
                "session_id": self.session.session_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_searches": self.session.total_searches,
                "search_history": [
                    {
                        "query_id": query.query_id,
                        "query_text": query.query_text,
                        "search_type": query.search_type.value,
                        "timestamp": query.timestamp.isoformat(),
                        "case_id": query.case_id,
                        "filters": {
                            "document_types": query.document_types,
                            "relevance_threshold": query.relevance_threshold,
                            "max_results": query.max_results
                        }
                    }
                    for query in self.session.search_history
                ]
            }
            
            # Save to file (this would typically trigger a download)
            filename = f"search_history_{self.session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Implementation would depend on deployment environment
            
            self.logger.info(f"Search history exported: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export search history: {str(e)}")
    
    async def _handle_search_progress(self, message: WebSocketMessage) -> None:
        """
        Handle search progress updates from WebSocket.
        
        Args:
            message: WebSocket message with progress data
        """
        try:
            data = message.data
            query_id = data.get("query_id")
            
            # Only handle progress for current query
            if (self.session.current_query and 
                self.session.current_query.query_id == query_id):
                
                progress = data.get("progress", 0)
                status = data.get("status", "Processing...")
                
                # Update progress indicator
                # This would integrate with Gradio's progress system
                
        except Exception as e:
            self.logger.error(f"Error handling search progress: {str(e)}")
    
    async def _handle_search_completed(self, message: WebSocketMessage) -> None:
        """
        Handle search completion from WebSocket.
        
        Args:
            message: WebSocket message with completion data
        """
        try:
            data = message.data
            query_id = data.get("query_id")
            
            # Only handle completion for current query
            if (self.session.current_query and 
                self.session.current_query.query_id == query_id):
                
                results_count = data.get("results_count", 0)
                search_time = data.get("search_time", 0)
                
                # Update UI with completed results
                self.session.search_status = SearchStatus.COMPLETED
                
                # This would trigger a UI update in the actual implementation
                
        except Exception as e:
            self.logger.error(f"Error handling search completion: {str(e)}")
    
    async def _handle_search_error(self, message: WebSocketMessage) -> None:
        """
        Handle search error from WebSocket.
        
        Args:
            message: WebSocket message with error data
        """
        try:
            data = message.data
            query_id = data.get("query_id")
            error = data.get("error", "Unknown error")
            
            # Only handle errors for current query
            if (self.session.current_query and 
                self.session.current_query.query_id == query_id):
                
                self.session.search_status = SearchStatus.ERROR
                
                # This would trigger a UI error display in the actual implementation
                
        except Exception as e:
            self.logger.error(f"Error handling search error: {str(e)}")
    
    def _render_searching_state(self) -> str:
        """
        Render the searching state HTML.
        
        Returns:
            HTML string for searching state
        """
        return """
        <div class="search-progress">
            <div class="progress-spinner"></div>
            <div class="progress-text">
                Searching documents...
                <br><small>This may take a few moments for large document sets</small>
            </div>
        </div>
        """
    
    def _render_search_results(self, results: List[SearchResult]) -> str:
        """
        Render search results as HTML.
        
        Args:
            results: List of search results to render
            
        Returns:
            HTML string for search results
        """
        if not results:
            return "<div class='no-results'>No results found for your query</div>"
        
        html_parts = ["<div class='search-results-list'>"]
        
        for i, result in enumerate(results):
            html_parts.append(f"""
            <div class="result-item" data-document-id="{result.document_id}">
                <div class="result-header">
                    <div class="result-title">{result.document_name}</div>
                    <div class="result-score">Score: {result.relevance_score:.2f}</div>
                </div>
                <div class="result-meta">
                    <span class="doc-type">{result.document_type}</span>
                    {f'<span class="page-num">Page {result.page_number}</span>' if result.page_number else ''}
                </div>
                <div class="result-snippet">{result.snippet}</div>
                <div class="result-actions">
                    <button onclick="viewDocument('{result.document_id}', '{result.chunk_id}')">
                        📄 View Document
                    </button>
                    <button onclick="findSimilar('{result.document_id}')">
                        🔍 Find Similar
                    </button>
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _render_search_history(self) -> str:
        """
        Render search history as HTML.
        
        Returns:
            HTML string for search history
        """
        if not self.session.search_history:
            return "<div class='no-history'>No searches yet</div>"
        
        html_parts = ["<div class='search-history-list'>"]
        
        # Show last 10 searches
        recent_searches = self.session.search_history[-10:]
        
        for query in reversed(recent_searches):
            timestamp = format_timestamp(query.timestamp)
            html_parts.append(f"""
            <div class="history-item" data-query-id="{query.query_id}">
                <div class="history-query">{query.query_text}</div>
                <div class="history-meta">
                    <span class="search-type">{query.search_type.value}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _serialize_search_result(self, result: SearchResult) -> Dict[str, Any]:
        """
        Serialize search result for WebSocket transmission.
        
        Args:
            result: Search result to serialize
            
        Returns:
            Serialized result dictionary
        """
        return {
            "document_id": result.document_id,
            "document_name": result.document_name,
            "document_type": result.document_type,
            "relevance_score": result.relevance_score,
            "snippet": result.snippet,
            "highlighted_text": result.highlighted_text,
            "chunk_id": result.chunk_id,
            "page_number": result.page_number,
            "created_at": result.created_at.isoformat(),
            "metadata": result.metadata
        }
    
    def set_current_case(self, case_id: Optional[str]) -> None:
        """
        Set the current case for search scope.
        
        Args:
            case_id: ID of the current case or None for global search
        """
        self.session.current_case_id = case_id
        self.logger.info(f"Search scope changed to case: {case_id}")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search statistics for the current session.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            "session_id": self.session.session_id,
            "total_searches": self.session.total_searches,
            "last_search_time": self.session.last_search_time.isoformat() if self.session.last_search_time else None,
            "current_status": self.session.search_status.value,
            "results_count": len(self.session.current_results),
            "history_count": len(self.session.search_history)
        }


def create_search_pane() -> gr.Column:
    """
    Factory function to create a search pane component.
    
    Returns:
        Configured SearchPane Gradio component
    """
    search_pane = SearchPane()
    return search_pane.create_component()


# Export for use in main.py
__all__ = ["SearchPane", "create_search_pane", "SearchType", "SearchStatus"]