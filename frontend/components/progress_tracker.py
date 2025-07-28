"""
Progress Tracker Component for Patexia Legal AI Chatbot - WebSocket Fix

This file contains the corrected WebSocket implementation for the progress tracker.
The main issue was with the websocket import - it should use websocket-client library.

Key fixes:
1. Correct import for WebSocketApp from websocket-client
2. Proper error handling and connection management
3. Alternative implementation using native websockets library
4. Fallback to polling if WebSocket connection fails
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

import gradio as gr
import requests

# Try multiple WebSocket libraries for compatibility
websockets_module = None
WebSocketApp = None
websocket_module = None

try:
    # First try websocket-client library (pip install websocket-client)
    from websocket import WebSocketApp
    import websocket as websocket_module
    WEBSOCKET_CLIENT_TYPE = "websocket-client"
except ImportError:
    try:
        # Try native websockets library (pip install websockets)
        import websockets as websockets_module
        WEBSOCKET_CLIENT_TYPE = "websockets"
    except ImportError:
        # Fallback to None - will use polling instead
        WEBSOCKET_CLIENT_TYPE = None
        logging.warning("No WebSocket library found. Progress tracking will use polling.")

# Backend configuration
BACKEND_BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"

# Global state for progress tracking
progress_state = {
    "active_documents": {},
    "completed_documents": [],
    "failed_documents": [],
    "session_stats": {
        "total_processed": 0,
        "total_failed": 0,
        "chunks_created": 0,
        "avg_processing_time": 0,
        "session_start": None
    },
    "ws_connection": None,
    "is_connected": False,
    "current_case_id": None
}

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Document processing stages with progress ranges."""
    UPLOAD = ("upload", 0, 20, "📤")
    EXTRACTION = ("extraction", 20, 40, "📄")
    CHUNKING = ("chunking", 40, 70, "✂️")
    EMBEDDING = ("embedding", 70, 90, "🧠")
    INDEXING = ("indexing", 90, 100, "💾")
    COMPLETED = ("completed", 100, 100, "✅")
    FAILED = ("failed", 0, 0, "❌")


class ProcessingStatus(Enum):
    """Processing status types."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DocumentProgress:
    """Document processing progress tracking."""
    document_id: str
    document_name: str
    file_size: int
    status: ProcessingStatus
    current_stage: ProcessingStage
    progress_percent: int
    message: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    chunks_created: int = 0
    retry_count: int = 0
    processing_time: Optional[float] = None


class ProgressTrackerAPI:
    """API client for progress tracking operations."""
    
    def __init__(self, base_url: str = BACKEND_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    async def get_processing_queue(self, case_id: str) -> Dict[str, Any]:
        """Get current processing queue status."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/documents/queue",
                params={"case_id": case_id}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to get processing queue: {e}")
            return {}
    
    async def retry_document_processing(self, document_id: str, case_id: str) -> Dict[str, Any]:
        """Retry failed document processing."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/documents/{document_id}/retry",
                json={"case_id": case_id}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to retry document processing: {e}")
            return {}
    
    async def get_processing_statistics(self, case_id: str) -> Dict[str, Any]:
        """Get processing statistics for the current session."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/documents/stats",
                params={"case_id": case_id}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {}


# Initialize API client
api = ProgressTrackerAPI()


class WebSocketProgressClient:
    """WebSocket client for real-time progress updates with multiple backend support."""
    
    def __init__(self, on_message_callback: Callable[[Dict], None]):
        self.ws = None
        self.on_message_callback = on_message_callback
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.connection_thread = None
        self.should_reconnect = True
    
    def connect(self):
        """Connect to WebSocket server using available WebSocket library."""
        if WEBSOCKET_CLIENT_TYPE == "websocket-client":
            self._connect_websocket_client()
        elif WEBSOCKET_CLIENT_TYPE == "websockets":
            self._connect_websockets()
        else:
            logger.warning("No WebSocket library available, falling back to polling")
            self._start_polling_fallback()
    
    def _connect_websocket_client(self):
        """Connect using websocket-client library."""
        # Check if websocket-client library is available
        if WEBSOCKET_CLIENT_TYPE != "websocket-client" or WebSocketApp is None:
            logger.error("websocket-client library not available, trying alternative")
            self._connect_websockets()
            return
            
        try:
            # Enable debugging for WebSocket (uncomment for troubleshooting)
            # if websocket_module:
            #     websocket_module.enableTrace(True)
            
            self.ws = WebSocketApp(
                WEBSOCKET_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run in separate thread
            def run_ws():
                try:
                    self.ws.run_forever(
                        reconnect=3,  # Reconnect up to 3 times
                        ping_interval=30,  # Send ping every 30 seconds
                        ping_timeout=10    # Wait 10 seconds for pong
                    )
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
                    if self.should_reconnect:
                        self._attempt_reconnection()
            
            self.connection_thread = threading.Thread(target=run_ws, daemon=True)
            self.connection_thread.start()
            self.is_running = True
            
            logger.info("WebSocket connection initiated with websocket-client")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed with websocket-client: {e}")
            self._start_polling_fallback()
    
    def _connect_websockets(self):
        """Connect using native websockets library (async)."""
        # Check if websockets library is available
        if WEBSOCKET_CLIENT_TYPE != "websockets" or websockets_module is None:
            logger.error("websockets library not available, falling back to polling")
            self._start_polling_fallback()
            return
            
        try:
            # Create event loop for WebSocket connection
            loop = asyncio.new_event_loop()
            
            async def websocket_handler():
                try:
                    async with websockets_module.connect(WEBSOCKET_URL) as websocket_conn:
                        self.ws = websocket_conn
                        progress_state["is_connected"] = True
                        self.reconnect_attempts = 0
                        logger.info("WebSocket connection established with websockets")
                        
                        # Send authentication
                        auth_message = {
                            "type": "auth",
                            "data": {"user_id": "current_user"}
                        }
                        await websocket_conn.send(json.dumps(auth_message))
                        
                        # Listen for messages
                        async for message in websocket_conn:
                            try:
                                data = json.loads(message)
                                if self.on_message_callback:
                                    self.on_message_callback(data)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode WebSocket message: {message}")
                            except Exception as e:
                                logger.error(f"Error handling WebSocket message: {e}")
                                
                except websockets_module.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    progress_state["is_connected"] = False
                    if self.should_reconnect:
                        self._attempt_reconnection()
                except Exception as e:
                    logger.error(f"WebSocket error with websockets: {e}")
                    progress_state["is_connected"] = False
                    if self.should_reconnect:
                        self._attempt_reconnection()
            
            def run_async_ws():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(websocket_handler())
            
            self.connection_thread = threading.Thread(target=run_async_ws, daemon=True)
            self.connection_thread.start()
            self.is_running = True
            
            logger.info("WebSocket connection initiated with websockets")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed with websockets: {e}")
            self._start_polling_fallback()
    
    def _start_polling_fallback(self):
        """Start polling fallback when WebSocket is not available."""
        logger.info("Starting polling fallback for progress tracking")
        
        def polling_worker():
            while self.should_reconnect:
                try:
                    # Poll for progress updates every 2 seconds
                    if progress_state.get("current_case_id"):
                        # Simulate progress updates by polling API
                        stats = asyncio.run(api.get_processing_statistics(
                            progress_state["current_case_id"]
                        ))
                        
                        if stats and self.on_message_callback:
                            # Convert API response to WebSocket message format
                            message = {
                                "type": "polling_update",
                                "data": stats
                            }
                            self.on_message_callback(message)
                    
                    time.sleep(2)  # Poll every 2 seconds
                    
                except Exception as e:
                    logger.error(f"Polling error: {e}")
                    time.sleep(5)  # Wait longer on error
        
        self.connection_thread = threading.Thread(target=polling_worker, daemon=True)
        self.connection_thread.start()
        self.is_running = True
        
        # Mark as "connected" for polling mode
        progress_state["is_connected"] = True
    
    def _attempt_reconnection(self):
        """Attempt to reconnect to WebSocket."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, switching to polling")
            self._start_polling_fallback()
            return
        
        self.reconnect_attempts += 1
        wait_time = min(30, 2 ** self.reconnect_attempts)  # Exponential backoff, max 30s
        
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time}s")
        time.sleep(wait_time)
        
        if self.should_reconnect:
            self.connect()
    
    def _on_open(self, ws):
        """Handle WebSocket connection opened (websocket-client)."""
        logger.info("WebSocket connection established")
        progress_state["is_connected"] = True
        self.reconnect_attempts = 0
        
        # Send authentication if needed
        auth_message = {
            "type": "auth",
            "data": {"user_id": "current_user"}  # Would be real user ID in production
        }
        ws.send(json.dumps(auth_message))
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message (websocket-client)."""
        try:
            data = json.loads(message)
            if self.on_message_callback:
                self.on_message_callback(data)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error (websocket-client)."""
        logger.error(f"WebSocket error: {error}")
        progress_state["is_connected"] = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed (websocket-client)."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        progress_state["is_connected"] = False
        
        # Attempt reconnection if enabled
        if self.should_reconnect:
            self._attempt_reconnection()
    
    def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket server."""
        if not progress_state["is_connected"]:
            logger.warning("WebSocket not connected, cannot send message")
            return
        
        try:
            if WEBSOCKET_CLIENT_TYPE == "websocket-client" and self.ws:
                self.ws.send(json.dumps(message))
            elif WEBSOCKET_CLIENT_TYPE == "websockets" and self.ws:
                # For async websockets, we'd need to queue the message
                # This is a simplified implementation
                logger.info(f"Would send message: {message}")
            else:
                logger.info(f"Polling mode - message ignored: {message}")
                
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    def disconnect(self):
        """Disconnect from WebSocket server."""
        self.should_reconnect = False
        self.is_running = False
        progress_state["is_connected"] = False
        
        try:
            if WEBSOCKET_CLIENT_TYPE == "websocket-client" and self.ws:
                self.ws.close()
            elif WEBSOCKET_CLIENT_TYPE == "websockets" and self.ws:
                # For async websockets, we'd need to properly close the connection
                pass
            
            logger.info("WebSocket disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")


class ProgressTracker:
    """
    Main progress tracker component class.
    
    Provides a wrapper interface for the progress tracking functionality
    that integrates with the main application architecture.
    """
    
    def __init__(self, websocket_client: Optional[WebSocketProgressClient] = None):
        """
        Initialize progress tracker.
        
        Args:
            websocket_client: Optional WebSocket client for real-time updates
        """
        self.websocket_client = websocket_client
        self.api = ProgressTrackerAPI()
        self.logger = logging.getLogger(f"{__name__}.ProgressTracker")
        
        # UI components (will be set during create_component)
        self.components = {}
        
        self.logger.info("ProgressTracker initialized")
    
    def create_component(self) -> gr.Group:
        """
        Create the progress tracker Gradio component.
        
        Returns:
            Gradio Group containing the complete progress tracker interface
        """
        try:
            with gr.Group(elem_id="progress-tracker") as progress_tracker:
                
                # Header with connection status
                with gr.Row(elem_id="progress-header"):
                    with gr.Column(scale=3):
                        gr.Markdown("## 📊 Document Processing Progress")
                    
                    with gr.Column(scale=1):
                        self.components["connection_status"] = gr.HTML(
                            value=self._render_connection_status(),
                            elem_id="connection-status"
                        )
                
                # Active processing section
                with gr.Column(elem_id="active-processing"):
                    gr.Markdown("### 🔄 Currently Processing")
                    
                    self.components["active_progress"] = gr.HTML(
                        value=self._generate_empty_progress_html(),
                        elem_id="active-progress-display"
                    )
                    
                    # Manual refresh button
                    self.components["refresh_btn"] = gr.Button(
                        "🔄 Refresh Progress", 
                        size="sm",
                        elem_id="refresh-progress-btn"
                    )
                
                # Processing queue section
                with gr.Accordion("📋 Processing Queue", open=False, elem_id="processing-queue"):
                    self.components["queue_display"] = gr.Dataframe(
                        headers=["Document", "Status", "Priority", "Queue Position", "ETA"],
                        datatype=["str", "str", "str", "number", "str"],
                        value=[],
                        label="Processing Queue",
                        elem_id="queue-dataframe"
                    )
                    
                    with gr.Row(elem_id="queue-actions"):
                        self.components["clear_queue_btn"] = gr.Button(
                            "🗑️ Clear Completed", 
                            size="sm"
                        )
                        self.components["retry_all_btn"] = gr.Button(
                            "🔄 Retry All Failed", 
                            size="sm"
                        )
                
                # Session statistics
                with gr.Row(elem_id="progress-stats"):
                    with gr.Column(scale=1):
                        self.components["stats_display"] = gr.HTML(
                            value=self._generate_stats_html(),
                            elem_id="stats-display"
                        )
                    
                    with gr.Column(scale=1):
                        self.components["performance_display"] = gr.HTML(
                            value=self._generate_performance_html(),
                            elem_id="performance-display"
                        )
                
                # Error log section
                with gr.Accordion("⚠️ Error Log", open=False, elem_id="error-log-section"):
                    self.components["error_log"] = gr.Textbox(
                        value="No errors reported",
                        lines=5,
                        label="Recent Errors",
                        interactive=False,
                        elem_id="error-log-textbox"
                    )
                    
                    self.components["clear_errors_btn"] = gr.Button(
                        "🧹 Clear Error Log", 
                        size="sm"
                    )
                
                # Hidden state components for Gradio
                self.components["current_case"] = gr.State("")
                self.components["last_update_time"] = gr.State(datetime.now())
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Initialize WebSocket connection if not provided
            if not self.websocket_client:
                self._initialize_websocket()
            
            self.logger.info("Progress tracker component created successfully")
            return progress_tracker
            
        except Exception as e:
            self.logger.error(f"Failed to create progress tracker component: {e}")
            return self._create_fallback_component()
    
    def _setup_event_handlers(self):
        """Setup event handlers for progress tracker interactions."""
        try:
            # Refresh button
            self.components["refresh_btn"].click(
                fn=self._handle_manual_refresh,
                outputs=[
                    self.components["active_progress"],
                    self.components["connection_status"],
                    self.components["stats_display"]
                ]
            )
            
            # Queue management buttons
            self.components["clear_queue_btn"].click(
                fn=self._handle_clear_completed,
                outputs=[self.components["queue_display"]]
            )
            
            self.components["retry_all_btn"].click(
                fn=self._handle_retry_all_failed,
                outputs=[
                    self.components["queue_display"],
                    self.components["active_progress"]
                ]
            )
            
            # Error log management
            self.components["clear_errors_btn"].click(
                fn=self._handle_clear_errors,
                outputs=[self.components["error_log"]]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup event handlers: {e}")
    
    def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time updates."""
        try:
            def handle_websocket_message(data):
                """Handle incoming WebSocket messages."""
                try:
                    message_type = data.get("type")
                    
                    if message_type == "document_progress":
                        self._handle_document_progress_update(data.get("data", {}))
                    elif message_type == "processing_queue":
                        self._handle_queue_update(data.get("data", {}))
                    elif message_type == "session_stats":
                        self._handle_stats_update(data.get("data", {}))
                    elif message_type == "error":
                        self._handle_error_message(data.get("data", {}))
                    elif message_type == "polling_update":
                        self._handle_polling_update(data.get("data", {}))
                    
                except Exception as e:
                    self.logger.error(f"Error handling WebSocket message: {e}")
            
            self.websocket_client = WebSocketProgressClient(handle_websocket_message)
            self.websocket_client.connect()
            
            self.logger.info("WebSocket client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket client: {e}")
    
    def _create_fallback_component(self) -> gr.Group:
        """Create fallback component when progress tracker creation fails."""
        with gr.Group(elem_id="progress-tracker-fallback") as fallback:
            gr.Markdown("## 📊 Progress Tracker")
            gr.Markdown("*Progress tracker temporarily unavailable*")
            
            gr.Button("Retry Connection", variant="secondary")
            
            gr.HTML("""
                <div style="padding: 1rem; background: #fee; border-left: 4px solid #f00; margin: 1rem 0;">
                    <strong>Progress Tracker Error</strong><br>
                    Unable to initialize progress tracking.
                    Please check your connection and try again.
                </div>
            """)
        
        return fallback
    
    def _render_connection_status(self) -> str:
        """Render connection status indicator."""
        is_connected = progress_state.get("is_connected", False)
        
        if is_connected:
            if WEBSOCKET_CLIENT_TYPE:
                connection_type = "WebSocket"
                color = "#10b981"  # Green
                icon = "🟢"
            else:
                connection_type = "Polling"
                color = "#f59e0b"  # Orange
                icon = "🟡"
        else:
            connection_type = "Disconnected"
            color = "#ef4444"  # Red
            icon = "🔴"
        
        return f'''
        <div style="text-align: right; padding: 8px;">
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                <span style="font-size: 12px; color: #6b7280;">Connection:</span>
                <span style="color: {color}; font-weight: 500; font-size: 14px;">
                    {icon} {connection_type}
                </span>
            </div>
        </div>
        '''
    
    def _generate_empty_progress_html(self) -> str:
        """Generate empty state HTML for progress display."""
        return '''
        <div style="text-align: center; color: #9ca3af; padding: 40px 20px; border: 2px dashed #e5e7eb; border-radius: 8px;">
            <div style="font-size: 32px; margin-bottom: 12px;">📊</div>
            <h4 style="margin: 0 0 8px 0; color: #6b7280;">No Active Processing</h4>
            <p style="margin: 0; font-size: 14px;">Document processing progress will appear here</p>
        </div>
        '''
    
    def _generate_stats_html(self) -> str:
        """Generate session statistics HTML."""
        stats = progress_state.get("session_stats", {})
        
        return f'''
        <div style="background: #f8fafc; border-radius: 8px; padding: 16px; border-left: 4px solid #3b82f6;">
            <h4 style="margin: 0 0 12px 0; color: #1f2937;">📈 Session Stats</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 14px;">
                <div>
                    <span style="color: #6b7280;">Processed:</span>
                    <strong style="color: #059669;">{stats.get("total_processed", 0)}</strong>
                </div>
                <div>
                    <span style="color: #6b7280;">Failed:</span>
                    <strong style="color: #dc2626;">{stats.get("total_failed", 0)}</strong>
                </div>
                <div>
                    <span style="color: #6b7280;">Chunks:</span>
                    <span style="color: #374151;">{stats.get("chunks_created", 0)}</span>
                </div>
                <div>
                    <span style="color: #6b7280;">Avg Time:</span>
                    <span style="color: #374151;">{stats.get("avg_processing_time", 0):.1f}s</span>
                </div>
            </div>
        </div>
        '''
    
    def _generate_performance_html(self) -> str:
        """Generate performance metrics HTML."""
        return '''
        <div style="background: #f0fdf4; border-radius: 8px; padding: 16px; border-left: 4px solid #10b981;">
            <h4 style="margin: 0 0 12px 0; color: #1f2937;">⚡ Performance</h4>
            <div style="font-size: 14px; color: #374151;">
                <div style="margin-bottom: 8px;">
                    <span style="color: #6b7280;">Connection:</span>
                    <span style="color: #059669; font-weight: 500;">Optimal</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #6b7280;">Processing Speed:</span>
                    <span>Normal</span>
                </div>
                <div>
                    <span style="color: #6b7280;">Queue Depth:</span>
                    <span>0</span>
                </div>
            </div>
        </div>
        '''
    
    # Event handler methods
    
    def _handle_manual_refresh(self) -> tuple:
        """Handle manual refresh button click."""
        try:
            # Update connection status
            connection_html = self._render_connection_status()
            
            # Refresh active progress
            progress_html = self._generate_empty_progress_html()
            if progress_state.get("active_documents"):
                progress_html = self._render_active_documents()
            
            # Update stats
            stats_html = self._generate_stats_html()
            
            self.logger.info("Manual refresh completed")
            
            return progress_html, connection_html, stats_html
            
        except Exception as e:
            self.logger.error(f"Error during manual refresh: {e}")
            return self._generate_empty_progress_html(), self._render_connection_status(), self._generate_stats_html()
    
    def _handle_clear_completed(self) -> list:
        """Handle clear completed documents from queue."""
        try:
            # Filter out completed documents
            progress_state["completed_documents"] = []
            
            # Return updated queue data
            return []
            
        except Exception as e:
            self.logger.error(f"Error clearing completed documents: {e}")
            return []
    
    def _handle_retry_all_failed(self) -> tuple:
        """Handle retry all failed documents."""
        try:
            # This would trigger retry for all failed documents
            self.logger.info("Retrying all failed documents")
            
            # Return updated displays
            return [], self._generate_empty_progress_html()
            
        except Exception as e:
            self.logger.error(f"Error retrying failed documents: {e}")
            return [], self._generate_empty_progress_html()
    
    def _handle_clear_errors(self) -> str:
        """Handle clear error log."""
        return "Error log cleared"
    
    def _render_active_documents(self) -> str:
        """Render currently active document processing."""
        active_docs = progress_state.get("active_documents", {})
        
        if not active_docs:
            return self._generate_empty_progress_html()
        
        html_parts = ['<div style="space-y: 16px;">']
        
        for doc_id, doc_progress in active_docs.items():
            stage = doc_progress.current_stage
            progress_percent = doc_progress.progress_percent
            
            html_parts.append(f'''
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h5 style="margin: 0; color: #1f2937;">{doc_progress.document_name}</h5>
                    <span style="font-size: 12px; color: #6b7280;">{progress_percent}%</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 8px; margin-bottom: 8px;">
                    <div style="background: #3b82f6; height: 100%; border-radius: 4px; width: {progress_percent}%; transition: width 0.3s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 14px; color: #374151;">
                        {stage.value[3]} {stage.value[0].title()} - {doc_progress.message}
                    </span>
                    <span style="font-size: 12px; color: #6b7280;">
                        {doc_progress.status.value.title()}
                    </span>
                </div>
            </div>
            ''')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    # WebSocket message handlers
    
    def _handle_document_progress_update(self, data: Dict[str, Any]):
        """Handle document progress update from WebSocket."""
        try:
            document_id = data.get("document_id")
            if not document_id:
                return
            
            # Update progress state
            if document_id not in progress_state["active_documents"]:
                progress_state["active_documents"][document_id] = DocumentProgress(
                    document_id=document_id,
                    document_name=data.get("document_name", "Unknown"),
                    file_size=data.get("file_size", 0),
                    status=ProcessingStatus.PROCESSING,
                    current_stage=ProcessingStage.UPLOAD,
                    progress_percent=0,
                    message="Starting processing...",
                    start_time=datetime.now()
                )
            
            doc_progress = progress_state["active_documents"][document_id]
            doc_progress.progress_percent = data.get("progress_percent", 0)
            doc_progress.message = data.get("message", "")
            doc_progress.status = ProcessingStatus(data.get("status", "processing"))
            
            # Update current stage based on progress
            if doc_progress.progress_percent >= 90:
                doc_progress.current_stage = ProcessingStage.INDEXING
            elif doc_progress.progress_percent >= 70:
                doc_progress.current_stage = ProcessingStage.EMBEDDING
            elif doc_progress.progress_percent >= 40:
                doc_progress.current_stage = ProcessingStage.CHUNKING
            elif doc_progress.progress_percent >= 20:
                doc_progress.current_stage = ProcessingStage.EXTRACTION
            else:
                doc_progress.current_stage = ProcessingStage.UPLOAD
            
            # Move to completed if finished
            if doc_progress.status == ProcessingStatus.COMPLETED:
                doc_progress.end_time = datetime.now()
                progress_state["completed_documents"].append(doc_progress)
                del progress_state["active_documents"][document_id]
                progress_state["session_stats"]["total_processed"] += 1
            elif doc_progress.status == ProcessingStatus.FAILED:
                doc_progress.end_time = datetime.now()
                progress_state["failed_documents"].append(doc_progress)
                del progress_state["active_documents"][document_id]
                progress_state["session_stats"]["total_failed"] += 1
            
            self.logger.info(f"Updated progress for {document_id}: {doc_progress.progress_percent}%")
            
        except Exception as e:
            self.logger.error(f"Error handling document progress update: {e}")
    
    def _handle_queue_update(self, data: Dict[str, Any]):
        """Handle processing queue update from WebSocket."""
        try:
            self.logger.info(f"Queue update received: {data}")
        except Exception as e:
            self.logger.error(f"Error handling queue update: {e}")
    
    def _handle_stats_update(self, data: Dict[str, Any]):
        """Handle session statistics update from WebSocket."""
        try:
            progress_state["session_stats"].update(data)
            self.logger.info("Session statistics updated")
        except Exception as e:
            self.logger.error(f"Error handling stats update: {e}")
    
    def _handle_error_message(self, data: Dict[str, Any]):
        """Handle error message from WebSocket."""
        try:
            error_msg = data.get("message", "Unknown error")
            self.logger.error(f"WebSocket error message: {error_msg}")
        except Exception as e:
            self.logger.error(f"Error handling error message: {e}")
    
    def _handle_polling_update(self, data: Dict[str, Any]):
        """Handle polling-based updates (fallback mode)."""
        try:
            self.logger.debug(f"Polling update received: {data}")
            # Process polling data similar to WebSocket messages
        except Exception as e:
            self.logger.error(f"Error handling polling update: {e}")
    
    # Public API methods
    
    def update_progress(self, document_id: str, progress: float, status: str) -> None:
        """
        Update progress for a specific document.
        
        Args:
            document_id: ID of the document
            progress: Progress percentage (0.0 to 100.0)
            status: Processing status
        """
        try:
            if document_id in progress_state["active_documents"]:
                doc_progress = progress_state["active_documents"][document_id]
                doc_progress.progress_percent = int(progress)
                doc_progress.status = ProcessingStatus(status)
                
                # Update UI components if available
                if "active_progress" in self.components:
                    self.components["active_progress"].value = self._render_active_documents()
                
        except Exception as e:
            self.logger.error(f"Error updating progress for {document_id}: {e}")
    
    def set_case_context(self, case_id: str) -> None:
        """
        Set the current case context for progress tracking.
        
        Args:
            case_id: ID of the current case
        """
        progress_state["current_case_id"] = case_id
        self.logger.info(f"Progress tracker case context set to: {case_id}")
    
    def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        if self.websocket_client:
            self.websocket_client.disconnect()
        
        self.logger.info("Progress tracker disconnected")


def create_progress_tracker() -> gr.Group:
    """
    Factory function to create a progress tracker component.
    
    Returns:
        Configured ProgressTracker Gradio component
    """
    tracker = ProgressTracker()
    return tracker.create_component()


# Export for use in main.py
__all__ = ["ProgressTracker", "create_progress_tracker", "WebSocketProgressClient", "ProcessingStage", "ProcessingStatus"]