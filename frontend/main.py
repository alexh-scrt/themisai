"""
Patexia Legal AI Chatbot - Gradio Frontend Application

This module serves as the main entry point for the Gradio-based web interface of the
Legal AI Chatbot system. It orchestrates the complete user interface including case
management, document processing, search functionality, and administration features.

Key Features:
- Two-pane legal search interface with document viewer
- Real-time WebSocket integration for progress tracking
- Case management sidebar with visual case switching
- Document upload and processing with batch support
- Administrative configuration panel with resource monitoring
- Search history and analytics dashboard
- Multi-user session management
- Responsive design with legal professional UX optimization

Architecture Integration:
- Connects to FastAPI backend via RESTful APIs
- Real-time communication through WebSocket connections
- Modular component architecture for maintainability
- Integrated error handling and user feedback systems
- Session state management across components
- Performance monitoring and analytics collection

UI Components:
- Case Navigation Sidebar: Visual case management and switching
- Search Pane: Hybrid search interface with advanced filtering
- Document Viewer: Integrated document display with highlighting
- Admin Panel: System configuration and resource monitoring
- Progress Tracker: Real-time operation status and feedback
- Upload Interface: Batch document processing with validation

Dependencies:
- gradio: Modern web interface framework
- asyncio: Asynchronous operations support
- logging: Structured application logging
- httpx: HTTP client for API communication
- websockets: Real-time communication support
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import uuid
import signal
import threading
import json

import gradio as gr
from gradio.themes import Soft, Base
import httpx

# Add project paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import frontend components
from frontend.components.sidebar import create_sidebar, SidebarComponent
from frontend.components.search_pane import create_search_pane, SearchPane
from frontend.components.document_viewer import create_document_viewer, DocumentViewer
from frontend.components.admin_panel import create_admin_panel, AdminPanel
from frontend.components.progress_tracker import create_progress_tracker, ProgressTracker

# Import utility modules
from frontend.utils.api_client import APIClient, APIError
from frontend.utils.websocket_client import WebSocketClient
from frontend.utils.session_manager import SessionManager
from frontend.utils.ui_helpers import (
    setup_custom_css, setup_custom_js, create_notification_system,
    format_error_message, validate_environment
)


# Global application state
APP_STATE = {
    "initialized": False,
    "api_client": None,
    "websocket_client": None,
    "session_manager": None,
    "components": {},
    "background_tasks": [],
    "startup_time": None
}

# Configuration constants
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860
DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_WEBSOCKET_URL = "ws://localhost:8000/ws"
APP_TITLE = "Patexia Legal AI Chatbot"
APP_DESCRIPTION = "AI-Powered Legal Document Analysis and Search"


class LegalAIChatbotApp:
    """
    Main application class for the Legal AI Chatbot frontend.
    
    Manages the complete Gradio application lifecycle including initialization,
    component orchestration, WebSocket connections, and shutdown procedures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Legal AI Chatbot application.
        
        Args:
            config: Optional configuration dictionary for custom settings
        """
        self.config = self._load_config(config)
        self.logger = self._setup_logging()
        self.demo = None
        self.api_client = None
        self.websocket_client = None
        self.session_manager = None
        self.components = {}
        self.background_tasks = []
        self.shutdown_event = threading.Event()
        
        # Track application state
        self.startup_time = datetime.now(timezone.utc)
        self.is_running = False
        self.connection_status = "disconnected"
        
        self.logger.info(
            "Legal AI Chatbot application initialized",
            backend_url=self.config["backend_url"],
            websocket_url=self.config["websocket_url"],
            debug_mode=self.config["debug"]
        )
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate application configuration."""
        default_config = {
            "backend_url": os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL),
            "websocket_url": os.getenv("WEBSOCKET_URL", DEFAULT_WEBSOCKET_URL),
            "server_host": os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST),
            "server_port": int(os.getenv("SERVER_PORT", DEFAULT_SERVER_PORT)),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            "max_concurrent_uploads": int(os.getenv("MAX_CONCURRENT_UPLOADS", "5")),
            "session_timeout_minutes": int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
            "enable_analytics": os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
            "theme": os.getenv("UI_THEME", "soft"),
            "custom_css_path": os.getenv("CUSTOM_CSS_PATH"),
            "custom_js_path": os.getenv("CUSTOM_JS_PATH")
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the application."""
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/frontend.log", mode="a") if Path("logs").exists() else logging.NullHandler()
            ]
        )
        
        logger = logging.getLogger(f"{__name__}.LegalAIChatbotApp")
        logger.info(f"Logging initialized at {self.config['log_level']} level")
        return logger
    
    async def initialize_clients(self) -> None:
        """Initialize API and WebSocket clients with backend connectivity."""
        try:
            # Initialize API client
            self.api_client = APIClient(
                base_url=self.config["backend_url"],
                timeout=30.0,
                max_retries=3
            )
            
            # Test backend connectivity
            health_response = await self.api_client.get("/health")
            if not health_response.get("healthy", False):
                raise ConnectionError("Backend health check failed")
            
            self.logger.info("API client connected successfully")
            
            # Initialize WebSocket client
            self.websocket_client = WebSocketClient(
                url=self.config["websocket_url"],
                auto_reconnect=True,
                max_reconnect_attempts=10
            )
            
            # Setup WebSocket event handlers
            self._setup_websocket_handlers()
            
            # Connect WebSocket
            await self.websocket_client.connect()
            self.connection_status = "connected"
            
            self.logger.info("WebSocket client connected successfully")
            
            # Initialize session manager
            self.session_manager = SessionManager(
                api_client=self.api_client,
                websocket_client=self.websocket_client,
                timeout_minutes=self.config["session_timeout_minutes"]
            )
            
            # Update global state
            APP_STATE.update({
                "api_client": self.api_client,
                "websocket_client": self.websocket_client,
                "session_manager": self.session_manager,
                "startup_time": self.startup_time
            })
            
        except Exception as e:
            self.logger.error(f"Failed to initialize clients: {e}")
            self.connection_status = "error"
            raise
    
    def _setup_websocket_handlers(self) -> None:
        """Setup WebSocket message handlers for real-time updates."""
        # Document processing progress
        self.websocket_client.add_handler(
            "document_progress",
            self._handle_document_progress
        )
        
        # Search operation updates
        self.websocket_client.add_handler(
            "search_progress",
            self._handle_search_progress
        )
        
        # System notifications
        self.websocket_client.add_handler(
            "system_notification",
            self._handle_system_notification
        )
        
        # Configuration updates
        self.websocket_client.add_handler(
            "config_update",
            self._handle_config_update
        )
        
        # Error notifications
        self.websocket_client.add_handler(
            "error",
            self._handle_error_notification
        )
    
    async def _handle_document_progress(self, message: Dict[str, Any]) -> None:
        """Handle document processing progress updates."""
        try:
            progress_data = message.get("data", {})
            document_id = progress_data.get("document_id")
            progress_percent = progress_data.get("progress", 0)
            status = progress_data.get("status", "processing")
            
            # Update progress tracker component
            if "progress_tracker" in self.components:
                await self.components["progress_tracker"].update_progress(
                    document_id, progress_percent, status
                )
                
            self.logger.debug(
                f"Document progress update: {document_id} - {progress_percent}% ({status})"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling document progress: {e}")
    
    async def _handle_search_progress(self, message: Dict[str, Any]) -> None:
        """Handle search operation progress updates."""
        try:
            search_data = message.get("data", {})
            query_id = search_data.get("query_id")
            progress = search_data.get("progress", 0)
            stage = search_data.get("stage", "searching")
            
            # Update search pane component
            if "search_pane" in self.components:
                await self.components["search_pane"].update_search_progress(
                    query_id, progress, stage
                )
                
        except Exception as e:
            self.logger.error(f"Error handling search progress: {e}")
    
    async def _handle_system_notification(self, message: Dict[str, Any]) -> None:
        """Handle system-wide notifications."""
        try:
            notification = message.get("data", {})
            level = notification.get("level", "info")
            title = notification.get("title", "System Notification")
            content = notification.get("message", "")
            
            # Display notification to user
            self.logger.info(f"System notification: {title} - {content}")
            
        except Exception as e:
            self.logger.error(f"Error handling system notification: {e}")
    
    async def _handle_config_update(self, message: Dict[str, Any]) -> None:
        """Handle configuration update notifications."""
        try:
            config_data = message.get("data", {})
            
            # Update admin panel if available
            if "admin_panel" in self.components:
                await self.components["admin_panel"].handle_config_update(config_data)
                
        except Exception as e:
            self.logger.error(f"Error handling config update: {e}")
    
    async def _handle_error_notification(self, message: Dict[str, Any]) -> None:
        """Handle error notifications from backend."""
        try:
            error_data = message.get("data", {})
            error_type = error_data.get("type", "unknown")
            error_message = error_data.get("message", "An error occurred")
            
            self.logger.warning(f"Backend error notification: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handling error notification: {e}")
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface with all components.
        
        Returns:
            Configured Gradio Blocks interface
        """
        # Determine theme
        if self.config["theme"] == "soft":
            theme = Soft()
        else:
            theme = Base()
        
        # Create main interface
        with gr.Blocks(
            title=APP_TITLE,
            theme=theme,
            css=self._load_custom_css(),
            js=self._load_custom_js(),
            head=self._create_head_content()
        ) as demo:
            
            # Application header
            with gr.Row(elem_id="app-header"):
                gr.HTML(f"""
                    <div class="app-header">
                        <h1>⚖️ {APP_TITLE}</h1>
                        <p class="app-description">{APP_DESCRIPTION}</p>
                        <div class="connection-status" id="connection-status">
                            <span class="status-indicator {self.connection_status}"></span>
                            Connection: {self.connection_status.title()}
                        </div>
                    </div>
                """)
            
            # Global notification area
            self.notification_area = gr.HTML(
                value="",
                visible=False,
                elem_id="notification-area"
            )
            
            # Main application tabs
            with gr.Tabs(elem_id="main-tabs") as main_tabs:
                
                # Legal Search Tab
                with gr.TabItem("🔍 Legal Search", elem_id="search-tab"):
                    with gr.Row(elem_id="search-interface", equal_height=True):
                        
                        # Case navigation sidebar
                        with gr.Column(scale=1, elem_id="sidebar-container"):
                            sidebar_component = SidebarComponent(
                                api_client=self.api_client,
                                websocket_client=self.websocket_client
                            )
                            sidebar = sidebar_component.create_component()
                            self.components["sidebar"] = sidebar_component
                        
                        # Search pane
                        with gr.Column(scale=2, elem_id="search-container"):
                            search_component = SearchPane(
                                api_client=self.api_client,
                                websocket_client=self.websocket_client,
                                session_manager=self.session_manager
                            )
                            search_pane = search_component.create_component()
                            self.components["search_pane"] = search_component
                        
                        # Document viewer pane
                        with gr.Column(scale=2, elem_id="viewer-container"):
                            viewer_component = DocumentViewer(
                                api_client=self.api_client,
                                websocket_client=self.websocket_client
                            )
                            document_viewer = viewer_component.create_component()
                            self.components["document_viewer"] = viewer_component
                
                # Document Upload Tab
                with gr.TabItem("📄 Document Upload", elem_id="upload-tab"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Case selector for upload
                            upload_case_selector = gr.Dropdown(
                                label="Select Case for Upload",
                                choices=[],
                                elem_id="upload-case-selector"
                            )
                        
                        with gr.Column(scale=2):
                            # File upload interface
                            file_upload = gr.Files(
                                label="Upload Legal Documents",
                                file_count="multiple",
                                file_types=[".pdf", ".txt"],
                                elem_id="document-upload"
                            )
                            
                            upload_button = gr.Button(
                                "🚀 Start Processing",
                                variant="primary",
                                elem_id="upload-button"
                            )
                    
                    # Progress tracking for uploads
                    progress_component = ProgressTracker(
                        websocket_client=self.websocket_client
                    )
                    progress_tracker = progress_component.create_component()
                    self.components["progress_tracker"] = progress_component
                
                # Analytics Dashboard Tab
                with gr.TabItem("📊 Analytics", elem_id="analytics-tab"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Search Analytics")
                            search_stats = gr.JSON(
                                label="Search Statistics",
                                elem_id="search-stats"
                            )
                        
                        with gr.Column():
                            gr.Markdown("## System Metrics")
                            system_metrics = gr.JSON(
                                label="System Performance",
                                elem_id="system-metrics"
                            )
                
                # Admin Panel Tab
                with gr.TabItem("⚙️ Administration", elem_id="admin-tab"):
                    admin_component = AdminPanel(
                        api_client=self.api_client,
                        websocket_client=self.websocket_client
                    )
                    admin_panel = admin_component.create_component()
                    self.components["admin_panel"] = admin_component
            
            # Setup component interactions
            self._setup_component_interactions(
                sidebar_component, search_component, viewer_component
            )
        
        return demo
    
    def _setup_component_interactions(
        self,
        sidebar: SidebarComponent,
        search_pane: SearchPane,
        document_viewer: DocumentViewer
    ) -> None:
        """Setup interactions between components."""
        # Case selection updates search context
        sidebar.on_case_selected(search_pane.set_case_context)
        
        # Search result selection updates document viewer
        search_pane.on_result_selected(document_viewer.display_document)
        
        # Document viewer requests update search highlighting
        document_viewer.on_highlight_request(search_pane.update_highlighting)
    
    def _load_custom_css(self) -> str:
        """Load custom CSS styles for the application."""
        css_content = """
        /* Legal AI Chatbot Custom Styles */
        .app-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: bold;
        }
        
        .app-description {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .connection-status {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .status-indicator.connected {
            background-color: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }
        
        .status-indicator.disconnected {
            background-color: #ef4444;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
        }
        
        .status-indicator.error {
            background-color: #f59e0b;
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
        }
        
        #search-interface {
            min-height: 600px;
        }
        
        #sidebar-container {
            border-right: 1px solid #e5e7eb;
            padding-right: 1rem;
        }
        
        #search-container {
            padding: 0 1rem;
        }
        
        #viewer-container {
            border-left: 1px solid #e5e7eb;
            padding-left: 1rem;
        }
        
        .search-result-item {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .search-result-item:hover {
            border-color: #3b82f6;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        }
        
        .progress-bar {
            background-color: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            height: 8px;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #3b82f6, #10b981);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        #notification-area {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        }
        
        .notification {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6;
        }
        """
        
        # Load additional custom CSS if specified
        if self.config.get("custom_css_path"):
            try:
                css_path = Path(self.config["custom_css_path"])
                if css_path.exists():
                    css_content += css_path.read_text()
            except Exception as e:
                self.logger.warning(f"Failed to load custom CSS: {e}")
        
        return css_content
    
    def _load_custom_js(self) -> str:
        """Load custom JavaScript for enhanced interactivity."""
        js_content = """
        // Legal AI Chatbot Custom JavaScript
        function updateConnectionStatus(status) {
            const indicator = document.getElementById('connection-status');
            if (indicator) {
                const statusSpan = indicator.querySelector('.status-indicator');
                if (statusSpan) {
                    statusSpan.className = `status-indicator ${status}`;
                    indicator.querySelector('span:last-child').textContent = 
                        `Connection: ${status.charAt(0).toUpperCase() + status.slice(1)}`;
                }
            }
        }
        
        function showNotification(title, message, level = 'info') {
            const notificationArea = document.getElementById('notification-area');
            if (!notificationArea) return;
            
            const notification = document.createElement('div');
            notification.className = `notification ${level}`;
            notification.innerHTML = `
                <h4>${title}</h4>
                <p>${message}</p>
                <button onclick="this.parentElement.remove()" style="float: right;">×</button>
            `;
            
            notificationArea.appendChild(notification);
            notificationArea.style.display = 'block';
            
            // Auto-remove after 5 seconds
            setTimeout(() => notification.remove(), 5000);
        }
        
        // WebSocket connection monitoring
        let wsCheckInterval;
        
        function startConnectionMonitoring() {
            wsCheckInterval = setInterval(() => {
                // This would be updated by the WebSocket client
                // Implementation depends on Gradio's WebSocket integration
            }, 5000);
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            startConnectionMonitoring();
            console.log('Legal AI Chatbot interface initialized');
        });
        """
        
        # Load additional custom JavaScript if specified
        if self.config.get("custom_js_path"):
            try:
                js_path = Path(self.config["custom_js_path"])
                if js_path.exists():
                    js_content += js_path.read_text()
            except Exception as e:
                self.logger.warning(f"Failed to load custom JavaScript: {e}")
        
        return js_content
    
    def _create_head_content(self) -> str:
        """Create additional HTML head content."""
        return """
        <meta name="description" content="AI-powered legal document analysis and search system">
        <meta name="keywords" content="legal AI, document analysis, legal search, patent analysis">
        <meta name="author" content="Patexia">
        <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
        """
    
    async def startup(self) -> None:
        """Application startup procedures."""
        try:
            self.logger.info("Starting Legal AI Chatbot application...")
            
            # Validate environment
            await self._validate_environment()
            
            # Initialize clients
            await self.initialize_clients()
            
            # Create interface
            self.demo = self.create_interface()
            
            # Mark as initialized
            APP_STATE["initialized"] = True
            self.is_running = True
            
            self.logger.info(
                "Legal AI Chatbot application started successfully",
                startup_time_ms=(datetime.now(timezone.utc) - self.startup_time).total_seconds() * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            raise
    
    async def _validate_environment(self) -> None:
        """Validate required environment and dependencies."""
        required_dirs = ["logs", "temp", "static"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
    
    async def shutdown(self) -> None:
        """Application shutdown procedures."""
        try:
            self.logger.info("Shutting down Legal AI Chatbot application...")
            
            self.is_running = False
            self.shutdown_event.set()
            
            # Close WebSocket connection
            if self.websocket_client:
                await self.websocket_client.disconnect()
            
            # Close API client
            if self.api_client:
                await self.api_client.close()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            self.logger.info("Application shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def launch(
        self,
        server_name: Optional[str] = None,
        server_port: Optional[int] = None,
        share: bool = False,
        debug: Optional[bool] = None,
        **kwargs
    ) -> None:
        """
        Launch the Gradio application.
        
        Args:
            server_name: Server host address
            server_port: Server port number
            share: Whether to create public link
            debug: Enable debug mode
            **kwargs: Additional Gradio launch parameters
        """
        # Use config defaults if not specified
        host = server_name or self.config["server_host"]
        port = server_port or self.config["server_port"]
        debug_mode = debug if debug is not None else self.config["debug"]
        
        try:
            # Run startup in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.startup())
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, initiating shutdown...")
                loop.run_until_complete(self.shutdown())
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Launch Gradio interface
            self.demo.launch(
                server_name=host,
                server_port=port,
                share=share,
                debug=debug_mode,
                favicon_path="static/favicon.ico",
                **kwargs
            )
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application launch failed: {e}")
            raise
        finally:
            # Ensure cleanup
            if loop.is_running():
                loop.run_until_complete(self.shutdown())


def create_interface(config: Optional[Dict[str, Any]] = None) -> gr.Blocks:
    """
    Factory function to create a Gradio interface.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Gradio Blocks interface
    """
    app = LegalAIChatbotApp(config)
    
    # Run startup synchronously for interface creation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(app.startup())
    
    return app.demo


def main():
    """Main entry point for the application."""
    app = LegalAIChatbotApp()
    app.launch()


if __name__ == "__main__":
    main()