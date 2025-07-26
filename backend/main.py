"""
FastAPI Application Entry Point for Patexia Legal AI Chatbot

This module serves as the main entry point for the FastAPI backend server,
providing a comprehensive legal AI platform for document processing and search.
It initializes all core components, configures middleware, and sets up routing
for the complete legal document management system.

Key Features:
- Production-ready FastAPI application with async lifecycle management
- Comprehensive middleware stack for security, CORS, logging, and error handling
- Hot-reload configuration management with real-time updates
- WebSocket connection pooling for multi-user real-time communication
- Database initialization and connection management (MongoDB, Weaviate)
- Resource monitoring and performance optimization
- Security headers and authentication middleware
- Request/response logging with correlation IDs
- Graceful startup and shutdown procedures

Architecture Components:
- API Gateway Layer: FastAPI with middleware and routing
- Service Layer: Business logic and workflow coordination
- Data Layer: MongoDB and Weaviate database connections
- AI Layer: Ollama integration for embedding and language models
- Real-time Layer: WebSocket manager for live updates
- Configuration Layer: Hot-reload settings management

Security Features:
- CORS configuration for cross-origin requests
- Security headers for XSS and CSRF protection
- Request rate limiting and throttling
- Error sanitization to prevent information leakage
- WebSocket authentication and authorization
- Comprehensive audit logging for compliance

Performance Optimizations:
- Async/await throughout for non-blocking operations
- Connection pooling for database and external services
- Request/response caching for frequent operations
- Resource monitoring with automatic alerting
- Graceful degradation under high load
"""

import asyncio
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException as FastAPIHTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import application components
from backend.app.api.routes import cases, documents, search, admin, websocket
from backend.app.api.middleware.cors import setup_cors_middleware
from backend.app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.error_handler import ErrorHandlerMiddleware
from backend.config.settings import get_settings
from backend.app.core.config_watcher import ConfigurationWatcher
from backend.app.core.database import DatabaseManager, get_database_manager
from backend.app.core.websocket_manager import WebSocketManager, get_websocket_manager
from backend.app.core.resource_monitor import ResourceMonitor, get_resource_monitor
from backend.app.core.ollama_client import OllamaClient, get_ollama_client
from backend.app.core.exceptions import (
    BaseCustomException, ErrorCode, ConfigurationError,
    DatabaseError, get_exception_response_data
)
from backend.app.utils.logging import get_logger, initialize_logging_from_settings
from backend.app.utils.security import SecurityHeaders

# Initialize logging as early as possible
initialize_logging_from_settings()
logger = get_logger(__name__)

# Global instances for lifecycle management
_database_manager: Optional[DatabaseManager] = None
_websocket_manager: Optional[WebSocketManager] = None
_resource_monitor: Optional[ResourceMonitor] = None
_config_watcher: Optional[ConfigurationWatcher] = None
_ollama_client: Optional[OllamaClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown procedures.
    
    Manages the complete lifecycle of the application including:
    - Service initialization and configuration
    - Database connections and schema validation
    - WebSocket manager setup and connection pooling
    - Resource monitoring and performance tracking
    - Hot-reload configuration management
    - Graceful shutdown and cleanup procedures
    """
    global _database_manager, _websocket_manager, _resource_monitor, _config_watcher, _ollama_client
    
    # Startup procedures
    logger.info("=== Patexia Legal AI Backend Starting Up ===")
    
    try:
        # 1. Load and validate configuration
        settings = get_settings()
        logger.info("Configuration loaded successfully", environment="development", debug_mode=settings.debug)
        
        # 2. Initialize Ollama client first (required for embeddings)
        logger.info("Initializing Ollama client...")
        _ollama_client = OllamaClient()
        await _ollama_client.initialize()
        logger.info("Ollama client initialized successfully")
        
        # 3. Initialize database connections
        logger.info("Initializing database connections...")
        _database_manager = DatabaseManager()
        await _database_manager.initialize()
        logger.info("Database connections established successfully")
        
        # 4. Initialize WebSocket manager
        logger.info("Initializing WebSocket manager...")
        _websocket_manager = WebSocketManager()
        await _websocket_manager.initialize()
        logger.info("WebSocket manager initialized successfully")
        
        # 5. Initialize resource monitoring
        logger.info("Initializing resource monitor...")
        _resource_monitor = ResourceMonitor()
        await _resource_monitor.start_monitoring()
        logger.info("Resource monitoring started successfully")
        
        # 6. Start configuration hot-reload watcher
        logger.info("Starting configuration watcher...")
        _config_watcher = ConfigurationWatcher()
        await _config_watcher.start()
        logger.info("Configuration watcher started successfully")
        
        # 7. Validate system readiness
        await _validate_system_readiness()
        
        logger.info("=== Patexia Legal AI Backend Started Successfully ===")
        
        # Application is ready - yield control
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", error=str(e), traceback=traceback.format_exc())
        raise
    
    finally:
        # Shutdown procedures
        logger.info("=== Patexia Legal AI Backend Shutting Down ===")
        
        # Stop configuration watcher
        if _config_watcher:
            try:
                await _config_watcher.stop()
                logger.info("Configuration watcher stopped")
            except Exception as e:
                logger.error(f"Error stopping configuration watcher: {e}")
        
        # Stop resource monitoring
        if _resource_monitor:
            try:
                await _resource_monitor.stop_monitoring()
                logger.info("Resource monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping resource monitor: {e}")
        
        # Disconnect all WebSocket connections
        if _websocket_manager:
            try:
                await _websocket_manager.disconnect_all()
                logger.info("All WebSocket connections closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket connections: {e}")
        
        # Close database connections
        if _database_manager:
            try:
                await _database_manager.close()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")
        
        # Close Ollama client
        if _ollama_client:
            try:
                await _ollama_client.close()
                logger.info("Ollama client closed")
            except Exception as e:
                logger.error(f"Error closing Ollama client: {e}")
        
        logger.info("=== Patexia Legal AI Backend Shutdown Complete ===")


async def _validate_system_readiness():
    """Validate that all system components are ready for operation."""
    logger.info("Validating system readiness...")
    
    # Validate database connections
    if not await _database_manager.health_check():
        raise DatabaseError("Database health check failed")
    
    # Validate Ollama availability
    if not await _ollama_client.health_check():
        raise ConfigurationError("Ollama service is not available")
    
    # Validate embedding model availability
    settings = get_settings()
    if not await _ollama_client.validate_model(settings.ollama_settings.embedding_model):
        logger.warning(f"Embedding model {settings.ollama_settings.embedding_model} not available, attempting to pull...")
        try:
            await _ollama_client.pull_model(settings.ollama_settings.embedding_model)
        except Exception as e:
            raise ConfigurationError(f"Failed to load embedding model: {e}")
    
    logger.info("System readiness validation completed successfully")


# Create FastAPI application
def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="Patexia Legal AI API",
        description="AI-powered legal document processing and search platform",
        version="1.0.0",
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Configure middleware
    configure_middleware(app)
    
    # Configure routes
    configure_routes(app)
    
    # Configure exception handlers
    configure_exception_handlers(app)
    
    return app


def configure_middleware(app: FastAPI) -> None:
    """Configure application middleware stack."""
    settings = get_settings()
    
    # 1. Trusted Host Middleware (security)
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
        )
    
    # 2. Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        security_headers = SecurityHeaders.get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    # 3. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:7860", "http://127.0.0.1:7860"],  # Gradio frontend
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # 4. GZip Compression Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 5. Custom Error Handler Middleware
    app.add_middleware(ErrorHandlerMiddleware)
    
    # 6. Request Logging Middleware
    app.add_middleware(LoggingMiddleware)
    
    logger.info("Middleware configuration completed")


def configure_routes(app: FastAPI) -> None:
    """Configure application routes and API endpoints."""
    
    # Health check endpoint
    @app.get("/health", tags=["system"], include_in_schema=False)
    async def health_check():
        """System health check endpoint."""
        try:
            # Check database health
            db_healthy = await _database_manager.health_check() if _database_manager else False
            
            # Check Ollama health
            ollama_healthy = await _ollama_client.health_check() if _ollama_client else False
            
            # Check WebSocket manager
            ws_healthy = _websocket_manager.is_healthy() if _websocket_manager else False
            
            return {
                "status": "healthy" if all([db_healthy, ollama_healthy, ws_healthy]) else "degraded",
                "timestamp": "2025-01-28T00:00:00Z",
                "services": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "ollama": "healthy" if ollama_healthy else "unhealthy",
                    "websocket": "healthy" if ws_healthy else "unhealthy"
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "Health check failed",
                    "timestamp": "2025-01-28T00:00:00Z"
                }
            )
    
    # API version endpoint
    @app.get("/", tags=["system"], include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Patexia Legal AI API",
            "version": "1.0.0",
            "description": "AI-powered legal document processing and search",
            "docs_url": "/docs",
            "health_url": "/health"
        }
    
    # Include API routers
    app.include_router(
        cases.router,
        prefix="/api/v1/cases",
        tags=["cases"]
    )
    
    app.include_router(
        documents.router,
        prefix="/api/v1/documents",
        tags=["documents"]
    )
    
    app.include_router(
        search.router,
        prefix="/api/v1/search",
        tags=["search"]
    )
    
    app.include_router(
        admin.router,
        prefix="/api/v1/admin",
        tags=["admin"]
    )
    
    app.include_router(
        websocket.router,
        prefix="/ws",
        tags=["websocket"]
    )
    
    logger.info("Routes configuration completed")


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""
    
    @app.exception_handler(BaseCustomException)
    async def custom_exception_handler(request: Request, exc: BaseCustomException):
        """Handle custom application exceptions."""
        logger.error(
            f"Custom exception: {exc.error_code}",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
            method=request.method,
            correlation_id=exc.correlation_id
        )
        
        response_data = get_exception_response_data(exc)
        return JSONResponse(
            status_code=exc.http_status_code,
            content=response_data
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        logger.warning(
            "Request validation failed",
            path=request.url.path,
            method=request.method,
            errors=exc.errors()
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.CONFIG_INVALID_VALUE,
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions."""
        logger.warning(
            f"HTTP exception: {exc.status_code}",
            path=request.url.path,
            method=request.method,
            detail=exc.detail
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "details": {}
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected exception occurred",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            traceback=traceback.format_exc()
        )
        
        # Don't expose internal error details in production
        settings = get_settings()
        error_detail = str(exc) if settings.debug else "Internal server error"
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": error_detail,
                    "details": {}
                }
            }
        )
    
    logger.info("Exception handlers configuration completed")


# Create the application instance
app = create_application()


# Development dependency injection helpers
def get_app_database_manager() -> DatabaseManager:
    """Get database manager instance for dependency injection."""
    if _database_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _database_manager


def get_app_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance for dependency injection."""
    if _websocket_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return _websocket_manager


def get_app_resource_monitor() -> ResourceMonitor:
    """Get resource monitor instance for dependency injection."""
    if _resource_monitor is None:
        raise RuntimeError("Resource monitor not initialized")
    return _resource_monitor


def get_app_ollama_client() -> OllamaClient:
    """Get Ollama client instance for dependency injection."""
    if _ollama_client is None:
        raise RuntimeError("Ollama client not initialized")
    return _ollama_client


# Update global getters to use application instances
get_database_manager = get_app_database_manager
get_websocket_manager = get_app_websocket_manager
get_resource_monitor = get_app_resource_monitor
get_ollama_client = get_app_ollama_client


# Development server entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get settings for server configuration
    settings = get_settings()
    
    # Configure logging for uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
        }
    }
    
    # Run the development server
    logger.info("Starting Patexia Legal AI development server...")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["backend/app", "backend/config"],
            log_config=log_config,
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        logger.info("Development server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start development server: {e}")
        sys.exit(1)