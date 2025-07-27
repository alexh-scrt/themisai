"""
FastAPI Logging Middleware for LegalAI Document Processing System

This module provides comprehensive logging middleware for the FastAPI application,
enabling structured request/response logging, performance monitoring, correlation
ID tracking, and error context capture.

Key Features:
- Structured request and response logging
- Automatic correlation ID generation and propagation
- Performance timing and metrics collection
- Error context capture and correlation
- Legal document processing event tracking
- WebSocket connection logging support
- Configurable log levels and filtering
- Integration with existing logging utility system

Architecture Integration:
- Uses the centralized logging system from backend/app/utils/logging.py
- Integrates with correlation ID system for request tracking
- Supports both API and WebSocket request logging
- Provides context for error handling middleware
- Enables performance monitoring and alerting

Performance Considerations:
- Minimal overhead with structured logging
- Efficient correlation ID generation
- Configurable detail levels for different environments
- Async-safe implementation for FastAPI
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.config.settings import get_settings, Settings
from ...utils.logging import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    log_api_request,
    log_api_response,
    performance_context,
    clear_correlation_id,
    websocket_logger,
)
from ...core.exceptions import BaseCustomException

import re
from fastapi import HTTPException


logger = get_logger(__name__)


class RequestLoggingConfig:
    """Configuration for request logging behavior."""
    
    def __init__(
        self,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024,  # Max bytes to log
        excluded_paths: Optional[Set[str]] = None,
        excluded_headers: Optional[Set[str]] = None,
        log_query_params: bool = True,
        log_user_agent: bool = True,
        mask_sensitive_data: bool = True,
    ):
        """
        Initialize request logging configuration.
        
        Args:
            log_request_body: Whether to log request body content
            log_response_body: Whether to log response body content
            max_body_size: Maximum body size to log in bytes
            excluded_paths: Set of paths to exclude from detailed logging
            excluded_headers: Set of header names to exclude from logs
            log_query_params: Whether to log query parameters
            log_user_agent: Whether to log user agent information
            mask_sensitive_data: Whether to mask sensitive data in logs
        """
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.excluded_paths = excluded_paths or {
            "/health",
            "/metrics",
            "/favicon.ico",
            "/docs",
            "/openapi.json",
            "/redoc",
        }
        self.excluded_headers = excluded_headers or {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
        }
        self.log_query_params = log_query_params
        self.log_user_agent = log_user_agent
        self.mask_sensitive_data = mask_sensitive_data


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to establish request context with correlation IDs and timing.
    
    This middleware should be applied first to ensure all other middleware
    and route handlers have access to the request context.
    """
    
    def __init__(self, app: ASGIApp, config: Optional[RequestLoggingConfig] = None):
        """
        Initialize request context middleware.
        
        Args:
            app: FastAPI application instance
            config: Request logging configuration
        """
        super().__init__(app)
        self.config = config or RequestLoggingConfig()
        self.logger = get_logger("middleware.context")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and establish context.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response with context information
        """
        # Generate or extract correlation ID
        correlation_id = self._get_or_generate_correlation_id(request)
        
        # Set correlation ID in context
        set_correlation_id(correlation_id)
        
        # Store request start time
        start_time = time.time()
        
        # Add context to request state
        request.state.correlation_id = correlation_id
        request.state.start_time = start_time
        
        # Log basic request information
        self.logger.info(
            "Request context established",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", "unknown")[:100],
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as exc:
            # Log exception with context
            self.logger.error(
                "Request processing failed",
                method=request.method,
                path=request.url.path,
                correlation_id=correlation_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
    
    def _get_or_generate_correlation_id(self, request: Request) -> str:
        """Get correlation ID from headers or generate new one."""
        # Check for existing correlation ID in headers
        correlation_id = request.headers.get("x-correlation-id")
        
        if not correlation_id:
            # Generate new correlation ID
            correlation_id = str(uuid.uuid4())
        
        return correlation_id
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive request and response logging middleware.
    
    Provides detailed logging of HTTP requests and responses with
    configurable detail levels and performance metrics.
    """
    
    def __init__(self, app: ASGIApp, config: Optional[RequestLoggingConfig] = None):
        """
        Initialize request/response logging middleware.
        
        Args:
            app: FastAPI application instance
            config: Request logging configuration
        """
        super().__init__(app)
        self.config = config or RequestLoggingConfig()
        self.logger = get_logger("middleware.request_response")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive logging.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response with logging information
        """
        # Skip logging for excluded paths
        if request.url.path in self.config.excluded_paths:
            return await call_next(request)
        
        # Start performance tracking
        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        
        # Log incoming request
        await self._log_request(request, correlation_id)
        
        # Prepare request body capture if needed
        request_body = None
        if self.config.log_request_body and self._should_log_body(request):
            request_body = await self._capture_request_body(request)
        
        try:
            # Process request with performance context
            with performance_context(
                "http_request",
                method=request.method,
                path=request.url.path,
                correlation_id=correlation_id
            ):
                response = await call_next(request)
            
            # Calculate timing
            duration = time.time() - start_time
            
            # Log successful response
            await self._log_response(
                request,
                response,
                duration,
                correlation_id,
                request_body=request_body
            )
            
            return response
            
        except Exception as exc:
            # Calculate timing for failed requests
            duration = time.time() - start_time
            
            # Log error response
            await self._log_error_response(
                request,
                exc,
                duration,
                correlation_id,
                request_body=request_body
            )
            
            raise
    
    async def _log_request(self, request: Request, correlation_id: str) -> None:
        """Log incoming request details."""
        # Prepare request context
        context = {
            "method": request.method,
            "path": request.url.path,
            "correlation_id": correlation_id,
            "client_ip": self._get_client_ip(request),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }
        
        # Add query parameters if enabled
        if self.config.log_query_params and request.query_params:
            context["query_params"] = dict(request.query_params)
        
        # Add user agent if enabled
        if self.config.log_user_agent:
            context["user_agent"] = request.headers.get("user-agent", "unknown")[:100]
        
        # Add filtered headers
        context["headers"] = self._filter_headers(dict(request.headers))
        
        # Log with convenience function
        log_api_request(
            method=request.method,
            path=request.url.path,
            user_id=self._extract_user_id(request),
            request_size=self._get_content_length(request)
        )
        
        self.logger.info("HTTP request received", **context)
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
        correlation_id: str,
        request_body: Optional[bytes] = None
    ) -> None:
        """Log successful response details."""
        # Prepare response context
        context = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
            "correlation_id": correlation_id,
            "response_size": self._get_response_size(response),
        }
        
        # Add request body if captured
        if request_body and self.config.log_request_body:
            context["request_body_size"] = len(request_body)
            if len(request_body) <= self.config.max_body_size:
                context["request_body"] = self._decode_body(request_body)
        
        # Add response body if enabled
        if self.config.log_response_body:
            response_body = await self._capture_response_body(response)
            if response_body:
                context["response_body_size"] = len(response_body)
                if len(response_body) <= self.config.max_body_size:
                    context["response_body"] = self._decode_body(response_body)
        
        # Log with convenience function
        log_api_response(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            response_size=self._get_response_size(response)
        )
        
        self.logger.info("HTTP response sent", **context)
    
    async def _log_error_response(
        self,
        request: Request,
        exception: Exception,
        duration: float,
        correlation_id: str,
        request_body: Optional[bytes] = None
    ) -> None:
        """Log error response details."""
        # Determine error details
        if isinstance(exception, BaseCustomException):
            error_code = exception.error_code
            status_code = exception.http_status_code
            error_context = exception.details
        else:
            error_code = "INTERNAL_SERVER_ERROR"
            status_code = 500
            error_context = {}
        
        # Prepare error context
        context = {
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration": duration,
            "correlation_id": correlation_id,
            "error_code": error_code,
            "error_message": str(exception),
            "error_type": type(exception).__name__,
            "error_context": error_context,
        }
        
        # Add request body if captured
        if request_body and self.config.log_request_body:
            context["request_body_size"] = len(request_body)
            if len(request_body) <= self.config.max_body_size:
                context["request_body"] = self._decode_body(request_body)
        
        # Log with convenience function
        log_api_response(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration=duration
        )
        
        self.logger.error("HTTP request failed", **context)
    
    async def _capture_request_body(self, request: Request) -> Optional[bytes]:
        """Capture request body for logging."""
        try:
            body = await request.body()
            return body if len(body) <= self.config.max_body_size * 2 else None
        except Exception as exc:
            self.logger.warning(
                "Failed to capture request body",
                error=str(exc),
                correlation_id=getattr(request.state, "correlation_id", "unknown")
            )
            return None
    
    async def _capture_response_body(self, response: Response) -> Optional[bytes]:
        """Capture response body for logging."""
        try:
            if hasattr(response, "body") and response.body:
                body = response.body
                return body if len(body) <= self.config.max_body_size * 2 else None
            return None
        except Exception as exc:
            self.logger.warning(
                "Failed to capture response body",
                error=str(exc)
            )
            return None
    
    def _should_log_body(self, request: Request) -> bool:
        """Determine if request body should be logged."""
        content_type = request.headers.get("content-type", "")
        
        # Only log text-based content types
        text_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "text/",
        }
        
        return any(content_type.startswith(t) for t in text_types)
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers from logging."""
        if not self.config.mask_sensitive_data:
            return headers
        
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.config.excluded_headers:
                filtered[key] = "***MASKED***"
            else:
                filtered[key] = value
        
        return filtered
    
    def _decode_body(self, body: bytes) -> str:
        """Decode body bytes to string for logging."""
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary data: {len(body)} bytes>"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available."""
        # This would typically come from authentication middleware
        return getattr(request.state, "user_id", None)
    
    def _get_content_length(self, request: Request) -> Optional[int]:
        """Get request content length."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        return None
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response content size."""
        if hasattr(response, "body") and response.body:
            return len(response.body)
        
        # Check content-length header
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        return None


class WebSocketLoggingMiddleware:
    """
    WebSocket connection and message logging middleware.
    
    Provides structured logging for WebSocket connections, messages,
    and error conditions with correlation ID support.
    """
    
    def __init__(self, config: Optional[RequestLoggingConfig] = None):
        """
        Initialize WebSocket logging middleware.
        
        Args:
            config: Request logging configuration
        """
        self.config = config or RequestLoggingConfig()
        self.logger = get_logger("middleware.websocket")
        self.ws_logger = websocket_logger
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        ASGI middleware for WebSocket logging.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "websocket":
            # Not a WebSocket connection, skip
            return
        
        # Extract connection information
        connection_id = str(uuid.uuid4())
        client_ip = self._get_client_ip_from_scope(scope)
        path = scope.get("path", "unknown")
        
        # Set correlation ID
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
        # Log connection establishment
        self.ws_logger.connection_established(
            connection_id=connection_id,
            user_id=None,  # Would be set by auth middleware
            case_id=None   # Would be extracted from path/query
        )
        
        self.logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            correlation_id=correlation_id,
            client_ip=client_ip,
            path=path,
        )
        
        # Wrap receive to log incoming messages
        async def logged_receive():
            message = await receive()
            
            if message["type"] == "websocket.receive":
                message_size = len(message.get("text", message.get("bytes", b"")))
                self.ws_logger.message_received(
                    connection_id=connection_id,
                    message_type="text" if "text" in message else "binary",
                    message_size=message_size
                )
            
            return message
        
        # Wrap send to log outgoing messages
        async def logged_send(message):
            if message["type"] == "websocket.send":
                message_size = len(message.get("text", message.get("bytes", b"")))
                self.ws_logger.message_sent(
                    connection_id=connection_id,
                    message_type="text" if "text" in message else "binary",
                    message_size=message_size
                )
            elif message["type"] == "websocket.close":
                close_code = message.get("code", 1000)
                self.ws_logger.connection_closed(
                    connection_id=connection_id,
                    code=close_code,
                    reason="Normal closure"
                )
            
            await send(message)
        
        # Continue with WebSocket handling
        # Note: This would typically be integrated with FastAPI's WebSocket handling
        # For now, we provide the logging framework that can be used by WebSocket routes
    
    def _get_client_ip_from_scope(self, scope: Scope) -> str:
        """Extract client IP from ASGI scope."""
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"


def setup_logging_middleware(
    app: FastAPI,
    config: Optional[RequestLoggingConfig] = None
) -> None:
    """
    Set up all logging middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Request logging configuration
    """
    # Use settings to configure logging behavior
    settings:Settings = get_settings()
    
    if not config:
        config = RequestLoggingConfig(
            log_request_body=settings.logging.log_request_body,
            log_response_body=settings.logging.log_response_body,
            log_query_params=settings.logging.log_query_params,
            mask_sensitive_data=True,  # Always mask sensitive data
        )
    
    # Add middleware in reverse order (last added = first executed)
    
    # Request/Response logging (outermost layer)
    app.add_middleware(RequestResponseLoggingMiddleware, config=config)
    
    # Request context establishment (innermost layer)
    app.add_middleware(RequestContextMiddleware, config=config)
    
    logger.info(
        "Logging middleware configured",
        log_request_body=config.log_request_body,
        log_response_body=config.log_response_body,
        excluded_paths=len(config.excluded_paths),
        mask_sensitive_data=config.mask_sensitive_data,
    )


def create_logging_config_from_settings() -> RequestLoggingConfig:
    """
    Create logging configuration from application settings.
    
    Returns:
        RequestLoggingConfig instance based on current settings
    """
    settings = get_settings()
    
    return RequestLoggingConfig(
        log_request_body=getattr(settings.logging, "log_request_body", False),
        log_response_body=getattr(settings.logging, "log_response_body", False),
        max_body_size=getattr(settings.logging, "max_body_size", 1024),
        log_query_params=getattr(settings.logging, "log_query_params", True),
        log_user_agent=getattr(settings.logging, "log_user_agent", True),
        mask_sensitive_data=True,  # Always enabled for security
    )


# Convenience functions for manual logging in routes

def log_route_entry(request: Request, **context) -> None:
    """Log route entry with context."""
    route_logger = get_logger("routes")
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    route_logger.info(
        "Route handler entered",
        method=request.method,
        path=request.url.path,
        correlation_id=correlation_id,
        **context
    )


def log_route_exit(request: Request, result: Any, **context) -> None:
    """Log route exit with result."""
    route_logger = get_logger("routes")
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    route_logger.info(
        "Route handler completed",
        method=request.method,
        path=request.url.path,
        correlation_id=correlation_id,
        result_type=type(result).__name__,
        **context
    )


def log_business_event(
    event_type: str,
    request: Request,
    **context
) -> None:
    """Log business logic events with request context."""
    business_logger = get_logger("business")
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    business_logger.info(
        f"Business event: {event_type}",
        event_type=event_type,
        correlation_id=correlation_id,
        request_path=request.url.path,
        **context
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for FastAPI requests and responses.
    
    This middleware provides:
    - Automatic correlation ID generation and propagation
    - Structured request/response logging with timing
    - Error context capture and correlation
    - Performance monitoring and metrics collection
    - Configurable logging levels and filtering
    - WebSocket connection logging support
    - Legal document processing event tracking
    
    Features:
    - Request/response timing and size tracking
    - Automatic correlation ID management
    - Sensitive data masking for security
    - Configurable path and header exclusions
    - Structured logging compatible with log aggregation
    - Error tracking with stack traces
    - Performance threshold monitoring
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RequestLoggingConfig] = None,
        enable_performance_monitoring: bool = True,
        slow_request_threshold_ms: float = 1000.0,
        large_request_threshold_bytes: int = 1024 * 1024,  # 1MB
        enable_correlation_ids: bool = True
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: ASGI application
            config: Request logging configuration
            enable_performance_monitoring: Enable performance tracking
            slow_request_threshold_ms: Threshold for slow request warnings
            large_request_threshold_bytes: Threshold for large request warnings
            enable_correlation_ids: Enable correlation ID generation
        """
        super().__init__(app)
        self.config = config or RequestLoggingConfig()
        self.enable_performance_monitoring = enable_performance_monitoring
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.large_request_threshold_bytes = large_request_threshold_bytes
        self.enable_correlation_ids = enable_correlation_ids
        
        # Performance tracking
        self.request_count = 0
        self.total_request_time = 0.0
        self.slow_request_count = 0
        
        # Sensitive data patterns for masking
        self.sensitive_patterns = [
            re.compile(r'(password|token|key|secret|auth)', re.IGNORECASE),
            re.compile(r'(authorization|x-api-key)', re.IGNORECASE),
        ]
        
        logger.info(
            "LoggingMiddleware initialized",
            slow_threshold_ms=slow_request_threshold_ms,
            large_request_threshold=large_request_threshold_bytes,
            correlation_ids_enabled=enable_correlation_ids
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through the logging middleware.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler
            
        Returns:
            HTTP response with logging context
        """
        # Start timing
        start_time = time.time()
        request_start = datetime.now(timezone.utc)
        
        # Generate and set correlation ID
        correlation_id = self._generate_correlation_id(request)
        if self.enable_correlation_ids:
            set_correlation_id(correlation_id)
            request.state.correlation_id = correlation_id
        
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Check if this path should be logged
        if self._should_log_request(request):
            self._log_request_start(request, request_info, correlation_id)
        
        # Process request
        response = None
        error_occurred = False
        error_info = None
        
        try:
            response = await call_next(request)
            
        except Exception as exc:
            error_occurred = True
            error_info = self._extract_error_info(exc)
            
            # Log the error
            logger.error(
                "Request processing failed",
                method=request.method,
                path=str(request.url.path),
                error_type=type(exc).__name__,
                error_message=str(exc),
                correlation_id=correlation_id,
                **error_info
            )
            
            # Create error response
            response = self._create_error_response(exc, correlation_id)
        
        finally:
            # Calculate timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Extract response information
            response_info = self._extract_response_info(response) if response else {}
            
            # Update performance metrics
            if self.enable_performance_monitoring:
                self._update_performance_metrics(duration_ms, error_occurred)
            
            # Log response
            if self._should_log_request(request):
                self._log_request_complete(
                    request, response, request_info, response_info,
                    duration_ms, correlation_id, error_occurred, error_info
                )
            
            # Add correlation ID to response headers
            if response and self.enable_correlation_ids:
                response.headers["X-Correlation-ID"] = correlation_id
            
            # Clear correlation ID from thread local
            if self.enable_correlation_ids:
                clear_correlation_id()
        
        return response
    
    def _generate_correlation_id(self, request: Request) -> str:
        """
        Generate or extract correlation ID for request tracking.
        
        Args:
            request: HTTP request
            
        Returns:
            Correlation ID string
        """
        # Check if correlation ID is provided in headers
        existing_id = request.headers.get("X-Correlation-ID")
        if existing_id:
            return existing_id
        
        # Check for other common correlation headers
        trace_id = request.headers.get("X-Trace-ID")
        if trace_id:
            return trace_id
        
        request_id = request.headers.get("X-Request-ID")
        if request_id:
            return request_id
        
        # Generate new correlation ID
        return f"req_{str(uuid.uuid4())[:8]}_{int(time.time())}"
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """
        Extract request information for logging.
        
        Args:
            request: HTTP request
            
        Returns:
            Dictionary with request information
        """
        info = {
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params) if self.config.log_query_params else None,
            "client_ip": getattr(request.client, "host", "unknown") if request.client else "unknown",
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }
        
        # Add user agent if configured
        if self.config.log_user_agent:
            user_agent = request.headers.get("user-agent", "unknown")
            info["user_agent"] = user_agent[:200]  # Truncate long user agents
        
        # Add request body if configured and not too large
        if self.config.log_request_body:
            try:
                content_length = int(request.headers.get("content-length", 0))
                if content_length <= self.config.max_body_size:
                    # Note: This consumes the request body, so be careful with usage
                    # In production, you might want to avoid this or use streaming
                    body = await request.body()
                    if body:
                        info["request_body_size"] = len(body)
                        if content_length <= 512:  # Only log small bodies
                            try:
                                body_text = body.decode("utf-8")
                                info["request_body"] = self._mask_sensitive_data(body_text)
                            except UnicodeDecodeError:
                                info["request_body"] = "<binary data>"
            except Exception as e:
                logger.warning(f"Failed to extract request body: {e}")
        
        # Extract relevant headers (excluding sensitive ones)
        if hasattr(request, 'headers'):
            headers = {}
            for name, value in request.headers.items():
                if not self._is_sensitive_header(name):
                    headers[name.lower()] = value
            info["headers"] = headers
        
        return info
    
    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """
        Extract response information for logging.
        
        Args:
            response: HTTP response
            
        Returns:
            Dictionary with response information
        """
        info = {
            "status_code": getattr(response, "status_code", None),
            "content_type": None,
            "content_length": None,
        }
        
        # Extract response headers
        if hasattr(response, "headers"):
            info["content_type"] = response.headers.get("content-type")
            info["content_length"] = response.headers.get("content-length")
        
        # Add response body if configured and response is small
        if self.config.log_response_body and hasattr(response, "body"):
            try:
                if hasattr(response, "body"):
                    body_size = len(getattr(response, "body", b""))
                    info["response_body_size"] = body_size
                    
                    if body_size <= self.config.max_body_size:
                        # For small JSON responses, log the content
                        content_type = info.get("content_type", "")
                        if "application/json" in content_type and body_size <= 1024:
                            try:
                                body_text = response.body.decode("utf-8")
                                info["response_body"] = body_text
                            except (UnicodeDecodeError, AttributeError):
                                pass
            except Exception as e:
                logger.warning(f"Failed to extract response body: {e}")
        
        return info
    
    def _extract_error_info(self, exc: Exception) -> Dict[str, Any]:
        """
        Extract error information for logging.
        
        Args:
            exc: Exception that occurred
            
        Returns:
            Dictionary with error information
        """
        error_info = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_module": getattr(exc, "__module__", "unknown"),
        }
        
        # Add custom exception details if available
        if isinstance(exc, BaseCustomException):
            error_info.update({
                "error_code": exc.error_code,
                "error_details": exc.details,
                "http_status_code": exc.http_status_code,
            })
        
        # Add stack trace for debugging (truncated)
        try:
            import traceback
            stack_trace = traceback.format_exc()
            error_info["stack_trace"] = stack_trace[:2000]  # Truncate long traces
        except:
            pass
        
        return error_info
    
    def _should_log_request(self, request: Request) -> bool:
        """
        Determine if request should be logged based on configuration.
        
        Args:
            request: HTTP request
            
        Returns:
            True if request should be logged
        """
        path = str(request.url.path)
        
        # Check excluded paths
        if self.config.excluded_paths:
            for excluded_path in self.config.excluded_paths:
                if path.startswith(excluded_path):
                    return False
        
        # Skip health check and static file requests by default
        if path in ["/health", "/healthz", "/ping", "/metrics"]:
            return False
        
        if path.startswith("/static/") or path.startswith("/assets/"):
            return False
        
        return True
    
    def _is_sensitive_header(self, header_name: str) -> bool:
        """
        Check if header contains sensitive information.
        
        Args:
            header_name: Name of the header
            
        Returns:
            True if header is sensitive
        """
        header_lower = header_name.lower()
        
        # Default sensitive headers
        sensitive_headers = {
            "authorization", "x-api-key", "x-auth-token", "cookie",
            "x-forwarded-for", "x-real-ip"
        }
        
        if header_lower in sensitive_headers:
            return True
        
        # Check against configured excluded headers
        if self.config.excluded_headers:
            if header_lower in self.config.excluded_headers:
                return True
        
        # Check against sensitive patterns
        for pattern in self.sensitive_patterns:
            if pattern.search(header_name):
                return True
        
        return False
    
    def _mask_sensitive_data(self, data: str) -> str:
        """
        Mask sensitive information in request/response data.
        
        Args:
            data: Data string to mask
            
        Returns:
            Masked data string
        """
        if not self.config.mask_sensitive_data:
            return data
        
        # Simple masking patterns
        masked_data = data
        
        # Mask common sensitive patterns
        patterns = [
            (re.compile(r'"password"\s*:\s*"[^"]*"', re.IGNORECASE), '"password": "***"'),
            (re.compile(r'"token"\s*:\s*"[^"]*"', re.IGNORECASE), '"token": "***"'),
            (re.compile(r'"key"\s*:\s*"[^"]*"', re.IGNORECASE), '"key": "***"'),
            (re.compile(r'"secret"\s*:\s*"[^"]*"', re.IGNORECASE), '"secret": "***"'),
        ]
        
        for pattern, replacement in patterns:
            masked_data = pattern.sub(replacement, masked_data)
        
        return masked_data
    
    def _log_request_start(
        self,
        request: Request,
        request_info: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Log request start with context.
        
        Args:
            request: HTTP request
            request_info: Extracted request information
            correlation_id: Request correlation ID
        """
        log_context = {
            "event": "request_started",
            "correlation_id": correlation_id,
            **request_info
        }
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            **log_context
        )
    
    def _log_request_complete(
        self,
        request: Request,
        response: Optional[Response],
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        duration_ms: float,
        correlation_id: str,
        error_occurred: bool,
        error_info: Optional[Dict[str, Any]]
    ) -> None:
        """
        Log request completion with full context.
        
        Args:
            request: HTTP request
            response: HTTP response
            request_info: Request information
            response_info: Response information
            duration_ms: Request duration in milliseconds
            correlation_id: Request correlation ID
            error_occurred: Whether an error occurred
            error_info: Error information if applicable
        """
        log_context = {
            "event": "request_completed",
            "correlation_id": correlation_id,
            "duration_ms": round(duration_ms, 2),
            "error_occurred": error_occurred,
            **request_info,
            **response_info
        }
        
        if error_info:
            log_context.update(error_info)
        
        # Determine log level based on response status and timing
        log_level = "info"
        if error_occurred or (response and getattr(response, "status_code", 500) >= 500):
            log_level = "error"
        elif duration_ms > self.slow_request_threshold_ms:
            log_level = "warning"
        elif response and getattr(response, "status_code", 200) >= 400:
            log_level = "warning"
        
        # Create log message
        status_code = getattr(response, "status_code", "unknown") if response else "error"
        message = f"Request completed: {request.method} {request.url.path} - {status_code} ({duration_ms:.1f}ms)"
        
        # Log with appropriate level
        getattr(logger, log_level)(message, **log_context)
        
        # Log performance warnings
        if duration_ms > self.slow_request_threshold_ms:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path}",
                duration_ms=duration_ms,
                threshold_ms=self.slow_request_threshold_ms,
                correlation_id=correlation_id
            )
    
    def _create_error_response(self, exc: Exception, correlation_id: str) -> JSONResponse:
        """
        Create error response for unhandled exceptions.
        
        Args:
            exc: Exception that occurred
            correlation_id: Request correlation ID
            
        Returns:
            JSON error response
        """
        # Handle custom exceptions
        if isinstance(exc, BaseCustomException):
            error_data = {
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.user_message,
                    "details": exc.details
                },
                "correlation_id": correlation_id
            }
            return JSONResponse(
                status_code=exc.http_status_code,
                content=error_data
            )
        
        # Handle standard HTTP exceptions
        if isinstance(exc, HTTPException):
            error_data = {
                "success": False,
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail,
                    "details": {}
                },
                "correlation_id": correlation_id
            }
            return JSONResponse(
                status_code=exc.status_code,
                content=error_data
            )
        
        # Handle unexpected exceptions
        error_data = {
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error_type": type(exc).__name__}
            },
            "correlation_id": correlation_id
        }
        
        return JSONResponse(
            status_code=500,
            content=error_data
        )
    
    def _update_performance_metrics(self, duration_ms: float, error_occurred: bool) -> None:
        """
        Update performance tracking metrics.
        
        Args:
            duration_ms: Request duration in milliseconds
            error_occurred: Whether an error occurred
        """
        self.request_count += 1
        self.total_request_time += duration_ms
        
        if duration_ms > self.slow_request_threshold_ms:
            self.slow_request_count += 1
        
        # Log performance summary periodically
        if self.request_count % 1000 == 0:
            avg_duration = self.total_request_time / self.request_count
            slow_percentage = (self.slow_request_count / self.request_count) * 100
            
            logger.info(
                "Performance summary",
                total_requests=self.request_count,
                average_duration_ms=round(avg_duration, 2),
                slow_requests=self.slow_request_count,
                slow_percentage=round(slow_percentage, 2)
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_duration = (
            self.total_request_time / self.request_count
            if self.request_count > 0 else 0.0
        )
        
        return {
            "total_requests": self.request_count,
            "average_duration_ms": round(avg_duration, 2),
            "slow_requests": self.slow_request_count,
            "slow_percentage": round(
                (self.slow_request_count / self.request_count) * 100
                if self.request_count > 0 else 0.0, 2
            ),
            "slow_threshold_ms": self.slow_request_threshold_ms
        }


# Helper function to create and configure the middleware
def create_logging_middleware(
    enable_request_body_logging: bool = False,
    enable_response_body_logging: bool = False,
    slow_request_threshold_ms: float = 1000.0,
    excluded_paths: Optional[Set[str]] = None
) -> LoggingMiddleware:
    """
    Create and configure logging middleware with common settings.
    
    Args:
        enable_request_body_logging: Enable request body logging
        enable_response_body_logging: Enable response body logging
        slow_request_threshold_ms: Threshold for slow request warnings
        excluded_paths: Set of paths to exclude from logging
        
    Returns:
        Configured LoggingMiddleware instance
    """
    config = RequestLoggingConfig(
        log_request_body=enable_request_body_logging,
        log_response_body=enable_response_body_logging,
        max_body_size=1024,  # 1KB max for logged bodies
        excluded_paths=excluded_paths or {"/health", "/metrics", "/static"},
        excluded_headers={"authorization", "cookie", "x-api-key"},
        log_query_params=True,
        log_user_agent=True,
        mask_sensitive_data=True
    )
    
    return LoggingMiddleware(
        app=None,  # Will be set by FastAPI
        config=config,
        enable_performance_monitoring=True,
        slow_request_threshold_ms=slow_request_threshold_ms,
        large_request_threshold_bytes=1024 * 1024,  # 1MB
        enable_correlation_ids=True
    )
