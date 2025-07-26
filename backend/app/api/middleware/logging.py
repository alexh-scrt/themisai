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

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union
from urllib.parse import parse_qs, urlparse

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ...core.config import get_settings
from ...utils.logging import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    log_api_request,
    log_api_response,
    performance_context,
    websocket_logger,
)
from ...core.exceptions import BaseCustomException
from ...models.api.common_schemas import ErrorResponse


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
    settings = get_settings()
    
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