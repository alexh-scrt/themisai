"""
Global Error Handler Middleware for Patexia Legal AI Chatbot Backend

This module provides comprehensive error handling middleware for the FastAPI backend.
It catches, processes, and formats all exceptions into structured API responses
with proper HTTP status codes, error tracking, and user-friendly messages.

Key Features:
- Centralized exception handling for all API endpoints
- Custom exception mapping to structured error responses
- HTTP status code mapping for different error types
- Error correlation ID tracking for debugging
- Sanitized error messages for security and privacy
- Development vs production error detail control
- Legal document processing error specialization
- Performance monitoring and error metrics collection
- WebSocket error handling support
- Security event logging and alerting

Architecture Integration:
- Handles all custom exceptions from the core exception hierarchy
- Integrates with FastAPI's exception handling system
- Coordinates with logging middleware for error tracking
- Supports WebSocket error propagation
- Provides structured error responses for frontend consumption
- Maintains error correlation across service boundaries

Error Response Structure:
- Consistent JSON error format across all endpoints
- User-friendly messages for client display
- Technical details for debugging (development only)
- Error codes for programmatic handling
- Correlation IDs for request tracing
- HTTP status codes for proper client handling

Legal Document Considerations:
- Sensitive information sanitization in error messages
- Case-based error context preservation
- Document processing pipeline error handling
- Search operation error management
- User permission and access control errors
- Data privacy compliance in error reporting
"""

import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from enum import Enum

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response as StarletteResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

from ...core.exceptions import (
    BaseCustomException, ErrorCode, get_exception_response_data,
    is_retryable_error
)
from backend.config.settings import get_settings
from backend.app.utils.logging import get_logger


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification and routing."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    PROCESSING = "processing"
    SYSTEM = "system"
    EXTERNAL = "external"
    SECURITY = "security"


class ErrorMetrics:
    """Error metrics collection for monitoring."""
    
    def __init__(self):
        self.total_errors = 0
        self.error_counts_by_code: Dict[str, int] = {}
        self.error_counts_by_category: Dict[str, int] = {}
        self.error_counts_by_severity: Dict[str, int] = {}
        self.last_error_time: Optional[datetime] = None
        self.critical_errors_last_hour = 0
        self.response_time_errors = 0
    
    def record_error(
        self,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> None:
        """Record error occurrence for metrics."""
        self.total_errors += 1
        self.error_counts_by_code[error_code] = self.error_counts_by_code.get(error_code, 0) + 1
        self.error_counts_by_category[category.value] = self.error_counts_by_category.get(category.value, 0) + 1
        self.error_counts_by_severity[severity.value] = self.error_counts_by_severity.get(severity.value, 0) + 1
        self.last_error_time = datetime.now(timezone.utc)
        
        if severity == ErrorSeverity.CRITICAL:
            self.critical_errors_last_hour += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error metrics summary."""
        return {
            "total_errors": self.total_errors,
            "error_counts_by_code": self.error_counts_by_code,
            "error_counts_by_category": self.error_counts_by_category,
            "error_counts_by_severity": self.error_counts_by_severity,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "critical_errors_last_hour": self.critical_errors_last_hour
        }


class ErrorHandler:
    """Centralized error handling with classification and formatting."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.metrics = ErrorMetrics()
        self.is_development = self._is_development_environment()
        
        # Error classification mappings
        self._error_categories = self._build_error_category_mapping()
        self._error_severities = self._build_error_severity_mapping()
        
        self.logger.info(f"Error handler initialized for {'development' if self.is_development else 'production'} environment")
    
    def _is_development_environment(self) -> bool:
        """Determine if running in development environment."""
        import os
        env = os.getenv("ENVIRONMENT", "development").lower()
        debug_mode = os.getenv("DEBUG", "true").lower() == "true"
        return env in ["development", "dev", "local"] or debug_mode
    
    def _build_error_category_mapping(self) -> Dict[str, ErrorCategory]:
        """Build mapping of error codes to categories."""
        return {
            # Validation errors
            ErrorCode.CONFIG_INVALID_VALUE: ErrorCategory.VALIDATION,
            ErrorCode.DOCUMENT_INVALID_FORMAT: ErrorCategory.VALIDATION,
            ErrorCode.SEARCH_QUERY_INVALID: ErrorCategory.VALIDATION,
            
            # Authentication/Authorization errors
            ErrorCode.AUTH_INVALID_CREDENTIALS: ErrorCategory.AUTHENTICATION,
            ErrorCode.AUTH_TOKEN_EXPIRED: ErrorCategory.AUTHENTICATION,
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: ErrorCategory.AUTHORIZATION,
            ErrorCode.AUTH_USER_NOT_FOUND: ErrorCategory.AUTHENTICATION,
            ErrorCode.CASE_ACCESS_DENIED: ErrorCategory.AUTHORIZATION,
            
            # Resource errors
            ErrorCode.CASE_CAPACITY_EXCEEDED: ErrorCategory.RESOURCE,
            ErrorCode.RESOURCE_EXHAUSTED: ErrorCategory.RESOURCE,
            ErrorCode.RESOURCE_QUOTA_EXCEEDED: ErrorCategory.RESOURCE,
            ErrorCode.RESOURCE_UNAVAILABLE: ErrorCategory.RESOURCE,
            
            # Processing errors
            ErrorCode.DOCUMENT_PROCESSING_FAILED: ErrorCategory.PROCESSING,
            ErrorCode.DOCUMENT_EXTRACTION_FAILED: ErrorCategory.PROCESSING,
            ErrorCode.DOCUMENT_CHUNKING_FAILED: ErrorCategory.PROCESSING,
            ErrorCode.DOCUMENT_EMBEDDING_FAILED: ErrorCategory.PROCESSING,
            ErrorCode.SEARCH_ENGINE_ERROR: ErrorCategory.PROCESSING,
            ErrorCode.MODEL_INFERENCE_FAILED: ErrorCategory.PROCESSING,
            
            # System errors
            ErrorCode.DATABASE_CONNECTION_ERROR: ErrorCategory.SYSTEM,
            ErrorCode.DATABASE_TIMEOUT: ErrorCategory.SYSTEM,
            ErrorCode.CONFIG_VALIDATION_FAILED: ErrorCategory.SYSTEM,
            ErrorCode.MODEL_NOT_AVAILABLE: ErrorCategory.SYSTEM,
            ErrorCode.WEBSOCKET_CONNECTION_FAILED: ErrorCategory.SYSTEM,
            
            # External errors
            ErrorCode.MODEL_TIMEOUT: ErrorCategory.EXTERNAL,
            ErrorCode.MODEL_GPU_ERROR: ErrorCategory.EXTERNAL,
        }
    
    def _build_error_severity_mapping(self) -> Dict[str, ErrorSeverity]:
        """Build mapping of error codes to severity levels."""
        return {
            # Critical errors (system-wide impact)
            ErrorCode.DATABASE_CONNECTION_ERROR: ErrorSeverity.CRITICAL,
            ErrorCode.CONFIG_VALIDATION_FAILED: ErrorSeverity.CRITICAL,
            ErrorCode.MODEL_GPU_ERROR: ErrorSeverity.CRITICAL,
            
            # High severity errors (significant user impact)
            ErrorCode.MODEL_NOT_AVAILABLE: ErrorSeverity.HIGH,
            ErrorCode.RESOURCE_EXHAUSTED: ErrorSeverity.HIGH,
            ErrorCode.DATABASE_TIMEOUT: ErrorSeverity.HIGH,
            ErrorCode.DOCUMENT_PROCESSING_FAILED: ErrorSeverity.HIGH,
            
            # Medium severity errors (partial functionality impact)
            ErrorCode.SEARCH_ENGINE_ERROR: ErrorSeverity.MEDIUM,
            ErrorCode.MODEL_TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCode.CASE_CAPACITY_EXCEEDED: ErrorSeverity.MEDIUM,
            ErrorCode.WEBSOCKET_CONNECTION_FAILED: ErrorSeverity.MEDIUM,
            
            # Low severity errors (user errors, recoverable)
            ErrorCode.CASE_NOT_FOUND: ErrorSeverity.LOW,
            ErrorCode.DOCUMENT_NOT_FOUND: ErrorSeverity.LOW,
            ErrorCode.AUTH_INVALID_CREDENTIALS: ErrorSeverity.LOW,
            ErrorCode.SEARCH_QUERY_INVALID: ErrorSeverity.LOW,
            ErrorCode.DOCUMENT_INVALID_FORMAT: ErrorSeverity.LOW,
        }
    
    def classify_error(self, error_code: str) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.
        
        Args:
            error_code: Error code to classify
            
        Returns:
            Tuple of (category, severity)
        """
        category = self._error_categories.get(error_code, ErrorCategory.SYSTEM)
        severity = self._error_severities.get(error_code, ErrorSeverity.MEDIUM)
        return category, severity
    
    def handle_custom_exception(
        self,
        request: Request,
        exc: BaseCustomException
    ) -> JSONResponse:
        """
        Handle custom application exceptions.
        
        Args:
            request: HTTP request object
            exc: Custom exception to handle
            
        Returns:
            JSON error response
        """
        # Get correlation ID from request or exception
        correlation_id = self._get_correlation_id(request, exc)
        
        # Classify error
        category, severity = self.classify_error(exc.error_code)
        
        # Record metrics
        self.metrics.record_error(exc.error_code, category, severity)
        
        # Build error response
        error_response = self._build_error_response(
            exc=exc,
            request=request,
            correlation_id=correlation_id,
            category=category,
            severity=severity
        )
        
        # Log error with appropriate level
        self._log_error(exc, request, correlation_id, category, severity)
        
        # Send alerts for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._send_critical_error_alert(exc, request, correlation_id)
        
        return JSONResponse(
            status_code=exc.http_status_code,
            content=error_response,
            headers={"X-Correlation-ID": correlation_id}
        )
    
    def handle_http_exception(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """
        Handle FastAPI HTTP exceptions.
        
        Args:
            request: HTTP request object
            exc: HTTP exception to handle
            
        Returns:
            JSON error response
        """
        correlation_id = self._get_correlation_id(request)
        
        # Map HTTP status to error code
        error_code = self._map_status_to_error_code(exc.status_code)
        category, severity = self.classify_error(error_code)
        
        # Record metrics
        self.metrics.record_error(error_code, category, severity)
        
        error_response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": str(exc.detail),
                "user_message": self._sanitize_error_message(str(exc.detail)),
                "details": {"http_status": exc.status_code},
                "correlation_id": correlation_id,
            }
        }
        
        # Add debug information in development
        if self.is_development:
            error_response["error"]["debug"] = {
                "request_url": str(request.url),
                "request_method": request.method,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        self.logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={
                "correlation_id": correlation_id,
                "error_code": error_code,
                "status_code": exc.status_code,
                "url": str(request.url),
                "method": request.method,
                "category": category.value,
                "severity": severity.value
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers={"X-Correlation-ID": correlation_id}
        )
    
    def handle_validation_error(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors.
        
        Args:
            request: HTTP request object
            exc: Validation error to handle
            
        Returns:
            JSON error response
        """
        correlation_id = self._get_correlation_id(request)
        
        # Extract validation error details
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = {
            "success": False,
            "error": {
                "code": ErrorCode.CONFIG_INVALID_VALUE,
                "message": "Request validation failed",
                "user_message": "Please check your input and try again",
                "details": {
                    "validation_errors": validation_errors,
                    "error_count": len(validation_errors)
                },
                "correlation_id": correlation_id,
            }
        }
        
        # Add raw validation details in development
        if self.is_development:
            error_response["error"]["debug"] = {
                "raw_errors": exc.errors(),
                "request_body": str(exc.body) if hasattr(exc, 'body') else None,
            }
        
        self.logger.warning(
            f"Validation error: {len(validation_errors)} field(s) failed validation",
            extra={
                "correlation_id": correlation_id,
                "validation_errors": validation_errors,
                "url": str(request.url),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response,
            headers={"X-Correlation-ID": correlation_id}
        )
    
    def handle_unexpected_error(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Handle unexpected/unhandled exceptions.
        
        Args:
            request: HTTP request object
            exc: Unexpected exception
            
        Returns:
            JSON error response
        """
        correlation_id = self._get_correlation_id(request)
        
        # Classify as critical system error
        category = ErrorCategory.SYSTEM
        severity = ErrorSeverity.CRITICAL
        error_code = ErrorCode.DATABASE_OPERATION_FAILED  # Generic system error
        
        # Record metrics
        self.metrics.record_error(error_code, category, severity)
        
        error_response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": "Internal server error",
                "user_message": "An unexpected error occurred. Please try again later.",
                "details": {"error_type": type(exc).__name__},
                "correlation_id": correlation_id,
            }
        }
        
        # Add debug information in development only
        if self.is_development:
            error_response["error"]["debug"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "request_url": str(request.url),
                "request_method": request.method,
            }
        
        # Log as error with full context
        self.logger.error(
            f"Unexpected error: {type(exc).__name__}: {str(exc)}",
            extra={
                "correlation_id": correlation_id,
                "error_code": error_code,
                "exception_type": type(exc).__name__,
                "url": str(request.url),
                "method": request.method,
                "category": category.value,
                "severity": severity.value
            },
            exc_info=True
        )
        
        # Send critical error alert
        self._send_critical_error_alert(exc, request, correlation_id)
        
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response,
            headers={"X-Correlation-ID": correlation_id}
        )
    
    def _build_error_response(
        self,
        exc: BaseCustomException,
        request: Request,
        correlation_id: str,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> Dict[str, Any]:
        """Build structured error response."""
        # Start with base exception data
        response = get_exception_response_data(exc)
        
        # Ensure correlation ID is set
        response["error"]["correlation_id"] = correlation_id
        
        # Add classification metadata
        response["error"]["category"] = category.value
        response["error"]["severity"] = severity.value
        response["error"]["retryable"] = is_retryable_error(exc)
        
        # Sanitize user message for security
        response["error"]["user_message"] = self._sanitize_error_message(
            response["error"]["user_message"]
        )
        
        # Add debug information in development
        if self.is_development:
            response["error"]["debug"] = {
                "exception_type": type(exc).__name__,
                "technical_message": exc.message,
                "request_url": str(request.url),
                "request_method": request.method,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "traceback_preview": exc.traceback_info.split('\n')[-10:] if exc.traceback_info else None
            }
        
        return response
    
    def _get_correlation_id(
        self,
        request: Request,
        exc: Optional[BaseCustomException] = None
    ) -> str:
        """Get or generate correlation ID for request tracking."""
        # Try to get from exception first
        if exc and exc.correlation_id:
            return exc.correlation_id
        
        # Try to get from request headers
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            return correlation_id
        
        # Try to get from request state
        if hasattr(request.state, "correlation_id"):
            return request.state.correlation_id
        
        # Generate new correlation ID
        return str(uuid.uuid4())
    
    def _map_status_to_error_code(self, status_code: int) -> str:
        """Map HTTP status code to error code."""
        status_mapping = {
            400: ErrorCode.CONFIG_INVALID_VALUE,
            401: ErrorCode.AUTH_INVALID_CREDENTIALS,
            403: ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            404: ErrorCode.CASE_NOT_FOUND,
            409: ErrorCode.CASE_DUPLICATE_NAME,
            422: ErrorCode.CONFIG_INVALID_VALUE,
            429: ErrorCode.RESOURCE_QUOTA_EXCEEDED,
            500: ErrorCode.DATABASE_OPERATION_FAILED,
            503: ErrorCode.RESOURCE_UNAVAILABLE,
            504: ErrorCode.MODEL_TIMEOUT,
        }
        return status_mapping.get(status_code, ErrorCode.DATABASE_OPERATION_FAILED)
    
    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information."""
        # Remove potential file paths
        import re
        message = re.sub(r'/[^\s]*', '[path]', message)
        
        # Remove potential connection strings
        message = re.sub(r'mongodb://[^\s]*', '[connection_string]', message)
        message = re.sub(r'postgresql://[^\s]*', '[connection_string]', message)
        
        # Remove potential API keys or tokens
        message = re.sub(r'[Tt]oken[:\s=]+[^\s]+', 'token=[redacted]', message)
        message = re.sub(r'[Kk]ey[:\s=]+[^\s]+', 'key=[redacted]', message)
        
        # Remove potential passwords
        message = re.sub(r'[Pp]assword[:\s=]+[^\s]+', 'password=[redacted]', message)
        
        return message
    
    def _log_error(
        self,
        exc: Exception,
        request: Request,
        correlation_id: str,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> None:
        """Log error with appropriate level and context."""
        log_data = {
            "correlation_id": correlation_id,
            "category": category.value,
            "severity": severity.value,
            "url": str(request.url),
            "method": request.method,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "client_ip": request.client.host if request.client else "unknown"
        }
        
        if isinstance(exc, BaseCustomException):
            log_data.update({
                "error_code": exc.error_code,
                "technical_message": exc.message,
                "user_message": exc.user_message,
                "details": exc.details
            })
        
        # Choose log level based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {str(exc)}", extra=log_data, exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {str(exc)}", extra=log_data, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {str(exc)}", extra=log_data)
        else:
            self.logger.info(f"Low severity error: {str(exc)}", extra=log_data)
    
    def _send_critical_error_alert(
        self,
        exc: Exception,
        request: Request,
        correlation_id: str
    ) -> None:
        """Send alert for critical errors (placeholder for actual alerting system)."""
        # In a production system, this would integrate with:
        # - Slack/Teams notifications
        # - PagerDuty/Opsgenie alerts
        # - Email notifications
        # - Monitoring systems (Datadog, NewRelic, etc.)
        
        alert_data = {
            "alert_type": "critical_error",
            "service": "patexia-legal-ai",
            "correlation_id": correlation_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "request_url": str(request.url),
            "request_method": request.method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Log the alert for now (replace with actual alerting)
        self.logger.critical(
            f"CRITICAL ERROR ALERT: {alert_data}",
            extra={"alert_data": alert_data}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error handler metrics."""
        return self.metrics.get_summary()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to catch and handle all exceptions globally."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_handler = ErrorHandler()
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Process request and handle any exceptions."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
            
        except BaseCustomException as exc:
            # Handle custom application exceptions
            return self.error_handler.handle_custom_exception(request, exc)
            
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            return self.error_handler.handle_http_exception(request, exc)
            
        except RequestValidationError as exc:
            # Handle Pydantic validation errors
            return self.error_handler.handle_validation_error(request, exc)
            
        except Exception as exc:
            # Handle unexpected exceptions
            return self.error_handler.handle_unexpected_error(request, exc)
        
        finally:
            # Log response time if it was an error
            duration = time.time() - start_time
            if duration > 5.0:  # Log slow requests that might indicate errors
                self.error_handler.metrics.response_time_errors += 1
                self.logger.warning(
                    f"Slow request: {request.method} {request.url} took {duration:.2f}s"
                )


def setup_error_handlers(app: FastAPI) -> None:
    """
    Setup global error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    error_handler = ErrorHandler()
    
    # Add global exception handlers
    @app.exception_handler(BaseCustomException)
    async def custom_exception_handler(request: Request, exc: BaseCustomException):
        return error_handler.handle_custom_exception(request, exc)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return error_handler.handle_http_exception(request, exc)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return error_handler.handle_validation_error(request, exc)
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        # Convert Starlette HTTPException to FastAPI HTTPException
        fastapi_exc = HTTPException(status_code=exc.status_code, detail=exc.detail)
        return error_handler.handle_http_exception(request, fastapi_exc)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return error_handler.handle_unexpected_error(request, exc)
    
    # Add error handler middleware
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Add health check endpoint for error handler
    @app.get("/api/v1/health/errors")
    async def error_handler_health():
        """Get error handler health and metrics."""
        return {
            "status": "healthy",
            "metrics": error_handler.get_metrics(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    logger = get_logger(__name__)
    logger.info("Global error handlers configured successfully")


# Export public interface
__all__ = [
    "ErrorHandler",
    "ErrorHandlerMiddleware",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorMetrics",
    "setup_error_handlers"
]