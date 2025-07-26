"""
Custom exception classes for Patexia Legal AI Chatbot.

This module defines a comprehensive exception hierarchy that provides:
- Domain-specific exceptions for legal document processing
- HTTP status code mapping for API responses
- Structured error information with context
- Exception chaining for debugging
- Graceful error handling patterns
"""

import traceback
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ErrorCode(str, Enum):
    """
    Standardized error codes for the legal AI system.
    
    These codes provide consistent error identification across the application
    and enable structured error handling in the frontend.
    """
    
    # Configuration Errors (1xxx)
    CONFIG_VALIDATION_FAILED = "1001"
    CONFIG_FILE_NOT_FOUND = "1002"
    CONFIG_PARSE_ERROR = "1003"
    CONFIG_INVALID_VALUE = "1004"
    CONFIG_MISSING_REQUIRED = "1005"
    
    # Database Errors (2xxx)
    DATABASE_CONNECTION_ERROR = "2001"
    DATABASE_OPERATION_FAILED = "2002"
    DATABASE_CONSTRAINT_VIOLATION = "2003"
    DATABASE_TIMEOUT = "2004"
    DATABASE_MIGRATION_ERROR = "2005"
    
    # Document Processing Errors (3xxx)
    DOCUMENT_NOT_FOUND = "3001"
    DOCUMENT_PROCESSING_FAILED = "3002"
    DOCUMENT_INVALID_FORMAT = "3003"
    DOCUMENT_TOO_LARGE = "3004"
    DOCUMENT_CORRUPTED = "3005"
    DOCUMENT_EXTRACTION_FAILED = "3006"
    DOCUMENT_CHUNKING_FAILED = "3007"
    DOCUMENT_EMBEDDING_FAILED = "3008"
    
    # Case Management Errors (4xxx)
    CASE_NOT_FOUND = "4001"
    CASE_ACCESS_DENIED = "4002"
    CASE_CAPACITY_EXCEEDED = "4003"
    CASE_INVALID_STATE = "4004"
    CASE_DUPLICATE_NAME = "4005"
    
    # Search Errors (5xxx)
    SEARCH_QUERY_INVALID = "5001"
    SEARCH_INDEX_NOT_READY = "5002"
    SEARCH_TIMEOUT = "5003"
    SEARCH_NO_RESULTS = "5004"
    SEARCH_ENGINE_ERROR = "5005"
    
    # AI Model Errors (6xxx)
    MODEL_NOT_AVAILABLE = "6001"
    MODEL_LOADING_FAILED = "6002"
    MODEL_INFERENCE_FAILED = "6003"
    MODEL_TIMEOUT = "6004"
    MODEL_GPU_ERROR = "6005"
    MODEL_UNSUPPORTED_FORMAT = "6006"
    
    # WebSocket Errors (7xxx)
    WEBSOCKET_CONNECTION_FAILED = "7001"
    WEBSOCKET_MESSAGE_INVALID = "7002"
    WEBSOCKET_AUTHENTICATION_FAILED = "7003"
    WEBSOCKET_RATE_LIMIT_EXCEEDED = "7004"
    
    # Authentication & Authorization Errors (8xxx)
    AUTH_INVALID_CREDENTIALS = "8001"
    AUTH_TOKEN_EXPIRED = "8002"
    AUTH_INSUFFICIENT_PERMISSIONS = "8003"
    AUTH_USER_NOT_FOUND = "8004"
    
    # Resource Errors (9xxx)
    RESOURCE_EXHAUSTED = "9001"
    RESOURCE_QUOTA_EXCEEDED = "9002"
    RESOURCE_UNAVAILABLE = "9003"
    RESOURCE_CONFLICT = "9004"


class BaseCustomException(Exception):
    """
    Base exception class for all custom exceptions in the legal AI system.
    
    Provides common functionality for error tracking, context preservation,
    and structured error information.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        http_status_code: int = 500,
        correlation_id: Optional[str] = None,
        user_message: Optional[str] = None
    ):
        """
        Initialize base exception with structured error information.
        
        Args:
            message: Technical error message for developers
            error_code: Standardized error code for identification
            details: Additional context and debugging information
            http_status_code: HTTP status code for API responses
            correlation_id: Request correlation ID for tracking
            user_message: User-friendly error message for display
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.http_status_code = http_status_code
        self.correlation_id = correlation_id
        self.user_message = user_message or self._generate_user_message()
        self.traceback_info = traceback.format_exc()
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly error message based on the error code."""
        user_messages = {
            ErrorCode.CONFIG_VALIDATION_FAILED: "Configuration validation failed. Please check your settings.",
            ErrorCode.DATABASE_CONNECTION_ERROR: "Unable to connect to the database. Please try again later.",
            ErrorCode.DOCUMENT_PROCESSING_FAILED: "Failed to process the document. Please check the file format.",
            ErrorCode.CASE_NOT_FOUND: "The requested case could not be found.",
            ErrorCode.CASE_CAPACITY_EXCEEDED: "Cannot add more documents to this case. Capacity limit reached.",
            ErrorCode.MODEL_NOT_AVAILABLE: "AI model is currently unavailable. Please try again later.",
            ErrorCode.SEARCH_TIMEOUT: "Search request timed out. Please try a simpler query.",
        }
        return user_messages.get(self.error_code, "An unexpected error occurred. Please contact support.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "http_status_code": self.http_status_code,
            "correlation_id": self.correlation_id,
        }
    
    def add_context(self, key: str, value: Any) -> None:
        """Add additional context to the exception details."""
        self.details[key] = value
    
    def __str__(self) -> str:
        """String representation of the exception."""
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(BaseCustomException):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIG_VALIDATION_FAILED,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        details = {
            "config_section": config_section,
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None,
        }
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=400,
            **kwargs
        )


class DatabaseError(BaseCustomException):
    """Exception raised for database-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_OPERATION_FAILED,
        database_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {
            "database_type": database_type,
            "collection_name": collection_name,
            "operation": operation,
        }
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=500,
            **kwargs
        )


class DocumentProcessingError(BaseCustomException):
    """Exception raised for document processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DOCUMENT_PROCESSING_FAILED,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        file_type: Optional[str] = None,
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        details = {
            "document_id": document_id,
            "document_name": document_name,
            "file_type": file_type,
            "processing_stage": processing_stage,
        }
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=422,
            **kwargs
        )


class CaseManagementError(BaseCustomException):
    """Exception raised for case management errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CASE_NOT_FOUND,
        case_id: Optional[str] = None,
        user_id: Optional[str] = None,
        case_name: Optional[str] = None,
        **kwargs
    ):
        details = {
            "case_id": case_id,
            "user_id": user_id,
            "case_name": case_name,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.CASE_NOT_FOUND: 404,
            ErrorCode.CASE_ACCESS_DENIED: 403,
            ErrorCode.CASE_CAPACITY_EXCEEDED: 429,
            ErrorCode.CASE_DUPLICATE_NAME: 409,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 400),
            **kwargs
        )


class SearchError(BaseCustomException):
    """Exception raised for search-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SEARCH_ENGINE_ERROR,
        query: Optional[str] = None,
        case_id: Optional[str] = None,
        search_type: Optional[str] = None,
        **kwargs
    ):
        details = {
            "query": query,
            "case_id": case_id,
            "search_type": search_type,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.SEARCH_QUERY_INVALID: 400,
            ErrorCode.SEARCH_NO_RESULTS: 404,
            ErrorCode.SEARCH_TIMEOUT: 408,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class ModelError(BaseCustomException):
    """Exception raised for AI model-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_FAILED,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {
            "model_name": model_name,
            "model_type": model_type,
            "operation": operation,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.MODEL_NOT_AVAILABLE: 503,
            ErrorCode.MODEL_TIMEOUT: 408,
            ErrorCode.MODEL_GPU_ERROR: 500,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class WebSocketError(BaseCustomException):
    """Exception raised for WebSocket-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.WEBSOCKET_CONNECTION_FAILED,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        details = {
            "connection_id": connection_id,
            "user_id": user_id,
        }
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=400,
            **kwargs
        )


class AuthenticationError(BaseCustomException):
    """Exception raised for authentication and authorization errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.AUTH_INVALID_CREDENTIALS,
        user_id: Optional[str] = None,
        **kwargs
    ):
        details = {
            "user_id": user_id,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.AUTH_INVALID_CREDENTIALS: 401,
            ErrorCode.AUTH_TOKEN_EXPIRED: 401,
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: 403,
            ErrorCode.AUTH_USER_NOT_FOUND: 404,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 401),
            **kwargs
        )


class ResourceError(BaseCustomException):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.RESOURCE_EXHAUSTED,
        resource_type: Optional[str] = None,
        current_usage: Optional[Union[str, int]] = None,
        limit: Optional[Union[str, int]] = None,
        **kwargs
    ):
        details = {
            "resource_type": resource_type,
            "current_usage": str(current_usage) if current_usage is not None else None,
            "limit": str(limit) if limit is not None else None,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.RESOURCE_QUOTA_EXCEEDED: 429,
            ErrorCode.RESOURCE_UNAVAILABLE: 503,
            ErrorCode.RESOURCE_CONFLICT: 409,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class ValidationError(BaseCustomException):
    """Exception raised for data validation errors."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        details = {
            "field_errors": field_errors or [],
        }
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIG_INVALID_VALUE,
            details=details,
            http_status_code=422,
            **kwargs
        )


# Convenience functions for common exception patterns

def raise_config_error(
    message: str,
    config_section: Optional[str] = None,
    config_key: Optional[str] = None,
    config_value: Optional[Any] = None
) -> None:
    """Raise a configuration error with context."""
    raise ConfigurationError(
        message=message,
        config_section=config_section,
        config_key=config_key,
        config_value=config_value
    )


def raise_database_error(
    message: str,
    database_type: Optional[str] = None,
    operation: Optional[str] = None,
    collection_name: Optional[str] = None
) -> None:
    """Raise a database error with context."""
    raise DatabaseError(
        message=message,
        database_type=database_type,
        operation=operation,
        collection_name=collection_name
    )


def raise_document_error(
    message: str,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    processing_stage: Optional[str] = None
) -> None:
    """Raise a document processing error with context."""
    raise DocumentProcessingError(
        message=message,
        document_id=document_id,
        document_name=document_name,
        processing_stage=processing_stage
    )


def raise_case_error(
    message: str,
    case_id: Optional[str] = None,
    user_id: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.CASE_NOT_FOUND
) -> None:
    """Raise a case management error with context."""
    raise CaseManagementError(
        message=message,
        error_code=error_code,
        case_id=case_id,
        user_id=user_id
    )


def raise_search_error(
    message: str,
    query: Optional[str] = None,
    case_id: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.SEARCH_ENGINE_ERROR
) -> None:
    """Raise a search error with context."""
    raise SearchError(
        message=message,
        error_code=error_code,
        query=query,
        case_id=case_id
    )


def raise_model_error(
    message: str,
    model_name: Optional[str] = None,
    operation: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_FAILED
) -> None:
    """Raise a model error with context."""
    raise ModelError(
        message=message,
        error_code=error_code,
        model_name=model_name,
        operation=operation
    )


def raise_capacity_error(
    message: str,
    resource_type: str,
    current_usage: Union[str, int],
    limit: Union[str, int]
) -> None:
    """Raise a capacity/quota exceeded error."""
    raise ResourceError(
        message=message,
        error_code=ErrorCode.RESOURCE_QUOTA_EXCEEDED,
        resource_type=resource_type,
        current_usage=current_usage,
        limit=limit
    )


# Exception middleware support

def get_exception_response_data(exc: BaseCustomException) -> Dict[str, Any]:
    """
    Extract response data from a custom exception for API responses.
    
    Args:
        exc: Custom exception instance
        
    Returns:
        Dictionary containing error response data
    """
    return {
        "success": False,
        "error": {
            "code": exc.error_code,
            "message": exc.user_message,
            "details": exc.details,
            "correlation_id": exc.correlation_id,
        }
    }


def is_retryable_error(exc: BaseCustomException) -> bool:
    """
    Determine if an error is retryable (for retry mechanisms).
    
    Args:
        exc: Custom exception instance
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_codes = {
        ErrorCode.DATABASE_TIMEOUT,
        ErrorCode.MODEL_TIMEOUT,
        ErrorCode.SEARCH_TIMEOUT,
        ErrorCode.WEBSOCKET_CONNECTION_FAILED,
        ErrorCode.RESOURCE_UNAVAILABLE,
    }
    return exc.error_code in retryable_codes


def get_retry_delay(exc: BaseCustomException, attempt: int) -> float:
    """
    Calculate retry delay for retryable errors with exponential backoff.
    
    Args:
        exc: Custom exception instance
        attempt: Current retry attempt number (1-based)
        
    Returns:
        Delay in seconds before next retry attempt
    """
    if not is_retryable_error(exc):
        return 0.0
    
    # Exponential backoff: 1s, 2s, 4s, 8s, max 30s
    base_delay = min(2 ** (attempt - 1), 30)
    
    # Add some jitter to prevent thundering herd
    import random
    jitter = random.uniform(0.1, 0.3)
    return base_delay + jitter