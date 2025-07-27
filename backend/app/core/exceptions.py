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
from datetime import datetime, timezone

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
    CONFIG_WATCHER_FAILED = "1006"
    
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
    DOCUMENT_UPLOAD_FAILED = "3009"
    DOCUMENT_UPDATE_FAILED = "3010"
    DOCUMENT_DELETE_FAILED = "3011"
    
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

    # Connection Errors (10xxx)
    CONNECTION_FAILED = "10001"

    # Case Errors (11xxx)
    CASE_UPDATE_FAILED = "11001"


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


# Add these classes to backend/app/core/exceptions.py after the existing exception classes
# and before the convenience functions section

# Additional Error Codes for missing exception types
class AdditionalErrorCodes:
    """Additional error codes for missing exception types."""
    
    # Embedding/Vector Processing Errors (3xxx series extension)
    EMBEDDING_MODEL_ERROR = "3101"
    EMBEDDING_GENERATION_FAILED = "3102"
    EMBEDDING_DIMENSION_MISMATCH = "3103"
    EMBEDDING_VALIDATION_FAILED = "3104"
    VECTOR_STORE_ERROR = "3105"
    
    # Relationship Processing Errors (3xxx series extension)
    RELATIONSHIP_EXTRACTION_FAILED = "3201"
    RELATIONSHIP_VALIDATION_FAILED = "3202"
    RELATIONSHIP_STORAGE_ERROR = "3203"
    GRAPH_OPERATION_FAILED = "3204"
    RELATIONSHIP_CYCLE_DETECTED = "3205"
    
    # Access Control Errors (8xxx series extension)
    ACCESS_DENIED = "8101"
    PERMISSION_INSUFFICIENT = "8102"
    RESOURCE_ACCESS_FORBIDDEN = "8103"
    CASE_ACCESS_DENIED = "8104"
    DOCUMENT_ACCESS_DENIED = "8105"
    
    # Performance and Resource Errors (9xxx series extension)
    PERFORMANCE_THRESHOLD_EXCEEDED = "9101"
    RESPONSE_TIME_EXCEEDED = "9102"
    MEMORY_USAGE_EXCEEDED = "9103"
    CPU_USAGE_EXCEEDED = "9104"
    CONCURRENT_LIMIT_EXCEEDED = "9105"
    
    # Task Processing Errors (1xxx series extension)
    TASK_EXECUTION_FAILED = "1101"
    TASK_TIMEOUT = "1102"
    TASK_QUEUE_FULL = "1103"
    TASK_RETRY_LIMIT_EXCEEDED = "1104"
    TASK_CANCELLATION_FAILED = "1105"
    
    # Notification Errors (7xxx series extension)
    NOTIFICATION_DELIVERY_FAILED = "7101"
    NOTIFICATION_TEMPLATE_ERROR = "7102"
    NOTIFICATION_CHANNEL_UNAVAILABLE = "7103"


# Add these new error codes to the main ErrorCode enum
# (This would be added to the existing ErrorCode enum)
ErrorCode.EMBEDDING_MODEL_ERROR = "3101"
ErrorCode.EMBEDDING_GENERATION_FAILED = "3102"
ErrorCode.EMBEDDING_DIMENSION_MISMATCH = "3103"
ErrorCode.EMBEDDING_VALIDATION_FAILED = "3104"
ErrorCode.VECTOR_STORE_ERROR = "3105"
ErrorCode.RELATIONSHIP_EXTRACTION_FAILED = "3201"
ErrorCode.RELATIONSHIP_VALIDATION_FAILED = "3202"
ErrorCode.RELATIONSHIP_STORAGE_ERROR = "3203"
ErrorCode.GRAPH_OPERATION_FAILED = "3204"
ErrorCode.RELATIONSHIP_CYCLE_DETECTED = "3205"
ErrorCode.ACCESS_DENIED = "8101"
ErrorCode.PERMISSION_INSUFFICIENT = "8102"
ErrorCode.RESOURCE_ACCESS_FORBIDDEN = "8103"
ErrorCode.CASE_ACCESS_DENIED = "8104"
ErrorCode.DOCUMENT_ACCESS_DENIED = "8105"
ErrorCode.PERFORMANCE_THRESHOLD_EXCEEDED = "9101"
ErrorCode.RESPONSE_TIME_EXCEEDED = "9102"
ErrorCode.MEMORY_USAGE_EXCEEDED = "9103"
ErrorCode.CPU_USAGE_EXCEEDED = "9104"
ErrorCode.CONCURRENT_LIMIT_EXCEEDED = "9105"
ErrorCode.TASK_EXECUTION_FAILED = "1101"
ErrorCode.TASK_TIMEOUT = "1102"
ErrorCode.TASK_QUEUE_FULL = "1103"
ErrorCode.TASK_RETRY_LIMIT_EXCEEDED = "1104"
ErrorCode.TASK_CANCELLATION_FAILED = "1105"
ErrorCode.NOTIFICATION_DELIVERY_FAILED = "7101"
ErrorCode.NOTIFICATION_TEMPLATE_ERROR = "7102"
ErrorCode.NOTIFICATION_CHANNEL_UNAVAILABLE = "7103"


class EmbeddingError(BaseCustomException):
    """Exception raised for embedding generation and vector processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.EMBEDDING_GENERATION_FAILED,
        model_name: Optional[str] = None,
        text_length: Optional[int] = None,
        expected_dimensions: Optional[int] = None,
        actual_dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        details = {
            "model_name": model_name,
            "text_length": text_length,
            "expected_dimensions": expected_dimensions,
            "actual_dimensions": actual_dimensions,
            "batch_size": batch_size,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.EMBEDDING_MODEL_ERROR: 503,
            ErrorCode.EMBEDDING_GENERATION_FAILED: 422,
            ErrorCode.EMBEDDING_DIMENSION_MISMATCH: 422,
            ErrorCode.EMBEDDING_VALIDATION_FAILED: 400,
            ErrorCode.VECTOR_STORE_ERROR: 500,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class RelationshipError(BaseCustomException):
    """Exception raised for document relationship processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.RELATIONSHIP_EXTRACTION_FAILED,
        document_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source_document: Optional[str] = None,
        target_document: Optional[str] = None,
        extraction_method: Optional[str] = None,
        confidence_score: Optional[float] = None,
        **kwargs
    ):
        details = {
            "document_id": document_id,
            "relationship_type": relationship_type,
            "source_document": source_document,
            "target_document": target_document,
            "extraction_method": extraction_method,
            "confidence_score": confidence_score,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.RELATIONSHIP_EXTRACTION_FAILED: 422,
            ErrorCode.RELATIONSHIP_VALIDATION_FAILED: 400,
            ErrorCode.RELATIONSHIP_STORAGE_ERROR: 500,
            ErrorCode.GRAPH_OPERATION_FAILED: 500,
            ErrorCode.RELATIONSHIP_CYCLE_DETECTED: 409,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class RelationshipExtractionError(RelationshipError):
    """Specific exception for relationship extraction failures."""
    
    def __init__(
        self,
        message: str,
        extraction_stage: Optional[str] = None,
        nlp_model: Optional[str] = None,
        pattern_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            "extraction_stage": extraction_stage,
            "nlp_model": nlp_model,
            "pattern_type": pattern_type,
        })
        kwargs['details'] = details
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RELATIONSHIP_EXTRACTION_FAILED,
            **kwargs
        )


class AccessError(BaseCustomException):
    """Exception raised for access control and permission errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.ACCESS_DENIED,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        required_permission: Optional[str] = None,
        current_permissions: Optional[List[str]] = None,
        **kwargs
    ):
        details = {
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "required_permission": required_permission,
            "current_permissions": current_permissions or [],
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.ACCESS_DENIED: 403,
            ErrorCode.PERMISSION_INSUFFICIENT: 403,
            ErrorCode.RESOURCE_ACCESS_FORBIDDEN: 403,
            ErrorCode.CASE_ACCESS_DENIED: 403,
            ErrorCode.DOCUMENT_ACCESS_DENIED: 403,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 403),
            **kwargs
        )


class PerformanceError(BaseCustomException):
    """Exception raised for performance threshold violations."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PERFORMANCE_THRESHOLD_EXCEEDED,
        operation_name: Optional[str] = None,
        actual_value: Optional[Union[float, int]] = None,
        threshold_value: Optional[Union[float, int]] = None,
        metric_type: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **kwargs
    ):
        details = {
            "operation_name": operation_name,
            "actual_value": actual_value,
            "threshold_value": threshold_value,
            "metric_type": metric_type,
            "duration_ms": duration_ms,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.PERFORMANCE_THRESHOLD_EXCEEDED: 429,
            ErrorCode.RESPONSE_TIME_EXCEEDED: 408,
            ErrorCode.MEMORY_USAGE_EXCEEDED: 507,
            ErrorCode.CPU_USAGE_EXCEEDED: 507,
            ErrorCode.CONCURRENT_LIMIT_EXCEEDED: 429,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class TaskError(BaseCustomException):
    """Exception raised for background task processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.TASK_EXECUTION_FAILED,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        retry_count: Optional[int] = None,
        max_retries: Optional[int] = None,
        execution_time_ms: Optional[float] = None,
        **kwargs
    ):
        details = {
            "task_id": task_id,
            "task_type": task_type,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "execution_time_ms": execution_time_ms,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.TASK_EXECUTION_FAILED: 500,
            ErrorCode.TASK_TIMEOUT: 408,
            ErrorCode.TASK_QUEUE_FULL: 503,
            ErrorCode.TASK_RETRY_LIMIT_EXCEEDED: 500,
            ErrorCode.TASK_CANCELLATION_FAILED: 500,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


class NotificationError(BaseCustomException):
    """Exception raised for notification system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.NOTIFICATION_DELIVERY_FAILED,
        notification_id: Optional[str] = None,
        channel: Optional[str] = None,
        recipient: Optional[str] = None,
        notification_type: Optional[str] = None,
        delivery_attempts: Optional[int] = None,
        **kwargs
    ):
        details = {
            "notification_id": notification_id,
            "channel": channel,
            "recipient": recipient,
            "notification_type": notification_type,
            "delivery_attempts": delivery_attempts,
        }
        
        # Set appropriate HTTP status code based on error type
        status_map = {
            ErrorCode.NOTIFICATION_DELIVERY_FAILED: 502,
            ErrorCode.NOTIFICATION_TEMPLATE_ERROR: 400,
            ErrorCode.NOTIFICATION_CHANNEL_UNAVAILABLE: 503,
        }
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            http_status_code=status_map.get(error_code, 500),
            **kwargs
        )


# Utility functions to check error types and determine retry behavior

def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable based on its type and code.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(error, BaseCustomException):
        # Retryable error codes
        retryable_codes = {
            ErrorCode.DATABASE_TIMEOUT,
            ErrorCode.MODEL_TIMEOUT,
            ErrorCode.SEARCH_TIMEOUT,
            ErrorCode.TASK_TIMEOUT,
            ErrorCode.WEBSOCKET_CONNECTION_FAILED,
            ErrorCode.RESOURCE_UNAVAILABLE,
            ErrorCode.NOTIFICATION_DELIVERY_FAILED,
            ErrorCode.NOTIFICATION_CHANNEL_UNAVAILABLE,
            ErrorCode.EMBEDDING_MODEL_ERROR,
            ErrorCode.VECTOR_STORE_ERROR,
        }
        return error.error_code in retryable_codes
    
    # Standard exceptions that are typically retryable
    retryable_types = (
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related OS errors
    )
    
    return isinstance(error, retryable_types)


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay for retry attempts.
    
    Args:
        attempt: Current attempt number (1-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds for the next retry
    """
    import random
    
    # Exponential backoff with jitter
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    
    # Add jitter to prevent thundering herd
    jitter = random.uniform(0.8, 1.2)
    
    return delay * jitter


def get_exception_response_data(exception: BaseCustomException) -> Dict[str, Any]:
    """
    Extract response data from a custom exception for API responses.
    
    Args:
        exception: Custom exception instance
        
    Returns:
        Dictionary containing structured error data
    """
    return {
        "success": False,
        "error": {
            "code": exception.error_code,
            "message": exception.user_message,
            "details": exception.details,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": exception.correlation_id,
    }


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
    error_code: Optional[str] = None
) -> None:
    """Raise a database error with context."""
    raise DatabaseError(
        message=message,
        database_type=database_type,
        operation=operation,
        error_code=error_code
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

# Add these helper functions to backend/app/core/exceptions.py
# after the existing convenience functions section

# Additional raise_* helper functions for consistent error handling

def raise_embedding_error(
    message: str,
    model_name: Optional[str] = None,
    text_length: Optional[int] = None,
    error_code: ErrorCode = ErrorCode.EMBEDDING_GENERATION_FAILED,
    **kwargs
) -> None:
    """Raise an embedding processing error with context."""
    raise EmbeddingError(
        message=message,
        error_code=error_code,
        model_name=model_name,
        text_length=text_length,
        **kwargs
    )


def raise_relationship_error(
    message: str,
    document_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.RELATIONSHIP_EXTRACTION_FAILED,
    **kwargs
) -> None:
    """Raise a relationship processing error with context."""
    raise RelationshipError(
        message=message,
        error_code=error_code,
        document_id=document_id,
        relationship_type=relationship_type,
        **kwargs
    )


def raise_access_error(
    message: str,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.ACCESS_DENIED,
    **kwargs
) -> None:
    """Raise an access control error with context."""
    raise AccessError(
        message=message,
        error_code=error_code,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        **kwargs
    )


def raise_performance_error(
    message: str,
    operation_name: Optional[str] = None,
    actual_value: Optional[Union[float, int]] = None,
    threshold_value: Optional[Union[float, int]] = None,
    error_code: ErrorCode = ErrorCode.PERFORMANCE_THRESHOLD_EXCEEDED,
    **kwargs
) -> None:
    """Raise a performance threshold error with context."""
    raise PerformanceError(
        message=message,
        error_code=error_code,
        operation_name=operation_name,
        actual_value=actual_value,
        threshold_value=threshold_value,
        **kwargs
    )


def raise_task_error(
    message: str,
    task_id: Optional[str] = None,
    task_type: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.TASK_EXECUTION_FAILED,
    **kwargs
) -> None:
    """Raise a task processing error with context."""
    raise TaskError(
        message=message,
        error_code=error_code,
        task_id=task_id,
        task_type=task_type,
        **kwargs
    )


def raise_notification_error(
    message: str,
    notification_id: Optional[str] = None,
    channel: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.NOTIFICATION_DELIVERY_FAILED,
    **kwargs
) -> None:
    """Raise a notification system error with context."""
    raise NotificationError(
        message=message,
        error_code=error_code,
        notification_id=notification_id,
        channel=channel,
        **kwargs
    )


def raise_validation_error(
    message: str,
    field_errors: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> None:
    """Raise a validation error with field-specific details."""
    raise ValidationError(
        message=message,
        field_errors=field_errors,
        **kwargs
    )


def raise_resource_error(
    message: str,
    resource_type: Optional[str] = None,
    current_usage: Optional[Union[str, int]] = None,
    limit: Optional[Union[str, int]] = None,
    error_code: ErrorCode = ErrorCode.RESOURCE_EXHAUSTED,
    **kwargs
) -> None:
    """Raise a resource-related error with context."""
    raise ResourceError(
        message=message,
        error_code=error_code,
        resource_type=resource_type,
        current_usage=current_usage,
        limit=limit,
        **kwargs
    )


def raise_search_error(
    message: str,
    query: Optional[str] = None,
    case_id: Optional[str] = None,
    search_type: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.SEARCH_ENGINE_ERROR,
    **kwargs
) -> None:
    """Raise a search-related error with context."""
    raise SearchError(
        message=message,
        error_code=error_code,
        query=query,
        case_id=case_id,
        search_type=search_type,
        **kwargs
    )


def raise_model_error(
    message: str,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    operation: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_FAILED,
    **kwargs
) -> None:
    """Raise an AI model error with context."""
    raise ModelError(
        message=message,
        error_code=error_code,
        model_name=model_name,
        model_type=model_type,
        operation=operation,
        **kwargs
    )


def raise_websocket_error(
    message: str,
    connection_id: Optional[str] = None,
    user_id: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.WEBSOCKET_CONNECTION_FAILED,
    **kwargs
) -> None:
    """Raise a WebSocket-related error with context."""
    raise WebSocketError(
        message=message,
        error_code=error_code,
        connection_id=connection_id,
        user_id=user_id,
        **kwargs
    )


def raise_auth_error(
    message: str,
    user_id: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.AUTH_INVALID_CREDENTIALS,
    **kwargs
) -> None:
    """Raise an authentication/authorization error with context."""
    raise AuthenticationError(
        message=message,
        error_code=error_code,
        user_id=user_id,
        **kwargs
    )


# Convenience functions for common error scenarios

def raise_case_not_found(case_id: str, user_id: Optional[str] = None) -> None:
    """Raise a case not found error."""
    raise_case_error(
        message=f"Case '{case_id}' not found",
        case_id=case_id,
        user_id=user_id,
        error_code=ErrorCode.CASE_NOT_FOUND
    )


def raise_document_not_found(document_id: str, case_id: Optional[str] = None) -> None:
    """Raise a document not found error."""
    raise_document_error(
        message=f"Document '{document_id}' not found",
        document_id=document_id,
        processing_stage="retrieval"
    )


def raise_case_capacity_exceeded(case_id: str, current_count: int, limit: int) -> None:
    """Raise a case capacity exceeded error."""
    raise_case_error(
        message=f"Case '{case_id}' has reached its document limit of {limit} documents",
        case_id=case_id,
        error_code=ErrorCode.CASE_CAPACITY_EXCEEDED
    )


def raise_invalid_search_query(query: str, reason: str) -> None:
    """Raise an invalid search query error."""
    raise_search_error(
        message=f"Invalid search query: {reason}",
        query=query,
        error_code=ErrorCode.SEARCH_QUERY_INVALID
    )


def raise_model_unavailable(model_name: str, model_type: str = "embedding") -> None:
    """Raise a model unavailable error."""
    raise_model_error(
        message=f"{model_type.title()} model '{model_name}' is not available",
        model_name=model_name,
        model_type=model_type,
        error_code=ErrorCode.MODEL_NOT_AVAILABLE
    )


def raise_embedding_dimension_mismatch(
    expected: int, 
    actual: int, 
    model_name: Optional[str] = None
) -> None:
    """Raise an embedding dimension mismatch error."""
    raise_embedding_error(
        message=f"Embedding dimension mismatch: expected {expected}, got {actual}",
        model_name=model_name,
        expected_dimensions=expected,
        actual_dimensions=actual,
        error_code=ErrorCode.EMBEDDING_DIMENSION_MISMATCH
    )


def raise_task_timeout(task_id: str, task_type: str, timeout_seconds: float) -> None:
    """Raise a task timeout error."""
    raise_task_error(
        message=f"Task '{task_id}' of type '{task_type}' timed out after {timeout_seconds} seconds",
        task_id=task_id,
        task_type=task_type,
        error_code=ErrorCode.TASK_TIMEOUT
    )


def raise_queue_full(queue_name: str, max_size: int) -> None:
    """Raise a queue full error."""
    raise_task_error(
        message=f"Task queue '{queue_name}' is full (max size: {max_size})",
        error_code=ErrorCode.TASK_QUEUE_FULL
    )


def raise_insufficient_permissions(
    user_id: str,
    resource_type: str,
    resource_id: str,
    required_permission: str
) -> None:
    """Raise an insufficient permissions error."""
    raise_access_error(
        message=f"User '{user_id}' lacks '{required_permission}' permission for {resource_type} '{resource_id}'",
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        required_permission=required_permission,
        error_code=ErrorCode.PERMISSION_INSUFFICIENT
    )


def raise_database_timeout(operation: str, timeout_seconds: float) -> None:
    """Raise a database timeout error."""
    raise_database_error(
        message=f"Database operation '{operation}' timed out after {timeout_seconds} seconds",
        operation=operation,
        error_code=ErrorCode.DATABASE_TIMEOUT
    )


def raise_connection_failed(service_name: str, host: str, port: int) -> None:
    """Raise a connection failed error."""
    raise_database_error(
        message=f"Failed to connect to {service_name} at {host}:{port}",
        database_type=service_name,
        operation="connect",
        error_code=ErrorCode.DATABASE_CONNECTION_ERROR
    )


def raise_relationship_cycle_detected(documents: List[str]) -> None:
    """Raise a relationship cycle detection error."""
    raise_relationship_error(
        message=f"Circular relationship detected involving documents: {', '.join(documents)}",
        error_code=ErrorCode.RELATIONSHIP_CYCLE_DETECTED
    )


def raise_notification_template_error(template_name: str, missing_variables: List[str]) -> None:
    """Raise a notification template error."""
    raise_notification_error(
        message=f"Notification template '{template_name}' missing variables: {', '.join(missing_variables)}",
        error_code=ErrorCode.NOTIFICATION_TEMPLATE_ERROR
    )


def raise_memory_exceeded(current_mb: float, limit_mb: float, operation: str) -> None:
    """Raise a memory usage exceeded error."""
    raise_performance_error(
        message=f"Memory usage exceeded during '{operation}': {current_mb:.1f}MB > {limit_mb:.1f}MB limit",
        operation_name=operation,
        actual_value=current_mb,
        threshold_value=limit_mb,
        metric_type="memory_mb",
        error_code=ErrorCode.MEMORY_USAGE_EXCEEDED
    )


def raise_response_time_exceeded(
    operation: str, 
    actual_ms: float, 
    threshold_ms: float
) -> None:
    """Raise a response time exceeded error."""
    raise_performance_error(
        message=f"Response time exceeded for '{operation}': {actual_ms:.1f}ms > {threshold_ms:.1f}ms threshold",
        operation_name=operation,
        actual_value=actual_ms,
        threshold_value=threshold_ms,
        metric_type="response_time_ms",
        error_code=ErrorCode.RESPONSE_TIME_EXCEEDED
    )


def raise_concurrent_limit_exceeded(operation: str, current: int, limit: int) -> None:
    """Raise a concurrent operations limit exceeded error."""
    raise_performance_error(
        message=f"Concurrent limit exceeded for '{operation}': {current} > {limit} limit",
        operation_name=operation,
        actual_value=current,
        threshold_value=limit,
        metric_type="concurrent_operations",
        error_code=ErrorCode.CONCURRENT_LIMIT_EXCEEDED
    )


# Error context builders for structured error details

def build_document_context(
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    case_id: Optional[str] = None,
    file_type: Optional[str] = None,
    file_size: Optional[int] = None
) -> Dict[str, Any]:
    """Build document context for error details."""
    return {
        "document_id": document_id,
        "document_name": document_name,
        "case_id": case_id,
        "file_type": file_type,
        "file_size": file_size,
    }


def build_search_context(
    query: Optional[str] = None,
    search_type: Optional[str] = None,
    case_id: Optional[str] = None,
    result_count: Optional[int] = None,
    execution_time_ms: Optional[float] = None
) -> Dict[str, Any]:
    """Build search context for error details."""
    return {
        "query": query,
        "search_type": search_type,
        "case_id": case_id,
        "result_count": result_count,
        "execution_time_ms": execution_time_ms,
    }


def build_model_context(
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    input_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    gpu_enabled: Optional[bool] = None
) -> Dict[str, Any]:
    """Build model context for error details."""
    return {
        "model_name": model_name,
        "model_version": model_version,
        "input_size": input_size,
        "batch_size": batch_size,
        "gpu_enabled": gpu_enabled,
    }


def build_task_context(
    task_id: Optional[str] = None,
    task_type: Optional[str] = None,
    priority: Optional[str] = None,
    retry_count: Optional[int] = None,
    queue_size: Optional[int] = None
) -> Dict[str, Any]:
    """Build task context for error details."""
    return {
        "task_id": task_id,
        "task_type": task_type,
        "priority": priority,
        "retry_count": retry_count,
        "queue_size": queue_size,
    }