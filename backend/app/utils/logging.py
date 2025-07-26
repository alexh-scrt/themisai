"""
Structured logging system for Patexia Legal AI Chatbot.

This module provides:
- Structured JSON logging with correlation IDs
- Context-aware logging for multi-user scenarios
- Performance monitoring and metrics collection
- Console-friendly output for POC development
- Integration with FastAPI request lifecycle
- WebSocket connection tracking
- Error tracking with stack traces
"""

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from backend.config.settings import get_settings


# Thread-local storage for correlation ID
_local = threading.local()

# Rich console for enhanced output
console = Console()


class CorrelationIDProcessor:
    """Structlog processor to add correlation IDs to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to the log event."""
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        return event_dict


class TimestampProcessor:
    """Structlog processor to add ISO timestamps to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add ISO timestamp to the log event."""
        event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return event_dict


class PerformanceProcessor:
    """Structlog processor to add performance metrics to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance context if available."""
        if hasattr(_local, 'performance_context'):
            event_dict.update(_local.performance_context)
        return event_dict


class LegalAILogFormatter:
    """
    Custom log formatter for structured JSON output with rich console support.
    
    Provides both machine-readable JSON logs and human-readable console output
    for development convenience.
    """
    
    def __init__(self, use_json: bool = False, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            use_json: Output structured JSON logs
            use_colors: Use colors in console output
        """
        self.use_json = use_json
        self.use_colors = use_colors
        self.console = Console(force_terminal=use_colors)
    
    def __call__(self, _, __, event_dict):
        """Format the log event for output."""
        if self.use_json:
            return json.dumps(event_dict, default=str)
        else:
            return self._format_console_output(event_dict)
    
    def _format_console_output(self, event_dict: Dict[str, Any]) -> str:
        """Format log event for console output with colors and structure."""
        timestamp = event_dict.get("timestamp", "")
        level = event_dict.get("level", "INFO").upper()
        logger_name = event_dict.get("logger", "")
        correlation_id = event_dict.get("correlation_id", "")
        event = event_dict.get("event", "")
        
        # Color mapping for log levels
        level_colors = {
            "DEBUG": "dim white",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }
        
        # Build the console output
        parts = []
        
        # Timestamp
        if timestamp:
            parts.append(f"[dim]{timestamp[:19]}[/dim]")
        
        # Log level with color
        level_color = level_colors.get(level, "white")
        parts.append(f"[{level_color}]{level:8}[/{level_color}]")
        
        # Logger name
        if logger_name:
            parts.append(f"[cyan]{logger_name}[/cyan]")
        
        # Correlation ID
        if correlation_id:
            parts.append(f"[magenta]{correlation_id[:8]}[/magenta]")
        
        # Main event message
        parts.append(f"[white]{event}[/white]")
        
        # Additional context (excluding standard fields)
        context_fields = {
            k: v for k, v in event_dict.items()
            if k not in {"timestamp", "level", "logger", "correlation_id", "event"}
        }
        
        if context_fields:
            context_str = " ".join([f"{k}={v}" for k, v in context_fields.items()])
            parts.append(f"[dim]{context_str}[/dim]")
        
        return " ".join(parts)


def setup_logging(
    level: str = "DEBUG",
    use_json: bool = False,
    log_file: Optional[Path] = None,
    enable_correlation_ids: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Output structured JSON logs
        log_file: Optional file path for log output
        enable_correlation_ids: Enable correlation ID tracking
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        TimestampProcessor(),
    ]
    
    if enable_correlation_ids:
        processors.append(CorrelationIDProcessor())
    
    processors.append(PerformanceProcessor())
    
    # Add JSON or console formatter
    if use_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(LegalAILogFormatter(use_json=use_json))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler with Rich formatting for development
    if not use_json:
        rich_handler = RichHandler(
            console=console,
            show_time=False,  # We add timestamp in structlog
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        rich_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(rich_handler)
    else:
        # Simple handler for JSON output
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog bound logger
    """
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current thread/request.
    
    Args:
        correlation_id: Optional correlation ID, generates UUID if None
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    _local.correlation_id = correlation_id
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the correlation ID for the current thread/request.
    
    Returns:
        Current correlation ID or None if not set
    """
    return getattr(_local, 'correlation_id', None)


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current thread/request."""
    if hasattr(_local, 'correlation_id'):
        delattr(_local, 'correlation_id')


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """
    Context manager for correlation ID scoping.
    
    Args:
        correlation_id: Optional correlation ID, generates UUID if None
        
    Usage:
        with correlation_context("req-123"):
            logger.info("This log will have correlation_id=req-123")
    """
    old_id = get_correlation_id()
    new_id = set_correlation_id(correlation_id)
    try:
        yield new_id
    finally:
        if old_id:
            set_correlation_id(old_id)
        else:
            clear_correlation_id()


@contextmanager
def performance_context(
    operation: str,
    **context: Any
):
    """
    Context manager for performance monitoring.
    
    Args:
        operation: Name of the operation being measured
        **context: Additional context to include in logs
        
    Usage:
        with performance_context("document_processing", document_id="123"):
            # Processing logic here
            pass
    """
    start_time = time.time()
    logger = get_logger("performance")
    
    # Set performance context
    perf_context = {
        "operation": operation,
        "start_time": start_time,
        **context
    }
    _local.performance_context = perf_context
    
    logger.info("Operation started", **perf_context)
    
    try:
        yield perf_context
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Operation failed",
            operation=operation,
            duration=duration,
            error=str(e),
            **context
        )
        raise
    else:
        duration = time.time() - start_time
        logger.info(
            "Operation completed",
            operation=operation,
            duration=duration,
            **context
        )
    finally:
        # Clear performance context
        if hasattr(_local, 'performance_context'):
            delattr(_local, 'performance_context')


def log_function_call(
    include_args: bool = False,
    include_result: bool = False,
    log_level: str = "DEBUG"
):
    """
    Decorator to log function calls with performance metrics.
    
    Args:
        include_args: Include function arguments in logs
        include_result: Include function result in logs
        log_level: Logging level for the decorator
        
    Usage:
        @log_function_call(include_args=True)
        def process_document(document_id: str):
            return "processed"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Prepare context
            context = {
                "function": func.__name__,
                "module": func.__module__,
            }
            
            if include_args:
                context["args"] = str(args)
                context["kwargs"] = str(kwargs)
            
            with performance_context(f"{func.__module__}.{func.__name__}", **context):
                result = func(*args, **kwargs)
                
                if include_result:
                    getattr(logger, log_level.lower())(
                        "Function completed",
                        function=func.__name__,
                        result=str(result)[:200]  # Truncate long results
                    )
                
                return result
        
        return wrapper
    return decorator


class WebSocketLogger:
    """Specialized logger for WebSocket connections and events."""
    
    def __init__(self):
        self.logger = get_logger("websocket")
    
    def connection_established(
        self,
        connection_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ):
        """Log WebSocket connection establishment."""
        self.logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            user_id=user_id,
            case_id=case_id,
            event_type="connection_established"
        )
    
    def connection_closed(
        self,
        connection_id: str,
        reason: Optional[str] = None,
        code: Optional[int] = None
    ):
        """Log WebSocket connection closure."""
        self.logger.info(
            "WebSocket connection closed",
            connection_id=connection_id,
            reason=reason,
            code=code,
            event_type="connection_closed"
        )
    
    def message_sent(
        self,
        connection_id: str,
        message_type: str,
        message_size: Optional[int] = None
    ):
        """Log WebSocket message sending."""
        self.logger.debug(
            "WebSocket message sent",
            connection_id=connection_id,
            message_type=message_type,
            message_size=message_size,
            event_type="message_sent"
        )
    
    def message_received(
        self,
        connection_id: str,
        message_type: str,
        message_size: Optional[int] = None
    ):
        """Log WebSocket message reception."""
        self.logger.debug(
            "WebSocket message received",
            connection_id=connection_id,
            message_type=message_type,
            message_size=message_size,
            event_type="message_received"
        )
    
    def error_occurred(
        self,
        connection_id: str,
        error: str,
        error_code: Optional[str] = None
    ):
        """Log WebSocket errors."""
        self.logger.error(
            "WebSocket error occurred",
            connection_id=connection_id,
            error=error,
            error_code=error_code,
            event_type="error"
        )


class DatabaseLogger:
    """Specialized logger for database operations."""
    
    def __init__(self):
        self.logger = get_logger("database")
    
    def query_executed(
        self,
        database_type: str,
        operation: str,
        collection: Optional[str] = None,
        duration: Optional[float] = None,
        result_count: Optional[int] = None
    ):
        """Log database query execution."""
        self.logger.debug(
            "Database query executed",
            database_type=database_type,
            operation=operation,
            collection=collection,
            duration=duration,
            result_count=result_count,
            event_type="query_executed"
        )
    
    def connection_established(self, database_type: str, database_name: str):
        """Log database connection establishment."""
        self.logger.info(
            "Database connection established",
            database_type=database_type,
            database_name=database_name,
            event_type="connection_established"
        )
    
    def connection_failed(self, database_type: str, error: str):
        """Log database connection failures."""
        self.logger.error(
            "Database connection failed",
            database_type=database_type,
            error=error,
            event_type="connection_failed"
        )


class ModelLogger:
    """Specialized logger for AI model operations."""
    
    def __init__(self):
        self.logger = get_logger("model")
    
    def model_loaded(self, model_name: str, model_type: str, load_time: Optional[float] = None):
        """Log model loading."""
        self.logger.info(
            "Model loaded successfully",
            model_name=model_name,
            model_type=model_type,
            load_time=load_time,
            event_type="model_loaded"
        )
    
    def model_inference(
        self,
        model_name: str,
        operation: str,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        duration: Optional[float] = None
    ):
        """Log model inference operations."""
        self.logger.debug(
            "Model inference completed",
            model_name=model_name,
            operation=operation,
            input_size=input_size,
            output_size=output_size,
            duration=duration,
            event_type="model_inference"
        )
    
    def model_error(self, model_name: str, operation: str, error: str):
        """Log model errors."""
        self.logger.error(
            "Model operation failed",
            model_name=model_name,
            operation=operation,
            error=error,
            event_type="model_error"
        )


# Global logger instances for common use cases
websocket_logger = WebSocketLogger()
database_logger = DatabaseLogger()
model_logger = ModelLogger()


def initialize_logging_from_settings() -> None:
    """Initialize logging using application settings."""
    settings = get_settings()
    
    setup_logging(
        level=settings.logging.level,
        use_json=False,  # Use rich console output for POC
        enable_correlation_ids=settings.logging.enable_correlation_ids
    )
    
    logger = get_logger(__name__)
    logger.info(
        "Logging system initialized",
        level=settings.logging.level,
        correlation_ids_enabled=settings.logging.enable_correlation_ids
    )


# Convenience functions for common logging patterns

def log_api_request(
    method: str,
    path: str,
    user_id: Optional[str] = None,
    request_size: Optional[int] = None
):
    """Log API request with standard fields."""
    logger = get_logger("api")
    logger.info(
        "API request received",
        method=method,
        path=path,
        user_id=user_id,
        request_size=request_size,
        event_type="api_request"
    )


def log_api_response(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    response_size: Optional[int] = None
):
    """Log API response with standard fields."""
    logger = get_logger("api")
    logger.info(
        "API response sent",
        method=method,
        path=path,
        status_code=status_code,
        duration=duration,
        response_size=response_size,
        event_type="api_response"
    )


def log_document_processing(
    document_id: str,
    document_name: str,
    stage: str,
    status: str,
    duration: Optional[float] = None,
    error: Optional[str] = None
):
    """Log document processing events."""
    logger = get_logger("document_processing")
    
    log_method = logger.info if status == "success" else logger.error
    
    log_method(
        f"Document processing {status}",
        document_id=document_id,
        document_name=document_name,
        stage=stage,
        status=status,
        duration=duration,
        error=error,
        event_type="document_processing"
    )


def log_search_query(
    query: str,
    case_id: str,
    search_type: str,
    result_count: int,
    duration: float,
    user_id: Optional[str] = None
):
    """Log search query execution."""
    logger = get_logger("search")
    logger.info(
        "Search query executed",
        query=query[:100],  # Truncate long queries
        case_id=case_id,
        search_type=search_type,
        result_count=result_count,
        duration=duration,
        user_id=user_id,
        event_type="search_query"
    )


def log_configuration_change(
    section: str,
    key: str,
    old_value: Any,
    new_value: Any,
    user_id: Optional[str] = None
):
    """Log configuration changes."""
    logger = get_logger("configuration")
    logger.info(
        "Configuration updated",
        section=section,
        key=key,
        old_value=str(old_value),
        new_value=str(new_value),
        user_id=user_id,
        event_type="config_change"
    )