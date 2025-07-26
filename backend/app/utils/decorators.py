"""
Common Decorators for Patexia Legal AI Chatbot

This module provides a comprehensive set of decorators for cross-cutting concerns
in the legal AI system, including error handling, performance monitoring, validation,
async operations, caching, and retry mechanisms.

Key Features:
- Performance monitoring with detailed metrics collection
- Comprehensive error handling with context preservation
- Async operation support with timeout management
- Input/output validation with legal document awareness
- Intelligent retry mechanisms with exponential backoff
- WebSocket progress tracking integration
- Caching with TTL and LRU strategies
- Resource usage monitoring and throttling
- Rate limiting for API endpoints
- Database transaction management

Decorator Categories:
1. Performance & Monitoring: @performance_monitor, @track_resource_usage
2. Error Handling: @error_handler, @safe_execute, @retry_on_failure
3. Validation: @validate_input, @validate_output, @validate_case_access
4. Async Operations: @async_timeout, @async_safe, @websocket_progress
5. Caching: @cache_result, @invalidate_cache, @cache_with_ttl
6. Rate Limiting: @rate_limit, @throttle_requests
7. Database: @transactional, @read_only, @with_connection

Architecture Integration:
- Integrates with logging system for structured monitoring
- Uses WebSocket manager for real-time progress updates
- Works with exception system for consistent error handling
- Supports configuration system for runtime parameter adjustment
- Implements security utilities for access control
"""

import asyncio
import functools
import hashlib
import inspect
import json
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import weakref

from cachetools import TTLCache, LRUCache
from pydantic import BaseModel, ValidationError as PydanticValidationError

from config.settings import get_settings
from ..core.websocket_manager import WebSocketManager
from ..core.exceptions import (
    BaseCustomException, ErrorCode, ValidationError, ResourceError,
    DatabaseError, ConfigurationError, ModelError, CaseManagementError,
    is_retryable_error, get_retry_delay
)
from ..utils.logging import get_logger, performance_context
from ..utils.validators import validate_case_access, validate_user_permissions

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

logger = get_logger(__name__)

# Global state for decorator functionality
_rate_limit_storage = defaultdict(deque)
_cache_storage = {}
_resource_monitors = {}
_progress_trackers = weakref.WeakValueDictionary()


class PerformanceMetrics(BaseModel):
    """Performance metrics collected by decorators."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    error_occurred: bool = False
    error_type: Optional[str] = None
    args_hash: Optional[str] = None
    result_size: Optional[int] = None


class RetryConfig(BaseModel):
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True
    retryable_exceptions: Set[str] = field(default_factory=lambda: {
        'DatabaseError', 'ModelError', 'ResourceError', 'TimeoutError'
    })


class CacheConfig(BaseModel):
    """Configuration for caching decorators."""
    ttl_seconds: int = 300
    max_size: int = 1000
    key_generator: Optional[Callable] = None
    invalidation_patterns: List[str] = field(default_factory=list)


# Performance Monitoring Decorators

def performance_monitor(
    operation_name: Optional[str] = None,
    include_memory: bool = False,
    include_cpu: bool = False,
    log_args: bool = False,
    log_result: bool = False,
    track_errors: bool = True
):
    """
    Monitor function performance with comprehensive metrics collection.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        include_memory: Track memory usage during execution
        include_cpu: Track CPU usage during execution
        log_args: Log function arguments (be careful with sensitive data)
        log_result: Log function result size
        track_errors: Track error occurrences and types
        
    Usage:
        @performance_monitor(operation_name="document_processing", include_memory=True)
        async def process_document(doc_id: str):
            return await heavy_processing(doc_id)
    """
    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _execute_with_monitoring(
                    func, args, kwargs, op_name, include_memory, 
                    include_cpu, log_args, log_result, track_errors
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(_execute_with_monitoring(
                    func, args, kwargs, op_name, include_memory,
                    include_cpu, log_args, log_result, track_errors
                ))
            return sync_wrapper
    return decorator


async def _execute_with_monitoring(
    func: Callable,
    args: tuple,
    kwargs: dict,
    op_name: str,
    include_memory: bool,
    include_cpu: bool,
    log_args: bool,
    log_result: bool,
    track_errors: bool
) -> Any:
    """Execute function with comprehensive monitoring."""
    import psutil
    import sys
    
    start_time = time.time()
    process = psutil.Process() if (include_memory or include_cpu) else None
    initial_memory = process.memory_info().rss / 1024 / 1024 if include_memory else None
    
    # Generate args hash for tracking
    args_hash = None
    if log_args:
        try:
            args_str = json.dumps({"args": args, "kwargs": kwargs}, default=str)
            args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
        except Exception:
            args_hash = "hash_failed"
    
    error_occurred = False
    error_type = None
    result = None
    
    with performance_context(op_name, args_hash=args_hash):
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            if track_errors:
                error_occurred = True
                error_type = type(e).__name__
                logger.error(
                    f"Error in monitored operation: {op_name}",
                    error_type=error_type,
                    error_message=str(e),
                    operation=op_name,
                    args_hash=args_hash
                )
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Collect resource metrics
            memory_usage = None
            cpu_percent = None
            
            if include_memory and process:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage = current_memory - (initial_memory or 0)
                except Exception:
                    pass
            
            if include_cpu and process:
                try:
                    cpu_percent = process.cpu_percent()
                except Exception:
                    pass
            
            # Calculate result size
            result_size = None
            if log_result and result is not None:
                try:
                    result_size = sys.getsizeof(result)
                except Exception:
                    pass
            
            # Create metrics object
            metrics = PerformanceMetrics(
                operation_name=op_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                error_occurred=error_occurred,
                error_type=error_type,
                args_hash=args_hash,
                result_size=result_size
            )
            
            # Log metrics
            logger.info(
                f"Operation completed: {op_name}",
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                error_occurred=error_occurred,
                operation=op_name
            )
    
    return result


def track_resource_usage(threshold_memory_mb: float = 100, threshold_cpu_percent: float = 80):
    """
    Track resource usage and warn if thresholds are exceeded.
    
    Args:
        threshold_memory_mb: Memory usage threshold in MB
        threshold_cpu_percent: CPU usage threshold as percentage
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_diff = final_memory - initial_memory
            cpu_percent = process.cpu_percent()
            
            if memory_diff > threshold_memory_mb:
                logger.warning(
                    f"High memory usage detected in {func.__name__}",
                    memory_usage_mb=memory_diff,
                    threshold_mb=threshold_memory_mb,
                    function=func.__name__
                )
            
            if cpu_percent > threshold_cpu_percent:
                logger.warning(
                    f"High CPU usage detected in {func.__name__}",
                    cpu_percent=cpu_percent,
                    threshold_percent=threshold_cpu_percent,
                    function=func.__name__
                )
            
            return result
        return wrapper
    return decorator


# Error Handling Decorators

def error_handler(
    error_types: Optional[Union[Exception, tuple]] = None,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = True,
    error_code: Optional[ErrorCode] = None
):
    """
    Comprehensive error handling with context preservation.
    
    Args:
        error_types: Specific exception types to handle (None for all)
        default_return: Value to return on error (if not reraising)
        log_errors: Whether to log caught errors
        reraise: Whether to reraise exceptions after handling
        error_code: Custom error code for wrapped exceptions
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if error_types and not isinstance(e, error_types):
                    raise
                
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        function=func.__name__,
                        traceback=traceback.format_exc()
                    )
                
                if isinstance(e, BaseCustomException):
                    if reraise:
                        raise
                    return default_return
                
                # Wrap in custom exception if needed
                if error_code and reraise:
                    raise BaseCustomException(
                        message=f"Error in {func.__name__}: {str(e)}",
                        error_code=error_code,
                        details={"original_error": str(e), "function": func.__name__}
                    ) from e
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def safe_execute(
    fallback_value: Any = None,
    log_errors: bool = True,
    error_message: Optional[str] = None
):
    """
    Execute function safely with fallback value on any error.
    
    Args:
        fallback_value: Value to return if function fails
        log_errors: Whether to log errors
        error_message: Custom error message for logs
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    message = error_message or f"Safe execution failed for {func.__name__}"
                    logger.warning(
                        message,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        function=func.__name__,
                        fallback_value=str(fallback_value)
                    )
                return fallback_value
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    max_delay: float = 30.0,
    retryable_errors: Optional[tuple] = None,
    jitter: bool = True
):
    """
    Retry function on failure with intelligent backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Base delay between retries in seconds
        exponential_backoff: Use exponential backoff strategy
        max_delay: Maximum delay between retries
        retryable_errors: Tuple of exception types to retry on
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if retryable_errors and not isinstance(e, retryable_errors):
                        logger.debug(f"Non-retryable error in {func.__name__}: {type(e).__name__}")
                        raise
                    
                    if isinstance(e, BaseCustomException) and not is_retryable_error(e):
                        logger.debug(f"Non-retryable custom error in {func.__name__}: {e.error_code}")
                        raise
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"All retry attempts failed for {func.__name__}",
                            attempts=max_attempts,
                            final_error=str(e),
                            function=func.__name__
                        )
                        raise
                    
                    # Calculate delay
                    if exponential_backoff:
                        retry_delay = min(delay * (2 ** (attempt - 1)), max_delay)
                    else:
                        retry_delay = delay
                    
                    # Add jitter
                    if jitter:
                        import random
                        retry_delay += random.uniform(0, retry_delay * 0.1)
                    
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}, retrying in {retry_delay:.2f}s",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=retry_delay,
                        error=str(e),
                        function=func.__name__
                    )
                    
                    await asyncio.sleep(retry_delay)
            
            # Should never reach here, but just in case
            raise last_exception
        return wrapper
    return decorator


# Validation Decorators

def validate_input(schema: Optional[BaseModel] = None, **validation_kwargs):
    """
    Validate function input using Pydantic models or custom validators.
    
    Args:
        schema: Pydantic model for validation
        **validation_kwargs: Additional validation parameters
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate using Pydantic schema if provided
            if schema:
                try:
                    # Combine args and kwargs for validation
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Validate against schema
                    validated_data = schema(**bound_args.arguments)
                    
                    # Update kwargs with validated data
                    kwargs.update(validated_data.dict())
                    
                except PydanticValidationError as e:
                    raise ValidationError(
                        message=f"Input validation failed for {func.__name__}",
                        field_errors=[
                            {"field": error["loc"], "message": error["msg"]}
                            for error in e.errors()
                        ]
                    )
            
            # Custom validation logic can be added here
            for key, validator in validation_kwargs.items():
                if key in kwargs and callable(validator):
                    if not validator(kwargs[key]):
                        raise ValidationError(
                            message=f"Validation failed for parameter {key} in {func.__name__}"
                        )
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_case_access_decorator(case_id_param: str = "case_id", user_id_param: str = "user_id"):
    """
    Validate user access to specific case.
    
    Args:
        case_id_param: Name of the case_id parameter
        user_id_param: Name of the user_id parameter
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            case_id = kwargs.get(case_id_param)
            user_id = kwargs.get(user_id_param)
            
            if case_id and user_id:
                if not await validate_case_access(case_id, user_id):
                    raise CaseManagementError(
                        message=f"Access denied to case {case_id}",
                        error_code=ErrorCode.CASE_ACCESS_DENIED,
                        case_id=case_id,
                        user_id=user_id
                    )
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Async Operation Decorators

def async_timeout(timeout_seconds: float = 30.0, error_message: Optional[str] = None):
    """
    Add timeout to async operations.
    
    Args:
        timeout_seconds: Timeout in seconds
        error_message: Custom error message for timeout
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                message = error_message or f"Operation {func.__name__} timed out after {timeout_seconds}s"
                logger.error(
                    message,
                    timeout_seconds=timeout_seconds,
                    function=func.__name__
                )
                raise ResourceError(
                    message=message,
                    error_code=ErrorCode.RESOURCE_UNAVAILABLE
                )
        return wrapper
    return decorator


def websocket_progress(
    operation_name: str,
    total_steps: Optional[int] = None,
    case_id_param: str = "case_id"
):
    """
    Track operation progress via WebSocket.
    
    Args:
        operation_name: Name of the operation for progress tracking
        total_steps: Total number of steps (if known)
        case_id_param: Parameter name containing case ID
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            case_id = kwargs.get(case_id_param)
            if not case_id:
                # Try to extract from args based on function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                case_id = bound_args.arguments.get(case_id_param)
            
            progress_id = f"{operation_name}_{case_id}_{int(time.time())}"
            websocket_manager = WebSocketManager()
            
            if case_id:
                await websocket_manager.send_progress_update(
                    case_id=case_id,
                    operation=operation_name,
                    progress=0,
                    total_steps=total_steps,
                    status="started",
                    message=f"Starting {operation_name}"
                )
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                if case_id:
                    await websocket_manager.send_progress_update(
                        case_id=case_id,
                        operation=operation_name,
                        progress=total_steps or 100,
                        total_steps=total_steps or 100,
                        status="completed",
                        message=f"Completed {operation_name}"
                    )
                
                return result
            except Exception as e:
                if case_id:
                    await websocket_manager.send_progress_update(
                        case_id=case_id,
                        operation=operation_name,
                        progress=0,
                        total_steps=total_steps or 100,
                        status="failed",
                        message=f"Failed {operation_name}: {str(e)}"
                    )
                raise
        return wrapper
    return decorator


# Caching Decorators

def cache_result(
    ttl_seconds: int = 300,
    max_size: int = 1000,
    key_generator: Optional[Callable] = None,
    skip_on_error: bool = True
):
    """
    Cache function results with TTL and LRU eviction.
    
    Args:
        ttl_seconds: Time to live for cached results
        max_size: Maximum number of cached items
        key_generator: Custom function to generate cache keys
        skip_on_error: Skip caching if function raises an exception
    """
    def decorator(func: F) -> F:
        cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        cache_lock = threading.Lock()
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            with cache_lock:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}", cache_key=cache_key)
                    return cached_result
            
            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                with cache_lock:
                    cache[cache_key] = result
                    logger.debug(f"Cached result for {func.__name__}", cache_key=cache_key)
                
                return result
            except Exception as e:
                if not skip_on_error:
                    # Cache the exception
                    with cache_lock:
                        cache[cache_key] = e
                raise
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function arguments."""
    try:
        key_data = {
            "function": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    except Exception:
        # Fallback to simple string representation
        return f"{func_name}_{hash((args, tuple(sorted(kwargs.items()))))}"


# Rate Limiting Decorators

def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    per_user: bool = True,
    user_id_param: str = "user_id"
):
    """
    Rate limit function calls per user or globally.
    
    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        per_user: Apply rate limiting per user
        user_id_param: Parameter name for user ID
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Determine rate limit key
            if per_user:
                user_id = kwargs.get(user_id_param)
                if not user_id:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    user_id = bound_args.arguments.get(user_id_param, "anonymous")
                rate_key = f"{func.__name__}_{user_id}"
            else:
                rate_key = func.__name__
            
            # Clean old entries
            request_times = _rate_limit_storage[rate_key]
            while request_times and current_time - request_times[0] > window_seconds:
                request_times.popleft()
            
            # Check rate limit
            if len(request_times) >= max_requests:
                logger.warning(
                    f"Rate limit exceeded for {func.__name__}",
                    rate_key=rate_key,
                    max_requests=max_requests,
                    window_seconds=window_seconds
                )
                raise ResourceError(
                    message=f"Rate limit exceeded for {func.__name__}",
                    error_code=ErrorCode.RESOURCE_QUOTA_EXCEEDED,
                    resource_type="api_calls",
                    current_usage=len(request_times),
                    limit=max_requests
                )
            
            # Record request
            request_times.append(current_time)
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Database Transaction Decorators

def transactional(rollback_on_error: bool = True, isolation_level: Optional[str] = None):
    """
    Execute function within a database transaction.
    
    Args:
        rollback_on_error: Automatically rollback on exception
        isolation_level: Transaction isolation level
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from ..core.database import get_database_manager
            
            db_manager = get_database_manager()
            
            async with db_manager.transaction(
                rollback_on_error=rollback_on_error,
                isolation_level=isolation_level
            ):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        return wrapper
    return decorator


# Utility Functions

def combine_decorators(*decorators):
    """
    Combine multiple decorators into a single decorator.
    
    Args:
        *decorators: Decorators to combine
        
    Usage:
        @combine_decorators(
            performance_monitor(include_memory=True),
            retry_on_failure(max_attempts=3),
            error_handler(log_errors=True)
        )
        def my_function():
            pass
    """
    def decorator(func: F) -> F:
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator


def conditional_decorator(condition: Union[bool, Callable], decorator_func: Callable):
    """
    Apply decorator conditionally based on condition.
    
    Args:
        condition: Boolean or callable that returns boolean
        decorator_func: Decorator to apply if condition is True
    """
    def decorator(func: F) -> F:
        should_apply = condition() if callable(condition) else condition
        if should_apply:
            return decorator_func(func)
        return func
    return decorator


# Context managers for use within decorated functions

@asynccontextmanager
async def progress_context(operation: str, case_id: str, total_steps: int = 100):
    """Context manager for progress tracking within functions."""
    websocket_manager = WebSocketManager()
    
    await websocket_manager.send_progress_update(
        case_id=case_id,
        operation=operation,
        progress=0,
        total_steps=total_steps,
        status="started"
    )
    
    try:
        yield
        await websocket_manager.send_progress_update(
            case_id=case_id,
            operation=operation,
            progress=total_steps,
            total_steps=total_steps,
            status="completed"
        )
    except Exception as e:
        await websocket_manager.send_progress_update(
            case_id=case_id,
            operation=operation,
            progress=0,
            total_steps=total_steps,
            status="failed",
            message=str(e)
        )
        raise


@contextmanager
def performance_timing(operation_name: str):
    """Context manager for performance timing."""
    start_time = time.time()
    logger.debug(f"Starting operation: {operation_name}")
    
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000
        logger.info(f"Operation completed: {operation_name}", duration_ms=duration)


# Export commonly used decorators and utilities
__all__ = [
    'performance_monitor',
    'track_resource_usage',
    'error_handler',
    'safe_execute',
    'retry_on_failure',
    'validate_input',
    'validate_case_access_decorator',
    'async_timeout',
    'websocket_progress',
    'cache_result',
    'rate_limit',
    'transactional',
    'combine_decorators',
    'conditional_decorator',
    'progress_context',
    'performance_timing',
    'PerformanceMetrics',
    'RetryConfig',
    'CacheConfig'
]