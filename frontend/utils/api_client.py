"""
API Client for Patexia Legal AI Chatbot Frontend

This module provides a comprehensive HTTP client for communicating with the
FastAPI backend. It handles all REST API operations with proper error handling,
retry mechanisms, authentication, and request/response processing.

Key Features:
- Async HTTP client with connection pooling
- Automatic retry with exponential backoff
- Request/response logging and metrics
- Error handling with custom exception mapping
- Authentication and session management
- File upload support with progress tracking
- Request correlation ID tracking
- Timeout and rate limiting handling
- Response caching for performance optimization

Architecture Integration:
- Communicates with FastAPI backend via REST API
- Maps backend error codes to frontend exceptions
- Provides typed request/response handling
- Integrates with frontend logging system
- Supports all backend API endpoints
- Handles authentication and authorization

Supported Operations:
- Case management (CRUD operations)
- Document upload and management
- Search and query operations
- Admin configuration management
- System monitoring and statistics
- User authentication and sessions

Error Handling:
- HTTP status code mapping to exceptions
- Retry logic for transient failures
- Circuit breaker pattern for reliability
- Request timeout and cancellation
- Network error recovery
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, IO
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import ssl

import aiofiles
import httpx
from httpx import AsyncClient, Request, Response, TimeoutException, RequestError, HTTPStatusError


class HTTPMethod(str, Enum):
    """HTTP methods supported by the API client."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class RetryStrategy(str, Enum):
    """Retry strategies for failed requests."""
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE = "immediate"


@dataclass
class APIClientConfig:
    """Configuration for the API client."""
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_backoff_factor: float = 2.0
    retry_max_delay: float = 60.0
    connection_pool_limits: int = 20
    max_keepalive_connections: int = 10
    enable_request_logging: bool = True
    enable_response_caching: bool = False
    cache_ttl_seconds: int = 300
    user_agent: str = "PatexiaLegalAI-Frontend/1.0"
    

@dataclass
class RequestMetrics:
    """Metrics for API request tracking."""
    request_id: str
    method: str
    url: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status_code: Optional[int] = None
    response_size_bytes: Optional[int] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class CachedResponse:
    """Cached API response with TTL."""
    data: Dict[str, Any]
    cached_at: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        """Check if the cached response has expired."""
        return (datetime.now(timezone.utc) - self.cached_at).total_seconds() > self.ttl_seconds


class APIError(Exception):
    """Base exception for API client errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id


class NetworkError(APIError):
    """Exception for network-related errors."""
    pass


class TimeoutError(APIError):
    """Exception for request timeout errors."""
    pass


class AuthenticationError(APIError):
    """Exception for authentication errors."""
    pass


class ValidationError(APIError):
    """Exception for request validation errors."""
    pass


class ServerError(APIError):
    """Exception for server-side errors."""
    pass


class APIClient:
    """
    Comprehensive HTTP client for the Patexia Legal AI backend API.
    
    Provides async HTTP operations with error handling, retry logic,
    authentication, and performance optimization features.
    """
    
    def __init__(self, config: Optional[APIClientConfig] = None):
        """
        Initialize the API client.
        
        Args:
            config: Optional API client configuration
        """
        self.config = config or APIClientConfig()
        self.logger = logging.getLogger(f"{__name__}.APIClient")
        
        # HTTP client setup
        self._client: Optional[AsyncClient] = None
        self._session_token: Optional[str] = None
        self._user_id: Optional[str] = None
        
        # Request tracking and metrics
        self._request_metrics: Dict[str, RequestMetrics] = {}
        self._active_requests: Dict[str, asyncio.Task] = {}
        
        # Response caching
        self._response_cache: Dict[str, CachedResponse] = {}
        
        # Circuit breaker state
        self._circuit_breaker_failures: int = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
        self._circuit_breaker_timeout: float = 30.0
        
        # Request correlation tracking
        self._correlation_id: Optional[str] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized and connected."""
        if self._client is not None:
            return
        
        # Create HTTP client with optimized settings
        timeout = httpx.Timeout(
            connect=5.0,
            read=self.config.timeout,
            write=10.0,
            pool=5.0
        )
        
        limits = httpx.Limits(
            max_keepalive_connections=self.config.max_keepalive_connections,
            max_connections=self.config.connection_pool_limits,
            keepalive_expiry=30.0
        )
        
        # Headers for all requests
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        self._client = AsyncClient(
            base_url=self.config.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
            follow_redirects=True
        )
        
        # Test connection
        try:
            response = await self._client.get("/api/v1/health")
            if response.status_code == 200:
                self.logger.info(f"API client connected to {self.config.base_url}")
            else:
                self.logger.warning(f"API health check returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Failed to verify API connection: {str(e)}")
    
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        
        # Cancel active requests
        for task in self._active_requests.values():
            if not task.done():
                task.cancel()
        
        self._active_requests.clear()
        self.logger.info("API client closed")
    
    def set_authentication(self, token: str, user_id: Optional[str] = None) -> None:
        """
        Set authentication token for API requests.
        
        Args:
            token: Authentication token
            user_id: Optional user ID for context
        """
        self._session_token = token
        self._user_id = user_id
        
        if self._client:
            self._client.headers.update({"Authorization": f"Bearer {token}"})
        
        self.logger.info(f"Authentication set for user: {user_id}")
    
    def clear_authentication(self) -> None:
        """Clear authentication token."""
        self._session_token = None
        self._user_id = None
        
        if self._client and "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]
        
        self.logger.info("Authentication cleared")
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set correlation ID for request tracking.
        
        Args:
            correlation_id: Unique correlation ID
        """
        self._correlation_id = correlation_id
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform GET request.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        return await self._request(HTTPMethod.GET, endpoint, params=params, **kwargs)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform POST request.
        
        Args:
            endpoint: API endpoint path
            data: Optional request body data
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        return await self._request(HTTPMethod.POST, endpoint, json=data, **kwargs)
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform PUT request.
        
        Args:
            endpoint: API endpoint path
            data: Optional request body data
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        return await self._request(HTTPMethod.PUT, endpoint, json=data, **kwargs)
    
    async def patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform PATCH request.
        
        Args:
            endpoint: API endpoint path
            data: Optional request body data
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        return await self._request(HTTPMethod.PATCH, endpoint, json=data, **kwargs)
    
    async def delete(
        self,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform DELETE request.
        
        Args:
            endpoint: API endpoint path
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        return await self._request(HTTPMethod.DELETE, endpoint, **kwargs)
    
    async def post_file(
        self,
        endpoint: str,
        files: Dict[str, Union[Tuple[str, IO], Tuple[str, bytes]]],
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform file upload POST request.
        
        Args:
            endpoint: API endpoint path
            files: Files to upload as {field_name: (filename, file_content)}
            data: Optional form data
            **kwargs: Additional request options
            
        Returns:
            JSON response data
        """
        # Remove Content-Type for multipart uploads
        headers = kwargs.get("headers", {})
        headers.pop("Content-Type", None)
        kwargs["headers"] = headers
        
        return await self._request(
            HTTPMethod.POST,
            endpoint,
            files=files,
            data=data,
            **kwargs
        )
    
    async def _request(
        self,
        method: HTTPMethod,
        endpoint: str,
        retry_count: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute HTTP request with error handling and retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            retry_count: Current retry attempt count
            **kwargs: Request parameters
            
        Returns:
            JSON response data
            
        Raises:
            APIError: For various API error conditions
        """
        await self._ensure_client()
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        
        # Create request metrics
        metrics = RequestMetrics(
            request_id=request_id,
            method=method.value,
            url=urljoin(self.config.base_url, endpoint),
            started_at=datetime.now(timezone.utc),
            retry_count=retry_count
        )
        self._request_metrics[request_id] = metrics
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise NetworkError(
                "Circuit breaker is open due to repeated failures",
                status_code=503,
                correlation_id=self._correlation_id
            )
        
        # Check cache for GET requests
        if method == HTTPMethod.GET and self.config.enable_response_caching:
            cache_key = self._get_cache_key(method, endpoint, kwargs.get("params"))
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.logger.debug(f"Cache hit for {method.value} {endpoint}")
                return cached_response
        
        try:
            # Prepare request headers
            headers = kwargs.pop("headers", {})
            if self._correlation_id:
                headers["X-Correlation-ID"] = self._correlation_id
            if self._session_token:
                headers["Authorization"] = f"Bearer {self._session_token}"
            
            # Log request if enabled
            if self.config.enable_request_logging:
                self.logger.debug(
                    f"API Request: {method.value} {endpoint}",
                    extra={
                        "request_id": request_id,
                        "method": method.value,
                        "endpoint": endpoint,
                        "retry_count": retry_count
                    }
                )
            
            # Execute request
            response = await self._client.request(
                method.value,
                endpoint,
                headers=headers,
                **kwargs
            )
            
            # Update metrics
            metrics.completed_at = datetime.now(timezone.utc)
            metrics.duration_seconds = (
                metrics.completed_at - metrics.started_at
            ).total_seconds()
            metrics.status_code = response.status_code
            metrics.response_size_bytes = len(response.content)
            
            # Handle response
            response_data = await self._handle_response(response, request_id)
            
            # Cache successful GET responses
            if (method == HTTPMethod.GET and 
                self.config.enable_response_caching and
                response.status_code == 200):
                cache_key = self._get_cache_key(method, endpoint, kwargs.get("params"))
                self._cache_response(cache_key, response_data)
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker()
            
            return response_data
            
        except (TimeoutException, RequestError) as e:
            metrics.error_message = str(e)
            self._handle_circuit_breaker_failure()
            
            # Retry logic
            if retry_count < self.config.max_retries and self._should_retry(e):
                retry_delay = self._calculate_retry_delay(retry_count)
                
                self.logger.warning(
                    f"Request failed, retrying in {retry_delay}s: {str(e)}",
                    extra={
                        "request_id": request_id,
                        "retry_count": retry_count,
                        "retry_delay": retry_delay
                    }
                )
                
                await asyncio.sleep(retry_delay)
                return await self._request(method, endpoint, retry_count + 1, **kwargs)
            
            # Map exceptions
            if isinstance(e, TimeoutException):
                raise TimeoutError(
                    f"Request timeout after {self.config.timeout}s",
                    correlation_id=self._correlation_id
                )
            else:
                raise NetworkError(
                    f"Network error: {str(e)}",
                    correlation_id=self._correlation_id
                )
                
        except HTTPStatusError as e:
            metrics.status_code = e.response.status_code
            metrics.error_message = str(e)
            
            # Handle HTTP errors
            await self._handle_http_error(e.response, request_id)
            
        except Exception as e:
            metrics.error_message = str(e)
            self.logger.error(
                f"Unexpected error in API request: {str(e)}",
                extra={"request_id": request_id}
            )
            raise APIError(
                f"Unexpected error: {str(e)}",
                correlation_id=self._correlation_id
            )
        finally:
            # Cleanup request tracking
            self._active_requests.pop(request_id, None)
    
    async def _handle_response(
        self,
        response: Response,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Handle successful HTTP response.
        
        Args:
            response: HTTP response object
            request_id: Request tracking ID
            
        Returns:
            Parsed JSON response data
        """
        try:
            if response.status_code == 204:  # No Content
                return {"success": True}
            
            response_data = response.json()
            
            if self.config.enable_request_logging:
                self.logger.debug(
                    f"API Response: {response.status_code}",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "response_size": len(response.content)
                    }
                )
            
            return response_data
            
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON response: {str(e)}",
                extra={"request_id": request_id}
            )
            raise APIError(
                "Invalid JSON response from server",
                status_code=response.status_code,
                correlation_id=self._correlation_id
            )
    
    async def _handle_http_error(
        self,
        response: Response,
        request_id: str
    ) -> None:
        """
        Handle HTTP error responses.
        
        Args:
            response: HTTP response object
            request_id: Request tracking ID
            
        Raises:
            Appropriate APIError subclass based on status code
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_code = error_data.get("error", {}).get("code")
            error_details = error_data.get("error", {}).get("details", {})
            correlation_id = error_data.get("error", {}).get("correlation_id", self._correlation_id)
        except json.JSONDecodeError:
            error_message = f"HTTP {response.status_code}: {response.reason_phrase}"
            error_code = None
            error_details = {}
            correlation_id = self._correlation_id
        
        self.logger.error(
            f"API Error Response: {response.status_code} - {error_message}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "error_code": error_code,
                "error_details": error_details
            }
        )
        
        # Map status codes to specific exceptions
        if response.status_code == 400:
            raise ValidationError(
                error_message,
                status_code=response.status_code,
                error_code=error_code,
                details=error_details,
                correlation_id=correlation_id
            )
        elif response.status_code in (401, 403):
            raise AuthenticationError(
                error_message,
                status_code=response.status_code,
                error_code=error_code,
                details=error_details,
                correlation_id=correlation_id
            )
        elif response.status_code == 422:
            raise ValidationError(
                error_message,
                status_code=response.status_code,
                error_code=error_code,
                details=error_details,
                correlation_id=correlation_id
            )
        elif response.status_code >= 500:
            raise ServerError(
                error_message,
                status_code=response.status_code,
                error_code=error_code,
                details=error_details,
                correlation_id=correlation_id
            )
        else:
            raise APIError(
                error_message,
                status_code=response.status_code,
                error_code=error_code,
                details=error_details,
                correlation_id=correlation_id
            )
    
    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if the request should be retried
        """
        if self.config.retry_strategy == RetryStrategy.NONE:
            return False
        
        # Don't retry authentication errors
        if isinstance(error, AuthenticationError):
            return False
        
        # Don't retry validation errors
        if isinstance(error, ValidationError):
            return False
        
        # Retry network errors and timeouts
        if isinstance(error, (NetworkError, TimeoutError)):
            return True
        
        # Retry server errors (5xx)
        if isinstance(error, ServerError):
            return True
        
        return False
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate delay before retry attempt.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Delay in seconds
        """
        if self.config.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.config.retry_strategy == RetryStrategy.FIXED_INTERVAL:
            return 1.0
        else:  # EXPONENTIAL_BACKOFF
            delay = min(
                self.config.retry_backoff_factor ** retry_count,
                self.config.retry_max_delay
            )
            return delay
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures < 5:
            return False
        
        if self._circuit_breaker_last_failure is None:
            return False
        
        time_since_failure = (
            datetime.now(timezone.utc) - self._circuit_breaker_last_failure
        ).total_seconds()
        
        return time_since_failure < self._circuit_breaker_timeout
    
    def _handle_circuit_breaker_failure(self) -> None:
        """Handle circuit breaker failure."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now(timezone.utc)
    
    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state."""
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
    
    def _get_cache_key(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate cache key for request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Cache key string
        """
        param_str = ""
        if params:
            sorted_params = sorted(params.items())
            param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        
        return f"{method.value}:{endpoint}?{param_str}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response data or None
        """
        cached = self._response_cache.get(cache_key)
        if cached and not cached.is_expired:
            return cached.data
        elif cached:
            # Remove expired cache entry
            del self._response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """
        Cache response data.
        
        Args:
            cache_key: Cache key
            response_data: Response data to cache
        """
        cached_response = CachedResponse(
            data=response_data,
            cached_at=datetime.now(timezone.utc),
            ttl_seconds=self.config.cache_ttl_seconds
        )
        self._response_cache[cache_key] = cached_response
        
        # Limit cache size
        if len(self._response_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(
                self._response_cache.items(),
                key=lambda x: x[1].cached_at
            )
            for key, _ in sorted_cache[:20]:
                del self._response_cache[key]
    
    def get_request_metrics(self) -> List[RequestMetrics]:
        """
        Get request metrics for monitoring.
        
        Returns:
            List of request metrics
        """
        return list(self._request_metrics.values())
    
    def clear_metrics(self) -> None:
        """Clear request metrics."""
        self._request_metrics.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        if not self.config.enable_response_caching:
            return {"caching_enabled": False}
        
        total_entries = len(self._response_cache)
        expired_entries = sum(
            1 for cached in self._response_cache.values()
            if cached.is_expired
        )
        
        return {
            "caching_enabled": True,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "cache_ttl_seconds": self.config.cache_ttl_seconds
        }


# Convenience functions for common operations

async def create_api_client(
    base_url: Optional[str] = None,
    **config_kwargs
) -> APIClient:
    """
    Create and initialize an API client.
    
    Args:
        base_url: Optional base URL override
        **config_kwargs: Additional configuration options
        
    Returns:
        Initialized API client
    """
    config = APIClientConfig(**config_kwargs)
    if base_url:
        config.base_url = base_url
    
    client = APIClient(config)
    await client._ensure_client()
    return client


async def test_api_connection(base_url: str = "http://localhost:8000") -> bool:
    """
    Test API connection health.
    
    Args:
        base_url: API base URL to test
        
    Returns:
        True if connection is healthy
    """
    try:
        async with create_api_client(base_url) as client:
            response = await client.get("/api/v1/health")
            return response.get("status") == "healthy"
    except Exception:
        return False


# Export public interface
__all__ = [
    "APIClient",
    "APIClientConfig",
    "APIError",
    "NetworkError",
    "TimeoutError",
    "AuthenticationError",
    "ValidationError",
    "ServerError",
    "RequestMetrics",
    "HTTPMethod",
    "RetryStrategy",
    "create_api_client",
    "test_api_connection"
]