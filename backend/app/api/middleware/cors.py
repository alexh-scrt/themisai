"""
CORS Middleware Configuration for Patexia Legal AI Chatbot Backend

This module provides comprehensive Cross-Origin Resource Sharing (CORS) middleware
configuration for the FastAPI backend. It handles secure cross-origin requests
between the Gradio frontend and FastAPI backend, with environment-specific
configurations for development and production deployments.

Key Features:
- Environment-aware CORS configuration (development vs production)
- Secure default settings with explicit origin whitelisting
- WebSocket CORS support for real-time communication
- Configurable allowed methods and headers
- Security headers integration for enhanced protection
- Development-friendly settings with production security
- Hot-reload configuration support
- Comprehensive request logging and monitoring
- Legal document upload CORS handling
- Admin panel access control

Security Considerations:
- Strict origin validation for production environments
- Secure credential handling for authenticated requests
- Protection against CSRF attacks
- Proper preflight request handling
- WebSocket upgrade CORS validation
- File upload security headers
- Legal document confidentiality protection

Architecture Integration:
- Integrates with FastAPI application middleware stack
- Supports hot-reload configuration via settings management
- Coordinates with authentication middleware
- Provides logging integration for security monitoring
- Handles both REST API and WebSocket connections
- Supports multi-environment deployment scenarios
"""

import logging
import re
from typing import List, Optional, Set, Dict, Any, Union
from urllib.parse import urlparse
import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from backend.config.settings import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class CORSConfiguration:
    """CORS configuration management with environment-specific settings."""
    
    def __init__(self):
        """Initialize CORS configuration based on environment."""
        self.settings = get_settings()
        self.is_development = self._is_development_environment()
        self.logger = logging.getLogger(f"{__name__}.CORSConfiguration")
        
        # Initialize CORS settings
        self._allowed_origins = self._get_allowed_origins()
        self._allowed_methods = self._get_allowed_methods()
        self._allowed_headers = self._get_allowed_headers()
        self._exposed_headers = self._get_exposed_headers()
        self._allow_credentials = self._get_allow_credentials()
        self._max_age = self._get_max_age()
        
        self.logger.info(
            f"CORS configuration initialized for {'development' if self.is_development else 'production'} environment"
        )
    
    def _is_development_environment(self) -> bool:
        """Determine if running in development environment."""
        env = os.getenv("ENVIRONMENT", "development").lower()
        debug_mode = os.getenv("DEBUG", "true").lower() == "true"
        return env in ["development", "dev", "local"] or debug_mode
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins based on environment and configuration."""
        if self.is_development:
            # Development: Allow common development ports and localhost
            development_origins = [
                "http://localhost:7860",    # Gradio frontend
                "http://127.0.0.1:7860",   # Gradio frontend (alternative)
                "http://localhost:8000",    # FastAPI docs
                "http://127.0.0.1:8000",   # FastAPI docs (alternative)
                "http://localhost:3000",    # Common React dev server
                "http://127.0.0.1:3000",   # Common React dev server (alternative)
                "http://localhost:8080",    # Common Vue dev server
                "http://127.0.0.1:8080",   # Common Vue dev server (alternative)
                "http://localhost:5173",    # Vite dev server
                "http://127.0.0.1:5173",   # Vite dev server (alternative)
            ]
            
            # Add any additional development origins from configuration
            config_origins = getattr(self.settings, 'cors_allowed_origins', [])
            if config_origins:
                development_origins.extend(config_origins)
            
            self.logger.info(f"Development CORS origins: {development_origins}")
            return development_origins
        else:
            # Production: Strict origin control
            production_origins = []
            
            # Get production origins from configuration
            config_origins = getattr(self.settings, 'cors_allowed_origins', [])
            if config_origins:
                production_origins.extend(config_origins)
            
            # Add environment-specific origins
            if hasattr(self.settings, 'frontend_url') and self.settings.frontend_url:
                production_origins.append(self.settings.frontend_url)
            
            # Fallback to localhost if no origins configured (for initial setup)
            if not production_origins:
                production_origins = [
                    "http://localhost:7860",
                    "https://localhost:7860"
                ]
                self.logger.warning("No production CORS origins configured, using localhost fallback")
            
            # Validate production origins
            validated_origins = []
            for origin in production_origins:
                if self._validate_origin(origin):
                    validated_origins.append(origin)
                else:
                    self.logger.warning(f"Invalid origin rejected: {origin}")
            
            self.logger.info(f"Production CORS origins: {validated_origins}")
            return validated_origins
    
    def _get_allowed_methods(self) -> List[str]:
        """Get allowed HTTP methods."""
        if self.is_development:
            # Development: Allow all common methods
            return ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
        else:
            # Production: Restrict to necessary methods
            return ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    
    def _get_allowed_headers(self) -> List[str]:
        """Get allowed request headers."""
        base_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-Correlation-ID",
            "Cache-Control",
            "Pragma",
            "Expires",
        ]
        
        # Add legal document specific headers
        legal_headers = [
            "X-Case-ID",
            "X-Document-Type",
            "X-Processing-Priority",
            "X-User-Role",
        ]
        
        # Add development headers if needed
        if self.is_development:
            dev_headers = [
                "X-Debug-Mode",
                "X-Trace-ID",
                "X-Request-ID",
            ]
            return base_headers + legal_headers + dev_headers
        
        return base_headers + legal_headers
    
    def _get_exposed_headers(self) -> List[str]:
        """Get headers exposed to the frontend."""
        base_exposed = [
            "Content-Type",
            "Content-Length",
            "X-Total-Count",
            "X-Correlation-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
        
        # Add legal document processing headers
        legal_exposed = [
            "X-Processing-Status",
            "X-Document-Count",
            "X-Search-Results-Count",
            "X-Case-ID",
        ]
        
        return base_exposed + legal_exposed
    
    def _get_allow_credentials(self) -> bool:
        """Determine if credentials should be allowed."""
        # Allow credentials for authenticated legal document access
        return True
    
    def _get_max_age(self) -> int:
        """Get preflight cache max age in seconds."""
        if self.is_development:
            # Development: Short cache for quick iteration
            return 300  # 5 minutes
        else:
            # Production: Longer cache for performance
            return 3600  # 1 hour
    
    def _validate_origin(self, origin: str) -> bool:
        """
        Validate origin URL format and security.
        
        Args:
            origin: Origin URL to validate
            
        Returns:
            True if origin is valid and secure
        """
        try:
            parsed = urlparse(origin)
            
            # Check for valid scheme
            if parsed.scheme not in ["http", "https"]:
                return False
            
            # In production, prefer HTTPS
            if not self.is_development and parsed.scheme != "https":
                self.logger.warning(f"Non-HTTPS origin in production: {origin}")
            
            # Check for valid hostname
            if not parsed.hostname:
                return False
            
            # Reject dangerous patterns
            dangerous_patterns = [
                r"javascript:",
                r"data:",
                r"vbscript:",
                r"file:",
                r"\*",
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, origin, re.IGNORECASE):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating origin {origin}: {e}")
            return False
    
    def get_cors_kwargs(self) -> Dict[str, Any]:
        """
        Get CORS middleware configuration kwargs.
        
        Returns:
            Dictionary of CORS configuration parameters
        """
        return {
            "allow_origins": self._allowed_origins,
            "allow_credentials": self._allow_credentials,
            "allow_methods": self._allowed_methods,
            "allow_headers": self._allowed_headers,
            "expose_headers": self._exposed_headers,
            "max_age": self._max_age,
        }
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if an origin is allowed.
        
        Args:
            origin: Origin to check
            
        Returns:
            True if origin is allowed
        """
        if not origin:
            return False
        
        # Exact match
        if origin in self._allowed_origins:
            return True
        
        # In development, be more permissive for localhost
        if self.is_development:
            parsed = urlparse(origin)
            if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                return True
        
        return False
    
    def log_cors_request(self, request: Request, response: Response) -> None:
        """
        Log CORS request details for monitoring.
        
        Args:
            request: HTTP request
            response: HTTP response
        """
        origin = request.headers.get("origin")
        method = request.method
        path = request.url.path
        
        if origin:
            allowed = self.is_origin_allowed(origin)
            self.logger.info(
                f"CORS request: {method} {path} from {origin} - {'allowed' if allowed else 'blocked'}"
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    def __init__(self, app: ASGIApp, cors_config: CORSConfiguration):
        super().__init__(app)
        self.cors_config = cors_config
        self.logger = logging.getLogger(f"{__name__}.SecurityHeadersMiddleware")
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        security_headers = self._get_security_headers(request)
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Log CORS request
        self.cors_config.log_cors_request(request, response)
        
        return response
    
    def _get_security_headers(self, request: Request) -> Dict[str, str]:
        """Get security headers based on request and environment."""
        headers = {}
        
        # Basic security headers
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        if self.cors_config.is_development:
            # Development: More permissive CSP
            csp = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' localhost:* 127.0.0.1:*; "
                "img-src 'self' data: blob: localhost:* 127.0.0.1:*; "
                "connect-src 'self' ws: wss: localhost:* 127.0.0.1:*; "
                "font-src 'self' data:; "
                "style-src 'self' 'unsafe-inline' localhost:* 127.0.0.1:*"
            )
        else:
            # Production: Strict CSP
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "connect-src 'self' wss:; "
                "font-src 'self' data:; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        
        headers["Content-Security-Policy"] = csp
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Permissions Policy (formerly Feature Policy)
        permissions_policy = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=()"
        )
        headers["Permissions-Policy"] = permissions_policy
        
        return headers


def setup_cors_middleware(app: FastAPI) -> None:
    """
    Setup CORS middleware and security headers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Initialize CORS configuration
    cors_config = CORSConfiguration()
    
    # Add trusted host middleware first
    if not cors_config.is_development:
        # Production: Strict host validation
        trusted_hosts = ["localhost", "127.0.0.1"]
        
        # Add configured hosts
        if hasattr(cors_config.settings, 'trusted_hosts'):
            trusted_hosts.extend(cors_config.settings.trusted_hosts)
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
        logger.info(f"Trusted hosts configured: {trusted_hosts}")
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware, cors_config=cors_config)
    
    # Add CORS middleware
    cors_kwargs = cors_config.get_cors_kwargs()
    app.add_middleware(CORSMiddleware, **cors_kwargs)
    
    logger.info("CORS middleware configured successfully")
    
    # Log configuration summary
    logger.info(
        f"CORS Configuration Summary:\n"
        f"  Environment: {'development' if cors_config.is_development else 'production'}\n"
        f"  Allowed Origins: {cors_kwargs['allow_origins']}\n"
        f"  Allow Credentials: {cors_kwargs['allow_credentials']}\n"
        f"  Allowed Methods: {cors_kwargs['allow_methods']}\n"
        f"  Max Age: {cors_kwargs['max_age']} seconds"
    )


def validate_cors_configuration() -> bool:
    """
    Validate CORS configuration for security issues.
    
    Returns:
        True if configuration is secure, False otherwise
    """
    try:
        cors_config = CORSConfiguration()
        
        # Check for wildcard origins in production
        if not cors_config.is_development:
            if "*" in cors_config._allowed_origins:
                logger.error("Wildcard origin (*) not allowed in production")
                return False
        
        # Validate each origin
        for origin in cors_config._allowed_origins:
            if not cors_config._validate_origin(origin):
                logger.error(f"Invalid origin in configuration: {origin}")
                return False
        
        # Check for HTTPS in production
        if not cors_config.is_development:
            http_origins = [o for o in cors_config._allowed_origins if o.startswith("http://")]
            if http_origins:
                logger.warning(f"HTTP origins in production (consider HTTPS): {http_origins}")
        
        logger.info("CORS configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"CORS configuration validation failed: {e}")
        return False


def get_cors_test_endpoints() -> List[Dict[str, str]]:
    """
    Get test endpoints for CORS validation.
    
    Returns:
        List of test endpoint configurations
    """
    return [
        {
            "method": "GET",
            "path": "/api/v1/health",
            "description": "Health check endpoint"
        },
        {
            "method": "POST",
            "path": "/api/v1/cases",
            "description": "Case creation endpoint"
        },
        {
            "method": "POST",
            "path": "/api/v1/documents/upload",
            "description": "Document upload endpoint"
        },
        {
            "method": "GET",
            "path": "/api/v1/search",
            "description": "Search endpoint"
        },
        {
            "method": "OPTIONS",
            "path": "/api/v1/cases",
            "description": "Preflight request test"
        }
    ]


# Export public interface
__all__ = [
    "CORSConfiguration",
    "SecurityHeadersMiddleware", 
    "setup_cors_middleware",
    "validate_cors_configuration",
    "get_cors_test_endpoints"
]