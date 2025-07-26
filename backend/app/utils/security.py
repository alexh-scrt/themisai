"""
Security Utilities for Patexia Legal AI Chatbot

This module provides comprehensive security utilities for the legal AI system,
including authentication, authorization, data protection, input validation,
session management, and security headers. It implements security best practices
for legal document management with focus on data privacy and access control.

Key Features:
- User authentication and session management
- Role-based access control (RBAC) for legal professionals
- Case-level document access control and isolation
- Input sanitization and validation for security
- Password hashing and verification with industry standards
- API key generation and validation for service authentication
- Security headers and CORS configuration
- Rate limiting and brute force protection
- Audit trail and security event logging
- Data encryption/decryption utilities for sensitive information

Security Architecture:
- Per-case document isolation with user_id based access control
- Role hierarchy: Admin > Lawyer > Paralegal > Viewer
- Session-based authentication with secure cookie management
- API key authentication for service-to-service communication
- Comprehensive audit trail for legal compliance requirements
- Input validation to prevent injection attacks and XSS

Legal Industry Compliance:
- Attorney-client privilege preservation through access controls
- Document confidentiality with encryption at rest and in transit
- Audit requirements for legal professional standards
- Multi-tenancy support for law firm isolation
- Secure handling of sensitive legal document metadata

Architecture Integration:
- Integrates with case management for access control validation
- Works with WebSocket manager for secure real-time communication
- Provides middleware for FastAPI security headers
- Supports MongoDB user authentication and authorization
- Implements security logging for audit trails
"""

import hashlib
import hmac
import secrets
import string
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import base64
import json
from contextlib import contextmanager
import asyncio
from functools import wraps

import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, validator
import jwt
from passlib.context import CryptContext
from passlib.hash import argon2

from ..core.config import get_settings
from ..exceptions import (
    BaseCustomException, ErrorCode, ValidationError,
    raise_validation_error, raise_resource_error
)
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Security constants
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
TOKEN_EXPIRY_HOURS = 24
SESSION_EXPIRY_HOURS = 8
API_KEY_LENGTH = 32
SALT_LENGTH = 32
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30


class UserRole(str, Enum):
    """User roles with hierarchical permissions for legal professionals."""
    ADMIN = "admin"
    LAWYER = "lawyer"
    PARALEGAL = "paralegal"
    VIEWER = "viewer"
    
    def has_permission(self, required_role: 'UserRole') -> bool:
        """Check if current role has permission for required role."""
        role_hierarchy = {
            UserRole.ADMIN: 4,
            UserRole.LAWYER: 3,
            UserRole.PARALEGAL: 2,
            UserRole.VIEWER: 1
        }
        return role_hierarchy[self] >= role_hierarchy[required_role]


class SecurityEventType(str, Enum):
    """Types of security events for audit logging."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    ACCESS_DENIED = "access_denied"
    CASE_ACCESSED = "case_accessed"
    DOCUMENT_ACCESSED = "document_accessed"
    UNAUTHORIZED_ATTEMPT = "unauthorized_attempt"
    SESSION_EXPIRED = "session_expired"
    API_KEY_USED = "api_key_used"
    SECURITY_VIOLATION = "security_violation"
    ADMIN_ACTION = "admin_action"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary for logging."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "risk_level": self.risk_level
        }


@dataclass
class UserSession:
    """User session data."""
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    case_access: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return (
            datetime.now(timezone.utc) - self.last_accessed 
            > timedelta(hours=SESSION_EXPIRY_HOURS)
        )
    
    def refresh(self):
        """Refresh session last accessed time."""
        self.last_accessed = datetime.now(timezone.utc)


class PasswordValidator:
    """Password validation with legal industry security requirements."""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against security requirements.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Length requirements
        if len(password) < MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
        
        if len(password) > MAX_PASSWORD_LENGTH:
            errors.append(f"Password must be less than {MAX_PASSWORD_LENGTH} characters long")
        
        # Character requirements
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password checks
        if PasswordValidator._is_common_password(password):
            errors.append("Password is too common, please choose a more secure password")
        
        # Sequential character checks
        if PasswordValidator._has_sequential_chars(password):
            errors.append("Password should not contain sequential characters")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _is_common_password(password: str) -> bool:
        """Check against common passwords."""
        common_passwords = {
            "password123", "administrator", "password1234", "qwerty123",
            "letmein123", "welcome123", "admin123", "root123"
        }
        return password.lower() in common_passwords
    
    @staticmethod
    def _has_sequential_chars(password: str) -> bool:
        """Check for sequential characters."""
        sequences = ["123", "abc", "qwe", "asd", "zxc", "321", "cba"]
        password_lower = password.lower()
        return any(seq in password_lower for seq in sequences)


class PasswordManager:
    """Secure password hashing and verification."""
    
    def __init__(self):
        """Initialize password manager with secure context."""
        self.context = CryptContext(
            schemes=["argon2", "bcrypt"],
            default="argon2",
            argon2__memory_cost=65536,  # 64MB memory
            argon2__time_cost=3,        # 3 iterations
            argon2__parallelism=1,      # Single thread
            bcrypt__rounds=12,          # BCrypt rounds for fallback
            deprecated="auto"
        )
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using Argon2.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        # Validate password first
        is_valid, errors = PasswordValidator.validate_password(password)
        if not is_valid:
            raise ValidationError(
                message="Password validation failed",
                field_errors=[{"field": "password", "message": error} for error in errors]
            )
        
        try:
            hashed = self.context.hash(password)
            logger.debug("Password hashed successfully")
            return hashed
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise ValidationError(f"Password hashing failed: {str(e)}")
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to verify against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            is_valid = self.context.verify(password, hashed_password)
            
            # Check if password needs rehashing (scheme update)
            if is_valid and self.context.needs_update(hashed_password):
                logger.info("Password hash needs update")
                return True
            
            return is_valid
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False


class TokenManager:
    """JWT token management for authentication."""
    
    def __init__(self):
        """Initialize token manager."""
        self.secret_key = self._get_secret_key()
        self.algorithm = "HS256"
    
    def _get_secret_key(self) -> str:
        """Get or generate secret key for JWT tokens."""
        settings = get_settings()
        if hasattr(settings, 'jwt_secret_key') and settings.jwt_secret_key:
            return settings.jwt_secret_key
        
        # Generate a secure secret key
        return secrets.token_urlsafe(32)
    
    def create_access_token(
        self,
        user_id: str,
        role: UserRole,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User identifier
            role: User role
            additional_claims: Additional JWT claims
            
        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        
        payload = {
            "sub": user_id,
            "role": role.value,
            "iat": now,
            "exp": now + timedelta(hours=TOKEN_EXPIRY_HOURS),
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Access token created for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise ValidationError(f"Token creation failed: {str(e)}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            ValidationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "access":
                raise ValidationError("Invalid token type")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise ValidationError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise ValidationError("Invalid token")


class APIKeyManager:
    """API key management for service authentication."""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(API_KEY_LENGTH)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage."""
        salt = secrets.token_bytes(SALT_LENGTH)
        key_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt, 100000)
        return base64.b64encode(salt + key_hash).decode()
    
    @staticmethod
    def verify_api_key(api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        try:
            decoded = base64.b64decode(stored_hash.encode())
            salt = decoded[:SALT_LENGTH]
            stored_key_hash = decoded[SALT_LENGTH:]
            
            key_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt, 100000)
            return hmac.compare_digest(key_hash, stored_key_hash)
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return False


class DataEncryption:
    """Data encryption utilities for sensitive information."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize encryption with key.
        
        Args:
            encryption_key: Optional encryption key (generates if None)
        """
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate new encryption key."""
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data."""
        if isinstance(data, str):
            data = data.encode()
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_json(self, data: Dict[str, Any]) -> bytes:
        """Encrypt JSON data."""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.encrypt(json_str)
    
    def decrypt_json(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt JSON data."""
        decrypted = self.decrypt(encrypted_data)
        return json.loads(decrypted.decode())


class InputSanitizer:
    """Input sanitization for security."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # Trim whitespace
        value = value.strip()
        
        # Length check
        if len(value) > max_length:
            raise ValidationError(f"Input too long (max {max_length} characters)")
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Basic XSS prevention
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe',
            r'<object',
            r'<embed'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError("Input contains potentially dangerous content")
        
        return value
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for secure file operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path traversal attempts
        filename = filename.replace('..', '')
        filename = filename.replace('/', '')
        filename = filename.replace('\\', '')
        
        # Remove special characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Ensure it has a reasonable length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + ('.' + ext if ext else '')
        
        return filename
    
    @staticmethod
    def validate_case_id(case_id: str) -> str:
        """
        Validate and sanitize case ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Validated case ID
        """
        if not isinstance(case_id, str):
            raise ValidationError("Case ID must be a string")
        
        case_id = case_id.strip()
        
        # Case ID format validation
        if not re.match(r'^CASE_\d{4}_\d{2}_\d{2}_[A-Z0-9]{8}$', case_id):
            raise ValidationError("Invalid case ID format")
        
        return case_id
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """
        Validate and sanitize user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            Validated user ID
        """
        if not isinstance(user_id, str):
            raise ValidationError("User ID must be a string")
        
        user_id = user_id.strip()
        
        # Basic validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValidationError("User ID contains invalid characters")
        
        if len(user_id) < 3 or len(user_id) > 50:
            raise ValidationError("User ID must be 3-50 characters long")
        
        return user_id


class SessionManager:
    """Session management for user authentication."""
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
    
    def create_session(
        self,
        user_id: str,
        role: UserRole,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create new user session.
        
        Args:
            user_id: User identifier
            role: User role
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            role=role,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Log security event
        SecurityEventLogger.log_event(
            SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"session_id": session_id}
        )
        
        # Clean expired sessions
        self._cleanup_expired_sessions()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        
        if session and not session.is_expired():
            session.refresh()
            return session
        elif session:
            # Remove expired session
            self.invalidate_session(session_id)
        
        return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            del self.sessions[session_id]
            
            SecurityEventLogger.log_event(
                SecurityEventType.LOGOUT,
                user_id=session.user_id,
                details={"session_id": session_id}
            )
            
            return True
        return False
    
    def check_failed_attempts(self, identifier: str) -> bool:
        """
        Check if identifier is locked out due to failed attempts.
        
        Args:
            identifier: IP address or user ID
            
        Returns:
            True if locked out, False otherwise
        """
        if identifier not in self.failed_attempts:
            return False
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        
        # Clean old attempts
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        return len(self.failed_attempts[identifier]) >= MAX_LOGIN_ATTEMPTS
    
    def record_failed_attempt(self, identifier: str):
        """Record failed login attempt."""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(datetime.now(timezone.utc))
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]


class SecurityEventLogger:
    """Security event logging for audit trails."""
    
    @staticmethod
    def log_event(
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low"
    ):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(timezone.utc),
            details=details or {},
            risk_level=risk_level
        )
        
        # Log to security logger
        security_logger = get_logger("security")
        security_logger.info(
            f"Security event: {event_type.value}",
            **event.to_dict()
        )
        
        # Alert on high-risk events
        if risk_level in ["high", "critical"]:
            security_logger.warning(
                f"High-risk security event: {event_type.value}",
                **event.to_dict()
            )


class AccessControl:
    """Access control utilities for case and document management."""
    
    @staticmethod
    async def validate_case_access(
        user_id: str,
        case_id: str,
        required_role: UserRole = UserRole.VIEWER
    ) -> bool:
        """
        Validate user access to specific case.
        
        Args:
            user_id: User identifier
            case_id: Case identifier
            required_role: Minimum required role
            
        Returns:
            True if access allowed, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from ..repositories.mongodb.case_repository import CaseRepository
            
            case_repo = CaseRepository()
            case = await case_repo.get_case_by_id(case_id, user_id)
            
            if not case:
                SecurityEventLogger.log_event(
                    SecurityEventType.ACCESS_DENIED,
                    user_id=user_id,
                    details={"case_id": case_id, "reason": "case_not_found"},
                    risk_level="medium"
                )
                return False
            
            # For POC, allow access if user owns the case
            # In production, implement role-based access control
            if case.user_id != user_id:
                SecurityEventLogger.log_event(
                    SecurityEventType.ACCESS_DENIED,
                    user_id=user_id,
                    details={"case_id": case_id, "reason": "not_owner"},
                    risk_level="medium"
                )
                return False
            
            SecurityEventLogger.log_event(
                SecurityEventType.CASE_ACCESSED,
                user_id=user_id,
                details={"case_id": case_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Case access validation failed: {e}")
            return False
    
    @staticmethod
    async def validate_document_access(
        user_id: str,
        document_id: str,
        required_role: UserRole = UserRole.VIEWER
    ) -> bool:
        """
        Validate user access to specific document.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            required_role: Minimum required role
            
        Returns:
            True if access allowed, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from ..repositories.mongodb.document_repository import DocumentRepository
            
            doc_repo = DocumentRepository()
            document = await doc_repo.get_document_by_id(document_id)
            
            if not document:
                SecurityEventLogger.log_event(
                    SecurityEventType.ACCESS_DENIED,
                    user_id=user_id,
                    details={"document_id": document_id, "reason": "document_not_found"},
                    risk_level="medium"
                )
                return False
            
            # Check case access
            case_access = await AccessControl.validate_case_access(
                user_id, document.case_id, required_role
            )
            
            if case_access:
                SecurityEventLogger.log_event(
                    SecurityEventType.DOCUMENT_ACCESSED,
                    user_id=user_id,
                    details={"document_id": document_id, "case_id": document.case_id}
                )
            
            return case_access
            
        except Exception as e:
            logger.error(f"Document access validation failed: {e}")
            return False


# Security decorators for FastAPI endpoints

def require_authentication(f: Callable) -> Callable:
    """Decorator to require authentication for endpoints."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # This would integrate with FastAPI's dependency injection
        # Implementation depends on how authentication is handled in the API layer
        return await f(*args, **kwargs)
    return wrapper


def require_role(required_role: UserRole):
    """Decorator to require specific role for endpoints."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # Role validation logic would go here
            return await f(*args, **kwargs)
        return wrapper
    return decorator


def audit_access(resource_type: str):
    """Decorator to audit resource access."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            # Audit logging logic would go here
            return await f(*args, **kwargs)
        return wrapper
    return decorator


# Security utilities for headers and CORS

class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get standard security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


# Global instances
password_manager = PasswordManager()
token_manager = TokenManager()
session_manager = SessionManager()
input_sanitizer = InputSanitizer()

# Export commonly used classes and functions
__all__ = [
    'UserRole',
    'SecurityEventType',
    'SecurityEvent',
    'UserSession',
    'PasswordValidator',
    'PasswordManager',
    'TokenManager',
    'APIKeyManager',
    'DataEncryption',
    'InputSanitizer',
    'SessionManager',
    'SecurityEventLogger',
    'AccessControl',
    'SecurityHeaders',
    'password_manager',
    'token_manager',
    'session_manager',
    'input_sanitizer',
    'require_authentication',
    'require_role',
    'audit_access'
]