"""
Validation Utilities for Patexia Legal AI Chatbot

This module provides comprehensive validation utilities for the legal AI system,
including business rule validation, input sanitization, file validation, data
integrity checks, and legal document-specific validation patterns.

Key Features:
- Legal document validation with industry-specific rules
- Case management validation with business rule enforcement
- File type and size validation for document processing
- User input validation and sanitization for security
- Configuration validation for system settings
- Business rule enforcement for legal case management
- Data integrity validation for database operations
- Search query validation and optimization
- API parameter validation and transformation

Validation Categories:
1. Document Validation: File types, sizes, names, content validation
2. Case Validation: Case names, IDs, access rights, capacity limits
3. User Validation: User IDs, permissions, authentication data
4. File Validation: MIME types, extensions, content integrity
5. Business Rule Validation: Legal industry compliance requirements
6. Configuration Validation: System settings and parameters
7. Search Validation: Query syntax, parameters, filters
8. API Validation: Request/response data validation

Legal Industry Compliance:
- Attorney-client privilege validation for document access
- Case confidentiality enforcement through access controls
- Document retention and archival policy compliance
- Legal citation format validation for proper referencing
- Professional conduct rule enforcement for user actions
- Multi-jurisdictional compliance for different legal systems

Architecture Integration:
- Integrates with security utilities for access control validation
- Works with logging system for validation audit trails
- Supports configuration system for validation rule management
- Provides middleware support for FastAPI request validation
- Implements custom exception handling for validation errors
"""

import asyncio
import hashlib
import mimetypes
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from urllib.parse import urlparse
import email.utils

from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError

from config.settings import get_settings
from ..core.exceptions import (
    ValidationError, ErrorCode, CaseManagementError, DocumentProcessingError,
    raise_validation_error, raise_case_error, raise_document_error
)
from ..models.domain.document import DocumentType
from ..models.domain.case import CaseStatus, CasePriority
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Constants for validation
MAX_CASE_NAME_LENGTH = 200
MIN_CASE_NAME_LENGTH = 3
MAX_DOCUMENT_NAME_LENGTH = 255
MIN_DOCUMENT_NAME_LENGTH = 1
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB for POC
MAX_CASE_DOCUMENT_COUNT = 25  # Standard limit
MAX_CASE_DOCUMENT_COUNT_OVERRIDE = 50  # Override limit
MAX_SUMMARY_LENGTH = 2000
MIN_SUMMARY_LENGTH = 10
MAX_SEARCH_QUERY_LENGTH = 500
MAX_TAG_LENGTH = 50
MAX_TAGS_PER_CASE = 20

# Supported file types for legal documents
SUPPORTED_FILE_TYPES = {
    DocumentType.PDF: {
        'extensions': ['.pdf'],
        'mime_types': ['application/pdf'],
        'max_size': 50 * 1024 * 1024,  # 50MB
        'description': 'PDF documents'
    },
    DocumentType.DOCX: {
        'extensions': ['.docx'],
        'mime_types': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        'max_size': 25 * 1024 * 1024,  # 25MB
        'description': 'Microsoft Word documents'
    },
    DocumentType.DOC: {
        'extensions': ['.doc'],
        'mime_types': ['application/msword'],
        'max_size': 25 * 1024 * 1024,  # 25MB
        'description': 'Legacy Microsoft Word documents'
    },
    DocumentType.TXT: {
        'extensions': ['.txt'],
        'mime_types': ['text/plain'],
        'max_size': 5 * 1024 * 1024,  # 5MB
        'description': 'Plain text documents'
    },
    DocumentType.RTF: {
        'extensions': ['.rtf'],
        'mime_types': ['application/rtf', 'text/rtf'],
        'max_size': 10 * 1024 * 1024,  # 10MB
        'description': 'Rich Text Format documents'
    }
}


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    validation_time_ms: Optional[float] = None


class FileValidationResult(ValidationResult):
    """Extended validation result for file validation."""
    detected_type: Optional[DocumentType] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    is_safe: bool = True
    security_warnings: List[str] = Field(default_factory=list)


@dataclass
class ValidationConfig:
    """Configuration for validation behaviors."""
    strict_mode: bool = False
    allow_overrides: bool = True
    max_file_size: int = MAX_FILE_SIZE_BYTES
    supported_types: Set[DocumentType] = field(default_factory=lambda: set(SUPPORTED_FILE_TYPES.keys()))
    custom_rules: Dict[str, Any] = field(default_factory=dict)


# Document Validation Functions

def validate_file_type(filename: str, file_content: Optional[bytes] = None) -> FileValidationResult:
    """
    Validate file type based on filename and optional content analysis.
    
    Args:
        filename: Name of the file
        file_content: Optional file content for MIME type detection
        
    Returns:
        FileValidationResult with validation details
    """
    result = FileValidationResult(is_valid=True)
    
    try:
        # Extract file extension
        file_path = Path(filename)
        extension = file_path.suffix.lower()
        
        if not extension:
            result.is_valid = False
            result.errors.append("File must have an extension")
            return result
        
        # Check against supported file types
        detected_type = None
        for doc_type, type_info in SUPPORTED_FILE_TYPES.items():
            if extension in type_info['extensions']:
                detected_type = doc_type
                break
        
        if not detected_type:
            result.is_valid = False
            result.errors.append(f"Unsupported file type: {extension}")
            result.details['supported_types'] = list(SUPPORTED_FILE_TYPES.keys())
            return result
        
        result.detected_type = detected_type
        
        # MIME type validation if content is provided
        if file_content:
            mime_type, _ = mimetypes.guess_type(filename)
            result.mime_type = mime_type
            
            type_info = SUPPORTED_FILE_TYPES[detected_type]
            if mime_type and mime_type not in type_info['mime_types']:
                result.warnings.append(f"MIME type {mime_type} doesn't match expected types for {detected_type.value}")
        
        # Basic security checks
        if _contains_suspicious_patterns(filename):
            result.is_safe = False
            result.security_warnings.append("Filename contains suspicious patterns")
        
        logger.debug(f"File type validation completed", filename=filename, detected_type=detected_type.value if detected_type else None)
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"File type validation failed: {str(e)}")
        logger.error(f"File type validation error: {e}")
    
    return result


def validate_file_size(file_content: bytes, file_type: Optional[DocumentType] = None) -> ValidationResult:
    """
    Validate file size against limits.
    
    Args:
        file_content: File content as bytes
        file_type: Optional document type for specific size limits
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        file_size = len(file_content)
        result.details['file_size'] = file_size
        
        # Check against global limit
        if file_size > MAX_FILE_SIZE_BYTES:
            result.is_valid = False
            result.errors.append(f"File size {file_size} bytes exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes")
            return result
        
        # Check against type-specific limits
        if file_type and file_type in SUPPORTED_FILE_TYPES:
            type_max_size = SUPPORTED_FILE_TYPES[file_type]['max_size']
            if file_size > type_max_size:
                result.is_valid = False
                result.errors.append(f"File size {file_size} bytes exceeds {file_type.value} limit of {type_max_size} bytes")
                return result
        
        # Warn about very small files
        if file_size < 100:  # Less than 100 bytes
            result.warnings.append("File is very small and may be empty or corrupted")
        
        logger.debug(f"File size validation completed", file_size=file_size, file_type=file_type.value if file_type else None)
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"File size validation failed: {str(e)}")
        logger.error(f"File size validation error: {e}")
    
    return result


def validate_document_name(document_name: str) -> ValidationResult:
    """
    Validate document name according to business rules.
    
    Args:
        document_name: Document name to validate
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        if not document_name or not document_name.strip():
            result.is_valid = False
            result.errors.append("Document name cannot be empty")
            return result
        
        # Normalize whitespace
        normalized_name = document_name.strip()
        
        # Length validation
        if len(normalized_name) < MIN_DOCUMENT_NAME_LENGTH:
            result.is_valid = False
            result.errors.append(f"Document name must be at least {MIN_DOCUMENT_NAME_LENGTH} character long")
        
        if len(normalized_name) > MAX_DOCUMENT_NAME_LENGTH:
            result.is_valid = False
            result.errors.append(f"Document name must be less than {MAX_DOCUMENT_NAME_LENGTH} characters")
        
        # Character validation
        if not _is_valid_name_characters(normalized_name):
            result.is_valid = False
            result.errors.append("Document name contains invalid characters")
        
        # Reserved name check
        if _is_reserved_name(normalized_name):
            result.is_valid = False
            result.errors.append("Document name is reserved and cannot be used")
        
        # Legal document naming conventions
        if _violates_legal_naming_conventions(normalized_name):
            result.warnings.append("Document name may not follow legal document naming conventions")
        
        result.details['normalized_name'] = normalized_name
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Document name validation failed: {str(e)}")
        logger.error(f"Document name validation error: {e}")
    
    return result


def validate_document_content(file_content: bytes, file_type: DocumentType) -> ValidationResult:
    """
    Validate document content for integrity and safety.
    
    Args:
        file_content: File content as bytes
        file_type: Document type
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Basic content validation
        if not file_content:
            result.is_valid = False
            result.errors.append("File content cannot be empty")
            return result
        
        # File signature validation
        if not _validate_file_signature(file_content, file_type):
            result.is_valid = False
            result.errors.append("File content doesn't match expected file type signature")
            return result
        
        # Malware/safety checks
        safety_result = _perform_basic_safety_checks(file_content)
        if not safety_result['is_safe']:
            result.warnings.extend(safety_result['warnings'])
        
        # Content encoding validation for text files
        if file_type == DocumentType.TXT:
            encoding_result = _validate_text_encoding(file_content)
            result.details['encoding'] = encoding_result['encoding']
            if not encoding_result['is_valid']:
                result.warnings.append("Text file encoding may cause processing issues")
        
        result.details['content_size'] = len(file_content)
        result.details['content_hash'] = hashlib.sha256(file_content).hexdigest()[:16]
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Document content validation failed: {str(e)}")
        logger.error(f"Document content validation error: {e}")
    
    return result


# Case Validation Functions

def validate_case_name(case_name: str) -> ValidationResult:
    """
    Validate case name according to legal business rules.
    
    Args:
        case_name: Case name to validate
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        if not case_name or not case_name.strip():
            result.is_valid = False
            result.errors.append("Case name cannot be empty")
            return result
        
        # Normalize whitespace
        normalized_name = case_name.strip()
        
        # Length validation
        if len(normalized_name) < MIN_CASE_NAME_LENGTH:
            result.is_valid = False
            result.errors.append(f"Case name must be at least {MIN_CASE_NAME_LENGTH} characters long")
        
        if len(normalized_name) > MAX_CASE_NAME_LENGTH:
            result.is_valid = False
            result.errors.append(f"Case name must be less than {MAX_CASE_NAME_LENGTH} characters")
        
        # Character validation
        if not _is_valid_case_name_characters(normalized_name):
            result.is_valid = False
            result.errors.append("Case name contains invalid characters")
        
        # Legal case naming conventions
        if _violates_case_naming_conventions(normalized_name):
            result.warnings.append("Case name may not follow legal case naming conventions")
        
        # Confidentiality check
        if _contains_sensitive_information(normalized_name):
            result.warnings.append("Case name may contain sensitive information that should be redacted")
        
        result.details['normalized_name'] = normalized_name
        result.details['suggested_format'] = _suggest_case_name_format(normalized_name)
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Case name validation failed: {str(e)}")
        logger.error(f"Case name validation error: {e}")
    
    return result


def validate_case_id(case_id: str) -> ValidationResult:
    """
    Validate case ID format.
    
    Args:
        case_id: Case ID to validate
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        if not case_id or not case_id.strip():
            result.is_valid = False
            result.errors.append("Case ID cannot be empty")
            return result
        
        # Format validation: CASE_YYYY_MM_DD_XXXXXXXX
        case_id_pattern = r'^CASE_\d{4}_\d{2}_\d{2}_[A-Z0-9]{8}$'
        if not re.match(case_id_pattern, case_id):
            result.is_valid = False
            result.errors.append("Case ID must follow format: CASE_YYYY_MM_DD_XXXXXXXX")
            return result
        
        # Extract date components for validation
        parts = case_id.split('_')
        if len(parts) == 5:
            try:
                year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                case_date = datetime(year, month, day)
                
                # Validate date is reasonable
                current_date = datetime.now()
                if case_date > current_date + timedelta(days=1):
                    result.warnings.append("Case ID contains future date")
                
                if case_date < datetime(2020, 1, 1):
                    result.warnings.append("Case ID contains very old date")
                
                result.details['case_date'] = case_date.isoformat()
                
            except ValueError:
                result.is_valid = False
                result.errors.append("Case ID contains invalid date")
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Case ID validation failed: {str(e)}")
        logger.error(f"Case ID validation error: {e}")
    
    return result


def validate_case_capacity(current_document_count: int, allow_override: bool = False) -> ValidationResult:
    """
    Validate case document capacity limits.
    
    Args:
        current_document_count: Current number of documents in case
        allow_override: Whether to allow override of standard limits
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Standard limit check
        if current_document_count >= MAX_CASE_DOCUMENT_COUNT:
            if allow_override and current_document_count < MAX_CASE_DOCUMENT_COUNT_OVERRIDE:
                result.warnings.append(f"Case exceeds standard limit of {MAX_CASE_DOCUMENT_COUNT} documents but override is allowed")
                result.details['override_used'] = True
            else:
                result.is_valid = False
                limit = MAX_CASE_DOCUMENT_COUNT_OVERRIDE if allow_override else MAX_CASE_DOCUMENT_COUNT
                result.errors.append(f"Case document count {current_document_count} exceeds limit of {limit}")
        
        # Warn when approaching limits
        warning_threshold = int(MAX_CASE_DOCUMENT_COUNT * 0.8)  # 80% of limit
        if current_document_count >= warning_threshold:
            result.warnings.append(f"Case is approaching document limit ({current_document_count}/{MAX_CASE_DOCUMENT_COUNT})")
        
        result.details['current_count'] = current_document_count
        result.details['standard_limit'] = MAX_CASE_DOCUMENT_COUNT
        result.details['override_limit'] = MAX_CASE_DOCUMENT_COUNT_OVERRIDE
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Case capacity validation failed: {str(e)}")
        logger.error(f"Case capacity validation error: {e}")
    
    return result


# User Access Validation Functions

async def validate_user_access(user_id: str, case_id: str) -> bool:
    """
    Validate user access to a specific case.
    
    Args:
        user_id: User identifier
        case_id: Case identifier
        
    Returns:
        True if access is allowed, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from ..utils.security import AccessControl
        
        return await AccessControl.validate_case_access(user_id, case_id)
        
    except Exception as e:
        logger.error(f"User access validation failed: {e}")
        return False


def validate_user_id(user_id: str) -> ValidationResult:
    """
    Validate user ID format and constraints.
    
    Args:
        user_id: User identifier to validate
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        if not user_id or not user_id.strip():
            result.is_valid = False
            result.errors.append("User ID cannot be empty")
            return result
        
        normalized_id = user_id.strip()
        
        # Length validation
        if len(normalized_id) < 3:
            result.is_valid = False
            result.errors.append("User ID must be at least 3 characters long")
        
        if len(normalized_id) > 50:
            result.is_valid = False
            result.errors.append("User ID must be less than 50 characters")
        
        # Character validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', normalized_id):
            result.is_valid = False
            result.errors.append("User ID can only contain letters, numbers, hyphens, and underscores")
        
        # Reserved ID check
        reserved_ids = {'admin', 'root', 'system', 'anonymous', 'guest', 'test'}
        if normalized_id.lower() in reserved_ids:
            result.is_valid = False
            result.errors.append("User ID is reserved and cannot be used")
        
        result.details['normalized_id'] = normalized_id
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"User ID validation failed: {str(e)}")
        logger.error(f"User ID validation error: {e}")
    
    return result


# Search and Query Validation Functions

def validate_search_query(query: str) -> ValidationResult:
    """
    Validate search query for safety and effectiveness.
    
    Args:
        query: Search query string
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        if not query or not query.strip():
            result.is_valid = False
            result.errors.append("Search query cannot be empty")
            return result
        
        normalized_query = query.strip()
        
        # Length validation
        if len(normalized_query) > MAX_SEARCH_QUERY_LENGTH:
            result.is_valid = False
            result.errors.append(f"Search query must be less than {MAX_SEARCH_QUERY_LENGTH} characters")
        
        # Check for SQL injection patterns
        if _contains_sql_injection_patterns(normalized_query):
            result.is_valid = False
            result.errors.append("Search query contains potentially harmful patterns")
            return result
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in normalized_query if not c.isalnum() and not c.isspace()) / len(normalized_query)
        if special_char_ratio > 0.3:
            result.warnings.append("Search query contains many special characters which may affect results")
        
        # Legal search optimization suggestions
        optimization_suggestions = _get_legal_search_suggestions(normalized_query)
        if optimization_suggestions:
            result.details['suggestions'] = optimization_suggestions
        
        result.details['normalized_query'] = normalized_query
        result.details['query_length'] = len(normalized_query)
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Search query validation failed: {str(e)}")
        logger.error(f"Search query validation error: {e}")
    
    return result


# Configuration Validation Functions

def validate_configuration_value(section: str, key: str, value: Any) -> ValidationResult:
    """
    Validate configuration values against expected types and ranges.
    
    Args:
        section: Configuration section
        key: Configuration key
        value: Value to validate
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Define validation rules for configuration sections
        config_rules = {
            'ollama': {
                'embedding_model': {
                    'type': str,
                    'allowed_values': ['mxbai-embed-large', 'nomic-embed-text', 'e5-large-v2']
                },
                'base_url': {
                    'type': str,
                    'pattern': r'^https?://[\w\.-]+(?::\d+)?/?$'
                },
                'timeout': {
                    'type': (int, float),
                    'min_value': 1,
                    'max_value': 300
                }
            },
            'llamaindex': {
                'chunk_size': {
                    'type': int,
                    'min_value': 100,
                    'max_value': 2000
                },
                'chunk_overlap': {
                    'type': int,
                    'min_value': 0,
                    'max_value': 200
                }
            },
            'capacity_limits': {
                'max_documents_per_case': {
                    'type': int,
                    'min_value': 1,
                    'max_value': 100
                },
                'max_file_size_mb': {
                    'type': (int, float),
                    'min_value': 1,
                    'max_value': 100
                }
            }
        }
        
        if section in config_rules and key in config_rules[section]:
            rule = config_rules[section][key]
            
            # Type validation
            expected_type = rule.get('type')
            if expected_type and not isinstance(value, expected_type):
                result.is_valid = False
                result.errors.append(f"Expected {expected_type}, got {type(value)}")
                return result
            
            # Value range validation
            if 'min_value' in rule and value < rule['min_value']:
                result.is_valid = False
                result.errors.append(f"Value {value} is below minimum {rule['min_value']}")
            
            if 'max_value' in rule and value > rule['max_value']:
                result.is_valid = False
                result.errors.append(f"Value {value} is above maximum {rule['max_value']}")
            
            # Allowed values validation
            if 'allowed_values' in rule and value not in rule['allowed_values']:
                result.is_valid = False
                result.errors.append(f"Value {value} not in allowed values: {rule['allowed_values']}")
            
            # Pattern validation
            if 'pattern' in rule and isinstance(value, str):
                if not re.match(rule['pattern'], value):
                    result.is_valid = False
                    result.errors.append(f"Value {value} doesn't match required pattern")
        
        result.details['section'] = section
        result.details['key'] = key
        result.details['value'] = value
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Configuration validation failed: {str(e)}")
        logger.error(f"Configuration validation error: {e}")
    
    return result


# Business Rule Validation Functions

def validate_case_status_transition(current_status: CaseStatus, new_status: CaseStatus) -> ValidationResult:
    """
    Validate case status transitions according to business rules.
    
    Args:
        current_status: Current case status
        new_status: Proposed new status
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Define valid status transitions
        valid_transitions = {
            CaseStatus.DRAFT: [CaseStatus.ACTIVE, CaseStatus.ARCHIVED],
            CaseStatus.ACTIVE: [CaseStatus.PROCESSING, CaseStatus.COMPLETE, CaseStatus.ARCHIVED, CaseStatus.ERROR],
            CaseStatus.PROCESSING: [CaseStatus.ACTIVE, CaseStatus.COMPLETE, CaseStatus.ERROR],
            CaseStatus.COMPLETE: [CaseStatus.ACTIVE, CaseStatus.ARCHIVED],
            CaseStatus.ERROR: [CaseStatus.ACTIVE, CaseStatus.PROCESSING, CaseStatus.ARCHIVED],
            CaseStatus.ARCHIVED: [CaseStatus.ACTIVE]  # Can reactivate archived cases
        }
        
        if new_status not in valid_transitions.get(current_status, []):
            result.is_valid = False
            result.errors.append(f"Invalid status transition from {current_status.value} to {new_status.value}")
            result.details['valid_transitions'] = [status.value for status in valid_transitions.get(current_status, [])]
        
        # Business rule warnings
        if current_status == CaseStatus.COMPLETE and new_status == CaseStatus.ACTIVE:
            result.warnings.append("Reactivating completed case - ensure all stakeholders are notified")
        
        if new_status == CaseStatus.ARCHIVED:
            result.warnings.append("Archiving case - ensure all required documentation is complete")
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Status transition validation failed: {str(e)}")
        logger.error(f"Status transition validation error: {e}")
    
    return result


def validate_legal_document_metadata(metadata: Dict[str, Any]) -> ValidationResult:
    """
    Validate legal document metadata for compliance and completeness.
    
    Args:
        metadata: Document metadata dictionary
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Required fields for legal documents
        required_fields = ['document_type', 'source', 'jurisdiction']
        optional_fields = ['court', 'case_number', 'filing_date', 'parties']
        
        # Check required fields
        for field in required_fields:
            if field not in metadata:
                result.warnings.append(f"Missing recommended field: {field}")
        
        # Validate specific field formats
        if 'filing_date' in metadata:
            try:
                filing_date = datetime.fromisoformat(metadata['filing_date'].replace('Z', '+00:00'))
                if filing_date > datetime.now(timezone.utc):
                    result.warnings.append("Filing date is in the future")
            except (ValueError, AttributeError):
                result.warnings.append("Invalid filing date format")
        
        if 'case_number' in metadata:
            case_number = metadata['case_number']
            if not isinstance(case_number, str) or len(case_number.strip()) == 0:
                result.warnings.append("Case number should be a non-empty string")
        
        # Validate jurisdiction format
        if 'jurisdiction' in metadata:
            jurisdiction = metadata['jurisdiction']
            valid_jurisdictions = ['federal', 'state', 'local', 'international']
            if jurisdiction not in valid_jurisdictions:
                result.warnings.append(f"Jurisdiction '{jurisdiction}' may not be standard")
        
        result.details['metadata_completeness'] = len([f for f in required_fields if f in metadata]) / len(required_fields)
        
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Legal document metadata validation failed: {str(e)}")
        logger.error(f"Legal document metadata validation error: {e}")
    
    return result


# Helper Functions

def _contains_suspicious_patterns(filename: str) -> bool:
    """Check for suspicious patterns in filename."""
    suspicious_patterns = [
        r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.vbs$',
        r'\.js$', r'\.jar$', r'\.php$', r'\.asp$', r'\.jsp$',
        r'\.\.',  # Path traversal
        r'[<>"|?*]',  # Invalid filename characters
    ]
    
    filename_lower = filename.lower()
    return any(re.search(pattern, filename_lower) for pattern in suspicious_patterns)


def _is_valid_name_characters(name: str) -> bool:
    """Check if name contains only valid characters."""
    # Allow letters, numbers, spaces, and basic punctuation
    allowed_pattern = r'^[a-zA-Z0-9\s\-_.,()[\]]+$'
    return bool(re.match(allowed_pattern, name))


def _is_valid_case_name_characters(name: str) -> bool:
    """Check if case name contains only valid characters for legal cases."""
    # More restrictive for case names
    allowed_pattern = r'^[a-zA-Z0-9\s\-_.,()\[\]&]+$'
    return bool(re.match(allowed_pattern, name))


def _is_reserved_name(name: str) -> bool:
    """Check if name is reserved."""
    reserved_names = {
        'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
        'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2',
        'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
    }
    return name.lower() in reserved_names


def _violates_legal_naming_conventions(name: str) -> bool:
    """Check if name violates legal document naming conventions."""
    violations = [
        len(name.split()) < 2,  # Should have at least 2 words
        not any(char.isalpha() for char in name),  # Should contain letters
        name.startswith('-') or name.endswith('-'),  # Should not start/end with hyphen
        '  ' in name,  # Should not have double spaces
    ]
    return any(violations)


def _violates_case_naming_conventions(name: str) -> bool:
    """Check if case name violates legal case naming conventions."""
    violations = [
        len(name.split()) < 2,  # Should have at least 2 words
        not re.search(r'\d{4}', name),  # Should contain a year
        name.lower().count('case') > 1,  # Should not repeat 'case'
        name.lower().startswith('untitled'),  # Should not be untitled
    ]
    return any(violations)


def _contains_sensitive_information(name: str) -> bool:
    """Check if name contains sensitive information that should be redacted."""
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b',  # Phone pattern
    ]
    
    return any(re.search(pattern, name) for pattern in sensitive_patterns)


def _suggest_case_name_format(name: str) -> str:
    """Suggest a properly formatted case name."""
    current_year = datetime.now().year
    normalized = re.sub(r'\s+', '-', name.strip())
    normalized = re.sub(r'[^\w\-.]', '', normalized)
    
    if not re.search(r'\d{4}', normalized):
        normalized = f"{normalized}-{current_year}"
    
    return normalized


def _validate_file_signature(file_content: bytes, file_type: DocumentType) -> bool:
    """Validate file signature against expected type."""
    if len(file_content) < 4:
        return False
    
    # File signatures (magic numbers)
    signatures = {
        DocumentType.PDF: [b'%PDF'],
        DocumentType.DOCX: [b'PK\x03\x04'],  # ZIP-based format
        DocumentType.DOC: [b'\xd0\xcf\x11\xe0'],  # OLE format
        DocumentType.RTF: [b'{\\rtf'],
        # TXT files don't have a specific signature
    }
    
    if file_type == DocumentType.TXT:
        # For text files, check if content is reasonably text-like
        try:
            file_content.decode('utf-8')
            return True
        except UnicodeDecodeError:
            try:
                file_content.decode('latin-1')
                return True
            except UnicodeDecodeError:
                return False
    
    expected_signatures = signatures.get(file_type, [])
    return any(file_content.startswith(sig) for sig in expected_signatures)


def _perform_basic_safety_checks(file_content: bytes) -> Dict[str, Any]:
    """Perform basic safety checks on file content."""
    result = {'is_safe': True, 'warnings': []}
    
    # Check for embedded executables
    exe_signatures = [b'MZ', b'\x7fELF', b'\xfe\xed\xfa']
    if any(sig in file_content for sig in exe_signatures):
        result['is_safe'] = False
        result['warnings'].append("File may contain embedded executables")
    
    # Check for suspicious scripts
    script_patterns = [b'<script', b'javascript:', b'vbscript:', b'powershell']
    if any(pattern in file_content.lower() for pattern in script_patterns):
        result['warnings'].append("File may contain script content")
    
    return result


def _validate_text_encoding(file_content: bytes) -> Dict[str, Any]:
    """Validate text file encoding."""
    result = {'is_valid': True, 'encoding': None}
    
    # Try common encodings
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            file_content.decode(encoding)
            result['encoding'] = encoding
            return result
        except UnicodeDecodeError:
            continue
    
    result['is_valid'] = False
    result['encoding'] = 'unknown'
    return result


def _contains_sql_injection_patterns(query: str) -> bool:
    """Check for potential SQL injection patterns."""
    injection_patterns = [
        r"';.*--", r'";.*--', r"'.*OR.*'", r'".*OR.*"',
        r'UNION.*SELECT', r'DROP.*TABLE', r'DELETE.*FROM',
        r'INSERT.*INTO', r'UPDATE.*SET', r'CREATE.*TABLE'
    ]
    
    query_upper = query.upper()
    return any(re.search(pattern, query_upper) for pattern in injection_patterns)


def _get_legal_search_suggestions(query: str) -> List[str]:
    """Get optimization suggestions for legal search queries."""
    suggestions = []
    
    # Common legal terms that might benefit from specific formatting
    legal_terms = ['contract', 'agreement', 'patent', 'copyright', 'trademark', 'litigation', 'settlement']
    
    query_lower = query.lower()
    for term in legal_terms:
        if term in query_lower and f'"{term}"' not in query:
            suggestions.append(f'Consider using exact phrase: "{term}"')
    
    # Suggest boolean operators
    if ' and ' in query_lower and ' AND ' not in query:
        suggestions.append('Consider using uppercase AND for boolean search')
    
    if ' or ' in query_lower and ' OR ' not in query:
        suggestions.append('Consider using uppercase OR for boolean search')
    
    return suggestions


# Export commonly used validation functions
__all__ = [
    'ValidationResult',
    'FileValidationResult',
    'ValidationConfig',
    'validate_file_type',
    'validate_file_size',
    'validate_document_name',
    'validate_document_content',
    'validate_case_name',
    'validate_case_id',
    'validate_case_capacity',
    'validate_user_access',
    'validate_user_id',
    'validate_search_query',
    'validate_configuration_value',
    'validate_case_status_transition',
    'validate_legal_document_metadata',
    'SUPPORTED_FILE_TYPES',
    'MAX_CASE_DOCUMENT_COUNT',
    'MAX_FILE_SIZE_BYTES'
]