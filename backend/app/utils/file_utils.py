"""
File Processing Utilities for Legal Document Management

This module provides comprehensive file handling utilities for the Patexia Legal AI Chatbot,
including file validation, metadata extraction, type detection, security checks, and content
analysis optimized for legal documents.

Key Features:
- Multi-format file type detection and validation
- Secure file content analysis and malware detection
- File metadata extraction (size, hash, timestamps)
- Legal document specific content validation
- MIME type verification and security checks
- File compression and decompression utilities
- Temporary file management with cleanup
- Performance monitoring and logging

Supported File Types:
- PDF documents (patents, contracts, briefs, regulations)
- Text files (plain text, legal transcripts)
- Future: DOCX, DOC, RTF formats
- Archive formats for batch processing (ZIP, TAR)

Security Features:
- File content validation and sanitization
- Malicious file detection and quarantine
- File size limits and resource protection
- Path traversal prevention
- Content type verification
- Hash-based deduplication and integrity checks

Legal Document Optimizations:
- Legal content detection and confidence scoring
- Document structure analysis and classification
- Citation and reference extraction
- Metadata preservation for legal compliance
- Audit trail and provenance tracking

Architecture Integration:
- Integrates with DocumentProcessor for file handling
- Supports GridFS for large file storage
- Provides validation for API endpoints
- Implements security checks for file uploads
- Offers performance monitoring and metrics
"""

import asyncio
import hashlib
import io
import logging
import magic
import mimetypes
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import gzip
import base64

from ..core.config import get_settings
from ..exceptions import (
    ValidationError, SecurityError, FileProcessingError,
    ErrorCode, raise_validation_error, raise_security_error, raise_file_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class FileType(str, Enum):
    """Supported file types for legal document processing."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    DOC = "doc"
    RTF = "rtf"
    HTML = "html"
    XML = "xml"
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gz"
    UNKNOWN = "unknown"


class SecurityThreat(str, Enum):
    """Types of security threats in file content."""
    MALWARE = "malware"
    SCRIPT_INJECTION = "script_injection"
    OVERSIZED_FILE = "oversized_file"
    INVALID_MIME = "invalid_mime"
    SUSPICIOUS_CONTENT = "suspicious_content"
    PATH_TRAVERSAL = "path_traversal"


class ContentCategory(str, Enum):
    """Categories of document content for classification."""
    LEGAL_DOCUMENT = "legal_document"
    PATENT = "patent"
    CONTRACT = "contract"
    COURT_FILING = "court_filing"
    REGULATION = "regulation"
    TECHNICAL_DOCUMENT = "technical_document"
    GENERAL_TEXT = "general_text"
    UNKNOWN = "unknown"


@dataclass
class FileMetadata:
    """Comprehensive file metadata extracted during processing."""
    filename: str
    file_size: int
    file_hash: str
    mime_type: str
    file_type: FileType
    encoding: Optional[str] = None
    creation_time: Optional[datetime] = None
    modification_time: Optional[datetime] = None
    access_time: Optional[datetime] = None
    permissions: Optional[str] = None
    is_compressed: bool = False
    is_encrypted: bool = False
    compression_ratio: Optional[float] = None
    content_preview: Optional[str] = None
    security_scan_result: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
    @property
    def is_large_file(self) -> bool:
        """Check if file is considered large (>10MB)."""
        return self.file_size_mb > 10.0


@dataclass
class ContentAnalysis:
    """Content analysis results for document classification."""
    content_category: ContentCategory
    confidence_score: float
    detected_patterns: List[str]
    language: Optional[str] = None
    text_quality_score: float = 0.0
    legal_indicators: List[str] = field(default_factory=list)
    citation_count: int = 0
    section_count: int = 0
    estimated_read_time_minutes: float = 0.0
    content_complexity: str = "medium"  # low, medium, high
    
    @property
    def is_legal_content(self) -> bool:
        """Check if content is classified as legal."""
        return self.content_category in [
            ContentCategory.LEGAL_DOCUMENT,
            ContentCategory.PATENT,
            ContentCategory.CONTRACT,
            ContentCategory.COURT_FILING,
            ContentCategory.REGULATION
        ]


@dataclass
class FileValidationResult:
    """Result of comprehensive file validation."""
    is_valid: bool
    file_metadata: FileMetadata
    content_analysis: Optional[ContentAnalysis] = None
    security_threats: List[SecurityThreat] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    @property
    def is_secure(self) -> bool:
        """Check if file passed security validation."""
        return len(self.security_threats) == 0
    
    @property
    def is_processable(self) -> bool:
        """Check if file can be processed."""
        return self.is_valid and self.is_secure


class FileValidator:
    """
    Comprehensive file validator with security and content analysis.
    
    Provides multi-layer validation including file type detection,
    content analysis, security scanning, and legal document classification.
    """
    
    def __init__(self):
        """Initialize file validator with security and analysis engines."""
        self.settings = get_settings()
        
        # File size limits (configurable)
        self.max_file_size = self.settings.legal_documents.max_file_size_mb * 1024 * 1024
        self.max_total_size = 500 * 1024 * 1024  # 500MB total limit
        
        # Supported MIME types
        self.supported_mime_types = {
            'application/pdf',
            'text/plain',
            'text/html',
            'text/xml',
            'application/rtf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'application/zip',
            'application/x-tar',
            'application/gzip'
        }
        
        # Security patterns
        self.malicious_patterns = [
            rb'<script[^>]*>',
            rb'javascript:',
            rb'vbscript:',
            rb'onload\s*=',
            rb'onerror\s*=',
            rb'eval\s*\(',
            rb'document\.write',
            rb'innerHTML\s*=',
            rb'<iframe[^>]*>',
            rb'<object[^>]*>',
            rb'<embed[^>]*>'
        ]
        
        # Legal content indicators
        self.legal_patterns = {
            'patent': [
                r'\bclaim\s+\d+\b',
                r'\bfield\s+of\s+invention\b',
                r'\bbackground\s+of\s+invention\b',
                r'\bpatent\s+application\b',
                r'\buspto\b',
                r'\bprior\s+art\b'
            ],
            'contract': [
                r'\bagreement\b',
                r'\bwhereas\b',
                r'\bnow\s+therefore\b',
                r'\bin\s+consideration\b',
                r'\bparty\s+of\s+the\s+first\s+part\b',
                r'\bwitnesseth\b'
            ],
            'court_filing': [
                r'\bplaintiff\b',
                r'\bdefendant\b',
                r'\bmotion\b',
                r'\bbrief\b',
                r'\bhonor\w*\s+court\b',
                r'\bcomes\s+now\b',
                r'\bcase\s+no\.\s*\d+\b'
            ],
            'regulation': [
                r'\bcode\s+of\s+federal\s+regulations\b',
                r'\bcfr\b',
                r'\bsection\s+\d+\b',
                r'\bauthority:\b',
                r'\bsource:\b'
            ]
        }
        
        logger.info(
            "FileValidator initialized",
            max_file_size_mb=self.max_file_size / (1024 * 1024),
            supported_types=len(self.supported_mime_types)
        )
    
    async def validate_file(
        self,
        file_content: bytes,
        filename: str,
        perform_content_analysis: bool = True,
        strict_security: bool = True
    ) -> FileValidationResult:
        """
        Perform comprehensive file validation with security and content analysis.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            perform_content_analysis: Whether to analyze file content
            strict_security: Whether to apply strict security checks
            
        Returns:
            Comprehensive validation result
            
        Raises:
            ValidationError: If validation fails critically
            SecurityError: If security threats are detected
        """
        start_time = time.time()
        
        try:
            # Extract file metadata
            metadata = await self._extract_file_metadata(file_content, filename)
            
            # Perform security scanning
            security_threats = await self._scan_for_security_threats(
                file_content, metadata, strict_security
            )
            
            # Perform content analysis if requested
            content_analysis = None
            if perform_content_analysis and len(security_threats) == 0:
                content_analysis = await self._analyze_content(file_content, metadata)
            
            # Validate file size
            validation_errors = []
            warnings = []
            
            if metadata.file_size > self.max_file_size:
                validation_errors.append(
                    f"File size {metadata.file_size_mb:.1f}MB exceeds maximum "
                    f"{self.max_file_size / (1024 * 1024):.1f}MB"
                )
            
            # Validate MIME type
            if metadata.mime_type not in self.supported_mime_types:
                validation_errors.append(f"Unsupported MIME type: {metadata.mime_type}")
            
            # Check for warnings
            if metadata.file_size_mb > 50:
                warnings.append("Large file may take longer to process")
            
            if metadata.is_encrypted:
                warnings.append("Encrypted file detected - may require special handling")
            
            # Determine overall validity
            is_valid = len(validation_errors) == 0 and len(security_threats) == 0
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = FileValidationResult(
                is_valid=is_valid,
                file_metadata=metadata,
                content_analysis=content_analysis,
                security_threats=security_threats,
                validation_errors=validation_errors,
                warnings=warnings,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(
                "File validation completed",
                filename=filename,
                file_size_mb=metadata.file_size_mb,
                is_valid=is_valid,
                security_threats=len(security_threats),
                processing_time_ms=processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "File validation failed",
                filename=filename,
                error=str(e)
            )
            raise_file_error(
                f"File validation failed: {str(e)}",
                ErrorCode.FILE_VALIDATION_FAILED,
                {"filename": filename}
            )
    
    async def _extract_file_metadata(
        self,
        file_content: bytes,
        filename: str
    ) -> FileMetadata:
        """Extract comprehensive metadata from file."""
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_content, filename)
        
        # Detect file type
        file_type = self._detect_file_type(filename, mime_type)
        
        # Detect encoding for text files
        encoding = None
        if mime_type.startswith('text/'):
            encoding = self._detect_encoding(file_content)
        
        # Check if compressed
        is_compressed = self._is_compressed_file(file_content, file_type)
        
        # Check if encrypted (basic detection)
        is_encrypted = self._is_encrypted_file(file_content)
        
        # Generate content preview for text files
        content_preview = None
        if mime_type.startswith('text/') and len(file_content) > 0:
            try:
                text_content = file_content.decode(encoding or 'utf-8', errors='ignore')
                content_preview = text_content[:500] + ('...' if len(text_content) > 500 else '')
            except:
                content_preview = None
        
        return FileMetadata(
            filename=filename,
            file_size=len(file_content),
            file_hash=file_hash,
            mime_type=mime_type,
            file_type=file_type,
            encoding=encoding,
            is_compressed=is_compressed,
            is_encrypted=is_encrypted,
            content_preview=content_preview
        )
    
    def _detect_mime_type(self, file_content: bytes, filename: str) -> str:
        """Detect MIME type using multiple methods."""
        # Try magic library first (most accurate)
        try:
            import magic
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type and mime_type != 'application/octet-stream':
                return mime_type
        except:
            pass
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
        
        # Fallback based on file extension
        extension = Path(filename).suffix.lower()
        extension_mapping = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.xml': 'text/xml',
            '.rtf': 'application/rtf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.zip': 'application/zip',
            '.tar': 'application/x-tar',
            '.gz': 'application/gzip'
        }
        
        return extension_mapping.get(extension, 'application/octet-stream')
    
    def _detect_file_type(self, filename: str, mime_type: str) -> FileType:
        """Detect file type from filename and MIME type."""
        extension = Path(filename).suffix.lower()
        
        # Direct extension mapping
        extension_mapping = {
            '.pdf': FileType.PDF,
            '.txt': FileType.TXT,
            '.docx': FileType.DOCX,
            '.doc': FileType.DOC,
            '.rtf': FileType.RTF,
            '.html': FileType.HTML,
            '.xml': FileType.XML,
            '.zip': FileType.ZIP,
            '.tar': FileType.TAR,
            '.gz': FileType.GZIP
        }
        
        if extension in extension_mapping:
            return extension_mapping[extension]
        
        # MIME type mapping
        mime_mapping = {
            'application/pdf': FileType.PDF,
            'text/plain': FileType.TXT,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCX,
            'application/msword': FileType.DOC,
            'application/rtf': FileType.RTF,
            'text/html': FileType.HTML,
            'text/xml': FileType.XML,
            'application/zip': FileType.ZIP,
            'application/x-tar': FileType.TAR,
            'application/gzip': FileType.GZIP
        }
        
        return mime_mapping.get(mime_type, FileType.UNKNOWN)
    
    def _detect_encoding(self, file_content: bytes) -> str:
        """Detect text encoding."""
        try:
            import chardet
            result = chardet.detect(file_content)
            if result and result['confidence'] > 0.7:
                return result['encoding']
        except:
            pass
        
        # Try common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                file_content.decode(encoding)
                return encoding
            except:
                continue
        
        return 'utf-8'  # Default fallback
    
    def _is_compressed_file(self, file_content: bytes, file_type: FileType) -> bool:
        """Check if file is compressed."""
        if file_type in [FileType.ZIP, FileType.TAR, FileType.GZIP]:
            return True
        
        # Check magic bytes for compression
        if file_content.startswith(b'PK'):  # ZIP magic
            return True
        if file_content.startswith(b'\x1f\x8b'):  # GZIP magic
            return True
        if file_content.startswith(b'ustar'):  # TAR magic
            return True
        
        return False
    
    def _is_encrypted_file(self, file_content: bytes) -> bool:
        """Basic check for encrypted content."""
        # Check for PDF encryption
        if b'/Encrypt' in file_content:
            return True
        
        # Check for high entropy (possible encryption)
        if len(file_content) > 1000:
            # Calculate byte frequency
            byte_counts = [0] * 256
            for byte in file_content[:1000]:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0
            for count in byte_counts:
                if count > 0:
                    p = count / 1000
                    entropy -= p * (p.bit_length() - 1)
            
            # High entropy might indicate encryption
            return entropy > 7.5
        
        return False
    
    async def _scan_for_security_threats(
        self,
        file_content: bytes,
        metadata: FileMetadata,
        strict_security: bool
    ) -> List[SecurityThreat]:
        """Scan file content for security threats."""
        threats = []
        
        # Check file size limits
        if metadata.file_size > self.max_file_size:
            threats.append(SecurityThreat.OVERSIZED_FILE)
        
        # Check MIME type whitelist
        if strict_security and metadata.mime_type not in self.supported_mime_types:
            threats.append(SecurityThreat.INVALID_MIME)
        
        # Scan for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, file_content, re.IGNORECASE):
                threats.append(SecurityThreat.SCRIPT_INJECTION)
                break
        
        # Check for suspicious binary content in text files
        if metadata.mime_type.startswith('text/'):
            if self._has_suspicious_binary_content(file_content):
                threats.append(SecurityThreat.SUSPICIOUS_CONTENT)
        
        # Path traversal check in filename
        if self._check_path_traversal(metadata.filename):
            threats.append(SecurityThreat.PATH_TRAVERSAL)
        
        return list(set(threats))  # Remove duplicates
    
    def _has_suspicious_binary_content(self, file_content: bytes) -> bool:
        """Check for suspicious binary content in text files."""
        try:
            text_content = file_content.decode('utf-8', errors='ignore')
            # Check for high ratio of non-printable characters
            printable_count = sum(1 for c in text_content if c.isprintable())
            if len(text_content) > 0:
                printable_ratio = printable_count / len(text_content)
                return printable_ratio < 0.7
        except:
            return True
        
        return False
    
    def _check_path_traversal(self, filename: str) -> bool:
        """Check for path traversal attempts in filename."""
        dangerous_patterns = ['../', '..\\', '/./', '\\.\\', '//', '\\\\']
        return any(pattern in filename for pattern in dangerous_patterns)
    
    async def _analyze_content(
        self,
        file_content: bytes,
        metadata: FileMetadata
    ) -> ContentAnalysis:
        """Analyze file content for classification and legal indicators."""
        # For text files, analyze content directly
        if metadata.mime_type.startswith('text/'):
            try:
                text_content = file_content.decode(metadata.encoding or 'utf-8', errors='ignore')
                return self._analyze_text_content(text_content)
            except:
                pass
        
        # For binary files, use filename and basic heuristics
        return self._analyze_binary_content(metadata)
    
    def _analyze_text_content(self, text_content: str) -> ContentAnalysis:
        """Analyze text content for legal classification."""
        text_lower = text_content.lower()
        
        # Count legal indicators
        legal_indicators = []
        pattern_scores = {}
        
        for category, patterns in self.legal_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_content, re.IGNORECASE))
                if matches > 0:
                    legal_indicators.append(f"{category}:{pattern}")
                    score += matches
            pattern_scores[category] = score
        
        # Determine content category
        if pattern_scores:
            best_category = max(pattern_scores.keys(), key=lambda k: pattern_scores[k])
            confidence = min(1.0, pattern_scores[best_category] / 10)
            
            category_mapping = {
                'patent': ContentCategory.PATENT,
                'contract': ContentCategory.CONTRACT,
                'court_filing': ContentCategory.COURT_FILING,
                'regulation': ContentCategory.REGULATION
            }
            
            content_category = category_mapping.get(best_category, ContentCategory.LEGAL_DOCUMENT)
        else:
            content_category = ContentCategory.GENERAL_TEXT
            confidence = 0.0
        
        # Calculate additional metrics
        word_count = len(text_content.split())
        citation_count = len(re.findall(r'\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b', text_content))
        section_count = len(re.findall(r'\bsection\s+\d+\b', text_content, re.IGNORECASE))
        estimated_read_time = word_count / 200  # 200 words per minute
        
        # Assess text quality
        sentences = text_content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        text_quality_score = min(1.0, avg_sentence_length / 20)  # Normalize to 20 words per sentence
        
        # Determine complexity
        if word_count < 1000:
            complexity = "low"
        elif word_count < 5000:
            complexity = "medium"
        else:
            complexity = "high"
        
        return ContentAnalysis(
            content_category=content_category,
            confidence_score=confidence,
            detected_patterns=legal_indicators,
            language="en",  # Simple assumption for now
            text_quality_score=text_quality_score,
            legal_indicators=legal_indicators,
            citation_count=citation_count,
            section_count=section_count,
            estimated_read_time_minutes=estimated_read_time,
            content_complexity=complexity
        )
    
    def _analyze_binary_content(self, metadata: FileMetadata) -> ContentAnalysis:
        """Analyze binary content based on metadata."""
        # Basic classification based on file type
        if metadata.file_type == FileType.PDF:
            return ContentAnalysis(
                content_category=ContentCategory.LEGAL_DOCUMENT,
                confidence_score=0.5,
                detected_patterns=["pdf_format"],
                content_complexity="medium"
            )
        
        return ContentAnalysis(
            content_category=ContentCategory.UNKNOWN,
            confidence_score=0.0,
            detected_patterns=[],
            content_complexity="unknown"
        )


class FileProcessor:
    """
    File processing utilities for legal document management.
    
    Provides file manipulation, compression, conversion, and management
    utilities optimized for legal document workflows.
    """
    
    def __init__(self):
        """Initialize file processor."""
        self.settings = get_settings()
        self.temp_dir = Path(tempfile.gettempdir()) / "patexia_legal_ai"
        self.temp_dir.mkdir(exist_ok=True)
        
        # File processing statistics
        self.processing_stats = {
            "files_processed": 0,
            "total_size_processed": 0,
            "compression_savings": 0,
            "processing_time_ms": 0
        }
        
        logger.info(
            "FileProcessor initialized",
            temp_dir=str(self.temp_dir)
        )
    
    async def create_temporary_file(
        self,
        file_content: bytes,
        filename: str,
        cleanup_after_seconds: int = 3600
    ) -> Path:
        """
        Create a temporary file with automatic cleanup.
        
        Args:
            file_content: File content to write
            filename: Original filename for extension detection
            cleanup_after_seconds: Seconds until automatic cleanup
            
        Returns:
            Path to temporary file
        """
        # Generate unique temporary filename
        file_extension = Path(filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = self.temp_dir / temp_filename
        
        try:
            # Write content to temporary file
            async with asyncio.Lock():  # Ensure thread safety
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
            
            # Schedule cleanup
            asyncio.create_task(self._cleanup_file_after_delay(temp_path, cleanup_after_seconds))
            
            logger.debug(
                "Temporary file created",
                temp_path=str(temp_path),
                file_size=len(file_content),
                cleanup_after=cleanup_after_seconds
            )
            
            return temp_path
            
        except Exception as e:
            # Clean up on error
            if temp_path.exists():
                temp_path.unlink()
            
            logger.error(
                "Failed to create temporary file",
                filename=filename,
                error=str(e)
            )
            raise_file_error(
                f"Failed to create temporary file: {str(e)}",
                ErrorCode.FILE_CREATION_FAILED,
                {"filename": filename}
            )
    
    async def compress_file(
        self,
        file_content: bytes,
        compression_level: int = 6
    ) -> Tuple[bytes, float]:
        """
        Compress file content using gzip.
        
        Args:
            file_content: Content to compress
            compression_level: Compression level (1-9)
            
        Returns:
            Tuple of (compressed_content, compression_ratio)
        """
        try:
            original_size = len(file_content)
            compressed_content = gzip.compress(file_content, compresslevel=compression_level)
            compressed_size = len(compressed_content)
            
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            logger.debug(
                "File compressed",
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio
            )
            
            return compressed_content, compression_ratio
            
        except Exception as e:
            logger.error("File compression failed", error=str(e))
            raise_file_error(
                f"File compression failed: {str(e)}",
                ErrorCode.FILE_COMPRESSION_FAILED
            )
    
    async def decompress_file(self, compressed_content: bytes) -> bytes:
        """
        Decompress gzip-compressed content.
        
        Args:
            compressed_content: Compressed content
            
        Returns:
            Decompressed content
        """
        try:
            return gzip.decompress(compressed_content)
        except Exception as e:
            logger.error("File decompression failed", error=str(e))
            raise_file_error(
                f"File decompression failed: {str(e)}",
                ErrorCode.FILE_DECOMPRESSION_FAILED
            )
    
    async def extract_archive(
        self,
        archive_content: bytes,
        archive_type: str = "zip",
        max_extracted_size: int = 100 * 1024 * 1024  # 100MB
    ) -> Dict[str, bytes]:
        """
        Extract files from archive with security checks.
        
        Args:
            archive_content: Archive file content
            archive_type: Type of archive (zip, tar)
            max_extracted_size: Maximum total extracted size
            
        Returns:
            Dictionary mapping filenames to content
        """
        extracted_files = {}
        total_size = 0
        
        try:
            if archive_type.lower() == "zip":
                with zipfile.ZipFile(io.BytesIO(archive_content), 'r') as zip_file:
                    for file_info in zip_file.infolist():
                        # Security checks
                        if file_info.filename.startswith('/') or '..' in file_info.filename:
                            logger.warning(
                                "Skipping potentially dangerous file in archive",
                                filename=file_info.filename
                            )
                            continue
                        
                        # Size check
                        if total_size + file_info.file_size > max_extracted_size:
                            logger.warning(
                                "Archive extraction stopped due to size limit",
                                total_size=total_size,
                                max_size=max_extracted_size
                            )
                            break
                        
                        # Extract file
                        with zip_file.open(file_info) as extracted_file:
                            content = extracted_file.read()
                            extracted_files[file_info.filename] = content
                            total_size += len(content)
            
            else:
                raise_validation_error(
                    f"Unsupported archive type: {archive_type}",
                    ErrorCode.FILE_TYPE_UNSUPPORTED,
                    {"archive_type": archive_type}
                )
            
            logger.info(
                "Archive extracted successfully",
                archive_type=archive_type,
                files_extracted=len(extracted_files),
                total_size=total_size
            )
            
            return extracted_files
            
        except zipfile.BadZipFile:
            raise_validation_error(
                "Invalid or corrupted archive file",
                ErrorCode.FILE_CORRUPTED
            )
        except Exception as e:
            logger.error("Archive extraction failed", error=str(e))
            raise_file_error(
                f"Archive extraction failed: {str(e)}",
                ErrorCode.FILE_EXTRACTION_FAILED
            )
    
    async def _cleanup_file_after_delay(self, file_path: Path, delay_seconds: int) -> None:
        """Clean up temporary file after delay."""
        try:
            await asyncio.sleep(delay_seconds)
            if file_path.exists():
                file_path.unlink()
                logger.debug("Temporary file cleaned up", file_path=str(file_path))
        except Exception as e:
            logger.warning(
                "Failed to clean up temporary file",
                file_path=str(file_path),
                error=str(e)
            )
    
    def cleanup_temp_directory(self) -> int:
        """
        Clean up temporary directory and return number of files removed.
        
        Returns:
            Number of files removed
        """
        removed_count = 0
        try:
            if self.temp_dir.exists():
                for file_path in self.temp_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        removed_count += 1
                
                logger.info(
                    "Temporary directory cleaned up",
                    files_removed=removed_count
                )
        except Exception as e:
            logger.warning(
                "Failed to clean up temporary directory",
                error=str(e)
            )
        
        return removed_count


# Convenience functions for common file operations

def calculate_file_hash(file_content: bytes, algorithm: str = "sha256") -> str:
    """
    Calculate hash of file content.
    
    Args:
        file_content: File content as bytes
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hexadecimal hash string
    """
    hash_algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    
    if algorithm not in hash_algorithms:
        raise_validation_error(
            f"Unsupported hash algorithm: {algorithm}",
            ErrorCode.VALIDATION_FAILED,
            {"algorithm": algorithm, "supported": list(hash_algorithms.keys())}
        )
    
    hash_obj = hash_algorithms[algorithm]()
    hash_obj.update(file_content)
    return hash_obj.hexdigest()


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        file_path: File path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or dangerous
    """
    path = Path(file_path).resolve()
    
    # Security checks
    if not path.exists():
        raise_validation_error(
            f"File does not exist: {file_path}",
            ErrorCode.FILE_NOT_FOUND,
            {"file_path": str(file_path)}
        )
    
    if not path.is_file():
        raise_validation_error(
            f"Path is not a file: {file_path}",
            ErrorCode.VALIDATION_FAILED,
            {"file_path": str(file_path)}
        )
    
    # Check for path traversal
    if '..' in str(path):
        raise_security_error(
            f"Potential path traversal in file path: {file_path}",
            ErrorCode.SECURITY_VIOLATION,
            {"file_path": str(file_path)}
        )
    
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    path = validate_file_path(file_path)
    return path.stat().st_size


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Validate file type against allowed types.
    
    Args:
        filename: File name to check
        allowed_types: List of allowed file extensions (e.g., ['.pdf', '.txt'])
        
    Returns:
        True if file type is allowed
    """
    file_extension = Path(filename).suffix.lower()
    return file_extension in [ext.lower() for ext in allowed_types]


def validate_file_size(file_size: int, max_size: int) -> bool:
    """
    Validate file size against maximum allowed size.
    
    Args:
        file_size: File size in bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if file size is within limits
    """
    return 0 < file_size <= max_size


def extract_file_metadata(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract basic metadata from file content.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        
    Returns:
        Dictionary with metadata
    """
    return {
        "filename": filename,
        "file_size": len(file_content),
        "file_hash": calculate_file_hash(file_content),
        "mime_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "file_extension": Path(filename).suffix.lower(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }


# Global instances for easy access
_file_validator = None
_file_processor = None


def get_file_validator() -> FileValidator:
    """Get global file validator instance."""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileValidator()
    return _file_validator


def get_file_processor() -> FileProcessor:
    """Get global file processor instance."""
    global _file_processor
    if _file_processor is None:
        _file_processor = FileProcessor()
    return _file_processor