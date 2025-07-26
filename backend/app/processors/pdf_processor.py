"""
Advanced PDF Text Extraction Processor for Legal Documents

This module provides sophisticated PDF processing capabilities optimized for legal documents,
including patents, contracts, briefs, and regulatory filings. It handles complex PDF structures,
extracts metadata, and provides high-quality text extraction with structure preservation.

Key Features:
- Multi-method PDF text extraction with quality validation
- Legal document structure recognition and preservation
- Metadata extraction (titles, authors, creation dates)
- Page-level processing with layout analysis
- Table and form detection and extraction
- OCR fallback for scanned documents (future enhancement)
- Confidence scoring for extraction quality
- Error recovery and corruption detection

Legal Document Optimizations:
- Patent document structure recognition (claims, specifications)
- Legal brief formatting preservation (headers, footnotes)
- Contract clause and section identification
- Citation and reference extraction
- Page numbering and header/footer handling
- Signature block and stamp recognition

Architecture Integration:
- Uses LlamaIndex PDFReader as the primary extraction engine
- Integrates with TextProcessor for semantic chunking
- Provides extracted text to DocumentProcessor pipeline
- Supports WebSocket progress tracking
- Implements comprehensive error handling and logging
"""

import io
import logging
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum

import PyPDF2
import pdfplumber
from llama_index.readers.file import PDFReader as LlamaPDFReader

from ..core.config import get_settings
from ..core.websocket_manager import WebSocketManager
from ..models.domain.document import DocumentType, LegalDocument
from ..utils.logging import get_logger
from ..utils.file_utils import validate_file_path, get_file_size, calculate_file_hash
from ..exceptions import (
    DocumentProcessingError,
    ErrorCode,
    ValidationError
)

logger = get_logger(__name__)


class PDFExtractionMethod(Enum):
    """PDF text extraction methods in order of preference."""
    LLAMA_INDEX = "llamaindex"
    PDFPLUMBER = "pdfplumber"
    PYPDF2 = "pypdf2"
    OCR = "ocr"  # Future enhancement


class PDFStructureType(Enum):
    """Types of PDF document structures for legal documents."""
    PATENT = "patent"
    LEGAL_BRIEF = "legal_brief"
    CONTRACT = "contract"
    REGULATION = "regulation"
    COURT_FILING = "court_filing"
    TECHNICAL_REPORT = "technical_report"
    UNKNOWN = "unknown"


@dataclass
class PDFMetadata:
    """PDF document metadata extracted during processing."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    keywords: List[str] = field(default_factory=list)
    page_count: int = 0
    is_encrypted: bool = False
    is_scanned: bool = False
    pdf_version: Optional[str] = None
    file_size_bytes: int = 0
    language: Optional[str] = None


@dataclass
class PDFPage:
    """Individual PDF page with extracted content and metadata."""
    page_number: int
    text_content: str
    text_length: int
    confidence_score: float
    has_images: bool = False
    has_tables: bool = False
    has_forms: bool = False
    layout_elements: List[str] = field(default_factory=list)
    extraction_method: PDFExtractionMethod = PDFExtractionMethod.LLAMA_INDEX
    extraction_time_ms: float = 0.0
    bbox_elements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PDFExtractionResult:
    """Complete PDF extraction result with text and metadata."""
    full_text: str
    pages: List[PDFPage]
    metadata: PDFMetadata
    structure_type: PDFStructureType
    extraction_confidence: float
    total_extraction_time_ms: float
    primary_method: PDFExtractionMethod
    fallback_methods_used: List[PDFExtractionMethod] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        return len(self.pages)
    
    @property
    def total_text_length(self) -> int:
        """Get total character count across all pages."""
        return len(self.full_text)
    
    @property
    def average_page_confidence(self) -> float:
        """Calculate average confidence across all pages."""
        if not self.pages:
            return 0.0
        return sum(page.confidence_score for page in self.pages) / len(self.pages)


class PDFQualityAnalyzer:
    """
    Analyzer for PDF text extraction quality assessment.
    
    Provides quality metrics and confidence scoring for extracted text
    to ensure reliable processing and identify potential issues.
    """
    
    def __init__(self):
        """Initialize PDF quality analyzer."""
        self.legal_indicators = self._compile_legal_indicators()
        self.quality_patterns = self._compile_quality_patterns()
    
    def _compile_legal_indicators(self) -> List[re.Pattern]:
        """Compile patterns that indicate legal document content."""
        return [
            re.compile(r'\b(?:CLAIM|Claims?)\s+\d+', re.IGNORECASE),
            re.compile(r'\b(?:USC|U\.S\.C\.)\s+§?\s*\d+', re.IGNORECASE),
            re.compile(r'\b(?:Patent|Application)\s+(?:No\.?\s*)?\d+', re.IGNORECASE),
            re.compile(r'\b(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)', re.IGNORECASE),
            re.compile(r'\b(?:Plaintiff|Defendant|Petitioner|Respondent)\b', re.IGNORECASE),
            re.compile(r'\b(?:§|Section|Article)\s+\d+', re.IGNORECASE),
            re.compile(r'\b(?:Court|Judge|Justice|Attorney)\b', re.IGNORECASE),
            re.compile(r'\b(?:Contract|Agreement|License|Amendment)\b', re.IGNORECASE),
        ]
    
    def _compile_quality_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for text quality assessment."""
        return {
            'garbage_chars': re.compile(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#$%&*+=<>~`|]+'),
            'repeated_chars': re.compile(r'(.)\1{10,}'),
            'excessive_spaces': re.compile(r'\s{5,}'),
            'malformed_words': re.compile(r'\b[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{8,}\b'),
            'encoding_errors': re.compile(r'[^\x00-\x7F\u00A0-\u017F\u0100-\u024F]+'),
            'line_breaks': re.compile(r'\n+'),
            'proper_sentences': re.compile(r'[A-Z][^.!?]*[.!?]'),
        }
    
    def analyze_text_quality(self, text: str) -> float:
        """
        Analyze text quality and return confidence score.
        
        Args:
            text: Extracted text to analyze
            
        Returns:
            Quality confidence score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        confidence = 1.0
        text_length = len(text)
        
        # Check for garbage characters
        garbage_matches = len(self.quality_patterns['garbage_chars'].findall(text))
        if garbage_matches > 0:
            confidence *= max(0.1, 1.0 - (garbage_matches / text_length * 100))
        
        # Check for repeated character sequences
        repeated_matches = len(self.quality_patterns['repeated_chars'].findall(text))
        if repeated_matches > 0:
            confidence *= max(0.3, 1.0 - (repeated_matches * 0.1))
        
        # Check for excessive spacing
        space_matches = len(self.quality_patterns['excessive_spaces'].findall(text))
        if space_matches > 0:
            confidence *= max(0.7, 1.0 - (space_matches / 50))
        
        # Check for malformed words
        malformed_matches = len(self.quality_patterns['malformed_words'].findall(text))
        word_count = len(text.split())
        if word_count > 0 and malformed_matches > 0:
            malformed_ratio = malformed_matches / word_count
            confidence *= max(0.5, 1.0 - malformed_ratio)
        
        # Check for proper sentence structure
        sentence_matches = len(self.quality_patterns['proper_sentences'].findall(text))
        if word_count > 20:  # Only check for longer texts
            expected_sentences = word_count / 15  # Rough estimate
            sentence_ratio = sentence_matches / max(1, expected_sentences)
            confidence *= min(1.0, max(0.3, sentence_ratio))
        
        # Check for encoding issues
        encoding_matches = len(self.quality_patterns['encoding_errors'].findall(text))
        if encoding_matches > 0:
            confidence *= max(0.2, 1.0 - (encoding_matches / text_length * 50))
        
        return max(0.0, min(1.0, confidence))
    
    def detect_legal_content(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detect if text contains legal content and return confidence.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (is_legal, confidence, matched_patterns)
        """
        matches = []
        total_matches = 0
        
        for pattern in self.legal_indicators:
            pattern_matches = pattern.findall(text)
            if pattern_matches:
                matches.extend(pattern_matches)
                total_matches += len(pattern_matches)
        
        # Calculate legal content confidence
        text_length = len(text.split())
        if text_length == 0:
            return False, 0.0, []
        
        # Higher match density indicates stronger legal content
        match_density = total_matches / max(1, text_length / 100)
        confidence = min(1.0, match_density)
        
        is_legal = confidence > 0.1 or total_matches > 2
        
        return is_legal, confidence, matches[:10]  # Limit matches returned
    
    def detect_document_structure(self, text: str, metadata: PDFMetadata) -> PDFStructureType:
        """
        Detect the type of legal document structure.
        
        Args:
            text: Full document text
            metadata: PDF metadata
            
        Returns:
            Detected document structure type
        """
        text_lower = text.lower()
        
        # Patent indicators
        if any(indicator in text_lower for indicator in [
            'claim', 'field of invention', 'background of invention',
            'detailed description', 'patent application', 'uspto'
        ]):
            return PDFStructureType.PATENT
        
        # Legal brief indicators
        if any(indicator in text_lower for indicator in [
            'plaintiff', 'defendant', 'motion', 'brief', 'court',
            'honorable', 'comes now', 'respectfully submitted'
        ]):
            return PDFStructureType.LEGAL_BRIEF
        
        # Contract indicators
        if any(indicator in text_lower for indicator in [
            'agreement', 'contract', 'whereas', 'now therefore',
            'in consideration', 'party of the first part', 'witnesseth'
        ]):
            return PDFStructureType.CONTRACT
        
        # Regulation indicators
        if any(indicator in text_lower for indicator in [
            'regulation', 'code of federal regulations', 'cfr',
            'section', 'subsection', 'authority:', 'source:'
        ]):
            return PDFStructureType.REGULATION
        
        # Court filing indicators
        if any(indicator in text_lower for indicator in [
            'case no.', 'civil action', 'criminal case',
            'docket', 'filed', 'clerk of court'
        ]):
            return PDFStructureType.COURT_FILING
        
        # Technical report indicators
        if any(indicator in text_lower for indicator in [
            'technical report', 'research', 'methodology',
            'results', 'conclusion', 'references', 'bibliography'
        ]):
            return PDFStructureType.TECHNICAL_REPORT
        
        return PDFStructureType.UNKNOWN


class PDFProcessor:
    """
    Advanced PDF processor for legal documents with multiple extraction methods.
    
    Provides robust PDF text extraction with quality validation, structure
    recognition, and fallback mechanisms for corrupted or complex documents.
    """
    
    def __init__(
        self,
        websocket_manager: Optional[WebSocketManager] = None,
        enable_ocr: bool = False
    ):
        """
        Initialize PDF processor.
        
        Args:
            websocket_manager: Optional WebSocket manager for progress updates
            enable_ocr: Whether to enable OCR fallback (future enhancement)
        """
        self.websocket_manager = websocket_manager
        self.enable_ocr = enable_ocr
        
        # Load configuration
        self.config = get_settings()
        self.processing_config = self.config.processing_settings
        
        # Initialize extraction engines
        self.llama_reader = LlamaPDFReader(
            return_full_document=False,
            concatenate_pages=False
        )
        
        # Initialize quality analyzer
        self.quality_analyzer = PDFQualityAnalyzer()
        
        # Processing statistics
        self._stats = {
            "pdfs_processed": 0,
            "total_pages_processed": 0,
            "total_extraction_time_ms": 0,
            "method_usage": {method.value: 0 for method in PDFExtractionMethod},
            "structure_types_detected": {stype.value: 0 for stype in PDFStructureType},
            "quality_scores": [],
            "error_counts": {}
        }
        
        logger.info(
            "PDFProcessor initialized",
            enable_ocr=enable_ocr,
            extraction_methods=list(PDFExtractionMethod),
            structure_types=list(PDFStructureType)
        )
    
    async def extract_text_from_pdf(
        self,
        file_path: Union[str, Path],
        document: Optional[LegalDocument] = None,
        user_id: Optional[str] = None
    ) -> PDFExtractionResult:
        """
        Extract text from PDF file using multiple methods with quality validation.
        
        Args:
            file_path: Path to PDF file
            document: Optional document object for progress tracking
            user_id: Optional user ID for WebSocket updates
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            logger.info(
                "Starting PDF text extraction",
                file_path=str(file_path),
                document_id=document.document_id if document else None,
                file_size=get_file_size(file_path)
            )
            
            # Validate file
            validate_file_path(file_path, [DocumentType.PDF])
            
            # Send initial progress update
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id,
                    document.document_id if document else "unknown",
                    "Analyzing PDF structure...",
                    10
                )
            
            # Extract metadata first
            metadata = await self._extract_pdf_metadata(file_path)
            
            # Try extraction methods in order of preference
            extraction_result = None
            methods_attempted = []
            
            for method in PDFExtractionMethod:
                if method == PDFExtractionMethod.OCR and not self.enable_ocr:
                    continue
                
                try:
                    methods_attempted.append(method)
                    
                    if self.websocket_manager and user_id:
                        await self._send_progress_update(
                            user_id,
                            document.document_id if document else "unknown",
                            f"Extracting text using {method.value}...",
                            30 + len(methods_attempted) * 20
                        )
                    
                    if method == PDFExtractionMethod.LLAMA_INDEX:
                        extraction_result = await self._extract_with_llamaindex(file_path, metadata)
                    elif method == PDFExtractionMethod.PDFPLUMBER:
                        extraction_result = await self._extract_with_pdfplumber(file_path, metadata)
                    elif method == PDFExtractionMethod.PYPDF2:
                        extraction_result = await self._extract_with_pypdf2(file_path, metadata)
                    elif method == PDFExtractionMethod.OCR:
                        extraction_result = await self._extract_with_ocr(file_path, metadata)
                    
                    # Validate extraction quality
                    if self._validate_extraction_quality(extraction_result):
                        extraction_result.primary_method = method
                        extraction_result.fallback_methods_used = methods_attempted[:-1]
                        break
                    else:
                        logger.warning(
                            "Extraction quality insufficient, trying next method",
                            method=method.value,
                            confidence=extraction_result.extraction_confidence if extraction_result else 0,
                            file_path=str(file_path)
                        )
                        extraction_result = None
                
                except Exception as e:
                    error_msg = f"Extraction failed with {method.value}: {str(e)}"
                    logger.warning(
                        "PDF extraction method failed",
                        method=method.value,
                        error=str(e),
                        file_path=str(file_path)
                    )
                    
                    if extraction_result is None:
                        extraction_result = PDFExtractionResult(
                            full_text="",
                            pages=[],
                            metadata=metadata,
                            structure_type=PDFStructureType.UNKNOWN,
                            extraction_confidence=0.0,
                            total_extraction_time_ms=0.0,
                            primary_method=method,
                            error_messages=[error_msg]
                        )
                    else:
                        extraction_result.error_messages.append(error_msg)
            
            # If all methods failed, raise error
            if extraction_result is None or extraction_result.extraction_confidence < 0.1:
                raise DocumentProcessingError(
                    f"All PDF extraction methods failed for {file_path.name}",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                    document_id=document.document_id if document else None,
                    stage="pdf_extraction"
                )
            
            # Analyze document structure
            structure_type = self.quality_analyzer.detect_document_structure(
                extraction_result.full_text,
                metadata
            )
            extraction_result.structure_type = structure_type
            
            # Calculate total extraction time
            total_time_ms = (time.time() - start_time) * 1000
            extraction_result.total_extraction_time_ms = total_time_ms
            
            # Update statistics
            self._update_processing_stats(extraction_result)
            
            # Send completion update
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id,
                    document.document_id if document else "unknown",
                    f"Extraction completed - {extraction_result.page_count} pages processed",
                    100
                )
            
            logger.info(
                "PDF text extraction completed",
                file_path=str(file_path),
                method=extraction_result.primary_method.value,
                page_count=extraction_result.page_count,
                text_length=extraction_result.total_text_length,
                confidence=extraction_result.extraction_confidence,
                structure_type=structure_type.value,
                extraction_time_ms=total_time_ms
            )
            
            return extraction_result
            
        except Exception as e:
            # Update error statistics
            error_type = type(e).__name__
            self._stats["error_counts"][error_type] = self._stats["error_counts"].get(error_type, 0) + 1
            
            logger.error(
                "PDF text extraction failed",
                file_path=str(file_path),
                error=str(e),
                exc_info=True
            )
            
            # Send error update
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id,
                    document.document_id if document else "unknown",
                    f"Extraction failed: {str(e)}",
                    -1
                )
            
            raise DocumentProcessingError(
                f"PDF text extraction failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document.document_id if document else None,
                stage="pdf_extraction"
            )
    
    async def _extract_pdf_metadata(self, file_path: Path) -> PDFMetadata:
        """Extract PDF metadata using PyPDF2."""
        metadata = PDFMetadata()
        metadata.file_size_bytes = get_file_size(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Basic metadata
                metadata.page_count = len(pdf_reader.pages)
                metadata.is_encrypted = pdf_reader.is_encrypted
                
                # Document info
                if pdf_reader.metadata:
                    info = pdf_reader.metadata
                    metadata.title = info.get('/Title', '').strip() if info.get('/Title') else None
                    metadata.author = info.get('/Author', '').strip() if info.get('/Author') else None
                    metadata.subject = info.get('/Subject', '').strip() if info.get('/Subject') else None
                    metadata.creator = info.get('/Creator', '').strip() if info.get('/Creator') else None
                    metadata.producer = info.get('/Producer', '').strip() if info.get('/Producer') else None
                    
                    # Parse dates
                    if info.get('/CreationDate'):
                        try:
                            metadata.creation_date = self._parse_pdf_date(info['/CreationDate'])
                        except Exception:
                            pass
                    
                    if info.get('/ModDate'):
                        try:
                            metadata.modification_date = self._parse_pdf_date(info['/ModDate'])
                        except Exception:
                            pass
                    
                    # Keywords
                    if info.get('/Keywords'):
                        keywords_str = info['/Keywords'].strip()
                        if keywords_str:
                            metadata.keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                
                # Check if scanned (simplified heuristic)
                if metadata.page_count > 0:
                    first_page = pdf_reader.pages[0]
                    try:
                        text = first_page.extract_text()
                        metadata.is_scanned = len(text.strip()) < 50  # Very little extractable text
                    except Exception:
                        metadata.is_scanned = True
                
        except Exception as e:
            logger.warning(
                "Failed to extract PDF metadata",
                file_path=str(file_path),
                error=str(e)
            )
        
        return metadata
    
    def _parse_pdf_date(self, date_str: str) -> datetime:
        """Parse PDF date string to datetime object."""
        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
        if isinstance(date_str, str) and date_str.startswith('D:'):
            date_str = date_str[2:]  # Remove 'D:' prefix
        
        # Extract just the date/time part (ignore timezone)
        date_part = date_str[:14]  # YYYYMMDDHHMMSS
        
        try:
            return datetime.strptime(date_part, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            # Try shorter formats
            for fmt in ['%Y%m%d%H%M', '%Y%m%d']:
                try:
                    return datetime.strptime(date_part[:len(fmt)], fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        
        raise ValueError(f"Unable to parse PDF date: {date_str}")
    
    async def _extract_with_llamaindex(self, file_path: Path, metadata: PDFMetadata) -> PDFExtractionResult:
        """Extract text using LlamaIndex PDFReader."""
        start_time = time.time()
        
        try:
            # Use LlamaIndex to load documents
            documents = self.llama_reader.load_data(file_path)
            
            pages = []
            full_text_parts = []
            
            for i, doc in enumerate(documents):
                page_start_time = time.time()
                page_text = doc.text
                
                # Analyze page quality
                confidence = self.quality_analyzer.analyze_text_quality(page_text)
                
                page = PDFPage(
                    page_number=i + 1,
                    text_content=page_text,
                    text_length=len(page_text),
                    confidence_score=confidence,
                    extraction_method=PDFExtractionMethod.LLAMA_INDEX,
                    extraction_time_ms=(time.time() - page_start_time) * 1000
                )
                
                pages.append(page)
                full_text_parts.append(page_text)
            
            full_text = '\n\n'.join(full_text_parts)
            
            # Calculate overall confidence
            overall_confidence = self.quality_analyzer.analyze_text_quality(full_text)
            
            return PDFExtractionResult(
                full_text=full_text,
                pages=pages,
                metadata=metadata,
                structure_type=PDFStructureType.UNKNOWN,  # Will be set later
                extraction_confidence=overall_confidence,
                total_extraction_time_ms=(time.time() - start_time) * 1000,
                primary_method=PDFExtractionMethod.LLAMA_INDEX
            )
            
        except Exception as e:
            logger.error(
                "LlamaIndex PDF extraction failed",
                file_path=str(file_path),
                error=str(e)
            )
            raise
    
    async def _extract_with_pdfplumber(self, file_path: Path, metadata: PDFMetadata) -> PDFExtractionResult:
        """Extract text using pdfplumber for better layout preservation."""
        start_time = time.time()
        
        try:
            pages = []
            full_text_parts = []
            
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_start_time = time.time()
                    
                    # Extract text with layout preservation
                    page_text = page.extract_text() or ""
                    
                    # Analyze page elements
                    has_tables = len(page.find_tables()) > 0
                    has_images = len(page.images) > 0
                    
                    # Get bounding box elements for advanced analysis
                    bbox_elements = []
                    try:
                        words = page.extract_words()
                        bbox_elements = [
                            {
                                'text': word['text'],
                                'bbox': word['bbox'],
                                'fontname': word.get('fontname', ''),
                                'size': word.get('size', 0)
                            }
                            for word in words[:100]  # Limit for performance
                        ]
                    except Exception:
                        pass
                    
                    # Calculate confidence
                    confidence = self.quality_analyzer.analyze_text_quality(page_text)
                    
                    page_obj = PDFPage(
                        page_number=i + 1,
                        text_content=page_text,
                        text_length=len(page_text),
                        confidence_score=confidence,
                        has_images=has_images,
                        has_tables=has_tables,
                        extraction_method=PDFExtractionMethod.PDFPLUMBER,
                        extraction_time_ms=(time.time() - page_start_time) * 1000,
                        bbox_elements=bbox_elements
                    )
                    
                    pages.append(page_obj)
                    full_text_parts.append(page_text)
            
            full_text = '\n\n'.join(full_text_parts)
            overall_confidence = self.quality_analyzer.analyze_text_quality(full_text)
            
            return PDFExtractionResult(
                full_text=full_text,
                pages=pages,
                metadata=metadata,
                structure_type=PDFStructureType.UNKNOWN,
                extraction_confidence=overall_confidence,
                total_extraction_time_ms=(time.time() - start_time) * 1000,
                primary_method=PDFExtractionMethod.PDFPLUMBER
            )
            
        except Exception as e:
            logger.error(
                "PDFPlumber extraction failed",
                file_path=str(file_path),
                error=str(e)
            )
            raise
    
    async def _extract_with_pypdf2(self, file_path: Path, metadata: PDFMetadata) -> PDFExtractionResult:
        """Extract text using PyPDF2 as fallback method."""
        start_time = time.time()
        
        try:
            pages = []
            full_text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(pdf_reader.pages):
                    page_start_time = time.time()
                    
                    try:
                        page_text = page.extract_text()
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from page",
                            page_number=i + 1,
                            error=str(e)
                        )
                        page_text = ""
                    
                    confidence = self.quality_analyzer.analyze_text_quality(page_text)
                    
                    page_obj = PDFPage(
                        page_number=i + 1,
                        text_content=page_text,
                        text_length=len(page_text),
                        confidence_score=confidence,
                        extraction_method=PDFExtractionMethod.PYPDF2,
                        extraction_time_ms=(time.time() - page_start_time) * 1000
                    )
                    
                    pages.append(page_obj)
                    full_text_parts.append(page_text)
            
            full_text = '\n\n'.join(full_text_parts)
            overall_confidence = self.quality_analyzer.analyze_text_quality(full_text)
            
            return PDFExtractionResult(
                full_text=full_text,
                pages=pages,
                metadata=metadata,
                structure_type=PDFStructureType.UNKNOWN,
                extraction_confidence=overall_confidence,
                total_extraction_time_ms=(time.time() - start_time) * 1000,
                primary_method=PDFExtractionMethod.PYPDF2
            )
            
        except Exception as e:
            logger.error(
                "PyPDF2 extraction failed",
                file_path=str(file_path),
                error=str(e)
            )
            raise
    
    async def _extract_with_ocr(self, file_path: Path, metadata: PDFMetadata) -> PDFExtractionResult:
        """Extract text using OCR for scanned documents (future enhancement)."""
        # Placeholder for OCR implementation
        # Would integrate with Tesseract or cloud OCR services
        
        logger.warning(
            "OCR extraction not yet implemented",
            file_path=str(file_path)
        )
        
        raise NotImplementedError("OCR extraction is not yet implemented")
    
    def _validate_extraction_quality(self, result: Optional[PDFExtractionResult]) -> bool:
        """Validate extraction quality meets minimum thresholds."""
        if result is None:
            return False
        
        min_confidence = self.processing_config.get("min_extraction_confidence", 0.3)
        min_text_length = self.processing_config.get("min_text_length", 50)
        
        # Check overall confidence
        if result.extraction_confidence < min_confidence:
            return False
        
        # Check text length
        if result.total_text_length < min_text_length:
            return False
        
        # Check that at least 50% of pages have decent confidence
        if result.pages:
            good_pages = sum(1 for page in result.pages if page.confidence_score >= min_confidence)
            if good_pages / len(result.pages) < 0.5:
                return False
        
        return True
    
    def _update_processing_stats(self, result: PDFExtractionResult) -> None:
        """Update processing statistics."""
        self._stats["pdfs_processed"] += 1
        self._stats["total_pages_processed"] += result.page_count
        self._stats["total_extraction_time_ms"] += result.total_extraction_time_ms
        self._stats["method_usage"][result.primary_method.value] += 1
        self._stats["structure_types_detected"][result.structure_type.value] += 1
        self._stats["quality_scores"].append(result.extraction_confidence)
        
        # Keep quality scores list manageable
        if len(self._stats["quality_scores"]) > 1000:
            self._stats["quality_scores"] = self._stats["quality_scores"][-500:]
    
    async def _send_progress_update(
        self,
        user_id: str,
        document_id: str,
        message: str,
        progress_percent: int
    ) -> None:
        """Send progress update via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            update_data = {
                "document_id": document_id,
                "stage": "pdf_extraction",
                "message": message,
                "progress_percent": progress_percent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.websocket_manager.broadcast_to_user(
                user_id,
                "pdf_extraction_progress",
                update_data
            )
            
        except Exception as e:
            logger.warning(
                "Failed to send PDF extraction progress update",
                user_id=user_id,
                document_id=document_id,
                error=str(e)
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get PDF processing statistics."""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats["pdfs_processed"] > 0:
            stats["average_pages_per_pdf"] = stats["total_pages_processed"] / stats["pdfs_processed"]
            stats["average_extraction_time_ms"] = stats["total_extraction_time_ms"] / stats["pdfs_processed"]
        else:
            stats["average_pages_per_pdf"] = 0
            stats["average_extraction_time_ms"] = 0
        
        if stats["quality_scores"]:
            stats["average_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
            stats["min_quality_score"] = min(stats["quality_scores"])
            stats["max_quality_score"] = max(stats["quality_scores"])
        else:
            stats["average_quality_score"] = 0
            stats["min_quality_score"] = 0
            stats["max_quality_score"] = 0
        
        return stats
    
    def get_supported_methods(self) -> List[PDFExtractionMethod]:
        """Get list of supported extraction methods."""
        methods = [
            PDFExtractionMethod.LLAMA_INDEX,
            PDFExtractionMethod.PDFPLUMBER,
            PDFExtractionMethod.PYPDF2
        ]
        
        if self.enable_ocr:
            methods.append(PDFExtractionMethod.OCR)
        
        return methods
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "pdfs_processed": 0,
            "total_pages_processed": 0,
            "total_extraction_time_ms": 0,
            "method_usage": {method.value: 0 for method in PDFExtractionMethod},
            "structure_types_detected": {stype.value: 0 for stype in PDFStructureType},
            "quality_scores": [],
            "error_counts": {}
        }
        
        logger.info("PDF processor statistics reset")