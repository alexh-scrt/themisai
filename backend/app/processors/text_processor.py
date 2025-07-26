"""
Legal Document Text Processing and Semantic Chunking Engine

This module provides specialized text processing capabilities for legal documents,
focusing on structure-aware semantic chunking that preserves legal document
hierarchy, citations, and contextual relationships.

Key Features:
- Legal document structure recognition (sections, paragraphs, citations)
- Semantic chunking with paragraph-based boundaries
- Citation extraction and preservation
- Section header detection and hierarchy maintenance
- Context preservation for proper attribution
- Configurable chunk sizing with overlap management
- Text cleaning and normalization for legal documents

Legal Document Structure Support:
- Patent documents (claims, specifications, backgrounds)
- Legal briefs (sections, subsections, footnotes)
- Contracts (clauses, definitions, schedules)
- Court filings (headers, numbered paragraphs)
- Regulations (sections, subsections, references)

Processing Approach:
- Paragraph-level semantic boundaries
- Preserve legal citations within chunks
- Maintain section hierarchy metadata
- Ensure minimum chunk coherence
- Optimize for embedding quality and search relevance
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set, Pattern
from enum import Enum

from config.settings import get_settings
from ..models.domain.document import DocumentChunk, DocumentType
from ..utils.logging import get_logger
from ..core.exceptions import DocumentProcessingError, ErrorCode

logger = get_logger(__name__)


class LegalDocumentSection(Enum):
    """Legal document section types for structure preservation."""
    TITLE = "title"
    ABSTRACT = "abstract"
    BACKGROUND = "background"
    FIELD_OF_INVENTION = "field_of_invention"
    SUMMARY = "summary"
    DETAILED_DESCRIPTION = "detailed_description"
    CLAIMS = "claims"
    DEFINITIONS = "definitions"
    EXHIBIT = "exhibit"
    SCHEDULE = "schedule"
    APPENDIX = "appendix"
    CONCLUSION = "conclusion"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk with legal document context."""
    section_type: LegalDocumentSection = LegalDocumentSection.UNKNOWN
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    paragraph_number: Optional[int] = None
    page_number: Optional[int] = None
    is_section_header: bool = False
    contains_citations: bool = False
    citation_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    confidence_score: float = 1.0


@dataclass
class ChunkingConfig:
    """Configuration for text chunking behavior."""
    target_chunk_size: int = 512
    max_chunk_size: int = 768
    min_chunk_size: int = 100
    chunk_overlap: int = 50
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    respect_section_boundaries: bool = True
    merge_short_chunks: bool = True
    split_oversized_chunks: bool = True
    citation_preservation_priority: bool = True


class LegalDocumentParser:
    """
    Parser for extracting structure from legal documents.
    
    Identifies sections, paragraphs, citations, and other structural
    elements to guide intelligent chunking decisions.
    """
    
    def __init__(self):
        """Initialize legal document parser with patterns."""
        self._section_patterns = self._compile_section_patterns()
        self._citation_patterns = self._compile_citation_patterns()
        self._paragraph_patterns = self._compile_paragraph_patterns()
    
    def _compile_section_patterns(self) -> Dict[LegalDocumentSection, List[Pattern]]:
        """Compile regex patterns for section detection."""
        patterns = {
            LegalDocumentSection.TITLE: [
                re.compile(r'^(.{1,100})\n', re.MULTILINE),
                re.compile(r'TITLE:\s*(.+)', re.IGNORECASE),
            ],
            LegalDocumentSection.ABSTRACT: [
                re.compile(r'(?:ABSTRACT|Summary)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:ABSTRACT|Summary)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
            LegalDocumentSection.BACKGROUND: [
                re.compile(r'(?:BACKGROUND|Prior Art|Related Work)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:BACKGROUND|Prior Art)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
            LegalDocumentSection.FIELD_OF_INVENTION: [
                re.compile(r'(?:FIELD OF INVENTION|Technical Field)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:FIELD|Technical Field)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
            LegalDocumentSection.SUMMARY: [
                re.compile(r'(?:SUMMARY|Brief Summary)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:SUMMARY|Brief Summary)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
            LegalDocumentSection.DETAILED_DESCRIPTION: [
                re.compile(r'(?:DETAILED DESCRIPTION|Description)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:DETAILED DESCRIPTION|Description)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
            LegalDocumentSection.CLAIMS: [
                re.compile(r'(?:CLAIMS?|What is claimed)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:CLAIMS?|What is claimed)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*\d+\.\s*(?:A|An|The)\s+', re.MULTILINE),  # Numbered claims
            ],
            LegalDocumentSection.DEFINITIONS: [
                re.compile(r'(?:DEFINITIONS|Glossary)\s*\n', re.IGNORECASE | re.MULTILINE),
                re.compile(r'^\s*(?:DEFINITIONS|Glossary)\s*:?\s*$', re.IGNORECASE | re.MULTILINE),
            ],
        }
        return patterns
    
    def _compile_citation_patterns(self) -> List[Pattern]:
        """Compile regex patterns for legal citation detection."""
        return [
            # USPTO/Patent citations
            re.compile(r'\b(?:U\.S\.\s*Patent\s*(?:No\.?\s*)?)?(\d{1,2}[,\s]*\d{3}[,\s]*\d{3})\b', re.IGNORECASE),
            re.compile(r'\bUS\s*(\d{8,10})\s*[AB]\d?\b', re.IGNORECASE),
            
            # USC citations
            re.compile(r'\b\d+\s+U\.S\.C\.\s+§?\s*\d+(?:\([a-z0-9]+\))*', re.IGNORECASE),
            
            # Federal court cases
            re.compile(r'\b\d+\s+F\.\s*(?:2d|3d|\d+d?)\s+\d+', re.IGNORECASE),
            re.compile(r'\b\d+\s+F\.\s*Supp\.\s*(?:2d|3d)?\s+\d+', re.IGNORECASE),
            
            # Supreme Court cases
            re.compile(r'\b\d+\s+U\.S\.\s+\d+', re.IGNORECASE),
            re.compile(r'\b\d+\s+S\.\s*Ct\.\s+\d+', re.IGNORECASE),
            
            # Case numbers
            re.compile(r'(?:Case|Civil|Criminal)\s+(?:No\.?\s*)?[\w\d\-\:]+', re.IGNORECASE),
            re.compile(r'\b\d{1,2}:\d{2}-cv-\d+', re.IGNORECASE),
            
            # CFR citations
            re.compile(r'\b\d+\s+C\.F\.R\.\s+§?\s*\d+(?:\.\d+)*', re.IGNORECASE),
            
            # State citations (example patterns)
            re.compile(r'\b\d+\s+[A-Z][a-z]+\.?\s*(?:2d|3d)?\s+\d+', re.IGNORECASE),
            
            # International patent citations
            re.compile(r'\bWO\s*\d{4}\/\d+', re.IGNORECASE),
            re.compile(r'\bEP\s*\d+\s*[AB]\d?', re.IGNORECASE),
        ]
    
    def _compile_paragraph_patterns(self) -> List[Pattern]:
        """Compile patterns for paragraph and section numbering."""
        return [
            # Numbered paragraphs/sections
            re.compile(r'^\s*(\d+)\.\s+', re.MULTILINE),
            re.compile(r'^\s*\((\d+)\)\s+', re.MULTILINE),
            re.compile(r'^\s*([IVX]+)\.\s+', re.MULTILINE),  # Roman numerals
            re.compile(r'^\s*([a-z])\.\s+', re.MULTILINE),   # Lettered subsections
            
            # Claim numbering patterns
            re.compile(r'^\s*(\d+)\.\s*(?:A|An|The)\s+', re.MULTILINE),
            re.compile(r'^\s*Claim\s+(\d+)[:\.]', re.IGNORECASE | re.MULTILINE),
        ]
    
    def detect_sections(self, text: str) -> List[Tuple[LegalDocumentSection, int, Optional[str]]]:
        """
        Detect sections in legal document text.
        
        Returns list of (section_type, start_position, title) tuples.
        """
        sections = []
        
        for section_type, patterns in self._section_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    title = match.group(1) if match.groups() else None
                    sections.append((section_type, match.start(), title))
        
        # Sort by position and remove duplicates
        sections.sort(key=lambda x: x[1])
        return sections
    
    def extract_citations(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract legal citations from text.
        
        Returns list of (citation_text, start_pos, end_pos) tuples.
        """
        citations = []
        
        for pattern in self._citation_patterns:
            for match in pattern.finditer(text):
                citation = match.group(0).strip()
                if len(citation) >= 5:  # Filter out very short matches
                    citations.append((citation, match.start(), match.end()))
        
        # Remove overlapping citations, keeping longer ones
        citations.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        filtered_citations = []
        last_end = -1
        
        for citation, start, end in citations:
            if start >= last_end:
                filtered_citations.append((citation, start, end))
                last_end = end
        
        return filtered_citations
    
    def detect_paragraph_boundaries(self, text: str) -> List[Tuple[int, Optional[str]]]:
        """
        Detect paragraph boundaries and numbering.
        
        Returns list of (position, paragraph_number) tuples.
        """
        boundaries = []
        
        # Natural paragraph breaks (double newlines)
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append((match.end(), None))
        
        # Numbered paragraphs/sections
        for pattern in self._paragraph_patterns:
            for match in pattern.finditer(text):
                number = match.group(1) if match.groups() else None
                boundaries.append((match.start(), number))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        return boundaries


class TextProcessor:
    """
    Advanced text processor for legal documents with semantic chunking.
    
    Provides intelligent text segmentation that respects legal document
    structure, preserves citations, and optimizes chunks for embedding
    and search quality.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize text processor.
        
        Args:
            config: Optional chunking configuration override
        """
        self.config = config or self._load_default_config()
        self.parser = LegalDocumentParser()
        
        # Processing statistics
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "citations_preserved": 0,
            "sections_detected": 0,
            "average_chunk_size": 0,
            "processing_errors": 0
        }
        
        logger.info(
            "TextProcessor initialized",
            target_chunk_size=self.config.target_chunk_size,
            preserve_paragraphs=self.config.preserve_paragraphs,
            preserve_sentences=self.config.preserve_sentences
        )
    
    def _load_default_config(self) -> ChunkingConfig:
        """Load default chunking configuration from settings."""
        settings = get_settings()
        processing_config = settings.processing_settings
        
        return ChunkingConfig(
            target_chunk_size=processing_config.get("chunk_size", 512),
            max_chunk_size=processing_config.get("max_chunk_size", 768),
            min_chunk_size=processing_config.get("min_chunk_size", 100),
            chunk_overlap=processing_config.get("chunk_overlap", 50),
            preserve_paragraphs=processing_config.get("preserve_paragraphs", True),
            preserve_sentences=processing_config.get("preserve_sentences", True),
            respect_section_boundaries=processing_config.get("respect_section_boundaries", True),
            merge_short_chunks=processing_config.get("merge_short_chunks", True),
            split_oversized_chunks=processing_config.get("split_oversized_chunks", True),
            citation_preservation_priority=processing_config.get("citation_preservation_priority", True)
        )
    
    def process_document_text(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType,
        document_name: str
    ) -> List[DocumentChunk]:
        """
        Process document text into semantic chunks.
        
        Args:
            text: Raw document text
            document_id: Document identifier
            document_type: Type of document
            document_name: Document name for metadata
            
        Returns:
            List of processed document chunks
        """
        try:
            logger.info(
                "Starting text processing",
                document_id=document_id,
                text_length=len(text),
                document_type=document_type.value
            )
            
            # Step 1: Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Step 2: Analyze document structure
            sections = self.parser.detect_sections(cleaned_text)
            citations = self.parser.extract_citations(cleaned_text)
            paragraph_boundaries = self.parser.detect_paragraph_boundaries(cleaned_text)
            
            self._stats["sections_detected"] += len(sections)
            self._stats["citations_preserved"] += len(citations)
            
            logger.debug(
                "Document structure analysis completed",
                document_id=document_id,
                sections_count=len(sections),
                citations_count=len(citations),
                paragraph_boundaries=len(paragraph_boundaries)
            )
            
            # Step 3: Create semantic chunks
            chunks = self._create_semantic_chunks(
                cleaned_text,
                document_id,
                sections,
                citations,
                paragraph_boundaries
            )
            
            # Step 4: Post-process chunks
            processed_chunks = self._post_process_chunks(chunks, document_id)
            
            # Update statistics
            self._stats["documents_processed"] += 1
            self._stats["chunks_created"] += len(processed_chunks)
            if processed_chunks:
                avg_size = sum(len(chunk.content) for chunk in processed_chunks) / len(processed_chunks)
                self._stats["average_chunk_size"] = avg_size
            
            logger.info(
                "Text processing completed",
                document_id=document_id,
                chunks_created=len(processed_chunks),
                average_chunk_size=self._stats["average_chunk_size"]
            )
            
            return processed_chunks
            
        except Exception as e:
            self._stats["processing_errors"] += 1
            logger.error(
                "Text processing failed",
                document_id=document_id,
                error=str(e),
                exc_info=True
            )
            raise DocumentProcessingError(
                f"Text processing failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document_id,
                stage="text_processing"
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\x0c]', '\n', text)
        
        # Clean up excessive newlines but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_semantic_chunks(
        self,
        text: str,
        document_id: str,
        sections: List[Tuple[LegalDocumentSection, int, Optional[str]]],
        citations: List[Tuple[str, int, int]],
        paragraph_boundaries: List[Tuple[int, Optional[str]]]
    ) -> List[Tuple[str, int, int, ChunkMetadata]]:
        """Create semantic chunks respecting document structure."""
        chunks = []
        text_length = len(text)
        
        # Create section boundaries map
        section_map = self._create_section_map(sections, text_length)
        citation_map = self._create_citation_map(citations)
        
        # Start chunking process
        current_pos = 0
        chunk_index = 0
        
        while current_pos < text_length:
            # Determine chunk boundaries
            chunk_start = current_pos
            chunk_end, metadata = self._find_optimal_chunk_end(
                text,
                chunk_start,
                section_map,
                citation_map,
                paragraph_boundaries
            )
            
            # Extract chunk content
            chunk_content = text[chunk_start:chunk_end].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append((chunk_content, chunk_start, chunk_end, metadata))
                chunk_index += 1
            
            # Move to next position with overlap
            if self.config.chunk_overlap > 0 and chunk_end < text_length:
                overlap_start = max(chunk_start, chunk_end - self.config.chunk_overlap)
                current_pos = self._find_sentence_boundary(text, overlap_start, direction="forward")
            else:
                current_pos = chunk_end
        
        return chunks
    
    def _create_section_map(
        self,
        sections: List[Tuple[LegalDocumentSection, int, Optional[str]]],
        text_length: int
    ) -> Dict[int, Tuple[LegalDocumentSection, Optional[str]]]:
        """Create position-to-section mapping."""
        section_map = {}
        
        for i, (section_type, pos, title) in enumerate(sections):
            # Determine section end position
            if i + 1 < len(sections):
                section_end = sections[i + 1][1]
            else:
                section_end = text_length
            
            # Map all positions in this section
            for j in range(pos, section_end):
                section_map[j] = (section_type, title)
        
        return section_map
    
    def _create_citation_map(self, citations: List[Tuple[str, int, int]]) -> Set[int]:
        """Create set of positions within citations."""
        citation_positions = set()
        for _, start, end in citations:
            citation_positions.update(range(start, end))
        return citation_positions
    
    def _find_optimal_chunk_end(
        self,
        text: str,
        start_pos: int,
        section_map: Dict[int, Tuple[LegalDocumentSection, Optional[str]]],
        citation_map: Set[int],
        paragraph_boundaries: List[Tuple[int, Optional[str]]]
    ) -> Tuple[int, ChunkMetadata]:
        """Find optimal end position for a chunk."""
        target_end = start_pos + self.config.target_chunk_size
        max_end = min(start_pos + self.config.max_chunk_size, len(text))
        
        # Start with target size
        best_end = min(target_end, len(text))
        
        # Respect section boundaries if configured
        if self.config.respect_section_boundaries:
            current_section = section_map.get(start_pos, (LegalDocumentSection.UNKNOWN, None))[0]
            for pos in range(start_pos, max_end):
                if pos in section_map:
                    section_at_pos = section_map[pos][0]
                    if section_at_pos != current_section:
                        best_end = min(best_end, pos)
                        break
        
        # Respect paragraph boundaries if configured
        if self.config.preserve_paragraphs:
            best_end = self._find_paragraph_boundary(text, best_end, paragraph_boundaries)
        
        # Respect sentence boundaries if configured
        if self.config.preserve_sentences:
            best_end = self._find_sentence_boundary(text, best_end, direction="backward")
        
        # Avoid breaking citations
        if self.config.citation_preservation_priority:
            best_end = self._avoid_citation_break(best_end, citation_map)
        
        # Ensure minimum chunk size
        if best_end - start_pos < self.config.min_chunk_size:
            best_end = min(start_pos + self.config.min_chunk_size, len(text))
        
        # Create metadata for this chunk
        metadata = self._create_chunk_metadata(
            text,
            start_pos,
            best_end,
            section_map,
            citation_map
        )
        
        return best_end, metadata
    
    def _find_paragraph_boundary(
        self,
        text: str,
        target_pos: int,
        paragraph_boundaries: List[Tuple[int, Optional[str]]]
    ) -> int:
        """Find nearest paragraph boundary."""
        # Find closest paragraph boundary before target position
        best_pos = target_pos
        for boundary_pos, _ in paragraph_boundaries:
            if boundary_pos <= target_pos:
                best_pos = boundary_pos
            else:
                break
        
        return best_pos
    
    def _find_sentence_boundary(self, text: str, pos: int, direction: str = "backward") -> int:
        """Find nearest sentence boundary."""
        sentence_endings = {'.', '!', '?', ';'}
        
        if direction == "backward":
            for i in range(pos, max(0, pos - 100), -1):
                if i < len(text) and text[i] in sentence_endings:
                    # Check that next character is whitespace or end of text
                    if i + 1 >= len(text) or text[i + 1].isspace():
                        return i + 1
        else:  # forward
            for i in range(pos, min(len(text), pos + 100)):
                if text[i] in sentence_endings:
                    if i + 1 >= len(text) or text[i + 1].isspace():
                        return i + 1
        
        return pos
    
    def _avoid_citation_break(self, pos: int, citation_map: Set[int]) -> int:
        """Adjust position to avoid breaking citations."""
        # If position is within a citation, move backward to start of citation
        if pos in citation_map:
            while pos > 0 and pos in citation_map:
                pos -= 1
        
        return pos
    
    def _create_chunk_metadata(
        self,
        text: str,
        start_pos: int,
        end_pos: int,
        section_map: Dict[int, Tuple[LegalDocumentSection, Optional[str]]],
        citation_map: Set[int]
    ) -> ChunkMetadata:
        """Create metadata for a text chunk."""
        chunk_text = text[start_pos:end_pos]
        
        # Determine section information
        section_info = section_map.get(start_pos, (LegalDocumentSection.UNKNOWN, None))
        section_type, section_title = section_info
        
        # Count citations in chunk
        citation_count = len([pos for pos in range(start_pos, end_pos) if pos in citation_map])
        
        # Calculate text statistics
        word_count = len(chunk_text.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', chunk_text) if s.strip()])
        
        # Check if this is a section header
        is_section_header = (
            len(chunk_text) < 100 and
            section_title and
            section_title.lower() in chunk_text.lower()
        )
        
        return ChunkMetadata(
            section_type=section_type,
            section_title=section_title,
            contains_citations=citation_count > 0,
            citation_count=citation_count,
            word_count=word_count,
            sentence_count=sentence_count,
            is_section_header=is_section_header,
            confidence_score=self._calculate_chunk_confidence(chunk_text, citation_count)
        )
    
    def _calculate_chunk_confidence(self, chunk_text: str, citation_count: int) -> float:
        """Calculate confidence score for chunk quality."""
        confidence = 1.0
        
        # Penalize very short chunks
        if len(chunk_text) < self.config.min_chunk_size:
            confidence *= 0.7
        
        # Penalize chunks with poor text quality indicators
        printable_ratio = sum(1 for c in chunk_text if c.isprintable()) / len(chunk_text)
        confidence *= printable_ratio
        
        # Boost chunks with legal citations
        if citation_count > 0:
            confidence = min(1.0, confidence + 0.1 * citation_count)
        
        # Penalize chunks with excessive whitespace
        whitespace_ratio = sum(1 for c in chunk_text if c.isspace()) / len(chunk_text)
        if whitespace_ratio > 0.3:
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def _post_process_chunks(
        self,
        chunks: List[Tuple[str, int, int, ChunkMetadata]],
        document_id: str
    ) -> List[DocumentChunk]:
        """Post-process chunks and create DocumentChunk objects."""
        processed_chunks = []
        
        # Merge short chunks if configured
        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)
        
        # Split oversized chunks if configured
        if self.config.split_oversized_chunks:
            chunks = self._split_oversized_chunks(chunks)
        
        # Create DocumentChunk objects
        for i, (content, start_pos, end_pos, metadata) in enumerate(chunks):
            # Extract legal citations from content
            legal_citations = self._extract_chunk_citations(content)
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{i:04d}",
                document_id=document_id,
                content=content,
                chunk_index=i,
                start_char=start_pos,
                end_char=end_pos,
                section_title=metadata.section_title,
                page_number=metadata.page_number,
                paragraph_number=metadata.paragraph_number,
                legal_citations=legal_citations
            )
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _merge_short_chunks(
        self,
        chunks: List[Tuple[str, int, int, ChunkMetadata]]
    ) -> List[Tuple[str, int, int, ChunkMetadata]]:
        """Merge adjacent short chunks."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            content, start_pos, end_pos, metadata = current_chunk
            next_content, next_start, next_end, next_metadata = next_chunk
            
            # Check if current chunk is short and can be merged
            if (len(content) < self.config.min_chunk_size and
                metadata.section_type == next_metadata.section_type and
                len(content + " " + next_content) <= self.config.max_chunk_size):
                
                # Merge chunks
                merged_content = content + " " + next_content
                merged_metadata = ChunkMetadata(
                    section_type=metadata.section_type,
                    section_title=metadata.section_title,
                    contains_citations=metadata.contains_citations or next_metadata.contains_citations,
                    citation_count=metadata.citation_count + next_metadata.citation_count,
                    word_count=metadata.word_count + next_metadata.word_count,
                    sentence_count=metadata.sentence_count + next_metadata.sentence_count,
                    confidence_score=min(metadata.confidence_score, next_metadata.confidence_score)
                )
                
                current_chunk = (merged_content, start_pos, next_end, merged_metadata)
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def _split_oversized_chunks(
        self,
        chunks: List[Tuple[str, int, int, ChunkMetadata]]
    ) -> List[Tuple[str, int, int, ChunkMetadata]]:
        """Split chunks that exceed maximum size."""
        split_chunks = []
        
        for content, start_pos, end_pos, metadata in chunks:
            if len(content) <= self.config.max_chunk_size:
                split_chunks.append((content, start_pos, end_pos, metadata))
            else:
                # Split oversized chunk
                sub_chunks = self._split_chunk_content(content, start_pos, metadata)
                split_chunks.extend(sub_chunks)
        
        return split_chunks
    
    def _split_chunk_content(
        self,
        content: str,
        base_start_pos: int,
        metadata: ChunkMetadata
    ) -> List[Tuple[str, int, int, ChunkMetadata]]:
        """Split a single chunk into smaller chunks."""
        sub_chunks = []
        content_length = len(content)
        current_pos = 0
        
        while current_pos < content_length:
            chunk_end = min(current_pos + self.config.target_chunk_size, content_length)
            
            # Find sentence boundary
            if chunk_end < content_length:
                chunk_end = self._find_sentence_boundary(content, chunk_end, direction="backward")
            
            sub_content = content[current_pos:chunk_end].strip()
            if sub_content:
                sub_metadata = ChunkMetadata(
                    section_type=metadata.section_type,
                    section_title=metadata.section_title,
                    contains_citations=any(pattern.search(sub_content) for pattern in self.parser._citation_patterns),
                    word_count=len(sub_content.split()),
                    sentence_count=len([s for s in re.split(r'[.!?]+', sub_content) if s.strip()]),
                    confidence_score=metadata.confidence_score * 0.9  # Slightly lower confidence for split chunks
                )
                
                sub_chunks.append((
                    sub_content,
                    base_start_pos + current_pos,
                    base_start_pos + chunk_end,
                    sub_metadata
                ))
            
            # Move to next position with overlap
            current_pos = chunk_end
            if self.config.chunk_overlap > 0 and current_pos < content_length:
                current_pos = max(current_pos - self.config.chunk_overlap, current_pos)
        
        return sub_chunks
    
    def _extract_chunk_citations(self, content: str) -> List[str]:
        """Extract legal citations from chunk content."""
        citations = []
        for pattern in self.parser._citation_patterns:
            for match in pattern.finditer(content):
                citation = match.group(0).strip()
                if len(citation) >= 5 and citation not in citations:
                    citations.append(citation)
        return citations
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get text processing statistics."""
        return {
            **self._stats,
            "config": {
                "target_chunk_size": self.config.target_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "min_chunk_size": self.config.min_chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "preserve_paragraphs": self.config.preserve_paragraphs,
                "preserve_sentences": self.config.preserve_sentences,
            }
        }
    
    def update_config(self, new_config: ChunkingConfig) -> None:
        """Update chunking configuration."""
        old_config = self.config
        self.config = new_config
        
        logger.info(
            "Text processor configuration updated",
            old_chunk_size=old_config.target_chunk_size,
            new_chunk_size=new_config.target_chunk_size,
            old_overlap=old_config.chunk_overlap,
            new_overlap=new_config.chunk_overlap
        )