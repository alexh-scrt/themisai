"""
UI Helper Functions for Patexia Legal AI Chatbot Frontend

This module provides comprehensive utility functions for the Gradio-based frontend interface.
It includes validation, formatting, data transformation, and UI component generation
functions optimized for legal document processing workflows.

Key Features:
- Input validation for legal case names, search queries, and file uploads
- Data formatting for timestamps, file sizes, and legal document metadata
- HTML rendering for search results, case lists, and progress indicators
- Visual marker management for case identification
- Legal document type detection and categorization
- Search suggestion generation and query preprocessing
- Error message formatting and user feedback
- Performance metrics display and monitoring
- File type validation and size formatting
- Legal citation parsing and formatting

Architecture Integration:
- Supports Gradio component rendering and validation
- Integrates with API client for data transformation
- Provides consistent UI patterns across components
- Handles legal document specific formatting requirements
- Supports real-time progress tracking and status updates
- Enables search result highlighting and snippet generation

Legal Document Optimizations:
- Case name validation following legal naming conventions
- Document type recognition for legal document categories
- Citation format validation and standardization
- Legal entity name parsing and validation
- Court document structure recognition
- Patent document specific formatting

UI Component Support:
- Dynamic HTML generation for complex layouts
- CSS class management for styling consistency
- JavaScript integration for interactive elements
- Progress indicator styling and animation
- Error state presentation and recovery guidance
- Success state confirmation and next steps
"""

import re
import html
import unicodedata
from typing import (
    Any,
    List,
    Dict,
    Set,
    Tuple,
    Optional,
    NamedTuple
)
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote, unquote
import base64

# Constants for legal document processing
LEGAL_CASE_NAME_PATTERN = r'^[a-zA-Z0-9\s\-_.,()&]+$'
LEGAL_CITATION_PATTERNS = [
    r'\d+\s+[A-Z][a-z.]+\s+\d+',  # Federal reporters (e.g., "123 F.3d 456")
    r'\d+\s+U\.S\.C\.\s+§?\s*\d+',  # U.S. Code citations
    r'\d+\s+C\.F\.R\.\s+§?\s*\d+',  # Code of Federal Regulations
]

# File size constants
BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * 1024
BYTES_PER_GB = BYTES_PER_MB * 1024

# Legal document type mappings
LEGAL_FILE_EXTENSIONS = {
    '.pdf': 'Portable Document Format',
    '.txt': 'Plain Text Document',
    '.doc': 'Microsoft Word Document',
    '.docx': 'Microsoft Word Document (XML)',
    '.rtf': 'Rich Text Format',
}

# Visual marker definitions
VISUAL_MARKER_COLORS = [
    ("#e74c3c", "Red", "Urgent/Litigation"),
    ("#27ae60", "Green", "Contract/Completed"),
    ("#3498db", "Blue", "IP/Patent"),
    ("#f39c12", "Orange", "Corporate/M&A"),
    ("#9b59b6", "Purple", "Regulatory/Compliance"),
    ("#1abc9c", "Teal", "Discovery/Investigation"),
    ("#e67e22", "Dark Orange", "Appeals/Motion"),
    ("#34495e", "Gray", "Archived/Reference"),
]

VISUAL_MARKER_ICONS = [
    ("📄", "Document", "General legal documents"),
    ("⚖️", "Legal", "Court filings and litigation"),
    ("🏢", "Corporate", "Business and corporate law"),
    ("💼", "Business", "Commercial transactions"),
    ("📋", "Contract", "Agreements and contracts"),
    ("🔍", "Investigation", "Discovery and research"),
    ("⚡", "Urgent", "High priority cases"),
    ("🎯", "Priority", "Focused cases"),
    ("📊", "Analytics", "Data-driven cases"),
    ("🔒", "Confidential", "Sensitive matters"),
]


class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    suggestions: Optional[List[str]] = None


class DocumentCategory(str, Enum):
    """Categories of legal documents for UI organization."""
    PATENT = "patent"
    CONTRACT = "contract"
    LITIGATION = "litigation"
    REGULATORY = "regulatory"
    CORPORATE = "corporate"
    DISCOVERY = "discovery"
    RESEARCH = "research"
    OTHER = "other"


@dataclass
class FormattedSearchResult:
    """Formatted search result for UI display."""
    title: str
    snippet: str
    document_type: str
    relevance_score: float
    page_number: Optional[int]
    highlighted_text: str
    document_id: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UITheme:
    """UI theme configuration for consistent styling."""
    primary_color: str = "#3498db"
    secondary_color: str = "#2c3e50"
    success_color: str = "#27ae60"
    warning_color: str = "#f39c12"
    error_color: str = "#e74c3c"
    info_color: str = "#1abc9c"
    background_color: str = "#f8f9fa"
    text_color: str = "#2c3e50"
    border_radius: str = "8px"
    font_family: str = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"


def clean_html_text(text: str, preserve_formatting: bool = True) -> str:
    """
    Clean and sanitize HTML text for safe display in Gradio components.
    
    This function removes potentially dangerous HTML tags while preserving
    safe formatting elements for legal document display.
    
    Args:
        text: Raw HTML or text content to clean
        preserve_formatting: Whether to preserve basic formatting tags
        
    Returns:
        Cleaned and sanitized text suitable for HTML display
    """
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text)
    
    # Remove null bytes and other control characters
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Define allowed HTML tags for legal documents
    if preserve_formatting:
        # Allow basic formatting tags that are safe and useful for legal documents
        allowed_tags = {
            'p', 'br', 'div', 'span', 'strong', 'b', 'em', 'i', 'u', 'sub', 'sup',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre', 'code',
            'ol', 'ul', 'li', 'table', 'tr', 'td', 'th', 'tbody', 'thead', 'tfoot'
        }
        
        # Define safe attributes for allowed tags
        safe_attributes = {
            'class', 'id', 'style', 'data-*', 'title', 'alt', 'colspan', 'rowspan'
        }
        
        # Remove dangerous tags while preserving safe ones
        text = _clean_html_tags(text, allowed_tags, safe_attributes)
    else:
        # Strip all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
    text = text.strip()
    
    # Escape any remaining potentially dangerous characters
    if not preserve_formatting:
        text = html.escape(text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")  # Smart apostrophe
    text = text.replace('â€œ', '"')  # Smart quote open
    text = text.replace('â€?', '"')  # Smart quote close
    text = text.replace('â€"', '–')  # En dash
    text = text.replace('â€"', '—')  # Em dash
    text = text.replace('Â', '')    # Non-breaking space artifacts
    
    return text


def _clean_html_tags(text: str, allowed_tags: Set[str], safe_attributes: Set[str]) -> str:
    """
    Helper function to clean HTML tags while preserving allowed ones.
    
    Args:
        text: HTML text to clean
        allowed_tags: Set of allowed HTML tag names
        safe_attributes: Set of safe attribute names
        
    Returns:
        Cleaned HTML text
    """
    # Pattern to match HTML tags with their attributes
    tag_pattern = re.compile(r'<(/?)([a-zA-Z0-9]+)([^>]*)>', re.IGNORECASE)
    
    def replace_tag(match):
        closing_slash = match.group(1)
        tag_name = match.group(2).lower()
        attributes = match.group(3)
        
        # If tag is not allowed, remove it entirely
        if tag_name not in allowed_tags:
            return ''
        
        # For closing tags, just return the tag if it's allowed
        if closing_slash:
            return f'</{tag_name}>'
        
        # Clean attributes for opening tags
        cleaned_attrs = _clean_html_attributes(attributes, safe_attributes)
        
        return f'<{tag_name}{cleaned_attrs}>'
    
    return tag_pattern.sub(replace_tag, text)


def _clean_html_attributes(attr_string: str, safe_attributes: Set[str]) -> str:
    """
    Clean HTML attributes, keeping only safe ones.
    
    Args:
        attr_string: String containing HTML attributes
        safe_attributes: Set of safe attribute names
        
    Returns:
        Cleaned attribute string
    """
    if not attr_string.strip():
        return ''
    
    # Pattern to match HTML attributes
    attr_pattern = re.compile(r'([a-zA-Z0-9-]+)\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
    safe_attrs = []
    
    for match in attr_pattern.finditer(attr_string):
        attr_name = match.group(1).lower()
        attr_value = match.group(2)
        
        # Check if attribute is safe
        is_safe = False
        for safe_attr in safe_attributes:
            if safe_attr.endswith('*'):
                # Wildcard attribute (e.g., data-*)
                if attr_name.startswith(safe_attr[:-1]):
                    is_safe = True
                    break
            elif attr_name == safe_attr:
                is_safe = True
                break
        
        if is_safe:
            # Additional validation for style attributes
            if attr_name == 'style':
                attr_value = _clean_css_style(attr_value)
                if attr_value:  # Only add if style has content after cleaning
                    safe_attrs.append(f'{attr_name}="{attr_value}"')
            else:
                # Escape attribute value
                attr_value = html.escape(attr_value, quote=True)
                safe_attrs.append(f'{attr_name}="{attr_value}"')
    
    return ' ' + ' '.join(safe_attrs) if safe_attrs else ''


def _clean_css_style(style: str) -> str:
    """
    Clean CSS style attribute, removing potentially dangerous properties.
    
    Args:
        style: CSS style string
        
    Returns:
        Cleaned CSS style string
    """
    if not style:
        return ''
    
    # Allowed CSS properties for legal document formatting
    allowed_properties = {
        'color', 'background-color', 'font-size', 'font-weight', 'font-style',
        'text-decoration', 'text-align', 'margin', 'padding', 'border',
        'border-radius', 'width', 'height', 'max-width', 'max-height',
        'display', 'float', 'clear', 'line-height', 'letter-spacing',
        'word-spacing', 'text-indent', 'white-space'
    }
    
    # Remove dangerous CSS (javascript:, expression(), etc.)
    dangerous_patterns = [
        r'javascript\s*:',
        r'expression\s*\(',
        r'behavior\s*:',
        r'@import',
        r'url\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        style = re.sub(pattern, '', style, flags=re.IGNORECASE)
    
    # Parse and clean individual CSS properties
    properties = []
    for prop in style.split(';'):
        if ':' in prop:
            prop_name, prop_value = prop.split(':', 1)
            prop_name = prop_name.strip().lower()
            prop_value = prop_value.strip()
            
            if prop_name in allowed_properties and prop_value:
                properties.append(f'{prop_name}: {prop_value}')
    
    return '; '.join(properties)


def extract_legal_citations(text: str, citation_types: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Extract legal citations from text using comprehensive regex patterns.
    
    This function identifies various types of legal citations commonly found
    in legal documents, including case citations, statutory citations, and
    regulatory citations.
    
    Args:
        text: Text content to extract citations from
        citation_types: Optional list of citation types to extract (default: all)
        
    Returns:
        List of dictionaries containing extracted citations with metadata
    """
    if not text:
        return []
    
    # Define comprehensive legal citation patterns
    citation_patterns = {
        'case_citation': [
            # Federal courts
            r'\b\d+\s+U\.S\.?\s+\d+\b',           # Supreme Court (e.g., "123 U.S. 456")
            r'\b\d+\s+S\.?\s*Ct\.?\s+\d+\b',      # Supreme Court Reporter
            r'\b\d+\s+F\.?\s*\d*d?\s+\d+\b',      # Federal Reporter (F., F.2d, F.3d)
            r'\b\d+\s+F\.?\s*Supp\.?\s*\d*\s+\d+\b', # Federal Supplement
            
            # State courts
            r'\b\d+\s+[A-Z][a-z]*\.?\s*\d*d?\s+\d+\b', # State reporters (e.g., "Cal.2d")
            r'\b\d+\s+P\.?\s*\d*d?\s+\d+\b',      # Pacific Reporter
            r'\b\d+\s+N\.E\.?\s*\d*d?\s+\d+\b',   # North Eastern Reporter
            r'\b\d+\s+S\.E\.?\s*\d*d?\s+\d+\b',   # South Eastern Reporter
            r'\b\d+\s+S\.W\.?\s*\d*d?\s+\d+\b',   # South Western Reporter
            r'\b\d+\s+N\.W\.?\s*\d*d?\s+\d+\b',   # North Western Reporter
            r'\b\d+\s+A\.?\s*\d*d?\s+\d+\b',      # Atlantic Reporter
            r'\b\d+\s+So\.?\s*\d*d?\s+\d+\b',     # Southern Reporter
        ],
        
        'statute_citation': [
            # Federal statutes
            r'\b\d+\s+U\.S\.C\.?\s+§?\s*\d+(?:\([a-z0-9]+\))?\b',  # U.S. Code
            r'\bPub\.?\s*L\.?\s*No\.?\s*\d+-\d+\b',                # Public Law
            r'\b\d+\s+Stat\.?\s+\d+\b',                           # Statutes at Large
            
            # State statutes
            r'\b[A-Z][a-z]*\.?\s*Rev\.?\s*Stat\.?\s*§?\s*\d+\b',  # State Revised Statutes
            r'\b[A-Z][a-z]*\.?\s*Code\s*§?\s*\d+\b',              # State Codes
        ],
        
        'regulation_citation': [
            # Federal regulations
            r'\b\d+\s+C\.F\.R\.?\s+§?\s*\d+(?:\.\d+)*\b',        # Code of Federal Regulations
            r'\b\d+\s+Fed\.?\s*Reg\.?\s+\d+\b',                   # Federal Register
        ],
        
        'patent_citation': [
            r'\bU\.?S\.?\s+Patent\s+No\.?\s+[\d,]+\b',             # US Patents
            r'\bU\.?S\.?\s+Pat\.?\s+No\.?\s+[\d,]+\b',             # US Patents (abbreviated)
            r'\bPat\.?\s+No\.?\s+[\d,]+\b',                        # Patent Numbers
            r'\bPatent\s+No\.?\s+[\d,]+\b',                        # Patent Numbers
            r'\bU\.?S\.?\s+Patent\s+Application\s+No\.?\s+[\d/,]+\b', # Patent Applications
        ],
        
        'court_citation': [
            r'\bNo\.?\s+\d{2,}-\d+\b',                             # Case numbers
            r'\bCiv\.?\s+No\.?\s+\d+-\d+\b',                      # Civil case numbers
            r'\bCrim\.?\s+No\.?\s+\d+-\d+\b',                     # Criminal case numbers
        ]
    }
    
    # Filter citation types if specified
    if citation_types:
        citation_patterns = {k: v for k, v in citation_patterns.items() if k in citation_types}
    
    citations = []
    
    for citation_type, patterns in citation_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation_text = match.group().strip()
                
                # Additional validation to avoid false positives
                if _validate_citation(citation_text, citation_type):
                    citations.append({
                        'text': citation_text,
                        'type': citation_type,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'confidence': _calculate_citation_confidence(citation_text, citation_type),
                        'normalized': _normalize_citation(citation_text, citation_type)
                    })
    
    # Remove duplicates and sort by position
    seen_citations = set()
    unique_citations = []
    
    for citation in sorted(citations, key=lambda x: x['start_pos']):
        citation_key = (citation['normalized'], citation['type'])
        if citation_key not in seen_citations:
            seen_citations.add(citation_key)
            unique_citations.append(citation)
    
    return unique_citations


def _validate_citation(citation_text: str, citation_type: str) -> bool:
    """
    Validate that a potential citation is likely to be a real legal citation.
    
    Args:
        citation_text: The citation text to validate
        citation_type: The type of citation
        
    Returns:
        True if the citation appears valid
    """
    # Basic length check
    if len(citation_text) < 5 or len(citation_text) > 100:
        return False
    
    # Type-specific validation
    if citation_type == 'case_citation':
        # Should have at least one number and one period or space
        if not re.search(r'\d+', citation_text) or not re.search(r'[\.\s]', citation_text):
            return False
        
        # Avoid common false positives like dates
        if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', citation_text):
            return False
    
    elif citation_type == 'statute_citation':
        # Should contain section symbol or "section"
        if not re.search(r'§|[Ss]ection', citation_text):
            return False
    
    elif citation_type == 'patent_citation':
        # Should contain "patent" or "pat"
        if not re.search(r'[Pp]at', citation_text):
            return False
    
    return True


def _calculate_citation_confidence(citation_text: str, citation_type: str) -> float:
    """
    Calculate confidence score for a legal citation.
    
    Args:
        citation_text: The citation text
        citation_type: The type of citation
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # Base confidence
    
    # Increase confidence for well-formed citations
    if citation_type == 'case_citation':
        if re.search(r'\d+\s+[A-Z][\w\.]+\s+\d+', citation_text):
            confidence += 0.3
        if re.search(r'F\.\d*d?|U\.S\.|S\.Ct\.', citation_text):
            confidence += 0.2
    
    elif citation_type == 'statute_citation':
        if re.search(r'U\.S\.C\.', citation_text):
            confidence += 0.3
        if re.search(r'§\s*\d+', citation_text):
            confidence += 0.2
    
    elif citation_type == 'patent_citation':
        if re.search(r'U\.?S\.?\s+Patent', citation_text):
            confidence += 0.3
        if re.search(r'No\.?\s+[\d,]+', citation_text):
            confidence += 0.2
    
    return min(1.0, confidence)


def _normalize_citation(citation_text: str, citation_type: str) -> str:
    """
    Normalize citation text for comparison and deduplication.
    
    Args:
        citation_text: The citation text to normalize
        citation_type: The type of citation
        
    Returns:
        Normalized citation text
    """
    # Basic normalization
    normalized = citation_text.upper().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
    
    # Type-specific normalization
    if citation_type == 'case_citation':
        # Standardize reporter abbreviations
        normalized = re.sub(r'F\.?\s*(\d*)D?', r'F.\1d', normalized)
        normalized = re.sub(r'U\.?S\.?', 'U.S.', normalized)
        normalized = re.sub(r'S\.?\s*CT\.?', 'S.Ct.', normalized)
    
    elif citation_type == 'statute_citation':
        # Standardize USC format
        normalized = re.sub(r'U\.?S\.?C\.?', 'U.S.C.', normalized)
        normalized = re.sub(r'§\s*', '§ ', normalized)
    
    elif citation_type == 'patent_citation':
        # Standardize patent format
        normalized = re.sub(r'U\.?S\.?\s+PAT(?:ENT)?', 'U.S. Patent', normalized)
        normalized = re.sub(r'NO\.?\s*', 'No. ', normalized)
    
    return normalized


def highlight_search_terms(
    text: str, 
    search_terms: List[str], 
    highlight_class: str = "search-highlight",
    case_sensitive: bool = False,
    whole_words_only: bool = False
) -> str:
    """
    Apply HTML highlighting to search terms within text content.
    
    This function adds HTML markup to highlight search terms in text,
    supporting multiple highlighting styles, case sensitivity options,
    and fuzzy matching for legal document search.
    
    Args:
        text: Text content to apply highlighting to
        search_terms: List of terms to highlight
        highlight_class: CSS class name for highlighting
        case_sensitive: Whether matching should be case sensitive
        whole_words_only: Whether to match only whole words
        
    Returns:
        Text with HTML highlighting markup applied
    """
    if not text or not search_terms:
        return text
    
    # Clean and prepare text
    text = str(text)
    
    # Define highlight colors for multiple terms
    highlight_colors = [
        "#fef3c7",  # Yellow
        "#ddd6fe",  # Purple
        "#fed7d7",  # Red
        "#d1fae5",  # Green
        "#fce7f3",  # Pink
        "#e0f2fe",  # Blue
        "#f0f9ff",  # Light blue
        "#f7fee7",  # Light green
    ]
    
    # Process each search term
    highlighted_text = text
    
    for i, term in enumerate(search_terms):
        if not term.strip():
            continue
        
        # Get highlight color for this term
        color = highlight_colors[i % len(highlight_colors)]
        
        # Create highlighting pattern
        if whole_words_only:
            # Match whole words only
            pattern = r'\b' + re.escape(term.strip()) + r'\b'
        else:
            # Match partial words
            pattern = re.escape(term.strip())
        
        # Set case sensitivity
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Create replacement function
        def replace_match(match):
            matched_text = match.group(0)
            return f'<mark class="{highlight_class}" style="background-color: {color}; padding: 1px 3px; border-radius: 3px; font-weight: 500;" data-term="{html.escape(term)}">{matched_text}</mark>'
        
        # Apply highlighting
        try:
            highlighted_text = re.sub(pattern, replace_match, highlighted_text, flags=flags)
        except re.error:
            # If regex fails (e.g., due to special characters), skip this term
            continue
    
    # Post-process to handle overlapping highlights
    highlighted_text = _merge_overlapping_highlights(highlighted_text)
    
    return highlighted_text


def _merge_overlapping_highlights(text: str) -> str:
    """
    Merge overlapping highlight marks to avoid nested HTML tags.
    
    Args:
        text: Text with potentially overlapping highlight marks
        
    Returns:
        Text with merged highlights
    """
    # This is a simplified version - a full implementation would need
    # more sophisticated parsing to handle all edge cases
    
    # Remove nested marks (basic cleanup)
    text = re.sub(r'<mark[^>]*>([^<]*)<mark[^>]*>([^<]*)</mark>([^<]*)</mark>', 
                  r'<mark class="search-highlight" style="background-color: #fef3c7; padding: 1px 3px; border-radius: 3px; font-weight: 500;">\1\2\3</mark>', 
                  text)
    
    return text


# Additional helper functions that support the main methods

def get_legal_term_suggestions(text: str, max_suggestions: int = 10) -> List[str]:
    """
    Get legal term suggestions based on input text.
    
    Args:
        text: Input text to analyze
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested legal terms
    """
    # Common legal terms and phrases
    legal_terms = [
        "contract", "agreement", "liability", "damages", "negligence", "breach",
        "patent", "trademark", "copyright", "infringement", "claim", "defendant",
        "plaintiff", "jurisdiction", "venue", "discovery", "deposition", "motion",
        "summary judgment", "preliminary injunction", "cease and desist",
        "intellectual property", "due diligence", "force majeure", "indemnification",
        "confidentiality", "non-disclosure", "licensing", "royalty", "settlement"
    ]
    
    # Find terms that match the input
    text_lower = text.lower()
    suggestions = []
    
    for term in legal_terms:
        if text_lower in term.lower() or term.lower().startswith(text_lower):
            suggestions.append(term)
        
        if len(suggestions) >= max_suggestions:
            break
    
    return suggestions


def create_citation_link(citation: Dict[str, str]) -> str:
    """
    Create a clickable link for a legal citation.
    
    Args:
        citation: Citation dictionary with text and type
        
    Returns:
        HTML link element for the citation
    """
    citation_text = citation.get('text', '')
    citation_type = citation.get('type', '')
    
    # Create different links based on citation type
    if citation_type == 'case_citation':
        # For now, just create a Google Scholar link
        search_query = citation_text.replace(' ', '+')
        url = f"https://scholar.google.com/scholar?q={search_query}"
    elif citation_type == 'statute_citation':
        # Link to relevant legal database
        search_query = citation_text.replace(' ', '+')
        url = f"https://www.law.cornell.edu/search?q={search_query}"
    elif citation_type == 'patent_citation':
        # Extract patent number and link to USPTO
        patent_num = re.search(r'[\d,]+', citation_text)
        if patent_num:
            clean_num = patent_num.group().replace(',', '')
            url = f"https://patents.uspto.gov/patent/{clean_num}"
        else:
            url = f"https://patents.uspto.gov/search"
    else:
        # Generic search
        search_query = citation_text.replace(' ', '+')
        url = f"https://scholar.google.com/scholar?q={search_query}"
    
    return f'<a href="{url}" target="_blank" class="citation-link" title="Look up {citation_text}">{html.escape(citation_text)}</a>'


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage and display.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed_file"
    
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s\-_\.]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + ('.' + ext if ext else '')
    
    return filename or "unnamed_file"





# Validation Functions

def validate_case_name(case_name: str) -> ValidationResult:
    """
    Validate legal case name according to legal naming conventions.
    
    Args:
        case_name: Case name to validate
        
    Returns:
        ValidationResult with validation status and suggestions
    """
    if not case_name or not case_name.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Case name cannot be empty",
            suggestions=["Enter a descriptive case name", "Use format: Type-Year-Subject"]
        )
    
    case_name = case_name.strip()
    
    # Check length constraints
    if len(case_name) < 3:
        return ValidationResult(
            is_valid=False,
            error_message="Case name must be at least 3 characters long",
            suggestions=["Add more descriptive text", "Include case type or year"]
        )
    
    if len(case_name) > 200:
        return ValidationResult(
            is_valid=False,
            error_message="Case name must be less than 200 characters",
            suggestions=["Shorten the case name", "Use abbreviations for common terms"]
        )
    
    # Check for valid characters
    if not re.match(LEGAL_CASE_NAME_PATTERN, case_name):
        return ValidationResult(
            is_valid=False,
            error_message="Case name contains invalid characters",
            suggestions=[
                "Use only letters, numbers, spaces, and common punctuation",
                "Remove special characters like @ # $ %"
            ]
        )
    
    # Check for legal naming best practices
    suggestions = []
    if not any(char.isdigit() for char in case_name):
        suggestions.append("Consider including a year for better organization")
    
    if len(case_name.split()) < 2:
        suggestions.append("Consider using multiple words for clarity")
    
    return ValidationResult(is_valid=True, suggestions=suggestions)


def validate_search_query(query: str) -> ValidationResult:
    """
    Validate search query for legal document search.
    
    Args:
        query: Search query to validate
        
    Returns:
        ValidationResult with validation status and suggestions
    """
    if not query or not query.strip():
        return ValidationResult(
            is_valid=False,
            error_message="Search query cannot be empty",
            suggestions=["Enter keywords or phrases", "Try legal terms or document names"]
        )
    
    query = query.strip()
    
    # Check length constraints
    if len(query) < 2:
        return ValidationResult(
            is_valid=False,
            error_message="Search query must be at least 2 characters long",
            suggestions=["Add more search terms", "Use specific legal terminology"]
        )
    
    if len(query) > 500:
        return ValidationResult(
            is_valid=False,
            error_message="Search query is too long (max 500 characters)",
            suggestions=["Shorten your query", "Focus on key terms"]
        )
    
    # Check for potential query improvements
    suggestions = []
    
    # Suggest adding quotes for exact phrases
    if ' ' in query and '"' not in query:
        suggestions.append('Use quotes for exact phrases: "intellectual property"')
    
    # Suggest boolean operators
    if len(query.split()) > 1 and not any(op in query.upper() for op in ['AND', 'OR', 'NOT']):
        suggestions.append("Use AND, OR, NOT for complex searches")
    
    return ValidationResult(is_valid=True, suggestions=suggestions)


def validate_file_upload(file_name: str, file_size: int, allowed_types: List[str] = None) -> ValidationResult:
    """
    Validate file upload for legal document processing.
    
    Args:
        file_name: Name of the uploaded file
        file_size: Size of the file in bytes
        allowed_types: List of allowed file extensions
        
    Returns:
        ValidationResult with validation status and suggestions
    """
    if allowed_types is None:
        allowed_types = ['.pdf', '.txt', '.doc', '.docx']
    
    if not file_name:
        return ValidationResult(
            is_valid=False,
            error_message="File name is required",
            suggestions=["Select a file to upload"]
        )
    
    # Check file extension
    file_ext = '.' + file_name.split('.')[-1].lower() if '.' in file_name else ''
    if file_ext not in allowed_types:
        return ValidationResult(
            is_valid=False,
            error_message=f"File type {file_ext} is not supported",
            suggestions=[
                f"Supported types: {', '.join(allowed_types)}",
                "Convert your document to PDF or TXT format"
            ]
        )
    
    # Check file size (50MB limit)
    max_size = 50 * BYTES_PER_MB
    if file_size > max_size:
        return ValidationResult(
            is_valid=False,
            error_message=f"File size {format_file_size(file_size)} exceeds 50MB limit",
            suggestions=[
                "Compress the document",
                "Split large documents into smaller files",
                "Contact administrator for large file support"
            ]
        )
    
    return ValidationResult(is_valid=True)


# Formatting Functions

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    elif size_bytes < BYTES_PER_MB:
        return f"{size_bytes / BYTES_PER_KB:.1f} KB"
    elif size_bytes < BYTES_PER_GB:
        return f"{size_bytes / BYTES_PER_MB:.1f} MB"
    else:
        return f"{size_bytes / BYTES_PER_GB:.1f} GB"


def format_timestamp(timestamp: datetime, relative: bool = True) -> str:
    """
    Format timestamp for UI display.
    
    Args:
        timestamp: Datetime object to format
        relative: Whether to show relative time (e.g., "2 hours ago")
        
    Returns:
        Formatted timestamp string
    """
    if not timestamp:
        return "Unknown"
    
    # Ensure timezone awareness
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    
    if relative:
        delta = now - timestamp
        
        if delta < timedelta(minutes=1):
            return "Just now"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta < timedelta(days=7):
            days = delta.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif delta < timedelta(days=30):
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            return timestamp.strftime("%b %d, %Y")
    else:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def format_relevance_score(score: float) -> str:
    """
    Format relevance score for UI display.
    
    Args:
        score: Relevance score (0.0 to 1.0)
        
    Returns:
        Formatted score string with visual indicator
    """
    percentage = int(score * 100)
    
    if percentage >= 90:
        return f"🟢 {percentage}%"
    elif percentage >= 70:
        return f"🟡 {percentage}%"
    elif percentage >= 50:
        return f"🟠 {percentage}%"
    else:
        return f"🔴 {percentage}%"


def format_legal_citation(citation: str) -> str:
    """
    Format legal citation for consistent display.
    
    Args:
        citation: Raw citation text
        
    Returns:
        Formatted citation string
    """
    # Basic citation cleanup
    citation = citation.strip()
    citation = re.sub(r'\s+', ' ', citation)  # Normalize whitespace
    
    # Add formatting for common citation patterns
    for pattern in LEGAL_CITATION_PATTERNS:
        if re.search(pattern, citation):
            return f"📖 {citation}"
    
    return citation


# HTML Rendering Functions

def render_search_result(result: FormattedSearchResult, highlight_terms: List[str] = None) -> str:
    """
    Render search result as HTML.
    
    Args:
        result: Formatted search result
        highlight_terms: Terms to highlight in the text
        
    Returns:
        HTML string for the search result
    """
    # Escape HTML in content
    title = html.escape(result.title)
    snippet = html.escape(result.snippet)
    
    # Apply highlighting
    if highlight_terms:
        for term in highlight_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            title = pattern.sub(f'<mark>{term}</mark>', title)
            snippet = pattern.sub(f'<mark>{term}</mark>', snippet)
    
    # Format relevance score
    score_display = format_relevance_score(result.relevance_score)
    
    # Page number display
    page_info = f"Page {result.page_number}" if result.page_number else "Document"
    
    return f"""
    <div class="search-result-item" data-document-id="{result.document_id}" data-chunk-id="{result.chunk_id}">
        <div class="search-result-header">
            <div class="search-result-title">{title}</div>
            <div class="search-result-score">{score_display}</div>
        </div>
        <div class="search-result-meta">
            <span class="document-type">{html.escape(result.document_type)}</span>
            <span class="page-info">{page_info}</span>
        </div>
        <div class="search-result-snippet">{snippet}</div>
        <div class="search-result-actions">
            <button class="btn-view-document" onclick="viewDocument('{result.document_id}', '{result.chunk_id}')">
                📄 View Document
            </button>
            <button class="btn-find-similar" onclick="findSimilar('{result.document_id}')">
                🔍 Find Similar
            </button>
        </div>
    </div>
    """


def render_progress_indicator(
    current: int,
    total: int,
    status: str = "Processing",
    show_percentage: bool = True
) -> str:
    """
    Render progress indicator as HTML.
    
    Args:
        current: Current progress value
        total: Total progress value
        status: Status message
        show_percentage: Whether to show percentage
        
    Returns:
        HTML string for progress indicator
    """
    if total == 0:
        percentage = 0
    else:
        percentage = min(100, max(0, (current / total) * 100))
    
    percentage_text = f" ({percentage:.0f}%)" if show_percentage else ""
    
    # Determine progress bar color
    if percentage == 100:
        bar_class = "progress-complete"
        status_icon = "✅"
    elif percentage > 0:
        bar_class = "progress-active"
        status_icon = "⚙️"
    else:
        bar_class = "progress-pending"
        status_icon = "⏳"
    
    return f"""
    <div class="progress-container">
        <div class="progress-header">
            <span class="progress-status">{status_icon} {html.escape(status)}</span>
            <span class="progress-text">{current}/{total}{percentage_text}</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar {bar_class}" style="width: {percentage}%"></div>
        </div>
    </div>
    """


def render_error_message(
    error_message: str,
    error_type: str = "Error",
    suggestions: List[str] = None
) -> str:
    """
    Render error message with suggestions.
    
    Args:
        error_message: Error message text
        error_type: Type of error (Error, Warning, Info)
        suggestions: List of suggestions to resolve the error
        
    Returns:
        HTML string for error display
    """
    # Determine icon and class based on error type
    type_config = {
        "Error": ("❌", "error"),
        "Warning": ("⚠️", "warning"),
        "Info": ("ℹ️", "info"),
        "Success": ("✅", "success")
    }
    
    icon, css_class = type_config.get(error_type, ("❌", "error"))
    
    suggestions_html = ""
    if suggestions:
        suggestions_items = "".join(f"<li>{html.escape(s)}</li>" for s in suggestions)
        suggestions_html = f"""
        <div class="error-suggestions">
            <strong>Suggestions:</strong>
            <ul>{suggestions_items}</ul>
        </div>
        """
    
    return f"""
    <div class="message-container {css_class}">
        <div class="message-header">
            <span class="message-icon">{icon}</span>
            <span class="message-type">{error_type}</span>
        </div>
        <div class="message-content">
            {html.escape(error_message)}
            {suggestions_html}
        </div>
    </div>
    """


def render_case_statistics(stats: Dict[str, Any]) -> str:
    """
    Render case statistics as HTML.
    
    Args:
        stats: Dictionary of statistics to display
        
    Returns:
        HTML string for statistics display
    """
    stats_items = []
    
    for key, value in stats.items():
        # Format the key as a readable label
        label = key.replace('_', ' ').title()
        
        # Format the value based on type
        if isinstance(value, int):
            formatted_value = f"{value:,}"
        elif isinstance(value, float):
            if 0 < value < 1:
                formatted_value = f"{value:.1%}"
            else:
                formatted_value = f"{value:.1f}"
        else:
            formatted_value = str(value)
        
        stats_items.append(f"""
        <div class="stat-item">
            <div class="stat-value">{formatted_value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """)
    
    return f"""
    <div class="statistics-container">
        {"".join(stats_items)}
    </div>
    """


# Visual Marker Functions

def get_available_visual_markers(used_markers: Set[Tuple[str, str]] = None) -> List[Dict[str, str]]:
    """
    Get list of available visual marker combinations.
    
    Args:
        used_markers: Set of (color, icon) tuples already in use
        
    Returns:
        List of available marker dictionaries
    """
    if used_markers is None:
        used_markers = set()
    
    available_markers = []
    
    for color, color_name, color_desc in VISUAL_MARKER_COLORS:
        for icon, icon_name, icon_desc in VISUAL_MARKER_ICONS:
            if (color, icon) not in used_markers:
                available_markers.append({
                    "color": color,
                    "color_name": color_name,
                    "color_description": color_desc,
                    "icon": icon,
                    "icon_name": icon_name,
                    "icon_description": icon_desc,
                    "display_name": f"{color_name} {icon_name}",
                    "full_description": f"{color_desc} - {icon_desc}"
                })
    
    return available_markers


def create_visual_marker_selector(used_markers: Set[Tuple[str, str]] = None) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Create visual marker selector options for Gradio components.
    
    Args:
        used_markers: Set of (color, icon) tuples already in use
        
    Returns:
        Tuple of (color_choices, icon_choices) for Gradio components
    """
    if used_markers is None:
        used_markers = set()
    
    # Color choices
    color_choices = []
    for color, color_name, color_desc in VISUAL_MARKER_COLORS:
        choice_label = f"{color_name} ({color_desc})"
        color_choices.append((choice_label, color))
    
    # Icon choices
    icon_choices = []
    for icon, icon_name, icon_desc in VISUAL_MARKER_ICONS:
        choice_label = f"{icon} {icon_name}"
        icon_choices.append((choice_label, icon))
    
    return color_choices, icon_choices


# Search Helper Functions

async def get_search_suggestions(query: str, case_id: Optional[str] = None) -> List[str]:
    """
    Generate search suggestions based on query and context.
    
    Args:
        query: Partial search query
        case_id: Optional case ID for context-specific suggestions
        
    Returns:
        List of search suggestions
    """
    suggestions = []
    query_lower = query.lower()
    
    # Legal term suggestions
    legal_terms = [
        "intellectual property", "patent application", "prior art",
        "contract terms", "licensing agreement", "merger agreement",
        "litigation discovery", "court filing", "regulatory compliance",
        "due diligence", "corporate governance", "employment law"
    ]
    
    for term in legal_terms:
        if query_lower in term.lower() and term not in suggestions:
            suggestions.append(term)
    
    # Document type suggestions
    if any(doc_type in query_lower for doc_type in ["patent", "contract", "brief"]):
        suggestions.extend([
            f"{query} analysis",
            f"{query} comparison",
            f"{query} summary"
        ])
    
    # Limit suggestions
    return suggestions[:5]


def preprocess_search_query(query: str) -> str:
    """
    Preprocess search query for better search results.
    
    Args:
        query: Raw search query
        
    Returns:
        Preprocessed search query
    """
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove common stop words for legal searches
    legal_stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = query.split()
    
    # Keep stop words if they're part of legal phrases
    if len(words) > 3:  # Only remove stop words from longer queries
        words = [word for word in words if word.lower() not in legal_stop_words]
    
    return ' '.join(words)


# Document Type Detection

def detect_document_category(file_name: str, content_preview: str = "") -> DocumentCategory:
    """
    Detect document category based on filename and content.
    
    Args:
        file_name: Name of the document file
        content_preview: Preview of document content
        
    Returns:
        Detected document category
    """
    file_name_lower = file_name.lower()
    content_lower = content_preview.lower()
    
    # Patent documents
    if any(term in file_name_lower for term in ['patent', 'application', 'uspto', 'prior_art']):
        return DocumentCategory.PATENT
    
    # Contract documents
    if any(term in file_name_lower for term in ['contract', 'agreement', 'license', 'nda']):
        return DocumentCategory.CONTRACT
    
    # Litigation documents
    if any(term in file_name_lower for term in ['complaint', 'motion', 'brief', 'filing']):
        return DocumentCategory.LITIGATION
    
    # Corporate documents
    if any(term in file_name_lower for term in ['merger', 'acquisition', 'corporate', 'bylaws']):
        return DocumentCategory.CORPORATE
    
    # Regulatory documents
    if any(term in file_name_lower for term in ['regulation', 'compliance', 'sec', 'filing']):
        return DocumentCategory.REGULATORY
    
    # Discovery documents
    if any(term in file_name_lower for term in ['discovery', 'deposition', 'interrogatory']):
        return DocumentCategory.DISCOVERY
    
    # Content-based detection
    if content_preview:
        # Look for patent-specific terms
        if any(term in content_lower for term in ['claim', 'specification', 'embodiment', 'invention']):
            return DocumentCategory.PATENT
        
        # Look for contract terms
        if any(term in content_lower for term in ['whereas', 'party', 'consideration', 'terminate']):
            return DocumentCategory.CONTRACT
    
    return DocumentCategory.OTHER


# Utility Functions

def sanitize_html(text: str) -> str:
    """
    Sanitize HTML content for safe display.
    
    Args:
        text: Text that may contain HTML
        
    Returns:
        Sanitized text safe for HTML display
    """
    return html.escape(text)


def truncate_text(text: str, max_length: int = 150, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of truncated text
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # Only break at word if not too short
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for search suggestions.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'are', 'was', 'were', 'been', 'have', 'has'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency and return most common
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, count in word_counts.most_common(max_keywords)]


def generate_case_id(case_name: str, user_id: str) -> str:
    """
    Generate a unique case ID based on case name and user.
    
    Args:
        case_name: Name of the case
        user_id: ID of the user creating the case
        
    Returns:
        Generated case ID
    """
    # Create a slug from the case name
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', case_name)
    slug = re.sub(r'\s+', '-', slug.strip()).lower()
    
    # Limit slug length
    if len(slug) > 50:
        slug = slug[:50].rstrip('-')
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d")
    
    return f"case-{slug}-{timestamp}"


# Export all utility functions
__all__ = [
    # Validation functions
    "validate_case_name",
    "validate_search_query", 
    "validate_file_upload",
    "ValidationResult",
    
    # Formatting functions
    "format_file_size",
    "format_timestamp",
    "format_duration",
    "format_relevance_score",
    "format_legal_citation",
    
    # HTML rendering functions
    "render_search_result",
    "render_progress_indicator",
    "render_error_message",
    "render_case_statistics",
    
    # Visual marker functions
    "get_available_visual_markers",
    "create_visual_marker_selector",
    
    # Search helper functions
    "get_search_suggestions",
    "preprocess_search_query",
    
    # Document type detection
    "detect_document_category",
    "DocumentCategory",
    
    # Utility functions
    "sanitize_html",
    "truncate_text",
    "extract_keywords",
    "generate_case_id",
    
    # Data classes
    "FormattedSearchResult",
    "UITheme",
    
    # Constants
    "VISUAL_MARKER_COLORS",
    "VISUAL_MARKER_ICONS",
    "LEGAL_FILE_EXTENSIONS"
]