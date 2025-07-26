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
import json
import logging
import math
import mimetypes
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set, NamedTuple
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