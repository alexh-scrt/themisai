"""
Domain models for search results in the Patexia Legal AI Chatbot.

This module defines the core domain entities for search results, rankings,
and search metadata. It provides immutable, business-logic-focused models
that encapsulate search result data and behavior independent of database
or API representation.

Key Features:
- Immutable search result entities with legal document context
- Search ranking and relevance scoring algorithms
- Citation-ready result formatting and highlighting
- Search performance metrics and analytics
- Result clustering and similarity grouping
- Legal document structure preservation in results
- Multi-faceted search result metadata

Search Result Architecture:
- SearchResult: Individual chunk result with ranking and metadata
- SearchResultCollection: Aggregated results with analytics
- SearchRanking: Relevance scoring and ranking algorithms  
- SearchHighlight: Content highlighting and context extraction
- SearchCitation: Legal citation formatting and source attribution
- SearchFacets: Faceted search result breakdown and filtering

Legal Document Context:
- Preserves legal document hierarchy in results
- Maintains citation context and source attribution
- Supports section-aware result presentation
- Enables legal professional workflow integration
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import re
import math
from uuid import uuid4

from ..domain.document import DocumentChunk, DocumentType
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SearchType(str, Enum):
    """Types of search operations supported."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"
    CITATION = "citation"
    SIMILARITY = "similarity"
    FUZZY = "fuzzy"


class SearchScope(str, Enum):
    """Scope of search operations."""
    CASE = "case"
    DOCUMENT = "document"
    GLOBAL = "global"
    CHUNK = "chunk"
    RECENT = "recent"


class RankingMethod(str, Enum):
    """Ranking methods for search results."""
    RELEVANCE = "relevance"
    RECIPROCAL_RANK_FUSION = "rrf"
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"
    LEARNING_TO_RANK = "ltr"


class HighlightStyle(str, Enum):
    """Highlighting styles for search results."""
    HTML_MARK = "html_mark"
    MARKDOWN_BOLD = "markdown_bold"
    PLAIN_BRACKETS = "plain_brackets"
    ANSI_COLOR = "ansi_color"


@dataclass(frozen=True)
class SearchHighlight:
    """
    Search result highlighting with context extraction.
    
    Provides highlighted content with configurable styles and
    context extraction for legal document citation purposes.
    """
    
    original_text: str
    highlighted_text: str
    highlight_style: HighlightStyle
    match_positions: List[Tuple[int, int]]
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    context_chars: int = 200
    
    def __post_init__(self):
        """Validate highlight data consistency."""
        if not self.original_text:
            raise ValueError("Original text cannot be empty")
        
        if not self.highlighted_text:
            object.__setattr__(self, 'highlighted_text', self.original_text)
        
        if not self.match_positions:
            object.__setattr__(self, 'match_positions', [])
    
    @property
    def match_count(self) -> int:
        """Get number of highlight matches."""
        return len(self.match_positions)
    
    @property
    def coverage_percentage(self) -> float:
        """Get percentage of text covered by highlights."""
        if not self.match_positions or not self.original_text:
            return 0.0
        
        total_highlighted = sum(end - start for start, end in self.match_positions)
        return (total_highlighted / len(self.original_text)) * 100
    
    def get_citation_context(self) -> str:
        """Get formatted citation context."""
        if self.context_before or self.context_after:
            before = f"...{self.context_before}" if self.context_before else ""
            after = f"{self.context_after}..." if self.context_after else ""
            return f"{before}{self.original_text}{after}"
        return self.original_text
    
    @classmethod
    def create_highlight(
        cls,
        text: str,
        query_terms: List[str],
        style: HighlightStyle = HighlightStyle.HTML_MARK,
        context_chars: int = 200,
        case_sensitive: bool = False
    ) -> "SearchHighlight":
        """
        Create highlighted text from query terms.
        
        Args:
            text: Original text to highlight
            query_terms: Terms to highlight in text
            style: Highlighting style to use
            context_chars: Characters of context to extract
            case_sensitive: Whether matching is case sensitive
            
        Returns:
            SearchHighlight instance with highlighted content
        """
        if not text or not query_terms:
            return cls(
                original_text=text,
                highlighted_text=text,
                highlight_style=style,
                match_positions=[],
                context_chars=context_chars
            )
        
        highlighted = text
        match_positions = []
        
        # Create highlight pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for term in query_terms:
            # Escape special regex characters
            escaped_term = re.escape(term)
            pattern = re.compile(r'\b' + escaped_term + r'\b', flags)
            
            # Find all matches
            for match in pattern.finditer(text):
                match_positions.append((match.start(), match.end()))
        
        # Sort and merge overlapping positions
        match_positions.sort()
        merged_positions = []
        for start, end in match_positions:
            if merged_positions and start <= merged_positions[-1][1]:
                # Merge overlapping highlights
                merged_positions[-1] = (merged_positions[-1][0], max(end, merged_positions[-1][1]))
            else:
                merged_positions.append((start, end))
        
        # Apply highlighting based on style
        if merged_positions:
            highlighted = cls._apply_highlighting(text, merged_positions, style)
        
        return cls(
            original_text=text,
            highlighted_text=highlighted,
            highlight_style=style,
            match_positions=merged_positions,
            context_chars=context_chars
        )
    
    @staticmethod
    def _apply_highlighting(text: str, positions: List[Tuple[int, int]], style: HighlightStyle) -> str:
        """Apply highlighting to text based on style."""
        if not positions:
            return text
        
        # Work backwards to avoid position shifts
        highlighted = text
        for start, end in reversed(positions):
            match_text = text[start:end]
            
            if style == HighlightStyle.HTML_MARK:
                replacement = f"<mark>{match_text}</mark>"
            elif style == HighlightStyle.MARKDOWN_BOLD:
                replacement = f"**{match_text}**"
            elif style == HighlightStyle.PLAIN_BRACKETS:
                replacement = f"[{match_text}]"
            elif style == HighlightStyle.ANSI_COLOR:
                replacement = f"\033[93m{match_text}\033[0m"  # Yellow highlight
            else:
                replacement = match_text
            
            highlighted = highlighted[:start] + replacement + highlighted[end:]
        
        return highlighted


@dataclass(frozen=True)
class SearchScore:
    """
    Comprehensive search scoring with multiple ranking signals.
    
    Combines semantic similarity, keyword matching, and document
    quality signals into unified relevance scores.
    """
    
    relevance_score: float
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    document_quality_score: Optional[float] = None
    recency_score: Optional[float] = None
    citation_boost: float = 0.0
    section_boost: float = 0.0
    
    def __post_init__(self):
        """Validate score ranges."""
        scores = [
            self.relevance_score, self.semantic_score, self.keyword_score,
            self.document_quality_score, self.recency_score
        ]
        
        for score in scores:
            if score is not None and not (0.0 <= score <= 1.0):
                raise ValueError(f"Scores must be between 0.0 and 1.0, got {score}")
    
    @property
    def composite_score(self) -> float:
        """Calculate composite score from all signals."""
        base_score = self.relevance_score
        
        # Apply boosts
        boosted_score = base_score + self.citation_boost + self.section_boost
        
        # Ensure final score stays in valid range
        return min(1.0, max(0.0, boosted_score))
    
    @property
    def has_semantic_component(self) -> bool:
        """Check if score includes semantic similarity."""
        return self.semantic_score is not None
    
    @property
    def has_keyword_component(self) -> bool:
        """Check if score includes keyword matching."""
        return self.keyword_score is not None
    
    @classmethod
    def create_semantic_score(cls, similarity: float, quality_boost: float = 0.0) -> "SearchScore":
        """Create score from semantic similarity."""
        return cls(
            relevance_score=similarity,
            semantic_score=similarity,
            document_quality_score=quality_boost
        )
    
    @classmethod
    def create_keyword_score(cls, bm25_score: float, max_possible: float = 10.0) -> "SearchScore":
        """Create score from BM25 keyword matching."""
        normalized = min(1.0, bm25_score / max_possible) if max_possible > 0 else 0.0
        return cls(
            relevance_score=normalized,
            keyword_score=normalized
        )
    
    @classmethod
    def create_hybrid_score(
        cls,
        semantic_score: float,
        keyword_score: float,
        alpha: float = 0.6,
        quality_boost: float = 0.0
    ) -> "SearchScore":
        """
        Create hybrid score combining semantic and keyword signals.
        
        Args:
            semantic_score: Semantic similarity score (0-1)
            keyword_score: Keyword matching score (0-1)  
            alpha: Weight for semantic score (1-alpha for keyword)
            quality_boost: Document quality boost
            
        Returns:
            Hybrid SearchScore instance
        """
        hybrid_relevance = alpha * semantic_score + (1 - alpha) * keyword_score
        
        return cls(
            relevance_score=hybrid_relevance,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            document_quality_score=quality_boost
        )


@dataclass(frozen=True)
class SearchResult:
    """
    Individual search result representing a document chunk match.
    
    Immutable search result entity containing the matched chunk,
    relevance scoring, highlighting, and legal document context
    for citation and presentation purposes.
    """
    
    result_id: str
    chunk: DocumentChunk
    score: SearchScore
    highlight: SearchHighlight
    search_query: str
    search_type: SearchType
    
    # Document context
    document_name: str
    document_type: DocumentType
    case_id: str
    
    # Ranking metadata
    rank: Optional[int] = None
    total_results: Optional[int] = None
    
    # Search metadata
    search_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    search_duration_ms: Optional[float] = None
    
    def __post_init__(self):
        """Validate search result consistency."""
        if not self.result_id:
            object.__setattr__(self, 'result_id', str(uuid4()))
        
        if not self.search_query:
            raise ValueError("Search query cannot be empty")
        
        if self.rank is not None and self.rank < 1:
            raise ValueError("Rank must be positive")
    
    @property
    def relevance_score(self) -> float:
        """Get primary relevance score."""
        return self.score.relevance_score
    
    @property
    def composite_score(self) -> float:
        """Get composite score with all boosts."""
        return self.score.composite_score
    
    @property
    def chunk_content(self) -> str:
        """Get chunk text content."""
        return self.chunk.content
    
    @property
    def highlighted_content(self) -> str:
        """Get highlighted chunk content."""
        return self.highlight.highlighted_text
    
    @property
    def document_id(self) -> str:
        """Get parent document ID."""
        return self.chunk.document_id
    
    @property
    def chunk_id(self) -> str:
        """Get chunk identifier."""
        return self.chunk.chunk_id
    
    @property
    def section_title(self) -> Optional[str]:
        """Get section title if available."""
        return self.chunk.section_title
    
    @property
    def page_number(self) -> Optional[int]:
        """Get page number if available."""
        return self.chunk.page_number
    
    @property
    def legal_citations(self) -> List[str]:
        """Get legal citations in chunk."""
        return self.chunk.legal_citations
    
    @property
    def has_citations(self) -> bool:
        """Check if result contains legal citations."""
        return len(self.legal_citations) > 0
    
    def get_citation_text(self, max_length: int = 200) -> str:
        """
        Get citation-ready text excerpt.
        
        Args:
            max_length: Maximum length of citation text
            
        Returns:
            Formatted citation text with context
        """
        citation_context = self.highlight.get_citation_context()
        
        if len(citation_context) <= max_length:
            return citation_context
        
        # Truncate at sentence boundary if possible
        truncated = citation_context[:max_length]
        last_sentence = truncated.rfind('.')
        
        if last_sentence > max_length // 2:
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    def get_source_attribution(self) -> Dict[str, Any]:
        """Get source attribution for citation."""
        return {
            "document_name": self.document_name,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "case_id": self.case_id,
            "legal_citations": self.legal_citations
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result_id": self.result_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type.value,
            "case_id": self.case_id,
            "content": self.chunk_content,
            "highlighted_content": self.highlighted_content,
            "relevance_score": self.relevance_score,
            "composite_score": self.composite_score,
            "semantic_score": self.score.semantic_score,
            "keyword_score": self.score.keyword_score,
            "chunk_index": self.chunk.chunk_index,
            "start_char": self.chunk.start_char,
            "end_char": self.chunk.end_char,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "legal_citations": self.legal_citations,
            "search_query": self.search_query,
            "search_type": self.search_type.value,
            "rank": self.rank,
            "total_results": self.total_results,
            "search_timestamp": self.search_timestamp.isoformat(),
            "search_duration_ms": self.search_duration_ms
        }


@dataclass
class SearchFacets:
    """
    Faceted search result breakdown for filtering and navigation.
    
    Provides categorical breakdowns of search results to support
    advanced filtering and faceted search interfaces.
    """
    
    document_types: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    case_ids: Dict[str, int] = field(default_factory=dict)
    sections: Dict[str, int] = field(default_factory=dict)
    date_ranges: Dict[str, int] = field(default_factory=dict)
    has_citations: Dict[str, int] = field(default_factory=dict)
    score_ranges: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: SearchResult) -> None:
        """Add a search result to facet counts."""
        # Document type facets
        doc_type = result.document_type.value
        self.document_types[doc_type] = self.document_types.get(doc_type, 0) + 1
        
        # File type facets (could derive from document_type)
        file_ext = doc_type.lower()
        self.file_types[file_ext] = self.file_types.get(file_ext, 0) + 1
        
        # Case facets
        case_id = result.case_id
        self.case_ids[case_id] = self.case_ids.get(case_id, 0) + 1
        
        # Section facets
        section = result.section_title or "Unknown"
        self.sections[section] = self.sections.get(section, 0) + 1
        
        # Citation facets
        citation_key = "Has Citations" if result.has_citations else "No Citations"
        self.has_citations[citation_key] = self.has_citations.get(citation_key, 0) + 1
        
        # Score range facets
        score = result.relevance_score
        if score >= 0.9:
            score_range = "Excellent (0.9-1.0)"
        elif score >= 0.7:
            score_range = "Good (0.7-0.9)"
        elif score >= 0.5:
            score_range = "Fair (0.5-0.7)"
        else:
            score_range = "Poor (0.0-0.5)"
        
        self.score_ranges[score_range] = self.score_ranges.get(score_range, 0) + 1
    
    def get_top_facets(self, facet_type: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top facet values for a given type."""
        facet_data = getattr(self, facet_type, {})
        return sorted(facet_data.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Convert to dictionary for serialization."""
        return {
            "document_types": self.document_types,
            "file_types": self.file_types,
            "case_ids": self.case_ids,
            "sections": self.sections,
            "date_ranges": self.date_ranges,
            "has_citations": self.has_citations,
            "score_ranges": self.score_ranges
        }


@dataclass
class SearchResultCollection:
    """
    Collection of search results with analytics and metadata.
    
    Aggregates individual search results with collection-level
    analytics, performance metrics, and search quality indicators
    for comprehensive search result management.
    """
    
    search_id: str
    query: str
    search_type: SearchType
    search_scope: SearchScope
    results: List[SearchResult] = field(default_factory=list)
    
    # Search metadata
    case_id: Optional[str] = None
    user_id: Optional[str] = None
    total_found: int = 0
    offset: int = 0
    limit: int = 50
    
    # Performance metrics
    search_duration_ms: float = 0.0
    database_time_ms: float = 0.0
    ranking_time_ms: float = 0.0
    highlighting_time_ms: float = 0.0
    
    # Quality metrics
    max_score: float = 0.0
    min_score: float = 1.0
    avg_score: float = 0.0
    score_variance: float = 0.0
    
    # Search coverage
    documents_searched: int = 0
    chunks_searched: int = 0
    documents_with_matches: int = 0
    
    # Faceted results
    facets: SearchFacets = field(default_factory=SearchFacets)
    
    # Search suggestions
    suggested_queries: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Initialize computed metrics."""
        if not self.search_id:
            object.__setattr__(self, 'search_id', str(uuid4()))
        
        self._compute_metrics()
    
    def add_result(self, result: SearchResult) -> None:
        """Add a search result to the collection."""
        # Set rank on result
        rank = len(self.results) + 1 + self.offset
        object.__setattr__(result, 'rank', rank)
        object.__setattr__(result, 'total_results', self.total_found)
        
        self.results.append(result)
        self.facets.add_result(result)
        self._compute_metrics()
    
    def add_results(self, results: List[SearchResult]) -> None:
        """Add multiple results to the collection."""
        for result in results:
            self.add_result(result)
    
    def _compute_metrics(self) -> None:
        """Compute collection-level metrics."""
        if not self.results:
            return
        
        # Score statistics
        scores = [r.relevance_score for r in self.results]
        object.__setattr__(self, 'max_score', max(scores))
        object.__setattr__(self, 'min_score', min(scores))
        object.__setattr__(self, 'avg_score', sum(scores) / len(scores))
        
        # Score variance
        mean_score = self.avg_score
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        object.__setattr__(self, 'score_variance', variance)
        
        # Document coverage
        unique_docs = set(r.document_id for r in self.results)
        object.__setattr__(self, 'documents_with_matches', len(unique_docs))
    
    @property
    def result_count(self) -> int:
        """Get number of results in collection."""
        return len(self.results)
    
    @property
    def has_more(self) -> bool:
        """Check if there are more results available."""
        return (self.offset + self.result_count) < self.total_found
    
    @property
    def is_empty(self) -> bool:
        """Check if collection has no results."""
        return self.result_count == 0
    
    @property
    def quality_score(self) -> float:
        """Get overall search quality score."""
        if self.is_empty:
            return 0.0
        
        # Quality based on score distribution and coverage
        avg_quality = self.avg_score
        coverage_quality = min(1.0, self.documents_with_matches / max(1, self.documents_searched))
        variance_penalty = max(0.0, 1.0 - self.score_variance)
        
        return (avg_quality + coverage_quality + variance_penalty) / 3
    
    def get_top_results(self, limit: int = 10) -> List[SearchResult]:
        """Get top N results by relevance score."""
        return sorted(self.results, key=lambda r: r.composite_score, reverse=True)[:limit]
    
    def filter_by_score(self, min_score: float) -> List[SearchResult]:
        """Filter results by minimum relevance score."""
        return [r for r in self.results if r.relevance_score >= min_score]
    
    def filter_by_document_type(self, doc_type: DocumentType) -> List[SearchResult]:
        """Filter results by document type."""
        return [r for r in self.results if r.document_type == doc_type]
    
    def filter_with_citations(self) -> List[SearchResult]:
        """Filter results that contain legal citations."""
        return [r for r in self.results if r.has_citations]
    
    def group_by_document(self) -> Dict[str, List[SearchResult]]:
        """Group results by document ID."""
        groups = {}
        for result in self.results:
            doc_id = result.document_id
            if doc_id not in groups:
                groups[doc_id] = []
            groups[doc_id].append(result)
        return groups
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "search_id": self.search_id,
            "query": self.query,
            "search_type": self.search_type.value,
            "search_scope": self.search_scope.value,
            "case_id": self.case_id,
            "user_id": self.user_id,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "result_count": self.result_count,
            "offset": self.offset,
            "limit": self.limit,
            "has_more": self.has_more,
            "search_duration_ms": self.search_duration_ms,
            "database_time_ms": self.database_time_ms,
            "ranking_time_ms": self.ranking_time_ms,
            "highlighting_time_ms": self.highlighting_time_ms,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "avg_score": self.avg_score,
            "score_variance": self.score_variance,
            "quality_score": self.quality_score,
            "documents_searched": self.documents_searched,
            "chunks_searched": self.chunks_searched,
            "documents_with_matches": self.documents_with_matches,
            "facets": self.facets.to_dict(),
            "suggested_queries": self.suggested_queries,
            "created_at": self.created_at.isoformat()
        }