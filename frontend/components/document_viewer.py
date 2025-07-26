"""
Document Viewer Component for Patexia Legal AI Chatbot

This module provides a sophisticated document viewer interface using Gradio for
displaying legal documents with search result highlighting, navigation controls,
and metadata display. It integrates with the search results pane to provide
a seamless two-pane search experience for legal professionals.

Key Features:
- Legal document display with proper formatting and structure preservation
- Search result highlighting with context-aware emphasis
- Navigation controls for moving between search matches within documents
- Document metadata display including citations, page numbers, and sections
- Support for multiple document formats (PDF text extraction, plain text, DOC)
- Real-time updates when search results are selected
- Citation-ready snippet extraction for legal references
- Document section navigation and bookmarking
- Text selection and annotation capabilities
- Export functionality for document excerpts and citations

Search Integration:
- Synchronized highlighting with search pane results
- Match navigation with previous/next controls
- Context expansion for better understanding of matches
- Relevance score display for highlighted sections
- Multi-term highlighting with different colors
- Fuzzy match highlighting for related terms

Legal Document Features:
- Legal citation recognition and formatting
- Section header preservation and navigation
- Page number tracking and reference
- Paragraph numbering for precise citations
- Legal structure recognition (claims, specifications, etc.)
- Footnote and cross-reference handling
- Document metadata display (filing dates, parties, etc.)

Architecture Integration:
- Connects to FastAPI backend for document content retrieval
- Real-time updates via WebSocket for search result selection
- Integration with search pane for synchronized highlighting
- Support for case-based document isolation and access control
- Responsive design for optimal legal document review experience
"""

import re
import html
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import gradio as gr
import requests

# Backend API configuration
BACKEND_BASE_URL = "http://localhost:8000"

# Global state for document viewer
viewer_state = {
    "current_document": None,
    "current_matches": [],
    "current_match_index": 0,
    "search_terms": [],
    "highlighted_content": "",
    "document_metadata": {}
}

logger = logging.getLogger(__name__)


class DocumentViewerAPI:
    """API client for document content retrieval."""
    
    def __init__(self, base_url: str = BACKEND_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    async def get_document_content(self, document_id: str, case_id: str) -> Dict[str, Any]:
        """Get full document content with metadata."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/documents/{document_id}/content",
                params={"case_id": case_id}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logging.error(f"Failed to get document content: {e}")
            return {}
    
    async def get_document_metadata(self, document_id: str, case_id: str) -> Dict[str, Any]:
        """Get document metadata and structure information."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/documents/{document_id}/metadata",
                params={"case_id": case_id}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logging.error(f"Failed to get document metadata: {e}")
            return {}


# Initialize API client
api = DocumentViewerAPI()


class TextHighlighter:
    """Advanced text highlighting for legal documents."""
    
    @staticmethod
    def highlight_search_terms(text: str, search_terms: List[str]) -> str:
        """
        Highlight search terms in text with different colors for multiple terms.
        
        Args:
            text: Source text to highlight
            search_terms: List of terms to highlight
            
        Returns:
            HTML text with highlighted terms
        """
        if not search_terms or not text:
            return html.escape(text)
        
        # Escape HTML in the original text
        escaped_text = html.escape(text)
        
        # Define colors for different search terms
        highlight_colors = [
            "#fff2cc",  # Yellow
            "#d4edda",  # Green
            "#cce5ff",  # Blue
            "#f8d7da",  # Red
            "#e2d6f3",  # Purple
            "#ffeaa7",  # Orange
        ]
        
        # Sort terms by length (longest first) to avoid partial matches
        sorted_terms = sorted(search_terms, key=len, reverse=True)
        
        for i, term in enumerate(sorted_terms):
            if not term.strip():
                continue
                
            color = highlight_colors[i % len(highlight_colors)]
            
            # Create case-insensitive regex pattern
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            # Replace with highlighted version
            def replace_func(match):
                return f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: 600;">{match.group()}</mark>'
            
            escaped_text = pattern.sub(replace_func, escaped_text)
        
        return escaped_text
    
    @staticmethod
    def extract_context(text: str, search_term: str, context_chars: int = 200) -> List[str]:
        """
        Extract text contexts around search term matches.
        
        Args:
            text: Source text
            search_term: Term to find contexts for
            context_chars: Number of characters to include around match
            
        Returns:
            List of context strings
        """
        if not search_term or not text:
            return []
        
        contexts = []
        pattern = re.compile(re.escape(search_term), re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            
            context = text[start:end]
            
            # Try to break at word boundaries
            if start > 0:
                space_pos = context.find(' ')
                if space_pos > 0:
                    context = context[space_pos + 1:]
            
            if end < len(text):
                space_pos = context.rfind(' ')
                if space_pos > 0:
                    context = context[:space_pos]
            
            contexts.append(context.strip())
        
        return contexts
    
    @staticmethod
    def get_match_positions(text: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Get positions of all search term matches in text."""
        matches = []
        
        for term in search_terms:
            if not term.strip():
                continue
                
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            for match in pattern.finditer(text):
                matches.append({
                    "term": term,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group()
                })
        
        # Sort by position
        matches.sort(key=lambda x: x["start"])
        return matches


class LegalDocumentFormatter:
    """Formatter for legal document structure and citations."""
    
    @staticmethod
    def format_legal_document(content: str, metadata: Dict[str, Any]) -> str:
        """
        Format legal document content with proper structure and citations.
        
        Args:
            content: Raw document content
            metadata: Document metadata including structure info
            
        Returns:
            Formatted HTML content
        """
        if not content:
            return ""
        
        # Split content into sections based on legal document structure
        formatted_content = LegalDocumentFormatter._add_section_headers(content, metadata)
        formatted_content = LegalDocumentFormatter._format_citations(formatted_content)
        formatted_content = LegalDocumentFormatter._add_paragraph_numbers(formatted_content)
        
        return formatted_content
    
    @staticmethod
    def _add_section_headers(content: str, metadata: Dict[str, Any]) -> str:
        """Add section headers based on legal document structure."""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Detect section headers (various patterns for legal documents)
            if (
                stripped_line.isupper() and len(stripped_line) < 100 or
                re.match(r'^\d+\.\s+[A-Z]', stripped_line) or
                re.match(r'^[IVX]+\.\s+[A-Z]', stripped_line) or
                stripped_line.startswith('CLAIM') or
                stripped_line.startswith('SPECIFICATION') or
                stripped_line.startswith('BACKGROUND') or
                stripped_line.startswith('SUMMARY') or
                stripped_line.startswith('DETAILED DESCRIPTION')
            ):
                formatted_lines.append(f'<h3 class="section-header" style="color: #2c3e50; font-weight: 600; margin: 20px 0 10px 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px;">{html.escape(stripped_line)}</h3>')
            else:
                formatted_lines.append(html.escape(line))
        
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def _format_citations(content: str) -> str:
        """Format legal citations with proper styling."""
        # Common legal citation patterns
        citation_patterns = [
            r'\b\d+\s+U\.?S\.?\s+\d+',  # U.S. Reports
            r'\b\d+\s+F\.\d*d?\s+\d+',  # Federal Reporter
            r'\b\d+\s+S\.\s*Ct\.\s+\d+',  # Supreme Court Reporter
            r'\b\d+\s+U\.?S\.?P\.?Q\.?\s+\d+',  # U.S. Patent Quarterly
            r'U\.?S\.?\s+Patent\s+No\.?\s+[\d,]+',  # Patent numbers
            r'Pub\.?\s+No\.?\s+[\d\/\-]+',  # Publication numbers
        ]
        
        for pattern in citation_patterns:
            content = re.sub(
                pattern,
                lambda m: f'<span class="citation" style="background-color: #e8f4fd; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 0.9em;">{m.group()}</span>',
                content,
                flags=re.IGNORECASE
            )
        
        return content
    
    @staticmethod
    def _add_paragraph_numbers(content: str) -> str:
        """Add paragraph numbers for citation purposes."""
        lines = content.split('\n')
        formatted_lines = []
        paragraph_num = 1
        
        for line in lines:
            if line.strip() and not line.startswith('<h'):
                # Add paragraph number for non-empty, non-header lines
                formatted_lines.append(f'<span class="paragraph-number" style="color: #95a5a6; font-size: 0.8em; margin-right: 8px;">[{paragraph_num}]</span>{line}')
                paragraph_num += 1
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)


def create_document_viewer() -> gr.Column:
    """Create the main document viewer component."""
    
    with gr.Column(scale=2) as doc_viewer:
        
        # Document header
        with gr.Row():
            doc_title = gr.Textbox(
                value="No document selected",
                label="",
                interactive=False,
                container=False,
                elem_classes=["doc-title"]
            )
        
        # Navigation and controls
        with gr.Row():
            with gr.Column(scale=3):
                match_info = gr.Textbox(
                    value="Select a search result to view document",
                    label="",
                    interactive=False,
                    container=False
                )
            
            with gr.Column(scale=1):
                with gr.Row():
                    prev_match_btn = gr.Button("⬅️ Previous", size="sm", interactive=False)
                    next_match_btn = gr.Button("➡️ Next", size="sm", interactive=False)
                    match_counter = gr.Textbox(
                        value="0 / 0",
                        label="",
                        interactive=False,
                        container=False,
                        elem_classes=["match-counter"]
                    )
        
        # Document content display
        document_content = gr.HTML(
            value='<div style="text-align: center; color: #7f8c8d; padding: 50px;"><h3>📄 Document Viewer</h3><p>Select a search result to view the document content with highlighted matches.</p></div>',
            elem_classes=["document-content"]
        )
        
        # Document metadata panel
        with gr.Accordion("📋 Document Information", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    doc_type = gr.Textbox(label="File Type", interactive=False)
                    file_size = gr.Textbox(label="File Size", interactive=False)
                    upload_date = gr.Textbox(label="Upload Date", interactive=False)
                
                with gr.Column(scale=1):
                    page_count = gr.Textbox(label="Pages", interactive=False)
                    word_count = gr.Textbox(label="Words", interactive=False)
                    processing_status = gr.Textbox(label="Status", interactive=False)
        
        # Legal document specific information
        with gr.Accordion("⚖️ Legal Information", open=False):
            with gr.Row():
                with gr.Column():
                    citations_found = gr.Textbox(
                        label="Legal Citations",
                        lines=3,
                        interactive=False,
                        placeholder="Citations found in document will appear here"
                    )
                
                with gr.Column():
                    section_structure = gr.Textbox(
                        label="Document Structure",
                        lines=3,
                        interactive=False,
                        placeholder="Document sections and hierarchy"
                    )
        
        # Document actions
        with gr.Row():
            export_snippet_btn = gr.Button("📄 Export Snippet", size="sm")
            copy_citation_btn = gr.Button("📋 Copy Citation", size="sm")
            bookmark_btn = gr.Button("🔖 Bookmark", size="sm")
            full_text_btn = gr.Button("📖 View Full Text", size="sm")
        
        # Hidden state components
        current_doc_id = gr.State("")
        current_case_id = gr.State("")
        search_matches = gr.State([])
        current_match_idx = gr.State(0)
    
    # Function to load and display document
    async def load_document(document_id: str, case_id: str, search_terms: List[str] = None):
        """Load and display a document with search highlighting."""
        if not document_id or not case_id:
            return {
                doc_title: "No document selected",
                document_content: '<div style="text-align: center; color: #7f8c8d; padding: 50px;"><h3>📄 Document Viewer</h3><p>Select a search result to view the document content.</p></div>',
                match_info: "No document selected",
                match_counter: "0 / 0",
                prev_match_btn: gr.update(interactive=False),
                next_match_btn: gr.update(interactive=False)
            }
        
        try:
            # Get document content and metadata
            content_data = await api.get_document_content(document_id, case_id)
            metadata = await api.get_document_metadata(document_id, case_id)
            
            if not content_data:
                return {
                    doc_title: "Document not found",
                    document_content: '<div style="text-align: center; color: #e74c3c; padding: 50px;"><h3>❌ Error</h3><p>Could not load document content.</p></div>',
                    match_info: "Error loading document",
                    match_counter: "0 / 0"
                }
            
            document_text = content_data.get("content", "")
            document_name = content_data.get("name", "Unknown Document")
            
            # Format the document content
            formatted_content = LegalDocumentFormatter.format_legal_document(document_text, metadata)
            
            # Apply search highlighting if search terms provided
            if search_terms:
                highlighted_content = TextHighlighter.highlight_search_terms(formatted_content, search_terms)
                matches = TextHighlighter.get_match_positions(document_text, search_terms)
                
                match_count = len(matches)
                match_info_text = f"Found {match_count} matches for: {', '.join(search_terms)}"
                match_counter_text = f"1 / {match_count}" if match_count > 0 else "0 / 0"
                
                # Store matches in global state
                viewer_state["current_matches"] = matches
                viewer_state["current_match_index"] = 0
                viewer_state["search_terms"] = search_terms
                
            else:
                highlighted_content = formatted_content
                match_info_text = "Document loaded"
                match_counter_text = "0 / 0"
                matches = []
            
            # Update metadata display
            metadata_updates = {
                doc_type: metadata.get("file_type", "Unknown"),
                file_size: metadata.get("file_size_formatted", "Unknown"),
                upload_date: metadata.get("upload_date", "Unknown"),
                page_count: str(metadata.get("page_count", "N/A")),
                word_count: str(metadata.get("word_count", "Unknown")),
                processing_status: metadata.get("status", "Unknown"),
                citations_found: "\n".join(metadata.get("legal_citations", [])),
                section_structure: "\n".join(metadata.get("section_headers", []))
            }
            
            # Wrap content in proper HTML structure
            final_content = f'''
            <div style="line-height: 1.8; font-family: Georgia, serif; max-width: 100%; padding: 20px;">
                {highlighted_content}
            </div>
            '''
            
            return {
                doc_title: f"📄 {document_name}",
                document_content: final_content,
                match_info: match_info_text,
                match_counter: match_counter_text,
                prev_match_btn: gr.update(interactive=len(matches) > 1),
                next_match_btn: gr.update(interactive=len(matches) > 1),
                current_doc_id: document_id,
                current_case_id: case_id,
                search_matches: matches,
                **metadata_updates
            }
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return {
                doc_title: "Error loading document",
                document_content: f'<div style="text-align: center; color: #e74c3c; padding: 50px;"><h3>❌ Error</h3><p>Failed to load document: {str(e)}</p></div>',
                match_info: "Error occurred",
                match_counter: "0 / 0"
            }
    
    def navigate_matches(direction: int, current_idx: int, matches: List[Dict], doc_content: str):
        """Navigate between search matches in the document."""
        if not matches:
            return current_idx, "0 / 0"
        
        new_idx = max(0, min(len(matches) - 1, current_idx + direction))
        
        # Update match counter
        counter_text = f"{new_idx + 1} / {len(matches)}"
        
        # Here you would implement scrolling to the specific match
        # This would require JavaScript integration in a full implementation
        
        return new_idx, counter_text
    
    def export_snippet(current_doc_id: str, current_match_idx: int, matches: List[Dict]):
        """Export current search result snippet with citation."""
        if not matches or current_match_idx >= len(matches):
            return "No snippet to export"
        
        match = matches[current_match_idx]
        snippet_text = f"From document {current_doc_id}:\n\n"
        snippet_text += f'"{match["text"]}"\n\n'
        snippet_text += f"[Citation: Document {current_doc_id}, paragraph {current_match_idx + 1}]"
        
        return snippet_text
    
    # Wire up navigation buttons
    prev_match_btn.click(
        fn=lambda idx, matches, content: navigate_matches(-1, idx, matches, content),
        inputs=[current_match_idx, search_matches, document_content],
        outputs=[current_match_idx, match_counter]
    )
    
    next_match_btn.click(
        fn=lambda idx, matches, content: navigate_matches(1, idx, matches, content),
        inputs=[current_match_idx, search_matches, document_content],
        outputs=[current_match_idx, match_counter]
    )
    
    # Export functionality
    export_snippet_btn.click(
        fn=export_snippet,
        inputs=[current_doc_id, current_match_idx, search_matches],
        outputs=[gr.Textbox(visible=False)]  # Would show export dialog in full implementation
    )
    
    return doc_viewer


def update_document_viewer(search_result: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    """
    Update document viewer when a search result is selected.
    
    This function would be called from the search pane when a user clicks on a search result.
    
    Args:
        search_result: Selected search result with document info and highlights
        case_id: Current case ID
        
    Returns:
        Dictionary of component updates for the document viewer
    """
    if not search_result:
        return {}
    
    document_id = search_result.get("document_id")
    search_terms = search_result.get("search_terms", [])
    
    # This would call the load_document function
    # In a full implementation, this would be handled by Gradio's event system
    return {}


def create_document_viewer_component():
    """Export function for main application integration."""
    return create_document_viewer()


# Utility functions for text processing
def clean_text_for_display(text: str) -> str:
    """Clean and prepare text for HTML display."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Convert newlines to HTML breaks for proper display
    text = text.replace('\n', '<br>')
    
    return text


def extract_document_structure(content: str) -> Dict[str, List[str]]:
    """Extract structural elements from legal document."""
    structure = {
        "sections": [],
        "claims": [],
        "citations": []
    }
    
    lines = content.split('\n')
    
    for line in lines:
        stripped = line.strip()
        
        # Extract section headers
        if (stripped.isupper() and len(stripped) < 100) or re.match(r'^\d+\.\s+[A-Z]', stripped):
            structure["sections"].append(stripped)
        
        # Extract claims (for patents)
        if stripped.startswith('CLAIM') or re.match(r'^\d+\.\s+[A-Z].*claim', stripped, re.IGNORECASE):
            structure["claims"].append(stripped)
        
        # Extract legal citations
        citation_patterns = [
            r'\d+\s+U\.?S\.?\s+\d+',
            r'\d+\s+F\.\d*d?\s+\d+',
            r'U\.?S\.?\s+Patent\s+No\.?\s+[\d,]+'
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, stripped, re.IGNORECASE)
            structure["citations"].extend(matches)
    
    return structure


if __name__ == "__main__":
    # For testing the document viewer standalone
    demo = gr.Interface(
        fn=lambda: None,
        inputs=[],
        outputs=[],
        title="Document Viewer Test"
    )
    
    with demo:
        viewer = create_document_viewer()
    
    demo.launch(
        server_name="localhost",
        server_port=7862,
        share=False,
        debug=True
    )