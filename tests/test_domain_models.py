"""
Unit tests for domain models.

This module contains comprehensive tests for the domain layer models that represent
the core business entities in the legal AI chatbot system.

Test Coverage:
- Case domain model validation and behavior
- Document domain model functionality
- SearchResult model operations
- Model relationships and constraints
- Serialization and deserialization
- Edge cases and error conditions

Dependencies:
- pytest for test framework
- faker for test data generation
- uuid for unique identifiers
"""

import pytest
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4, UUID
from unittest.mock import Mock, patch

# Import domain models to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.models.domain.case import Case, CaseStatus, CaseMetadata
from backend.app.models.domain.document import Document, DocumentType, DocumentMetadata
from backend.app.models.domain.search_result import SearchResult, SearchMetadata


class TestCase:
    """Test suite for Case domain model."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.case_id = uuid4()
        self.user_id = "test_user_123"
        self.case_name = "Patent Infringement Case 2024"
        self.description = "Testing patent infringement claims and prior art analysis"
        
    def test_case_creation_valid(self):
        """Test successful case creation with valid parameters."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name,
            description=self.description
        )
        
        assert case.case_id == self.case_id
        assert case.user_id == self.user_id
        assert case.name == self.case_name
        assert case.description == self.description
        assert case.status == CaseStatus.ACTIVE
        assert case.created_at is not None
        assert case.updated_at is not None
        assert case.document_count == 0
        assert case.metadata is not None
        assert isinstance(case.metadata, CaseMetadata)
        
    def test_case_creation_minimal(self):
        """Test case creation with minimal required parameters."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name
        )
        
        assert case.case_id == self.case_id
        assert case.user_id == self.user_id
        assert case.name == self.case_name
        assert case.description is None
        assert case.status == CaseStatus.ACTIVE
        
    def test_case_id_validation(self):
        """Test case ID validation and requirements."""
        # Test with None case_id
        with pytest.raises(ValueError, match="Case ID is required"):
            Case(
                case_id=None,
                user_id=self.user_id,
                name=self.case_name
            )
            
        # Test with invalid UUID format
        with pytest.raises(ValueError, match="Invalid case ID format"):
            Case(
                case_id="invalid-uuid",
                user_id=self.user_id,
                name=self.case_name
            )
            
    def test_user_id_validation(self):
        """Test user ID validation requirements."""
        # Test with None user_id
        with pytest.raises(ValueError, match="User ID is required"):
            Case(
                case_id=self.case_id,
                user_id=None,
                name=self.case_name
            )
            
        # Test with empty user_id
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            Case(
                case_id=self.case_id,
                user_id="",
                name=self.case_name
            )
            
        # Test with too long user_id
        with pytest.raises(ValueError, match="User ID too long"):
            Case(
                case_id=self.case_id,
                user_id="x" * 256,  # Assuming 255 char limit
                name=self.case_name
            )
            
    def test_case_name_validation(self):
        """Test case name validation requirements."""
        # Test with None name
        with pytest.raises(ValueError, match="Case name is required"):
            Case(
                case_id=self.case_id,
                user_id=self.user_id,
                name=None
            )
            
        # Test with empty name
        with pytest.raises(ValueError, match="Case name cannot be empty"):
            Case(
                case_id=self.case_id,
                user_id=self.user_id,
                name=""
            )
            
        # Test with too long name
        with pytest.raises(ValueError, match="Case name too long"):
            Case(
                case_id=self.case_id,
                user_id=self.user_id,
                name="x" * 501  # Assuming 500 char limit
            )
            
    def test_case_status_transitions(self):
        """Test valid case status transitions."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name
        )
        
        # Test ACTIVE -> ARCHIVED
        case.status = CaseStatus.ARCHIVED
        assert case.status == CaseStatus.ARCHIVED
        
        # Test ACTIVE -> DELETED
        case.status = CaseStatus.ACTIVE
        case.status = CaseStatus.DELETED
        assert case.status == CaseStatus.DELETED
        
        # Test invalid status
        with pytest.raises(ValueError, match="Invalid case status"):
            case.status = "INVALID_STATUS"
            
    def test_document_count_management(self):
        """Test document count tracking and limits."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name
        )
        
        # Test increment document count
        case.increment_document_count()
        assert case.document_count == 1
        
        # Test multiple increments
        for i in range(24):  # Assuming 25 document limit
            case.increment_document_count()
        assert case.document_count == 25
        
        # Test exceeding limit
        with pytest.raises(ValueError, match="Document count exceeds maximum"):
            case.increment_document_count()
            
        # Test decrement
        case.decrement_document_count()
        assert case.document_count == 24
        
        # Test decrement below zero
        case.document_count = 0
        with pytest.raises(ValueError, match="Document count cannot be negative"):
            case.decrement_document_count()
            
    def test_case_metadata_handling(self):
        """Test case metadata operations."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name
        )
        
        # Test adding metadata
        case.metadata.add_tag("patent")
        case.metadata.add_tag("infringement")
        assert "patent" in case.metadata.tags
        assert "infringement" in case.metadata.tags
        
        # Test duplicate tag
        case.metadata.add_tag("patent")  # Should not duplicate
        assert case.metadata.tags.count("patent") == 1
        
        # Test removing tag
        case.metadata.remove_tag("patent")
        assert "patent" not in case.metadata.tags
        
        # Test custom fields
        case.metadata.set_custom_field("priority", "high")
        case.metadata.set_custom_field("client", "Acme Corp")
        assert case.metadata.get_custom_field("priority") == "high"
        assert case.metadata.get_custom_field("client") == "Acme Corp"
        
    def test_case_timestamps(self):
        """Test case timestamp behavior."""
        before_creation = datetime.now(timezone.utc)
        
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name
        )
        
        after_creation = datetime.now(timezone.utc)
        
        # Test creation timestamp
        assert before_creation <= case.created_at <= after_creation
        assert before_creation <= case.updated_at <= after_creation
        
        # Test update timestamp
        original_updated = case.updated_at
        case.update_timestamp()
        assert case.updated_at > original_updated
        
    def test_case_serialization(self):
        """Test case model serialization to dict."""
        case = Case(
            case_id=self.case_id,
            user_id=self.user_id,
            name=self.case_name,
            description=self.description
        )
        
        case_dict = case.to_dict()
        
        assert case_dict["case_id"] == str(self.case_id)
        assert case_dict["user_id"] == self.user_id
        assert case_dict["name"] == self.case_name
        assert case_dict["description"] == self.description
        assert case_dict["status"] == CaseStatus.ACTIVE.value
        assert "created_at" in case_dict
        assert "updated_at" in case_dict
        assert "metadata" in case_dict
        
    def test_case_deserialization(self):
        """Test case model creation from dict."""
        case_data = {
            "case_id": str(self.case_id),
            "user_id": self.user_id,
            "name": self.case_name,
            "description": self.description,
            "status": CaseStatus.ACTIVE.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "document_count": 5,
            "metadata": {
                "tags": ["patent", "legal"],
                "custom_fields": {"priority": "high"}
            }
        }
        
        case = Case.from_dict(case_data)
        
        assert case.case_id == self.case_id
        assert case.user_id == self.user_id
        assert case.name == self.case_name
        assert case.description == self.description
        assert case.status == CaseStatus.ACTIVE
        assert case.document_count == 5


class TestDocument:
    """Test suite for Document domain model."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.document_id = uuid4()
        self.case_id = uuid4()
        self.filename = "patent_application.pdf"
        self.original_filename = "Patent Application - US12345678.pdf"
        self.file_size = 1024 * 1024  # 1MB
        self.content_type = "application/pdf"
        
    def test_document_creation_valid(self):
        """Test successful document creation with valid parameters."""
        document = Document(
            document_id=self.document_id,
            case_id=self.case_id,
            filename=self.filename,
            original_filename=self.original_filename,
            file_size=self.file_size,
            content_type=self.content_type,
            document_type=DocumentType.PDF
        )
        
        assert document.document_id == self.document_id
        assert document.case_id == self.case_id
        assert document.filename == self.filename
        assert document.original_filename == self.original_filename
        assert document.file_size == self.file_size
        assert document.content_type == self.content_type
        assert document.document_type == DocumentType.PDF
        assert document.is_processed is False
        assert document.processing_status == "pending"
        assert document.created_at is not None
        
    def test_document_type_validation(self):
        """Test document type validation and detection."""
        # Test PDF document
        pdf_doc = Document(
            document_id=self.document_id,
            case_id=self.case_id,
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF
        )
        assert pdf_doc.document_type == DocumentType.PDF
        
        # Test text document
        txt_doc = Document(
            document_id=uuid4(),
            case_id=self.case_id,
            filename="test.txt",
            content_type="text/plain",
            document_type=DocumentType.TEXT
        )
        assert txt_doc.document_type == DocumentType.TEXT
        
        # Test unsupported type
        with pytest.raises(ValueError, match="Unsupported document type"):
            Document(
                document_id=uuid4(),
                case_id=self.case_id,
                filename="test.docx",
                content_type="application/msword",
                document_type="UNSUPPORTED"
            )
            
    def test_file_size_validation(self):
        """Test file size validation and limits."""
        # Test valid file size
        document = Document(
            document_id=self.document_id,
            case_id=self.case_id,
            filename=self.filename,
            file_size=10 * 1024 * 1024  # 10MB
        )
        assert document.file_size == 10 * 1024 * 1024
        
        # Test zero file size
        with pytest.raises(ValueError, match="File size must be positive"):
            Document(
                document_id=self.document_id,
                case_id=self.case_id,
                filename=self.filename,
                file_size=0
            )
            
        # Test negative file size
        with pytest.raises(ValueError, match="File size must be positive"):
            Document(
                document_id=self.document_id,
                case_id=self.case_id,
                filename=self.filename,
                file_size=-1024
            )
            
        # Test excessive file size (assuming 100MB limit)
        with pytest.raises(ValueError, match="File size exceeds maximum"):
            Document(
                document_id=self.document_id,
                case_id=self.case_id,
                filename=self.filename,
                file_size=200 * 1024 * 1024  # 200MB
            )
            
    def test_processing_status_updates(self):
        """Test document processing status management."""
        document = Document(
            document_id=self.document_id,
            case_id=self.case_id,
            filename=self.filename
        )
        
        # Test initial status
        assert document.processing_status == "pending"
        assert document.is_processed is False
        
        # Test status transitions
        document.set_processing_status("processing")
        assert document.processing_status == "processing"
        
        document.set_processing_status("completed")
        assert document.processing_status == "completed"
        assert document.is_processed is True
        
        document.set_processing_status("failed", "Parsing error occurred")
        assert document.processing_status == "failed"
        assert document.processing_error == "Parsing error occurred"
        assert document.is_processed is False
        
    def test_document_metadata_operations(self):
        """Test document metadata handling."""
        document = Document(
            document_id=self.document_id,
            case_id=self.case_id,
            filename=self.filename
        )
        
        # Test adding extraction metadata
        document.metadata.set_extraction_info(
            page_count=25,
            word_count=5000,
            language="en"
        )
        
        assert document.metadata.page_count == 25
        assert document.metadata.word_count == 5000
        assert document.metadata.language == "en"
        
        # Test adding processing metadata
        document.metadata.set_processing_info(
            chunk_count=50,
            embedding_model="mxbai-embed-large",
            processing_time=45.2
        )
        
        assert document.metadata.chunk_count == 50
        assert document.metadata.embedding_model == "mxbai-embed-large"
        assert document.metadata.processing_time == 45.2


class TestSearchResult:
    """Test suite for SearchResult domain model."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.query = "patent infringement damages calculation"
        self.document_id = uuid4()
        self.case_id = uuid4()
        self.chunk_id = "chunk_123"
        self.content = "The court found that damages should be calculated based on..."
        self.score = 0.85
        
    def test_search_result_creation(self):
        """Test search result creation and validation."""
        result = SearchResult(
            query=self.query,
            document_id=self.document_id,
            case_id=self.case_id,
            chunk_id=self.chunk_id,
            content=self.content,
            score=self.score
        )
        
        assert result.query == self.query
        assert result.document_id == self.document_id
        assert result.case_id == self.case_id
        assert result.chunk_id == self.chunk_id
        assert result.content == self.content
        assert result.score == self.score
        assert result.metadata is not None
        
    def test_score_validation(self):
        """Test search score validation."""
        # Test valid score
        result = SearchResult(
            query=self.query,
            document_id=self.document_id,
            score=0.75
        )
        assert result.score == 0.75
        
        # Test invalid scores
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                query=self.query,
                document_id=self.document_id,
                score=-0.1
            )
            
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                query=self.query,
                document_id=self.document_id,
                score=1.1
            )
            
    def test_result_ranking(self):
        """Test search result comparison and ranking."""
        result1 = SearchResult(
            query=self.query,
            document_id=uuid4(),
            score=0.9
        )
        
        result2 = SearchResult(
            query=self.query,
            document_id=uuid4(),
            score=0.7
        )
        
        result3 = SearchResult(
            query=self.query,
            document_id=uuid4(),
            score=0.9
        )
        
        # Test comparison operators
        assert result1 > result2
        assert result2 < result1
        assert result1 == result3  # Same score
        
        # Test sorting
        results = [result2, result1, result3]
        sorted_results = sorted(results, reverse=True)
        
        assert sorted_results[0].score >= sorted_results[1].score
        assert sorted_results[1].score >= sorted_results[2].score


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])