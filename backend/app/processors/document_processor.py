"""
LlamaIndex Document Processing Pipeline for Legal Documents

This module provides the main document processing pipeline using LlamaIndex framework
for legal document ingestion, chunking, embedding, and indexing. It handles the
complete workflow from raw documents to searchable vector embeddings.

Key Features:
- Legal document structure-aware semantic chunking
- Progress tracking via WebSocket updates
- Error handling with retry mechanisms
- Support for PDF and text documents
- Metadata preservation for legal citations
- Per-case vector collection management

Processing Stages:
1. EXTRACTING: Document text extraction and validation
2. CHUNKING: Semantic chunking with legal structure awareness
3. EMBEDDING: Vector embedding generation using Ollama models
4. INDEXING: Storage in Weaviate vector database
5. COMPLETED: Processing finished successfully

Architecture Integration:
- Uses LlamaIndex for document processing and query engines
- Integrates with Ollama for embedding generation
- Stores vectors in Weaviate with per-case collections
- Updates MongoDB with document metadata and processing status
- Sends real-time progress via WebSocket to frontend
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llama_index.core import (
    Document as LlamaDocument,
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import VectorStore

from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor
)
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from backend.config.settings import get_settings
from backend.app.core.ollama_client import OllamaEmbeddingService
from backend.app.core.websocket_manager import WebSocketManager
from ..models.domain.document import (
    LegalDocument,
    ProcessingStatus,
    DocumentType,
    DocumentChunk
)
from ..models.api.document_schemas import DocumentProgressUpdate
from ..repositories.weaviate.vector_repository import VectorRepository
from ..repositories.mongodb.document_repository import DocumentRepository
from ..utils.logging import get_logger
from ..utils.validators import validate_file_type, validate_file_size
from ..core.exceptions import (
    DocumentProcessingError,
    ErrorCode,
    ConfigurationError,
    VectorStoreError
)

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Main document processing pipeline using LlamaIndex framework.
    
    Handles the complete workflow from document upload to vector indexing
    with legal document optimization and real-time progress tracking.
    """
    
    def __init__(
        self,
        embedding_service: OllamaEmbeddingService,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository,
        websocket_manager: WebSocketManager,
        settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize document processor with required services.
        
        Args:
            embedding_service: Ollama embedding service for vector generation
            vector_repository: Weaviate repository for vector storage
            document_repository: MongoDB repository for document metadata
            websocket_manager: WebSocket manager for real-time updates
            settings: Optional processing configuration override
        """
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.document_repository = document_repository
        self.websocket_manager = websocket_manager
        
        # Load configuration
        config = get_settings()
        self.settings = settings or config.processing_settings
        
        # Initialize LlamaIndex components
        self._setup_llamaindex()
        
        # Processing state
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._processing_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "current_active": 0
        }
        
        logger.info(
            "DocumentProcessor initialized",
            embedding_model=self.embedding_service.current_model,
            chunk_size=self.settings.get("chunk_size", 512),
            chunk_overlap=self.settings.get("chunk_overlap", 50)
        )
    
    def _setup_llamaindex(self) -> None:
        """Configure LlamaIndex components for legal document processing."""
        # Configure global settings
        Settings.embed_model = self.embedding_service
        Settings.chunk_size = self.settings.get("chunk_size", 512)
        Settings.chunk_overlap = self.settings.get("chunk_overlap", 50)
        
        # Initialize legal document-aware text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.settings.get("chunk_size", 512),
            chunk_overlap=self.settings.get("chunk_overlap", 50),
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]+",
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        # Initialize extractors for legal document metadata
        self.metadata_extractors = [
            TitleExtractor(nodes=5),
            SummaryExtractor(summaries=["prev", "self"]),
            QuestionsAnsweredExtractor(questions=3)
        ]
        
        # Configure PDF reader for legal documents
        self.pdf_reader = PDFReader(
            return_full_document=False,
            concatenate_pages=False
        )
        
        logger.debug("LlamaIndex components configured for legal document processing")
    
    async def process_document(
        self,
        document: LegalDocument,
        case_id: str,
        file_path: Union[str, Path],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Process a single document through the complete pipeline.
        
        Args:
            document: Legal document domain object
            case_id: Case identifier for vector collection
            file_path: Path to the document file
            user_id: Optional user identifier for WebSocket updates
            
        Returns:
            True if processing succeeded, False otherwise
        """
        task_id = f"{document.document_id}_{int(time.time())}"
        
        try:
            # Start processing and track task
            document.start_processing("LlamaIndex")
            await self.document_repository.update_document(document)
            
            self._active_tasks[task_id] = asyncio.current_task()
            self._processing_stats["current_active"] += 1
            
            # Send initial progress update
            await self._send_progress_update(
                document.document_id,
                ProcessingStatus.EXTRACTING,
                "Starting document processing...",
                user_id
            )
            
            # Stage 1: Extract text content
            await self._extract_document_content(document, file_path, user_id)
            
            # Stage 2: Create semantic chunks
            await self._create_document_chunks(document, case_id, user_id)
            
            # Stage 3: Generate embeddings
            await self._generate_embeddings(document, case_id, user_id)
            
            # Stage 4: Index in vector store
            await self._index_document(document, case_id, user_id)
            
            # Complete processing
            collection_name = f"LegalDocument_CASE_{case_id}"
            document.complete_processing(
                embedding_model=self.embedding_service.current_model,
                index_collection=collection_name
            )
            
            await self.document_repository.update_document(document)
            self._processing_stats["total_processed"] += 1
            
            await self._send_progress_update(
                document.document_id,
                ProcessingStatus.COMPLETED,
                f"Processing completed successfully - {document.chunk_count} chunks created",
                user_id
            )
            
            logger.info(
                "Document processing completed successfully",
                document_id=document.document_id,
                case_id=case_id,
                chunk_count=document.chunk_count,
                processing_time=document.processing_metadata.processing_duration_seconds
            )
            
            return True
            
        except Exception as e:
            await self._handle_processing_error(document, str(e), user_id)
            return False
            
        finally:
            # Cleanup task tracking
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
            self._processing_stats["current_active"] -= 1
    
    async def _extract_document_content(
        self,
        document: LegalDocument,
        file_path: Union[str, Path],
        user_id: Optional[str] = None
    ) -> None:
        """Extract text content from document file."""
        document.transition_to_status(ProcessingStatus.EXTRACTING)
        await self.document_repository.update_document(document)
        
        await self._send_progress_update(
            document.document_id,
            ProcessingStatus.EXTRACTING,
            "Extracting text content...",
            user_id,
            progress_percent=10
        )
        
        try:
            file_path = Path(file_path)
            
            # Validate file before processing
            validate_file_type(file_path, [DocumentType.PDF, DocumentType.TEXT])
            validate_file_size(file_path, self.settings.get("max_file_size_mb", 50))
            
            # Extract content based on file type
            if document.document_type == DocumentType.PDF:
                documents = self.pdf_reader.load_data(file_path)
                document.extracted_text = "\n\n".join([doc.text for doc in documents])
                document.page_count = len(documents)
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    document.extracted_text = f.read()
                document.page_count = 1
            
            # Update extraction metadata
            confidence_score = self._calculate_extraction_confidence(document.extracted_text)
            document.processing_metadata.extraction_confidence = confidence_score
            
            if confidence_score < self.settings.get("min_extraction_confidence", 0.8):
                logger.warning(
                    "Low extraction confidence",
                    document_id=document.document_id,
                    confidence=confidence_score
                )
            
            logger.debug(
                "Text extraction completed",
                document_id=document.document_id,
                text_length=len(document.extracted_text),
                confidence=confidence_score
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Text extraction failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document.document_id,
                stage="extraction"
            )
    
    async def _create_document_chunks(
        self,
        document: LegalDocument,
        case_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Create semantic chunks from extracted text."""
        document.transition_to_status(ProcessingStatus.CHUNKING)
        await self.document_repository.update_document(document)
        
        await self._send_progress_update(
            document.document_id,
            ProcessingStatus.CHUNKING,
            "Creating semantic chunks...",
            user_id,
            progress_percent=40
        )
        
        try:
            # Create LlamaIndex document
            llama_doc = LlamaDocument(
                text=document.extracted_text,
                metadata={
                    "document_id": document.document_id,
                    "document_name": document.document_name,
                    "case_id": case_id,
                    "document_type": document.document_type.value,
                    "upload_timestamp": document.upload_timestamp.isoformat(),
                    "file_size": document.file_size
                }
            )
            
            # Apply text splitting with legal document awareness
            nodes = self.text_splitter.get_nodes_from_documents([llama_doc])
            
            # Extract metadata using legal document extractors
            if self.settings.get("extract_metadata", True):
                for extractor in self.metadata_extractors:
                    nodes = await asyncio.to_thread(extractor.extract, nodes)
            
            # Convert nodes to document chunks
            for i, node in enumerate(nodes):
                chunk = DocumentChunk(
                    chunk_id=f"{document.document_id}_chunk_{i}",
                    document_id=document.document_id,
                    content=node.get_content(metadata_mode=MetadataMode.EMBED),
                    chunk_index=i,
                    start_char=node.start_char_idx or 0,
                    end_char=node.end_char_idx or len(node.get_content()),
                    metadata=node.metadata or {},
                    section_title=node.metadata.get("section_title"),
                    page_number=node.metadata.get("page_number"),
                    paragraph_number=node.metadata.get("paragraph_number"),
                    legal_citations=self._extract_legal_citations(node.get_content())
                )
                
                document.add_chunk(
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    paragraph_number=chunk.paragraph_number,
                    legal_citations=chunk.legal_citations
                )
            
            logger.info(
                "Document chunking completed",
                document_id=document.document_id,
                chunk_count=len(nodes),
                avg_chunk_length=sum(len(chunk.content) for chunk in document.chunks) // len(document.chunks) if document.chunks else 0
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Document chunking failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document.document_id,
                stage="chunking"
            )
    
    async def _generate_embeddings(
        self,
        document: LegalDocument,
        case_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Generate vector embeddings for document chunks."""
        document.transition_to_status(ProcessingStatus.EMBEDDING)
        await self.document_repository.update_document(document)
        
        await self._send_progress_update(
            document.document_id,
            ProcessingStatus.EMBEDDING,
            f"Generating embeddings using {self.embedding_service.current_model}...",
            user_id,
            progress_percent=60
        )
        
        try:
            # Prepare texts for embedding
            chunk_texts = [chunk.content for chunk in document.chunks]
            
            if not chunk_texts:
                raise DocumentProcessingError(
                    "No chunks available for embedding generation",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                    document_id=document.document_id,
                    stage="embedding"
                )
            
            # Generate embeddings in batches
            batch_size = self.settings.get("embedding_batch_size", 10)
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = await self.embedding_service.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = 60 + (30 * (i + len(batch_texts)) / len(chunk_texts))
                await self._send_progress_update(
                    document.document_id,
                    ProcessingStatus.EMBEDDING,
                    f"Generated {i + len(batch_texts)}/{len(chunk_texts)} embeddings...",
                    user_id,
                    progress_percent=int(progress)
                )
            
            # Store embeddings in chunks
            for chunk, embedding in zip(document.chunks, all_embeddings):
                chunk.embedding = embedding
            
            logger.info(
                "Embedding generation completed",
                document_id=document.document_id,
                chunk_count=len(all_embeddings),
                embedding_model=self.embedding_service.current_model,
                embedding_dimensions=len(all_embeddings[0]) if all_embeddings else 0
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Embedding generation failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document.document_id,
                stage="embedding"
            )
    
    async def _index_document(
        self,
        document: LegalDocument,
        case_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Index document chunks in Weaviate vector store."""
        document.transition_to_status(ProcessingStatus.INDEXING)
        await self.document_repository.update_document(document)
        
        await self._send_progress_update(
            document.document_id,
            ProcessingStatus.INDEXING,
            "Indexing in vector database...",
            user_id,
            progress_percent=90
        )
        
        try:
            collection_name = f"LegalDocument_CASE_{case_id}"
            
            # Ensure collection exists
            await self.vector_repository.ensure_collection_exists(collection_name)
            
            # Prepare vector objects for indexing
            vector_objects = []
            for chunk in document.chunks:
                if not chunk.embedding:
                    raise DocumentProcessingError(
                        f"Missing embedding for chunk {chunk.chunk_id}",
                        error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                        document_id=document.document_id,
                        stage="indexing"
                    )
                
                vector_obj = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": document.document_id,
                    "case_id": case_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "paragraph_number": chunk.paragraph_number,
                    "legal_citations": chunk.legal_citations or [],
                    "document_name": document.document_name,
                    "document_type": document.document_type.value,
                    "upload_timestamp": document.upload_timestamp.isoformat(),
                    "embedding": chunk.embedding
                }
                vector_objects.append(vector_obj)
            
            # Index in vector store
            await self.vector_repository.add_vectors(collection_name, vector_objects)
            
            logger.info(
                "Document indexing completed",
                document_id=document.document_id,
                case_id=case_id,
                collection_name=collection_name,
                vector_count=len(vector_objects)
            )
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Vector indexing failed: {str(e)}",
                error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                document_id=document.document_id,
                stage="indexing"
            )
    
    async def _handle_processing_error(
        self,
        document: LegalDocument,
        error_message: str,
        user_id: Optional[str] = None
    ) -> None:
        """Handle processing error and update document status."""
        # Determine the current stage from document status
        stage_map = {
            ProcessingStatus.EXTRACTING: "extraction",
            ProcessingStatus.CHUNKING: "chunking", 
            ProcessingStatus.EMBEDDING: "embedding",
            ProcessingStatus.INDEXING: "indexing"
        }
        stage = stage_map.get(document.status, "unknown")
        
        document.fail_processing(error_message, stage)
        await self.document_repository.update_document(document)
        self._processing_stats["total_failed"] += 1
        
        await self._send_progress_update(
            document.document_id,
            ProcessingStatus.FAILED,
            f"Processing failed: {error_message}",
            user_id,
            error_message=error_message
        )
        
        logger.error(
            "Document processing failed",
            document_id=document.document_id,
            error_message=error_message,
            stage=stage,
            exc_info=True
        )
    
    async def _send_progress_update(
        self,
        document_id: str,
        status: ProcessingStatus,
        message: str,
        user_id: Optional[str] = None,
        progress_percent: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Send progress update via WebSocket."""
        try:
            update = DocumentProgressUpdate(
                document_id=document_id,
                status=status.value,
                message=message,
                progress_percent=progress_percent,
                timestamp=datetime.now(timezone.utc),
                error_message=error_message
            )
            
            await self.websocket_manager.broadcast_to_user(
                user_id,
                "document_progress",
                update.dict()
            )
            
        except Exception as e:
            logger.warning(
                "Failed to send progress update",
                document_id=document_id,
                error=str(e)
            )
    
    def _calculate_extraction_confidence(self, text: str) -> float:
        """Calculate confidence score for text extraction quality."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Basic confidence metrics
        total_chars = len(text)
        printable_chars = sum(1 for c in text if c.isprintable())
        word_count = len(text.split())
        
        # Calculate confidence based on text quality indicators
        printable_ratio = printable_chars / total_chars if total_chars > 0 else 0
        avg_word_length = total_chars / word_count if word_count > 0 else 0
        
        confidence = min(1.0, printable_ratio * (avg_word_length / 10))
        return max(0.0, confidence)
    
    def _extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations from text content."""
        import re
        
        # Basic citation patterns for legal documents
        citation_patterns = [
            r'\b\d+\s+U\.S\.C\.\s+§?\s*\d+',  # USC citations
            r'\b\d+\s+F\.\d+d?\s+\d+',        # Federal reporters
            r'\b\d+\s+S\.Ct\.\s+\d+',         # Supreme Court
            r'Case\s+No\.\s+[\w\d-]+',        # Case numbers
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    async def retry_failed_document(
        self,
        document_id: str,
        case_id: str,
        file_path: Union[str, Path],
        user_id: Optional[str] = None,
        force_retry: bool = False
    ) -> bool:
        """
        Retry processing for a failed document.
        
        Args:
            document_id: Document identifier
            case_id: Case identifier
            file_path: Path to document file
            user_id: Optional user identifier
            force_retry: Force retry even if retry limit exceeded
            
        Returns:
            True if retry succeeded, False otherwise
        """
        try:
            document = await self.document_repository.get_document(document_id)
            if not document:
                raise DocumentProcessingError(
                    f"Document {document_id} not found",
                    error_code=ErrorCode.DOCUMENT_NOT_FOUND,
                    document_id=document_id
                )
            
            if document.status != ProcessingStatus.FAILED and not force_retry:
                raise DocumentProcessingError(
                    "Can only retry failed documents",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                    document_id=document_id
                )
            
            # Check retry limits
            max_retries = self.settings.get("max_retry_attempts", 3)
            if document.processing_metadata.retry_count >= max_retries and not force_retry:
                raise DocumentProcessingError(
                    f"Document has exceeded maximum retry attempts ({max_retries})",
                    error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
                    document_id=document_id
                )
            
            # Prepare for retry
            document.prepare_for_retry()
            await self.document_repository.update_document(document)
            
            # Process document again
            return await self.process_document(document, case_id, file_path, user_id)
            
        except Exception as e:
            logger.error(
                "Document retry failed",
                document_id=document_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self._processing_stats,
            "active_tasks": len(self._active_tasks),
            "embedding_model": self.embedding_service.current_model,
            "settings": self.settings
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown processor and cancel active tasks."""
        logger.info(f"Shutting down document processor with {len(self._active_tasks)} active tasks")
        
        # Cancel all active tasks
        for task_id, task in self._active_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._active_tasks.clear()
        logger.info("Document processor shutdown completed")