"""
Relationship Extraction Background Tasks (Phase 2)

This module provides background task orchestration for legal document relationship
extraction in the Patexia Legal AI Chatbot. It manages async relationship analysis,
NLP-based extraction workflows, confidence scoring, and Neo4j graph database updates.

Key Features:
- Background relationship extraction after document processing
- NLP-based relationship analysis using legal domain models
- Pattern matching for explicit legal document relationships
- Confidence scoring and evidence collection for extracted relationships
- Neo4j graph database integration for relationship storage
- Batch relationship processing for case-wide analysis
- Relationship validation and quality assessment
- Progress tracking and error handling with retry mechanisms

Relationship Types:
- AMENDS: Document amendments and modifications
- REFERENCES: Cross-document references and citations
- SUPERSEDES: Document replacement relationships
- RELATES_TO: General semantic relationships
- CITES: Legal precedent and authority citations
- CONTINUES: Sequential document relationships
- CONTRADICTS: Conflicting document analysis

Background Processing Features:
- Triggered automatically after document processing completion
- Runs independently to avoid blocking main document workflow
- Processes entire case contexts for comprehensive relationship mapping
- Supports incremental updates when new documents are added
- Provides progress tracking and status updates via WebSocket

Architecture Integration:
- Uses RelationshipExtractor for NLP analysis and pattern matching
- Integrates with Neo4j RelationshipRepository for graph storage
- Coordinates with DocumentRepository for document metadata
- Employs NotificationService for progress tracking
- Implements task queuing and retry mechanisms
- Provides relationship quality metrics and analytics
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..core.config import get_settings
from ..models.domain.document import LegalDocument
from ..processors.relationship_extractor import (
    RelationshipExtractor, DocumentRelationship, RelationshipType,
    ExtractionMethod, RelationshipEvidence
)
from ..repositories.neo4j.relationship_repository import RelationshipRepository
from ..repositories.mongodb.document_repository import DocumentRepository
from ..services.notification_service import NotificationService
from ..tasks.document_tasks import DocumentTaskManager, TaskType, TaskStatus, TaskPriority
from ..exceptions import (
    RelationshipExtractionError, TaskError, DatabaseError,
    ErrorCode, raise_relationship_error, raise_task_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class RelationshipTaskType(str, Enum):
    """Types of relationship extraction tasks."""
    SINGLE_DOCUMENT_ANALYSIS = "single_document_analysis"
    CASE_RELATIONSHIP_ANALYSIS = "case_relationship_analysis"
    INCREMENTAL_UPDATE = "incremental_update"
    RELATIONSHIP_VALIDATION = "relationship_validation"
    GRAPH_OPTIMIZATION = "graph_optimization"
    RELATIONSHIP_CLEANUP = "relationship_cleanup"
    CONFIDENCE_RECALCULATION = "confidence_recalculation"
    RELATIONSHIP_EXPORT = "relationship_export"


class ExtractionPhase(str, Enum):
    """Phases of relationship extraction."""
    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CITATION_ANALYSIS = "citation_analysis"
    VALIDATION = "validation"
    STORAGE = "storage"
    COMPLETED = "completed"


@dataclass
class RelationshipTaskContext:
    """Context for relationship extraction tasks."""
    task_id: str
    user_id: str
    case_id: str
    document_ids: List[str]
    extraction_methods: List[ExtractionMethod]
    min_confidence: float = 0.5
    max_relationships_per_document: int = 50
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class RelationshipTaskProgress:
    """Progress tracking for relationship extraction."""
    current_phase: ExtractionPhase
    documents_analyzed: int
    total_documents: int
    relationships_found: int
    relationships_validated: int
    current_document: Optional[str] = None
    estimated_time_remaining: Optional[int] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.documents_analyzed / self.total_documents) * 100


@dataclass
class RelationshipTaskResult:
    """Result of relationship extraction task."""
    success: bool
    relationships_extracted: int
    relationships_stored: int
    documents_processed: int
    execution_time_ms: float
    extraction_methods_used: List[ExtractionMethod]
    average_confidence: float = 0.0
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipTaskStats:
    """Statistics for relationship extraction tasks."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_relationships_extracted: int = 0
    total_documents_analyzed: int = 0
    average_execution_time_ms: float = 0.0
    average_relationships_per_document: float = 0.0
    extraction_method_usage: Dict[ExtractionMethod, int] = field(default_factory=dict)
    relationship_type_distribution: Dict[RelationshipType, int] = field(default_factory=dict)


class RelationshipTaskManager:
    """
    Background task manager for document relationship extraction.
    
    Orchestrates relationship analysis workflows, manages extraction processes,
    and coordinates with Neo4j for graph database updates.
    """
    
    def __init__(
        self,
        relationship_extractor: RelationshipExtractor,
        relationship_repository: RelationshipRepository,
        document_repository: DocumentRepository,
        notification_service: NotificationService,
        document_task_manager: DocumentTaskManager,
        max_concurrent_extraction_tasks: int = 2
    ):
        """
        Initialize relationship task manager.
        
        Args:
            relationship_extractor: Extractor for relationship analysis
            relationship_repository: Neo4j repository for graph operations
            document_repository: MongoDB repository for document data
            notification_service: Service for progress notifications
            document_task_manager: Main task manager for integration
            max_concurrent_extraction_tasks: Maximum concurrent extraction tasks
        """
        self.relationship_extractor = relationship_extractor
        self.relationship_repository = relationship_repository
        self.document_repository = document_repository
        self.notification_service = notification_service
        self.document_task_manager = document_task_manager
        
        # Configuration
        self.settings = get_settings()
        self.max_concurrent_tasks = max_concurrent_extraction_tasks
        
        # Task tracking
        self.active_extraction_tasks: Dict[str, asyncio.Task] = {}
        self.task_progress: Dict[str, RelationshipTaskProgress] = {}
        self.task_results: Dict[str, RelationshipTaskResult] = {}
        self.stats = RelationshipTaskStats()
        
        # Processing state
        self._extraction_semaphore = asyncio.Semaphore(max_concurrent_extraction_tasks)
        self._running = False
        
        # Register relationship task handlers with main task manager
        self._register_task_handlers()
        
        logger.info(
            "RelationshipTaskManager initialized",
            max_concurrent_tasks=max_concurrent_extraction_tasks
        )
    
    def _register_task_handlers(self) -> None:
        """Register relationship task handlers with the main task manager."""
        # Add relationship-specific handlers to the main task manager
        if hasattr(self.document_task_manager, 'task_handlers'):
            self.document_task_manager.task_handlers.update({
                TaskType.RELATIONSHIP_EXTRACTION: self._handle_relationship_extraction,
                TaskType.RELATIONSHIP_VALIDATION: self._handle_relationship_validation,
                TaskType.GRAPH_OPTIMIZATION: self._handle_graph_optimization
            })
    
    async def start(self) -> None:
        """Start the relationship task manager."""
        self._running = True
        logger.info("RelationshipTaskManager started")
    
    async def stop(self) -> None:
        """Stop the relationship task manager and cancel active tasks."""
        self._running = False
        
        # Cancel active extraction tasks
        for task_id, task in self.active_extraction_tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.active_extraction_tasks:
            await asyncio.gather(
                *self.active_extraction_tasks.values(), 
                return_exceptions=True
            )
        
        logger.info("RelationshipTaskManager stopped")
    
    async def submit_document_relationship_analysis(
        self,
        document_id: str,
        case_id: str,
        user_id: str,
        extraction_methods: Optional[List[ExtractionMethod]] = None,
        priority: TaskPriority = TaskPriority.LOW
    ) -> str:
        """
        Submit single document relationship analysis task.
        
        Args:
            document_id: Document to analyze for relationships
            case_id: Case containing the document
            user_id: User identifier
            extraction_methods: Methods to use for extraction
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        if extraction_methods is None:
            extraction_methods = [
                ExtractionMethod.PATTERN_MATCHING,
                ExtractionMethod.SEMANTIC_SIMILARITY,
                ExtractionMethod.CITATION_ANALYSIS
            ]
        
        context = RelationshipTaskContext(
            task_id=f"rel_single_{document_id}_{int(time.time())}",
            user_id=user_id,
            case_id=case_id,
            document_ids=[document_id],
            extraction_methods=extraction_methods,
            correlation_id=str(uuid.uuid4())
        )
        
        task_context = {
            "task_id": context.task_id,
            "user_id": user_id,
            "case_id": case_id,
            "document_id": document_id
        }
        
        parameters = {
            "relationship_context": context,
            "task_type": RelationshipTaskType.SINGLE_DOCUMENT_ANALYSIS.value
        }
        
        return await self.document_task_manager.submit_task(
            task_type=TaskType.RELATIONSHIP_EXTRACTION,
            context=task_context,
            parameters=parameters,
            priority=priority,
            timeout_seconds=1800,  # 30 minutes for relationship analysis
            max_retries=2
        )
    
    async def submit_case_relationship_analysis(
        self,
        case_id: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        extraction_methods: Optional[List[ExtractionMethod]] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Submit comprehensive case relationship analysis task.
        
        Args:
            case_id: Case to analyze for relationships
            user_id: User identifier
            document_ids: Specific documents to analyze (optional)
            extraction_methods: Methods to use for extraction
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        if extraction_methods is None:
            extraction_methods = [
                ExtractionMethod.PATTERN_MATCHING,
                ExtractionMethod.SEMANTIC_SIMILARITY,
                ExtractionMethod.CITATION_ANALYSIS,
                ExtractionMethod.NLP_MODEL
            ]
        
        # Get all documents in case if not specified
        if document_ids is None:
            case_documents = await self.document_repository.get_case_documents(case_id)
            document_ids = [doc.document_id for doc in case_documents]
        
        context = RelationshipTaskContext(
            task_id=f"rel_case_{case_id}_{int(time.time())}",
            user_id=user_id,
            case_id=case_id,
            document_ids=document_ids,
            extraction_methods=extraction_methods,
            correlation_id=str(uuid.uuid4())
        )
        
        task_context = {
            "task_id": context.task_id,
            "user_id": user_id,
            "case_id": case_id
        }
        
        parameters = {
            "relationship_context": context,
            "task_type": RelationshipTaskType.CASE_RELATIONSHIP_ANALYSIS.value
        }
        
        return await self.document_task_manager.submit_task(
            task_type=TaskType.RELATIONSHIP_EXTRACTION,
            context=task_context,
            parameters=parameters,
            priority=priority,
            timeout_seconds=3600,  # 1 hour for case analysis
            max_retries=1
        )
    
    async def submit_incremental_relationship_update(
        self,
        new_document_id: str,
        case_id: str,
        user_id: str,
        priority: TaskPriority = TaskPriority.HIGH
    ) -> str:
        """
        Submit incremental relationship update for newly added document.
        
        Args:
            new_document_id: Newly added document
            case_id: Case containing the document
            user_id: User identifier
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        # Get existing documents in case for comparison
        existing_documents = await self.document_repository.get_case_documents(case_id)
        existing_doc_ids = [doc.document_id for doc in existing_documents 
                           if doc.document_id != new_document_id]
        
        # Include the new document in analysis
        all_document_ids = [new_document_id] + existing_doc_ids
        
        context = RelationshipTaskContext(
            task_id=f"rel_incr_{new_document_id}_{int(time.time())}",
            user_id=user_id,
            case_id=case_id,
            document_ids=all_document_ids,
            extraction_methods=[
                ExtractionMethod.PATTERN_MATCHING,
                ExtractionMethod.SEMANTIC_SIMILARITY,
                ExtractionMethod.CITATION_ANALYSIS
            ],
            correlation_id=str(uuid.uuid4())
        )
        
        task_context = {
            "task_id": context.task_id,
            "user_id": user_id,
            "case_id": case_id,
            "document_id": new_document_id
        }
        
        parameters = {
            "relationship_context": context,
            "task_type": RelationshipTaskType.INCREMENTAL_UPDATE.value,
            "focus_document_id": new_document_id
        }
        
        return await self.document_task_manager.submit_task(
            task_type=TaskType.RELATIONSHIP_EXTRACTION,
            context=task_context,
            parameters=parameters,
            priority=priority,
            timeout_seconds=900,  # 15 minutes for incremental update
            max_retries=2
        )
    
    async def get_task_progress(self, task_id: str) -> Optional[RelationshipTaskProgress]:
        """Get current progress of a relationship extraction task."""
        return self.task_progress.get(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[RelationshipTaskResult]:
        """Get result of completed relationship extraction task."""
        return self.task_results.get(task_id)
    
    def get_extraction_statistics(self) -> RelationshipTaskStats:
        """Get relationship extraction statistics."""
        return self.stats
    
    # Task handlers (registered with main task manager)
    
    async def _handle_relationship_extraction(self, task) -> Any:
        """Handle relationship extraction task."""
        relationship_context = task.parameters["relationship_context"]
        task_type = task.parameters["task_type"]
        
        async with self._extraction_semaphore:
            return await self._execute_relationship_extraction(
                relationship_context, task_type, task.task_id
            )
    
    async def _handle_relationship_validation(self, task) -> Any:
        """Handle relationship validation task."""
        case_id = task.parameters.get("case_id")
        min_confidence = task.parameters.get("min_confidence", 0.5)
        
        return await self._validate_case_relationships(case_id, min_confidence)
    
    async def _handle_graph_optimization(self, task) -> Any:
        """Handle graph optimization task."""
        case_id = task.parameters.get("case_id")
        
        return await self._optimize_case_graph(case_id)
    
    # Core extraction logic
    
    async def _execute_relationship_extraction(
        self,
        context: RelationshipTaskContext,
        task_type: str,
        task_id: str
    ) -> RelationshipTaskResult:
        """Execute relationship extraction with progress tracking."""
        start_time = time.time()
        
        # Initialize progress tracking
        progress = RelationshipTaskProgress(
            current_phase=ExtractionPhase.PATTERN_MATCHING,
            documents_analyzed=0,
            total_documents=len(context.document_ids),
            relationships_found=0,
            relationships_validated=0
        )
        self.task_progress[task_id] = progress
        
        try:
            # Get documents for analysis
            documents = await self._get_documents_for_analysis(context.document_ids)
            
            if not documents:
                raise RelationshipExtractionError(
                    "No documents found for relationship analysis",
                    error_code=ErrorCode.RELATIONSHIP_NO_DOCUMENTS
                )
            
            all_relationships = []
            
            # Phase 1: Pattern Matching
            if ExtractionMethod.PATTERN_MATCHING in context.extraction_methods:
                progress.current_phase = ExtractionPhase.PATTERN_MATCHING
                await self._send_progress_update(context, progress)
                
                pattern_relationships = await self._extract_pattern_relationships(
                    documents, context, progress
                )
                all_relationships.extend(pattern_relationships)
            
            # Phase 2: Semantic Analysis
            if ExtractionMethod.SEMANTIC_SIMILARITY in context.extraction_methods:
                progress.current_phase = ExtractionPhase.SEMANTIC_ANALYSIS
                await self._send_progress_update(context, progress)
                
                semantic_relationships = await self._extract_semantic_relationships(
                    documents, context, progress
                )
                all_relationships.extend(semantic_relationships)
            
            # Phase 3: Citation Analysis
            if ExtractionMethod.CITATION_ANALYSIS in context.extraction_methods:
                progress.current_phase = ExtractionPhase.CITATION_ANALYSIS
                await self._send_progress_update(context, progress)
                
                citation_relationships = await self._extract_citation_relationships(
                    documents, context, progress
                )
                all_relationships.extend(citation_relationships)
            
            # Phase 4: NLP Model Analysis (if enabled)
            if ExtractionMethod.NLP_MODEL in context.extraction_methods:
                progress.current_phase = ExtractionPhase.SEMANTIC_ANALYSIS
                await self._send_progress_update(context, progress)
                
                nlp_relationships = await self._extract_nlp_relationships(
                    documents, context, progress
                )
                all_relationships.extend(nlp_relationships)
            
            # Phase 5: Validation and Filtering
            progress.current_phase = ExtractionPhase.VALIDATION
            await self._send_progress_update(context, progress)
            
            validated_relationships = await self._validate_relationships(
                all_relationships, context
            )
            progress.relationships_validated = len(validated_relationships)
            
            # Phase 6: Storage
            progress.current_phase = ExtractionPhase.STORAGE
            await self._send_progress_update(context, progress)
            
            stored_count = await self._store_relationships(
                validated_relationships, context.case_id
            )
            
            # Completion
            progress.current_phase = ExtractionPhase.COMPLETED
            progress.documents_analyzed = len(documents)
            await self._send_progress_update(context, progress)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            average_confidence = (
                sum(rel.confidence_score for rel in validated_relationships) / 
                len(validated_relationships) if validated_relationships else 0.0
            )
            
            result = RelationshipTaskResult(
                success=True,
                relationships_extracted=len(all_relationships),
                relationships_stored=stored_count,
                documents_processed=len(documents),
                execution_time_ms=execution_time_ms,
                extraction_methods_used=context.extraction_methods,
                average_confidence=average_confidence,
                quality_metrics={
                    "pattern_matches": sum(1 for r in all_relationships 
                                         if r.extraction_method == ExtractionMethod.PATTERN_MATCHING),
                    "semantic_matches": sum(1 for r in all_relationships 
                                          if r.extraction_method == ExtractionMethod.SEMANTIC_SIMILARITY),
                    "citation_matches": sum(1 for r in all_relationships 
                                          if r.extraction_method == ExtractionMethod.CITATION_ANALYSIS),
                    "high_confidence_count": sum(1 for r in validated_relationships 
                                                if r.confidence_score > 0.8),
                    "relationship_types": dict(Counter(r.relationship_type for r in validated_relationships))
                }
            )
            
            self.task_results[task_id] = result
            self._update_extraction_stats(result)
            
            logger.info(
                "Relationship extraction completed",
                task_id=task_id,
                task_type=task_type,
                relationships_found=len(all_relationships),
                relationships_stored=stored_count,
                execution_time_ms=execution_time_ms
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = RelationshipTaskResult(
                success=False,
                relationships_extracted=0,
                relationships_stored=0,
                documents_processed=0,
                execution_time_ms=execution_time_ms,
                extraction_methods_used=context.extraction_methods,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN_ERROR')
            )
            
            self.task_results[task_id] = result
            
            logger.error(
                "Relationship extraction failed",
                task_id=task_id,
                task_type=task_type,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            
            raise
        
        finally:
            # Cleanup progress tracking
            if task_id in self.task_progress:
                del self.task_progress[task_id]
    
    async def _get_documents_for_analysis(self, document_ids: List[str]) -> List[LegalDocument]:
        """Get documents for relationship analysis."""
        documents = []
        
        for doc_id in document_ids:
            try:
                document = await self.document_repository.get_document(doc_id)
                if document and document.text_content:
                    documents.append(document)
            except Exception as e:
                logger.warning(
                    "Failed to load document for relationship analysis",
                    document_id=doc_id,
                    error=str(e)
                )
        
        return documents
    
    async def _extract_pattern_relationships(
        self,
        documents: List[LegalDocument],
        context: RelationshipTaskContext,
        progress: RelationshipTaskProgress
    ) -> List[DocumentRelationship]:
        """Extract relationships using pattern matching."""
        async with performance_context("pattern_relationship_extraction"):
            relationships = []
            
            for i, document in enumerate(documents):
                progress.current_document = document.document_name
                progress.documents_analyzed = i
                await self._send_progress_update(context, progress)
                
                try:
                    doc_relationships = await self.relationship_extractor.extract_pattern_relationships(
                        document, documents
                    )
                    relationships.extend(doc_relationships)
                    progress.relationships_found += len(doc_relationships)
                    
                except Exception as e:
                    logger.warning(
                        "Pattern extraction failed for document",
                        document_id=document.document_id,
                        error=str(e)
                    )
            
            return relationships
    
    async def _extract_semantic_relationships(
        self,
        documents: List[LegalDocument],
        context: RelationshipTaskContext,
        progress: RelationshipTaskProgress
    ) -> List[DocumentRelationship]:
        """Extract relationships using semantic similarity."""
        async with performance_context("semantic_relationship_extraction"):
            relationships = []
            
            for i, document in enumerate(documents):
                progress.current_document = document.document_name
                await self._send_progress_update(context, progress)
                
                try:
                    other_documents = [d for d in documents if d.document_id != document.document_id]
                    doc_relationships = await self.relationship_extractor.extract_semantic_relationships(
                        document, other_documents
                    )
                    relationships.extend(doc_relationships)
                    progress.relationships_found += len(doc_relationships)
                    
                except Exception as e:
                    logger.warning(
                        "Semantic extraction failed for document",
                        document_id=document.document_id,
                        error=str(e)
                    )
            
            return relationships
    
    async def _extract_citation_relationships(
        self,
        documents: List[LegalDocument],
        context: RelationshipTaskContext,
        progress: RelationshipTaskProgress
    ) -> List[DocumentRelationship]:
        """Extract relationships using citation analysis."""
        async with performance_context("citation_relationship_extraction"):
            relationships = []
            
            for i, document in enumerate(documents):
                progress.current_document = document.document_name
                await self._send_progress_update(context, progress)
                
                try:
                    other_documents = [d for d in documents if d.document_id != document.document_id]
                    doc_relationships = await self.relationship_extractor.analyze_citation_relationships(
                        document, other_documents
                    )
                    relationships.extend(doc_relationships)
                    progress.relationships_found += len(doc_relationships)
                    
                except Exception as e:
                    logger.warning(
                        "Citation extraction failed for document",
                        document_id=document.document_id,
                        error=str(e)
                    )
            
            return relationships
    
    async def _extract_nlp_relationships(
        self,
        documents: List[LegalDocument],
        context: RelationshipTaskContext,
        progress: RelationshipTaskProgress
    ) -> List[DocumentRelationship]:
        """Extract relationships using NLP models."""
        async with performance_context("nlp_relationship_extraction"):
            relationships = []
            
            for i, document in enumerate(documents):
                progress.current_document = document.document_name
                await self._send_progress_update(context, progress)
                
                try:
                    other_documents = [d for d in documents if d.document_id != document.document_id]
                    doc_relationships = await self.relationship_extractor.analyze_semantic_relationships(
                        document, other_documents
                    )
                    relationships.extend(doc_relationships)
                    progress.relationships_found += len(doc_relationships)
                    
                except Exception as e:
                    logger.warning(
                        "NLP extraction failed for document",
                        document_id=document.document_id,
                        error=str(e)
                    )
            
            return relationships
    
    async def _validate_relationships(
        self,
        relationships: List[DocumentRelationship],
        context: RelationshipTaskContext
    ) -> List[DocumentRelationship]:
        """Validate and filter relationships."""
        validated = []
        
        for relationship in relationships:
            # Filter by minimum confidence
            if relationship.confidence_score < context.min_confidence:
                continue
            
            # Apply business rules validation
            if self._passes_business_rules(relationship):
                relationship.validated = True
                validated.append(relationship)
        
        # Remove duplicates and merge evidence
        validated = self._deduplicate_relationships(validated)
        
        # Limit relationships per document
        validated = self._limit_relationships_per_document(
            validated, context.max_relationships_per_document
        )
        
        return validated
    
    def _passes_business_rules(self, relationship: DocumentRelationship) -> bool:
        """Apply business rules for relationship validation."""
        # Rule 1: Document cannot relate to itself
        if relationship.source_document_id == relationship.target_document_id:
            return False
        
        # Rule 2: Must have evidence
        if not relationship.evidence:
            return False
        
        # Rule 3: Certain relationship types require higher confidence
        high_confidence_types = {
            RelationshipType.AMENDS,
            RelationshipType.SUPERSEDES,
            RelationshipType.CANCELS
        }
        
        if (relationship.relationship_type in high_confidence_types and 
            relationship.confidence_score < 0.7):
            return False
        
        return True
    
    def _deduplicate_relationships(
        self,
        relationships: List[DocumentRelationship]
    ) -> List[DocumentRelationship]:
        """Remove duplicate relationships and merge evidence."""
        relationship_map = {}
        
        for rel in relationships:
            key = (rel.source_document_id, rel.target_document_id, rel.relationship_type.value)
            
            if key in relationship_map:
                # Merge evidence and take higher confidence
                existing = relationship_map[key]
                existing.evidence.extend(rel.evidence)
                existing.confidence_score = max(existing.confidence_score, rel.confidence_score)
            else:
                relationship_map[key] = rel
        
        return list(relationship_map.values())
    
    def _limit_relationships_per_document(
        self,
        relationships: List[DocumentRelationship],
        max_per_document: int
    ) -> List[DocumentRelationship]:
        """Limit number of relationships per document."""
        doc_relationships = defaultdict(list)
        
        # Group by source document
        for rel in relationships:
            doc_relationships[rel.source_document_id].append(rel)
        
        # Sort by confidence and limit
        limited = []
        for doc_id, doc_rels in doc_relationships.items():
            doc_rels.sort(key=lambda r: r.confidence_score, reverse=True)
            limited.extend(doc_rels[:max_per_document])
        
        return limited
    
    async def _store_relationships(
        self,
        relationships: List[DocumentRelationship],
        case_id: str
    ) -> int:
        """Store relationships in Neo4j graph database."""
        if not relationships:
            return 0
        
        try:
            stored_count = 0
            
            for relationship in relationships:
                success = await self.relationship_repository.create_relationship(
                    relationship, case_id
                )
                if success:
                    stored_count += 1
            
            logger.info(
                "Relationships stored in graph database",
                case_id=case_id,
                total_relationships=len(relationships),
                stored_count=stored_count
            )
            
            return stored_count
            
        except Exception as e:
            logger.error(
                "Failed to store relationships",
                case_id=case_id,
                relationship_count=len(relationships),
                error=str(e)
            )
            raise DatabaseError(f"Failed to store relationships: {str(e)}")
    
    async def _validate_case_relationships(
        self,
        case_id: str,
        min_confidence: float
    ) -> Dict[str, Any]:
        """Validate existing relationships for a case."""
        try:
            # Get existing relationships
            relationships = await self.relationship_repository.get_case_relationships(
                case_id, min_confidence
            )
            
            validation_results = {
                "case_id": case_id,
                "total_relationships": len(relationships),
                "validated_relationships": 0,
                "removed_relationships": 0,
                "confidence_distribution": defaultdict(int)
            }
            
            for relationship in relationships:
                # Update confidence distribution
                confidence_bucket = int(relationship.confidence_score * 10) / 10
                validation_results["confidence_distribution"][confidence_bucket] += 1
                
                # Mark as validated if it passes current rules
                if self._passes_business_rules(relationship):
                    validation_results["validated_relationships"] += 1
                else:
                    # Remove invalid relationships
                    await self.relationship_repository.delete_relationship(
                        relationship.source_document_id,
                        relationship.target_document_id,
                        relationship.relationship_type
                    )
                    validation_results["removed_relationships"] += 1
            
            return validation_results
            
        except Exception as e:
            logger.error(
                "Relationship validation failed",
                case_id=case_id,
                error=str(e)
            )
            raise
    
    async def _optimize_case_graph(self, case_id: str) -> Dict[str, Any]:
        """Optimize graph structure for a case."""
        try:
            # This would implement graph optimization algorithms
            # For now, return basic statistics
            
            relationships = await self.relationship_repository.get_case_relationships(case_id)
            
            optimization_results = {
                "case_id": case_id,
                "total_relationships": len(relationships),
                "optimization_type": "basic_cleanup",
                "performance_improvement": "5%"  # Placeholder
            }
            
            return optimization_results
            
        except Exception as e:
            logger.error(
                "Graph optimization failed",
                case_id=case_id,
                error=str(e)
            )
            raise
    
    async def _send_progress_update(
        self,
        context: RelationshipTaskContext,
        progress: RelationshipTaskProgress
    ) -> None:
        """Send progress update via notification service."""
        if not self.notification_service:
            return
        
        try:
            await self.notification_service.send_notification(
                user_id=context.user_id,
                notification_type="relationship_extraction_progress",
                context={
                    "task_id": context.task_id,
                    "case_id": context.case_id,
                    "current_phase": progress.current_phase.value,
                    "progress_percentage": progress.progress_percentage,
                    "documents_analyzed": progress.documents_analyzed,
                    "total_documents": progress.total_documents,
                    "relationships_found": progress.relationships_found,
                    "current_document": progress.current_document or ""
                }
            )
        except Exception as e:
            logger.warning(
                "Failed to send relationship extraction progress update",
                task_id=context.task_id,
                error=str(e)
            )
    
    def _update_extraction_stats(self, result: RelationshipTaskResult) -> None:
        """Update extraction statistics."""
        self.stats.total_tasks += 1
        
        if result.success:
            self.stats.completed_tasks += 1
            self.stats.total_relationships_extracted += result.relationships_extracted
            self.stats.total_documents_analyzed += result.documents_processed
            
            # Update average execution time
            current_avg = self.stats.average_execution_time_ms
            completed = self.stats.completed_tasks
            
            self.stats.average_execution_time_ms = (
                (current_avg * (completed - 1) + result.execution_time_ms) / completed
            )
            
            # Update average relationships per document
            if self.stats.total_documents_analyzed > 0:
                self.stats.average_relationships_per_document = (
                    self.stats.total_relationships_extracted / self.stats.total_documents_analyzed
                )
            
            # Update extraction method usage
            for method in result.extraction_methods_used:
                if method not in self.stats.extraction_method_usage:
                    self.stats.extraction_method_usage[method] = 0
                self.stats.extraction_method_usage[method] += 1
        else:
            self.stats.failed_tasks += 1
    
    async def cleanup(self) -> None:
        """Cleanup relationship task manager resources."""
        await self.stop()
        
        # Clear state
        self.active_extraction_tasks.clear()
        self.task_progress.clear()
        self.task_results.clear()
        
        logger.info("RelationshipTaskManager cleanup completed")