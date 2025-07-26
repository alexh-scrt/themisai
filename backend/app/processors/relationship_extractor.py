"""
Document Relationship Extractor for Legal Documents (Phase 2)

This module provides sophisticated NLP-based relationship extraction capabilities
for legal documents, identifying complex inter-document relationships and storing
them in a Neo4j graph database for enhanced search and analysis.

Key Features:
- Automated detection of document relationships (AMENDS, REFERENCES, SUPERSEDES, etc.)
- NLP-based relationship extraction using legal domain models
- Confidence scoring and evidence tracking for relationships
- Neo4j graph database integration for relationship storage
- Background processing to avoid blocking main document flow
- Legal citation and reference analysis
- Document version and amendment tracking
- Entity relationship mapping (parties, cases, contracts)

Relationship Types:
- AMENDS: One document amends or modifies another
- REFERENCES: Document contains references to another document
- SUPERSEDES: Document replaces or supersedes another
- RELATES_TO: General relationship between documents
- CITES: Document cites legal precedent or authority
- CONTINUES: Document continues or follows from another
- CONTRADICTS: Documents contain conflicting information

Legal Document Optimizations:
- Patent application relationship detection
- Contract amendment and modification tracking
- Legal brief citation analysis
- Regulatory reference mapping
- Court filing relationship chains

Architecture Integration:
- Runs as background task after document processing
- Integrates with Neo4j for graph storage
- Uses Ollama for NLP relationship extraction
- Provides relationship data for enhanced search
- Supports confidence-based filtering
"""

import asyncio
import logging
import re
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from enum import Enum
from pathlib import Path

from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable, TransientError

from ..core.config import get_settings
from ..core.ollama_client import OllamaClient
from ..core.websocket_manager import WebSocketManager
from ..models.domain.document import LegalDocument, DocumentChunk
from ..repositories.mongodb.document_repository import DocumentRepository
from ..repositories.neo4j.relationship_repository import RelationshipRepository
from ..utils.logging import get_logger
from ..exceptions import (
    RelationshipExtractionError,
    ErrorCode,
    DatabaseError
)

logger = get_logger(__name__)


class RelationshipType(Enum):
    """Types of relationships between legal documents."""
    AMENDS = "AMENDS"
    REFERENCES = "REFERENCES"
    SUPERSEDES = "SUPERSEDES"
    RELATES_TO = "RELATES_TO"
    CITES = "CITES"
    CONTINUES = "CONTINUES"
    CONTRADICTS = "CONTRADICTS"
    IMPLEMENTS = "IMPLEMENTS"
    DEFINES = "DEFINES"
    CANCELS = "CANCELS"


class ExtractionMethod(Enum):
    """Methods used for relationship extraction."""
    NLP_MODEL = "nlp_model"
    PATTERN_MATCHING = "pattern_matching"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CITATION_ANALYSIS = "citation_analysis"
    METADATA_ANALYSIS = "metadata_analysis"
    MANUAL = "manual"


@dataclass
class RelationshipEvidence:
    """Evidence supporting a document relationship."""
    text_snippet: str
    confidence_score: float
    source_location: str  # chunk_id or page reference
    extraction_method: ExtractionMethod
    pattern_matched: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None


@dataclass
class DocumentRelationship:
    """Relationship between two legal documents."""
    source_document_id: str
    target_document_id: str
    relationship_type: RelationshipType
    confidence_score: float
    evidence: List[RelationshipEvidence]
    extraction_method: ExtractionMethod
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated: bool = False
    validation_score: Optional[float] = None
    
    @property
    def has_high_confidence(self) -> bool:
        """Check if relationship has high confidence."""
        return self.confidence_score >= 0.8
    
    @property
    def evidence_count(self) -> int:
        """Get number of evidence items."""
        return len(self.evidence)
    
    @property
    def average_evidence_confidence(self) -> float:
        """Calculate average confidence across all evidence."""
        if not self.evidence:
            return 0.0
        return sum(e.confidence_score for e in self.evidence) / len(self.evidence)


class LegalPatternMatcher:
    """
    Pattern matching engine for legal document relationships.
    
    Uses regex patterns and legal terminology to identify
    explicit relationships between documents.
    """
    
    def __init__(self):
        """Initialize legal pattern matcher."""
        self.relationship_patterns = self._compile_relationship_patterns()
        self.citation_patterns = self._compile_citation_patterns()
        self.reference_patterns = self._compile_reference_patterns()
    
    def _compile_relationship_patterns(self) -> Dict[RelationshipType, List[re.Pattern]]:
        """Compile regex patterns for relationship detection."""
        patterns = {
            RelationshipType.AMENDS: [
                re.compile(r'\bamends?\b.*?(?:agreement|contract|document|section)', re.IGNORECASE),
                re.compile(r'\bmodif(?:y|ies|ied)\b.*?(?:agreement|contract|document)', re.IGNORECASE),
                re.compile(r'\bfirst amendment\b.*?\bto\b', re.IGNORECASE),
                re.compile(r'\baddendum\b.*?\bto\b', re.IGNORECASE),
                re.compile(r'\bsupplement\b.*?\bto\b', re.IGNORECASE),
            ],
            
            RelationshipType.SUPERSEDES: [
                re.compile(r'\bsupersedes?\b.*?(?:agreement|contract|document)', re.IGNORECASE),
                re.compile(r'\breplaces?\b.*?(?:agreement|contract|document)', re.IGNORECASE),
                re.compile(r'\bsubstitutes?\b.*?\bfor\b', re.IGNORECASE),
                re.compile(r'\bnull and void\b.*?(?:agreement|contract)', re.IGNORECASE),
                re.compile(r'\bterminates?\b.*?(?:agreement|contract)', re.IGNORECASE),
            ],
            
            RelationshipType.REFERENCES: [
                re.compile(r'\bas referenced in\b', re.IGNORECASE),
                re.compile(r'\baccording to\b.*?(?:document|agreement|section)', re.IGNORECASE),
                re.compile(r'\bas defined in\b', re.IGNORECASE),
                re.compile(r'\bpursuant to\b', re.IGNORECASE),
                re.compile(r'\bin accordance with\b', re.IGNORECASE),
            ],
            
            RelationshipType.CITES: [
                re.compile(r'\bciting\b.*?(?:case|precedent|authority)', re.IGNORECASE),
                re.compile(r'\brelying on\b.*?(?:case|precedent)', re.IGNORECASE),
                re.compile(r'\bfollowing\b.*?(?:case|precedent)', re.IGNORECASE),
                re.compile(r'\bdistinguishing\b.*?(?:case)', re.IGNORECASE),
            ],
            
            RelationshipType.CONTINUES: [
                re.compile(r'\bcontinuation of\b', re.IGNORECASE),
                re.compile(r'\bfollows from\b', re.IGNORECASE),
                re.compile(r'\bbuilds upon\b', re.IGNORECASE),
                re.compile(r'\bextends\b.*?(?:agreement|contract)', re.IGNORECASE),
            ],
            
            RelationshipType.IMPLEMENTS: [
                re.compile(r'\bimplements?\b.*?(?:agreement|contract|plan)', re.IGNORECASE),
                re.compile(r'\bexecutes?\b.*?(?:agreement|contract)', re.IGNORECASE),
                re.compile(r'\bcarries out\b.*?(?:terms|provisions)', re.IGNORECASE),
            ],
            
            RelationshipType.CONTRADICTS: [
                re.compile(r'\bcontradicts?\b.*?(?:agreement|statement)', re.IGNORECASE),
                re.compile(r'\bconflicts? with\b', re.IGNORECASE),
                re.compile(r'\binconsistent with\b', re.IGNORECASE),
                re.compile(r'\bopposed to\b', re.IGNORECASE),
            ]
        }
        
        return patterns
    
    def _compile_citation_patterns(self) -> List[re.Pattern]:
        """Compile patterns for legal citation detection."""
        return [
            # Case citations
            re.compile(r'\b\w+\s+v\.\s+\w+', re.IGNORECASE),
            re.compile(r'\b\d+\s+[A-Z][a-z]+\.?\s*(?:2d|3d)?\s+\d+', re.IGNORECASE),
            
            # Statutory citations
            re.compile(r'\b\d+\s+U\.S\.C\.\s+§?\s*\d+', re.IGNORECASE),
            re.compile(r'\b\d+\s+C\.F\.R\.\s+§?\s*\d+', re.IGNORECASE),
            
            # Patent citations
            re.compile(r'\bU\.S\.\s+Patent\s+(?:No\.?\s*)?\d+', re.IGNORECASE),
            re.compile(r'\bPatent\s+(?:Application\s+)?(?:No\.?\s*)?\d+', re.IGNORECASE),
        ]
    
    def _compile_reference_patterns(self) -> List[re.Pattern]:
        """Compile patterns for document reference detection."""
        return [
            # Document references
            re.compile(r'\bDocument\s+(?:No\.?\s*)?\w+', re.IGNORECASE),
            re.compile(r'\bExhibit\s+[A-Z]', re.IGNORECASE),
            re.compile(r'\bSchedule\s+\d+', re.IGNORECASE),
            re.compile(r'\bAnnex\s+[A-Z]', re.IGNORECASE),
            re.compile(r'\bAppendix\s+[A-Z]', re.IGNORECASE),
            
            # Contract references
            re.compile(r'\bAgreement\s+dated\s+\w+', re.IGNORECASE),
            re.compile(r'\bContract\s+(?:No\.?\s*)?\w+', re.IGNORECASE),
            re.compile(r'\bAmendment\s+(?:No\.?\s*)?\d+', re.IGNORECASE),
        ]
    
    def extract_relationships(
        self,
        document: LegalDocument,
        candidate_documents: List[LegalDocument]
    ) -> List[DocumentRelationship]:
        """
        Extract relationships using pattern matching.
        
        Args:
            document: Source document to analyze
            candidate_documents: Potential target documents
            
        Returns:
            List of discovered relationships
        """
        relationships = []
        document_text = document.text_content or ""
        
        # Create mapping of document names/identifiers to document IDs
        doc_name_map = self._create_document_name_mapping(candidate_documents)
        
        # Search for relationship patterns
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(document_text):
                    # Find potential target documents mentioned near the match
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(document_text), match.end() + 200)
                    context = document_text[context_start:context_end]
                    
                    # Look for document references in context
                    target_docs = self._find_referenced_documents(context, doc_name_map)
                    
                    for target_doc_id in target_docs:
                        evidence = RelationshipEvidence(
                            text_snippet=match.group(),
                            confidence_score=self._calculate_pattern_confidence(rel_type, match.group()),
                            source_location=f"text_position_{match.start()}",
                            extraction_method=ExtractionMethod.PATTERN_MATCHING,
                            pattern_matched=pattern.pattern,
                            context_before=document_text[context_start:match.start()],
                            context_after=document_text[match.end():context_end]
                        )
                        
                        relationship = DocumentRelationship(
                            source_document_id=document.document_id,
                            target_document_id=target_doc_id,
                            relationship_type=rel_type,
                            confidence_score=evidence.confidence_score,
                            evidence=[evidence],
                            extraction_method=ExtractionMethod.PATTERN_MATCHING
                        )
                        
                        relationships.append(relationship)
        
        return self._deduplicate_relationships(relationships)
    
    def _create_document_name_mapping(self, documents: List[LegalDocument]) -> Dict[str, str]:
        """Create mapping from document names/titles to document IDs."""
        name_map = {}
        
        for doc in documents:
            # Map by document name
            if doc.document_name:
                name_map[doc.document_name.lower()] = doc.document_id
            
            # Map by filename
            if doc.original_filename:
                name_map[doc.original_filename.lower()] = doc.document_id
            
            # Extract and map potential document numbers/identifiers
            if doc.text_content:
                doc_numbers = re.findall(r'\b(?:Document|Agreement|Contract)\s+(?:No\.?\s*)?(\w+)', 
                                       doc.text_content, re.IGNORECASE)
                for doc_num in doc_numbers:
                    name_map[doc_num.lower()] = doc.document_id
        
        return name_map
    
    def _find_referenced_documents(self, context: str, doc_name_map: Dict[str, str]) -> List[str]:
        """Find document references in text context."""
        referenced_docs = []
        
        # Search for exact name matches
        for doc_name, doc_id in doc_name_map.items():
            if doc_name in context.lower():
                referenced_docs.append(doc_id)
        
        # Search for reference patterns
        for pattern in self.reference_patterns:
            for match in pattern.finditer(context):
                # Try to match the reference to known documents
                ref_text = match.group().lower()
                for doc_name, doc_id in doc_name_map.items():
                    if any(word in ref_text for word in doc_name.split()):
                        if doc_id not in referenced_docs:
                            referenced_docs.append(doc_id)
        
        return referenced_docs
    
    def _calculate_pattern_confidence(self, rel_type: RelationshipType, matched_text: str) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = 0.7
        
        # Boost confidence for specific relationship types
        if rel_type in [RelationshipType.AMENDS, RelationshipType.SUPERSEDES]:
            base_confidence = 0.8
        
        # Boost confidence for longer matches (more specific)
        if len(matched_text) > 50:
            base_confidence += 0.1
        
        # Boost confidence for legal terminology
        legal_terms = ['agreement', 'contract', 'amendment', 'supersedes', 'pursuant']
        if any(term in matched_text.lower() for term in legal_terms):
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _deduplicate_relationships(self, relationships: List[DocumentRelationship]) -> List[DocumentRelationship]:
        """Remove duplicate relationships and merge evidence."""
        relationship_map = {}
        
        for rel in relationships:
            key = (rel.source_document_id, rel.target_document_id, rel.relationship_type.value)
            
            if key in relationship_map:
                # Merge evidence and update confidence
                existing_rel = relationship_map[key]
                existing_rel.evidence.extend(rel.evidence)
                existing_rel.confidence_score = max(existing_rel.confidence_score, rel.confidence_score)
            else:
                relationship_map[key] = rel
        
        return list(relationship_map.values())


class SemanticRelationshipAnalyzer:
    """
    Semantic analysis engine for document relationships using embeddings.
    
    Uses document embeddings and semantic similarity to identify
    implicit relationships between documents.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """Initialize semantic analyzer."""
        self.ollama_client = ollama_client
        self.similarity_threshold = 0.7
        self.relationship_prompts = self._create_relationship_prompts()
    
    def _create_relationship_prompts(self) -> Dict[RelationshipType, str]:
        """Create prompts for LLM-based relationship analysis."""
        return {
            RelationshipType.AMENDS: """
            Analyze these two legal documents and determine if one amends the other.
            Look for explicit amendment language, modifications, or changes to terms.
            Provide confidence score (0.0-1.0) and brief explanation.
            """,
            
            RelationshipType.REFERENCES: """
            Determine if the first document references or relies upon the second document.
            Look for explicit references, citations, or dependencies.
            Provide confidence score (0.0-1.0) and brief explanation.
            """,
            
            RelationshipType.SUPERSEDES: """
            Analyze if the first document supersedes, replaces, or makes void the second document.
            Look for language indicating replacement, termination, or supersession.
            Provide confidence score (0.0-1.0) and brief explanation.
            """,
            
            RelationshipType.RELATES_TO: """
            Determine if these documents are related in subject matter, parties, or purpose.
            Consider thematic connections, shared entities, or complementary purposes.
            Provide confidence score (0.0-1.0) and brief explanation.
            """
        }
    
    async def analyze_semantic_relationships(
        self,
        document: LegalDocument,
        candidate_documents: List[LegalDocument],
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[DocumentRelationship]:
        """
        Analyze semantic relationships using LLM.
        
        Args:
            document: Source document
            candidate_documents: Potential target documents
            relationship_types: Types of relationships to analyze
            
        Returns:
            List of discovered relationships
        """
        if relationship_types is None:
            relationship_types = list(RelationshipType)
        
        relationships = []
        
        for candidate in candidate_documents:
            if candidate.document_id == document.document_id:
                continue
            
            # Get document excerpts for analysis
            source_excerpt = self._get_document_excerpt(document)
            target_excerpt = self._get_document_excerpt(candidate)
            
            for rel_type in relationship_types:
                try:
                    relationship = await self._analyze_relationship_with_llm(
                        document, candidate, rel_type, source_excerpt, target_excerpt
                    )
                    
                    if relationship and relationship.confidence_score > 0.5:
                        relationships.append(relationship)
                        
                except Exception as e:
                    logger.warning(
                        "Failed to analyze relationship with LLM",
                        source_doc=document.document_id,
                        target_doc=candidate.document_id,
                        rel_type=rel_type.value,
                        error=str(e)
                    )
        
        return relationships
    
    def _get_document_excerpt(self, document: LegalDocument, max_length: int = 2000) -> str:
        """Get representative excerpt from document for analysis."""
        text = document.text_content or ""
        
        if len(text) <= max_length:
            return text
        
        # Try to get a representative excerpt
        # Prefer beginning and ending of document
        first_half = text[:max_length//2]
        last_half = text[-(max_length//2):]
        
        return f"{first_half}\n\n[...document continues...]\n\n{last_half}"
    
    async def _analyze_relationship_with_llm(
        self,
        source_doc: LegalDocument,
        target_doc: LegalDocument,
        rel_type: RelationshipType,
        source_excerpt: str,
        target_excerpt: str
    ) -> Optional[DocumentRelationship]:
        """Analyze relationship using LLM."""
        prompt = f"""
        {self.relationship_prompts[rel_type]}
        
        Document 1 (Source): {source_doc.document_name}
        {source_excerpt}
        
        Document 2 (Target): {target_doc.document_name}
        {target_excerpt}
        
        Response format:
        Confidence: [0.0-1.0]
        Explanation: [brief explanation]
        Evidence: [specific text snippets if found]
        """
        
        try:
            response = await self.ollama_client.generate_text(
                prompt=prompt,
                model_name="llama3.1:8b",
                max_tokens=500
            )
            
            return self._parse_llm_response(response, source_doc, target_doc, rel_type)
            
        except Exception as e:
            logger.error(
                "LLM relationship analysis failed",
                error=str(e),
                source_doc=source_doc.document_id,
                target_doc=target_doc.document_id
            )
            return None
    
    def _parse_llm_response(
        self,
        response: str,
        source_doc: LegalDocument,
        target_doc: LegalDocument,
        rel_type: RelationshipType
    ) -> Optional[DocumentRelationship]:
        """Parse LLM response into relationship object."""
        try:
            # Extract confidence score
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
            if not confidence_match:
                return None
            
            confidence = float(confidence_match.group(1))
            if confidence < 0.3:  # Minimum threshold
                return None
            
            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n|Evidence:|$)', response, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            # Extract evidence
            evidence_match = re.search(r'Evidence:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            evidence_text = evidence_match.group(1).strip() if evidence_match else explanation
            
            # Create evidence object
            evidence = RelationshipEvidence(
                text_snippet=evidence_text[:500],  # Limit length
                confidence_score=confidence,
                source_location="llm_analysis",
                extraction_method=ExtractionMethod.NLP_MODEL,
                context_before=explanation
            )
            
            return DocumentRelationship(
                source_document_id=source_doc.document_id,
                target_document_id=target_doc.document_id,
                relationship_type=rel_type,
                confidence_score=confidence,
                evidence=[evidence],
                extraction_method=ExtractionMethod.NLP_MODEL
            )
            
        except Exception as e:
            logger.warning(
                "Failed to parse LLM relationship response",
                response=response[:200],
                error=str(e)
            )
            return None


class RelationshipExtractor:
    """
    Main relationship extraction processor for legal documents.
    
    Orchestrates multiple extraction methods to identify and validate
    relationships between legal documents, storing results in Neo4j.
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        document_repository: DocumentRepository,
        relationship_repository: RelationshipRepository,
        websocket_manager: Optional[WebSocketManager] = None
    ):
        """Initialize relationship extractor."""
        self.ollama_client = ollama_client
        self.document_repository = document_repository
        self.relationship_repository = relationship_repository
        self.websocket_manager = websocket_manager
        
        # Initialize extraction engines
        self.pattern_matcher = LegalPatternMatcher()
        self.semantic_analyzer = SemanticRelationshipAnalyzer(ollama_client)
        
        # Configuration
        self.config = get_settings()
        self.extraction_config = self.config.relationship_extraction_settings
        
        # Processing statistics
        self._stats = {
            "documents_processed": 0,
            "relationships_extracted": 0,
            "relationships_stored": 0,
            "extraction_time_total_ms": 0,
            "method_usage": {method.value: 0 for method in ExtractionMethod},
            "relationship_types": {rel_type.value: 0 for rel_type in RelationshipType},
            "error_counts": defaultdict(int)
        }
        
        logger.info(
            "RelationshipExtractor initialized",
            extraction_methods=list(ExtractionMethod),
            relationship_types=list(RelationshipType)
        )
    
    async def extract_relationships_for_case(
        self,
        case_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract relationships for all documents in a case.
        
        Args:
            case_id: Case identifier
            user_id: Optional user ID for progress tracking
            
        Returns:
            Summary of extraction results
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting relationship extraction for case",
                case_id=case_id,
                user_id=user_id
            )
            
            # Send initial progress update
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id, case_id, "Loading case documents...", 5
                )
            
            # Get all documents for the case
            documents = await self.document_repository.get_documents_by_case(case_id)
            
            if len(documents) < 2:
                logger.info(
                    "Insufficient documents for relationship extraction",
                    case_id=case_id,
                    document_count=len(documents)
                )
                return {
                    "case_id": case_id,
                    "documents_processed": len(documents),
                    "relationships_found": 0,
                    "message": "At least 2 documents required for relationship extraction"
                }
            
            # Process relationships for each document
            all_relationships = []
            total_documents = len(documents)
            
            for i, document in enumerate(documents):
                if self.websocket_manager and user_id:
                    progress = 10 + (70 * i / total_documents)
                    await self._send_progress_update(
                        user_id, case_id, 
                        f"Analyzing relationships for {document.document_name}...",
                        int(progress)
                    )
                
                # Extract relationships for this document
                doc_relationships = await self._extract_relationships_for_document(
                    document, documents
                )
                all_relationships.extend(doc_relationships)
            
            # Store relationships in Neo4j
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id, case_id, "Storing relationships in graph database...", 85
                )
            
            stored_count = await self._store_relationships(all_relationships, case_id)
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_processing_stats(len(documents), len(all_relationships), processing_time_ms)
            
            # Send completion update
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id, case_id, 
                    f"Completed - {stored_count} relationships extracted", 100
                )
            
            result = {
                "case_id": case_id,
                "documents_processed": len(documents),
                "relationships_found": len(all_relationships),
                "relationships_stored": stored_count,
                "processing_time_ms": processing_time_ms,
                "relationship_summary": self._create_relationship_summary(all_relationships)
            }
            
            logger.info(
                "Relationship extraction completed for case",
                case_id=case_id,
                **result
            )
            
            return result
            
        except Exception as e:
            self._stats["error_counts"]["extraction_error"] += 1
            
            logger.error(
                "Relationship extraction failed for case",
                case_id=case_id,
                error=str(e),
                exc_info=True
            )
            
            if self.websocket_manager and user_id:
                await self._send_progress_update(
                    user_id, case_id, f"Extraction failed: {str(e)}", -1
                )
            
            raise RelationshipExtractionError(
                f"Failed to extract relationships for case {case_id}: {str(e)}",
                error_code=ErrorCode.RELATIONSHIP_EXTRACTION_FAILED,
                case_id=case_id
            )
    
    async def _extract_relationships_for_document(
        self,
        document: LegalDocument,
        candidate_documents: List[LegalDocument]
    ) -> List[DocumentRelationship]:
        """Extract relationships for a single document."""
        all_relationships = []
        
        try:
            # Method 1: Pattern matching
            pattern_relationships = self.pattern_matcher.extract_relationships(
                document, candidate_documents
            )
            all_relationships.extend(pattern_relationships)
            self._stats["method_usage"][ExtractionMethod.PATTERN_MATCHING.value] += len(pattern_relationships)
            
            # Method 2: Semantic analysis (if enabled)
            if self.extraction_config.get("enable_semantic_analysis", True):
                semantic_relationships = await self.semantic_analyzer.analyze_semantic_relationships(
                    document, candidate_documents
                )
                all_relationships.extend(semantic_relationships)
                self._stats["method_usage"][ExtractionMethod.NLP_MODEL.value] += len(semantic_relationships)
            
            # Method 3: Citation analysis
            citation_relationships = await self._extract_citation_relationships(
                document, candidate_documents
            )
            all_relationships.extend(citation_relationships)
            self._stats["method_usage"][ExtractionMethod.CITATION_ANALYSIS.value] += len(citation_relationships)
            
            # Validate and filter relationships
            validated_relationships = self._validate_relationships(all_relationships)
            
            logger.debug(
                "Relationships extracted for document",
                document_id=document.document_id,
                pattern_count=len(pattern_relationships),
                semantic_count=len(semantic_relationships) if 'semantic_relationships' in locals() else 0,
                citation_count=len(citation_relationships),
                validated_count=len(validated_relationships)
            )
            
            return validated_relationships
            
        except Exception as e:
            logger.error(
                "Failed to extract relationships for document",
                document_id=document.document_id,
                error=str(e)
            )
            return []
    
    async def _extract_citation_relationships(
        self,
        document: LegalDocument,
        candidate_documents: List[LegalDocument]
    ) -> List[DocumentRelationship]:
        """Extract relationships based on citation analysis."""
        relationships = []
        
        # Extract citations from document text
        citations = self._extract_citations(document.text_content or "")
        
        # Find documents that contain these citations
        for candidate in candidate_documents:
            if candidate.document_id == document.document_id:
                continue
            
            candidate_citations = self._extract_citations(candidate.text_content or "")
            
            # Check for citation overlap
            common_citations = set(citations) & set(candidate_citations)
            
            if common_citations:
                confidence = min(0.9, len(common_citations) * 0.3)
                
                evidence = RelationshipEvidence(
                    text_snippet=f"Common citations: {', '.join(list(common_citations)[:3])}",
                    confidence_score=confidence,
                    source_location="citation_analysis",
                    extraction_method=ExtractionMethod.CITATION_ANALYSIS
                )
                
                relationship = DocumentRelationship(
                    source_document_id=document.document_id,
                    target_document_id=candidate.document_id,
                    relationship_type=RelationshipType.RELATES_TO,
                    confidence_score=confidence,
                    evidence=[evidence],
                    extraction_method=ExtractionMethod.CITATION_ANALYSIS
                )
                
                relationships.append(relationship)
        
        return relationships
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text."""
        citations = []
        
        # Use patterns from LegalPatternMatcher
        for pattern in self.pattern_matcher.citation_patterns:
            citations.extend(pattern.findall(text))
        
        return list(set(citations))  # Remove duplicates
    
    def _validate_relationships(
        self,
        relationships: List[DocumentRelationship]
    ) -> List[DocumentRelationship]:
        """Validate and filter relationships based on confidence and rules."""
        validated = []
        min_confidence = self.extraction_config.get("min_confidence", 0.5)
        
        for rel in relationships:
            # Filter by minimum confidence
            if rel.confidence_score < min_confidence:
                continue
            
            # Additional validation rules
            if self._passes_validation_rules(rel):
                rel.validated = True
                validated.append(rel)
        
        return validated
    
    def _passes_validation_rules(self, relationship: DocumentRelationship) -> bool:
        """Apply business rules for relationship validation."""
        # Rule 1: Must have evidence
        if not relationship.evidence:
            return False
        
        # Rule 2: High-confidence relationships with single evidence need strong evidence
        if (relationship.confidence_score > 0.8 and 
            len(relationship.evidence) == 1 and 
            relationship.evidence[0].confidence_score < 0.7):
            return False
        
        # Rule 3: Contradictory relationships need high confidence
        if (relationship.relationship_type == RelationshipType.CONTRADICTS and 
            relationship.confidence_score < 0.8):
            return False
        
        return True
    
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
                    # Update type statistics
                    self._stats["relationship_types"][relationship.relationship_type.value] += 1
            
            self._stats["relationships_stored"] += stored_count
            return stored_count
            
        except Exception as e:
            logger.error(
                "Failed to store relationships",
                case_id=case_id,
                relationship_count=len(relationships),
                error=str(e)
            )
            raise
    
    def _create_relationship_summary(
        self,
        relationships: List[DocumentRelationship]
    ) -> Dict[str, Any]:
        """Create summary of extracted relationships."""
        if not relationships:
            return {}
        
        # Count by type
        type_counts = Counter(rel.relationship_type.value for rel in relationships)
        
        # Count by extraction method
        method_counts = Counter(rel.extraction_method.value for rel in relationships)
        
        # Calculate average confidence
        avg_confidence = sum(rel.confidence_score for rel in relationships) / len(relationships)
        
        # Find high confidence relationships
        high_confidence = [rel for rel in relationships if rel.has_high_confidence]
        
        return {
            "total_relationships": len(relationships),
            "high_confidence_count": len(high_confidence),
            "average_confidence": round(avg_confidence, 3),
            "relationship_types": dict(type_counts),
            "extraction_methods": dict(method_counts),
            "confidence_distribution": {
                "high (>=0.8)": len([r for r in relationships if r.confidence_score >= 0.8]),
                "medium (0.5-0.8)": len([r for r in relationships if 0.5 <= r.confidence_score < 0.8]),
                "low (<0.5)": len([r for r in relationships if r.confidence_score < 0.5])
            }
        }
    
    def _update_processing_stats(
        self,
        documents_processed: int,
        relationships_found: int,
        processing_time_ms: float
    ) -> None:
        """Update processing statistics."""
        self._stats["documents_processed"] += documents_processed
        self._stats["relationships_extracted"] += relationships_found
        self._stats["extraction_time_total_ms"] += processing_time_ms
    
    async def _send_progress_update(
        self,
        user_id: str,
        case_id: str,
        message: str,
        progress_percent: int
    ) -> None:
        """Send progress update via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            update_data = {
                "case_id": case_id,
                "stage": "relationship_extraction",
                "message": message,
                "progress_percent": progress_percent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.websocket_manager.broadcast_to_user(
                user_id,
                "relationship_extraction_progress",
                update_data
            )
            
        except Exception as e:
            logger.warning(
                "Failed to send relationship extraction progress update",
                user_id=user_id,
                case_id=case_id,
                error=str(e)
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get relationship extraction statistics."""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats["documents_processed"] > 0:
            stats["average_relationships_per_document"] = (
                stats["relationships_extracted"] / stats["documents_processed"]
            )
            stats["average_processing_time_ms"] = (
                stats["extraction_time_total_ms"] / stats["documents_processed"]
            )
        else:
            stats["average_relationships_per_document"] = 0
            stats["average_processing_time_ms"] = 0
        
        return stats
    
    async def reprocess_case_relationships(
        self,
        case_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Reprocess relationships for a case, optionally forcing full reprocessing."""
        if force:
            # Clear existing relationships
            await self.relationship_repository.delete_case_relationships(case_id)
        
        return await self.extract_relationships_for_case(case_id)