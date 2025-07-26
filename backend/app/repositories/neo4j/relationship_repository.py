"""
Neo4j Relationship Repository for Legal Document Graph Operations (Phase 2)

This module provides a comprehensive repository for managing document relationships
in a Neo4j graph database. It handles the storage, retrieval, and analysis of
complex legal document relationships with support for confidence scoring,
evidence tracking, and advanced graph queries.

Key Features:
- Document node management with legal metadata
- Relationship creation with evidence and confidence scores
- Advanced Cypher queries for relationship discovery
- Graph traversal for relationship chains and patterns
- Confidence-based filtering and relationship validation
- Case-specific relationship isolation
- Performance optimization for large legal document graphs
- Transaction management for data consistency

Relationship Schema:
- Document nodes with properties (id, name, type, case_id, metadata)
- Relationship edges with properties (type, confidence, evidence, timestamps)
- Case nodes for organizational hierarchy
- Entity nodes for legal entities (parties, courts, etc.)

Graph Patterns:
- Amendment chains: Document -> AMENDS -> Document
- Citation networks: Document -> CITES -> LegalAuthority
- Reference webs: Document -> REFERENCES -> Document
- Supersession chains: Document -> SUPERSEDES -> Document
- Relationship confidence filtering and validation

Architecture Integration:
- Works with RelationshipExtractor for data ingestion
- Provides graph queries for enhanced search
- Supports relationship confidence scoring
- Implements transaction safety for concurrent operations
- Offers performance monitoring and optimization
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum

from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import (
    ServiceUnavailable, TransientError, ClientError,
    DatabaseError as Neo4jDatabaseError
)

from config.settings import get_settings
from ...models.domain.document import LegalDocument
from ...processors.relationship_extractor import (
    DocumentRelationship, RelationshipType, ExtractionMethod,
    RelationshipEvidence
)
from ...utils.logging import get_logger
from ...core.exceptions import (
    DatabaseError, RelationshipError, ErrorCode
)

logger = get_logger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the Neo4j graph."""
    node_id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the Neo4j graph."""
    start_node_id: str
    end_node_id: str
    relationship_type: str
    properties: Dict[str, Any]


@dataclass
class RelationshipPath:
    """Represents a path through the relationship graph."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    total_confidence: float
    path_length: int
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across path."""
        if not self.relationships:
            return 0.0
        confidences = [rel.properties.get('confidence', 0.0) for rel in self.relationships]
        return sum(confidences) / len(confidences)


class QueryBuilder:
    """
    Builder for constructing Cypher queries with safety and optimization.
    
    Provides a fluent interface for building complex graph queries
    with parameter binding and injection protection.
    """
    
    def __init__(self):
        """Initialize query builder."""
        self.query_parts = []
        self.parameters = {}
        self.return_clause = ""
        self.order_clause = ""
        self.limit_clause = ""
    
    def match(self, pattern: str) -> 'QueryBuilder':
        """Add MATCH clause."""
        self.query_parts.append(f"MATCH {pattern}")
        return self
    
    def where(self, condition: str) -> 'QueryBuilder':
        """Add WHERE clause."""
        if any("WHERE" in part for part in self.query_parts):
            self.query_parts.append(f"AND {condition}")
        else:
            self.query_parts.append(f"WHERE {condition}")
        return self
    
    def optional_match(self, pattern: str) -> 'QueryBuilder':
        """Add OPTIONAL MATCH clause."""
        self.query_parts.append(f"OPTIONAL MATCH {pattern}")
        return self
    
    def with_clause(self, expression: str) -> 'QueryBuilder':
        """Add WITH clause."""
        self.query_parts.append(f"WITH {expression}")
        return self
    
    def return_clause(self, expression: str) -> 'QueryBuilder':
        """Set RETURN clause."""
        self.return_clause = f"RETURN {expression}"
        return self
    
    def order_by(self, expression: str) -> 'QueryBuilder':
        """Set ORDER BY clause."""
        self.order_clause = f"ORDER BY {expression}"
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Set LIMIT clause."""
        self.limit_clause = f"LIMIT {count}"
        return self
    
    def add_parameter(self, key: str, value: Any) -> 'QueryBuilder':
        """Add query parameter."""
        self.parameters[key] = value
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build final query and parameters."""
        query_parts = self.query_parts.copy()
        
        if self.return_clause:
            query_parts.append(self.return_clause)
        
        if self.order_clause:
            query_parts.append(self.order_clause)
        
        if self.limit_clause:
            query_parts.append(self.limit_clause)
        
        query = "\n".join(query_parts)
        return query, self.parameters.copy()


class RelationshipRepository:
    """
    Neo4j repository for legal document relationship management.
    
    Provides high-level operations for storing, querying, and analyzing
    document relationships in a Neo4j graph database with optimizations
    for legal document workflows.
    """
    
    def __init__(self, connection_uri: Optional[str] = None):
        """
        Initialize relationship repository.
        
        Args:
            connection_uri: Optional Neo4j connection URI override
        """
        self.config = get_settings()
        self.neo4j_config = self.config.neo4j_settings
        
        # Connection configuration
        self.uri = connection_uri or self.neo4j_config.get("uri", "bolt://localhost:7687")
        self.username = self.neo4j_config.get("username", "neo4j")
        self.password = self.neo4j_config.get("password", "password")
        
        # Connection and session management
        self.driver = None
        self.is_connected = False
        self._connection_lock = asyncio.Lock()
        
        # Performance and monitoring
        self._query_stats = defaultdict(int)
        self._performance_metrics = {
            "queries_executed": 0,
            "total_query_time_ms": 0,
            "relationships_created": 0,
            "relationships_queried": 0,
            "nodes_created": 0,
            "errors": 0
        }
        
        # Query optimization settings
        self.default_timeout = self.neo4j_config.get("query_timeout", 30)
        self.max_retry_attempts = self.neo4j_config.get("max_retries", 3)
        self.batch_size = self.neo4j_config.get("batch_size", 100)
        
        logger.info(
            "RelationshipRepository initialized",
            uri=self.uri,
            username=self.username,
            timeout=self.default_timeout
        )
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        async with self._connection_lock:
            if self.is_connected:
                return
            
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60
                )
                
                # Test connection
                async with self.driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    await result.single()
                
                self.is_connected = True
                
                # Create indexes and constraints
                await self._create_schema()
                
                logger.info("Successfully connected to Neo4j database")
                
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                if self.driver:
                    await self.driver.close()
                    self.driver = None
                raise DatabaseError(
                    f"Neo4j connection failed: {str(e)}",
                    error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                    database_type="neo4j"
                )
    
    async def disconnect(self) -> None:
        """Close Neo4j database connection."""
        async with self._connection_lock:
            if self.driver:
                await self.driver.close()
                self.driver = None
                self.is_connected = False
                logger.info("Disconnected from Neo4j database")
    
    async def _create_schema(self) -> None:
        """Create database schema with indexes and constraints."""
        schema_queries = [
            # Unique constraints
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX document_case_idx IF NOT EXISTS FOR (d:Document) ON (d.case_id)",
            "CREATE INDEX document_name_idx IF NOT EXISTS FOR (d:Document) ON (d.document_name)",
            "CREATE INDEX relationship_confidence_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence)",
            "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relationship_type)",
            "CREATE INDEX relationship_extracted_at_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.extracted_at)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX document_text_search IF NOT EXISTS FOR (d:Document) ON EACH [d.content, d.document_name]",
            "CREATE FULLTEXT INDEX evidence_text_search IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON EACH [r.evidence_text]"
        ]
        
        async with self.driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.debug(f"Executed schema query: {query}")
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Schema query failed: {query}, error: {e}")
    
    async def create_document_node(
        self,
        document: LegalDocument,
        case_id: str
    ) -> bool:
        """
        Create or update a document node in the graph.
        
        Args:
            document: Legal document to create node for
            case_id: Associated case identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = """
            MERGE (c:Case {case_id: $case_id})
            ON CREATE SET 
                c.created_at = datetime(),
                c.document_count = 1
            ON MATCH SET 
                c.document_count = c.document_count + 1,
                c.updated_at = datetime()
            
            MERGE (d:Document {document_id: $document_id})
            ON CREATE SET 
                d.document_name = $document_name,
                d.original_filename = $original_filename,
                d.file_type = $file_type,
                d.file_size = $file_size,
                d.created_at = datetime($created_at),
                d.case_id = $case_id,
                d.content = $content,
                d.page_count = $page_count,
                d.processing_status = $processing_status,
                d.legal_citations = $legal_citations,
                d.section_headers = $section_headers
            ON MATCH SET 
                d.updated_at = datetime(),
                d.processing_status = $processing_status,
                d.content = $content
            
            MERGE (d)-[:BELONGS_TO]->(c)
            
            RETURN d.document_id as document_id
            """
            
            parameters = {
                "case_id": case_id,
                "document_id": document.document_id,
                "document_name": document.document_name,
                "original_filename": document.original_filename,
                "file_type": document.document_type.value,
                "file_size": document.file_size,
                "created_at": document.created_at.isoformat(),
                "content": (document.text_content or "")[:10000],  # Limit content size
                "page_count": document.page_count,
                "processing_status": document.status.value,
                "legal_citations": document.legal_citations,
                "section_headers": document.section_headers
            }
            
            result = await self._execute_query(query, parameters)
            if result:
                self._performance_metrics["nodes_created"] += 1
                logger.debug(f"Created/updated document node: {document.document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create document node: {e}")
            self._performance_metrics["errors"] += 1
        
        return False
    
    async def create_relationship(
        self,
        relationship: DocumentRelationship,
        case_id: str
    ) -> bool:
        """
        Create a relationship between documents.
        
        Args:
            relationship: Relationship to create
            case_id: Associated case identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare evidence data
            evidence_data = []
            for evidence in relationship.evidence:
                evidence_data.append({
                    "text_snippet": evidence.text_snippet[:500],  # Limit length
                    "confidence_score": evidence.confidence_score,
                    "source_location": evidence.source_location,
                    "extraction_method": evidence.extraction_method.value,
                    "pattern_matched": evidence.pattern_matched,
                    "context_before": evidence.context_before[:200] if evidence.context_before else None,
                    "context_after": evidence.context_after[:200] if evidence.context_after else None
                })
            
            query = """
            MATCH (source:Document {document_id: $source_id, case_id: $case_id})
            MATCH (target:Document {document_id: $target_id, case_id: $case_id})
            
            MERGE (source)-[r:RELATES_TO {
                relationship_type: $rel_type,
                source_id: $source_id,
                target_id: $target_id
            }]->(target)
            
            ON CREATE SET 
                r.confidence = $confidence,
                r.extraction_method = $extraction_method,
                r.extracted_at = datetime($extracted_at),
                r.evidence = $evidence,
                r.evidence_count = $evidence_count,
                r.validated = $validated,
                r.validation_score = $validation_score,
                r.created_at = datetime()
            
            ON MATCH SET 
                r.confidence = CASE 
                    WHEN $confidence > r.confidence THEN $confidence 
                    ELSE r.confidence 
                END,
                r.evidence = r.evidence + $evidence,
                r.evidence_count = r.evidence_count + $evidence_count,
                r.updated_at = datetime()
            
            RETURN r.confidence as final_confidence
            """
            
            parameters = {
                "source_id": relationship.source_document_id,
                "target_id": relationship.target_document_id,
                "case_id": case_id,
                "rel_type": relationship.relationship_type.value,
                "confidence": relationship.confidence_score,
                "extraction_method": relationship.extraction_method.value,
                "extracted_at": relationship.extracted_at.isoformat(),
                "evidence": evidence_data,
                "evidence_count": len(evidence_data),
                "validated": relationship.validated,
                "validation_score": relationship.validation_score
            }
            
            result = await self._execute_query(query, parameters)
            if result:
                self._performance_metrics["relationships_created"] += 1
                logger.debug(
                    f"Created relationship: {relationship.source_document_id} "
                    f"-[{relationship.relationship_type.value}]-> "
                    f"{relationship.target_document_id}"
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            self._performance_metrics["errors"] += 1
        
        return False
    
    async def find_document_relationships(
        self,
        document_id: str,
        case_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_confidence: float = 0.5,
        direction: str = "both",  # "outgoing", "incoming", "both"
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find relationships for a specific document.
        
        Args:
            document_id: Document identifier
            case_id: Case identifier
            relationship_types: Optional filter by relationship types
            min_confidence: Minimum confidence threshold
            direction: Relationship direction to query
            limit: Maximum results to return
            
        Returns:
            List of relationship dictionaries
        """
        try:
            builder = QueryBuilder()
            
            if direction == "outgoing":
                pattern = "(d:Document {document_id: $document_id, case_id: $case_id})-[r:RELATES_TO]->(target:Document)"
            elif direction == "incoming":
                pattern = "(source:Document)-[r:RELATES_TO]->(d:Document {document_id: $document_id, case_id: $case_id})"
            else:  # both
                pattern = "(d:Document {document_id: $document_id, case_id: $case_id})-[r:RELATES_TO]-(other:Document)"
            
            builder.match(pattern)
            builder.where("r.confidence >= $min_confidence")
            builder.add_parameter("document_id", document_id)
            builder.add_parameter("case_id", case_id)
            builder.add_parameter("min_confidence", min_confidence)
            
            if relationship_types:
                rel_type_values = [rt.value for rt in relationship_types]
                builder.where("r.relationship_type IN $relationship_types")
                builder.add_parameter("relationship_types", rel_type_values)
            
            if direction == "both":
                builder.return_clause("""
                    r.relationship_type as relationship_type,
                    r.confidence as confidence,
                    r.extraction_method as extraction_method,
                    r.extracted_at as extracted_at,
                    r.evidence as evidence,
                    r.validated as validated,
                    other.document_id as related_document_id,
                    other.document_name as related_document_name,
                    CASE WHEN startNode(r) = d THEN 'outgoing' ELSE 'incoming' END as direction
                """)
            else:
                target_var = "target" if direction == "outgoing" else "source"
                builder.return_clause(f"""
                    r.relationship_type as relationship_type,
                    r.confidence as confidence,
                    r.extraction_method as extraction_method,
                    r.extracted_at as extracted_at,
                    r.evidence as evidence,
                    r.validated as validated,
                    {target_var}.document_id as related_document_id,
                    {target_var}.document_name as related_document_name,
                    '{direction}' as direction
                """)
            
            builder.order_by("r.confidence DESC, r.extracted_at DESC")
            builder.limit(limit)
            
            query, parameters = builder.build()
            results = await self._execute_query(query, parameters, fetch_all=True)
            
            self._performance_metrics["relationships_queried"] += len(results) if results else 0
            
            return results or []
            
        except Exception as e:
            logger.error(f"Failed to find document relationships: {e}")
            self._performance_metrics["errors"] += 1
            return []
    
    async def find_relationship_paths(
        self,
        source_document_id: str,
        target_document_id: str,
        case_id: str,
        max_depth: int = 3,
        min_confidence: float = 0.5
    ) -> List[RelationshipPath]:
        """
        Find relationship paths between two documents.
        
        Args:
            source_document_id: Starting document
            target_document_id: Target document
            case_id: Case identifier
            max_depth: Maximum path length
            min_confidence: Minimum relationship confidence
            
        Returns:
            List of relationship paths
        """
        try:
            query = f"""
            MATCH path = (source:Document {{document_id: $source_id, case_id: $case_id}})
                        -[:RELATES_TO*1..{max_depth}]-
                        (target:Document {{document_id: $target_id, case_id: $case_id}})
            WHERE ALL(r IN relationships(path) WHERE r.confidence >= $min_confidence)
            WITH path, 
                 reduce(conf = 1.0, r IN relationships(path) | conf * r.confidence) as path_confidence,
                 length(path) as path_length
            ORDER BY path_confidence DESC, path_length ASC
            LIMIT 10
            RETURN nodes(path) as nodes, relationships(path) as relationships, 
                   path_confidence, path_length
            """
            
            parameters = {
                "source_id": source_document_id,
                "target_id": target_document_id,
                "case_id": case_id,
                "min_confidence": min_confidence
            }
            
            results = await self._execute_query(query, parameters, fetch_all=True)
            
            paths = []
            for result in results or []:
                # Convert Neo4j nodes and relationships to our data structures
                nodes = [
                    GraphNode(
                        node_id=node["document_id"],
                        labels=list(node.labels),
                        properties=dict(node)
                    )
                    for node in result["nodes"]
                ]
                
                relationships = [
                    GraphRelationship(
                        start_node_id=rel.start_node["document_id"],
                        end_node_id=rel.end_node["document_id"],
                        relationship_type=rel.type,
                        properties=dict(rel)
                    )
                    for rel in result["relationships"]
                ]
                
                path = RelationshipPath(
                    nodes=nodes,
                    relationships=relationships,
                    total_confidence=result["path_confidence"],
                    path_length=result["path_length"]
                )
                
                paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed to find relationship paths: {e}")
            return []
    
    async def get_case_relationship_graph(
        self,
        case_id: str,
        min_confidence: float = 0.5,
        relationship_types: Optional[List[RelationshipType]] = None,
        include_isolated_nodes: bool = False
    ) -> Dict[str, Any]:
        """
        Get the complete relationship graph for a case.
        
        Args:
            case_id: Case identifier
            min_confidence: Minimum relationship confidence
            relationship_types: Optional filter by relationship types
            include_isolated_nodes: Whether to include documents with no relationships
            
        Returns:
            Graph data with nodes and edges
        """
        try:
            # Build base query
            if include_isolated_nodes:
                base_query = """
                MATCH (d:Document {case_id: $case_id})
                OPTIONAL MATCH (d)-[r:RELATES_TO]-(other:Document {case_id: $case_id})
                WHERE r IS NULL OR r.confidence >= $min_confidence
                """
            else:
                base_query = """
                MATCH (d:Document {case_id: $case_id})-[r:RELATES_TO]-(other:Document {case_id: $case_id})
                WHERE r.confidence >= $min_confidence
                """
            
            # Add relationship type filter if specified
            if relationship_types:
                rel_type_values = [rt.value for rt in relationship_types]
                base_query += " AND r.relationship_type IN $relationship_types"
            
            query = base_query + """
            RETURN DISTINCT d.document_id as doc_id, d.document_name as doc_name, 
                   collect(DISTINCT {
                       target_id: other.document_id,
                       target_name: other.document_name,
                       relationship_type: r.relationship_type,
                       confidence: r.confidence,
                       extraction_method: r.extraction_method,
                       validated: r.validated
                   }) as relationships
            ORDER BY d.document_name
            """
            
            parameters = {
                "case_id": case_id,
                "min_confidence": min_confidence
            }
            
            if relationship_types:
                parameters["relationship_types"] = [rt.value for rt in relationship_types]
            
            results = await self._execute_query(query, parameters, fetch_all=True)
            
            # Process results into graph format
            nodes = []
            edges = []
            node_ids = set()
            
            for result in results or []:
                doc_id = result["doc_id"]
                doc_name = result["doc_name"]
                
                # Add node if not already added
                if doc_id not in node_ids:
                    nodes.append({
                        "id": doc_id,
                        "name": doc_name,
                        "type": "document"
                    })
                    node_ids.add(doc_id)
                
                # Add relationships as edges
                for rel in result["relationships"]:
                    if rel["target_id"]:  # Skip null relationships
                        edges.append({
                            "source": doc_id,
                            "target": rel["target_id"],
                            "relationship_type": rel["relationship_type"],
                            "confidence": rel["confidence"],
                            "extraction_method": rel["extraction_method"],
                            "validated": rel["validated"]
                        })
                        
                        # Ensure target node exists
                        if rel["target_id"] not in node_ids:
                            nodes.append({
                                "id": rel["target_id"],
                                "name": rel["target_name"],
                                "type": "document"
                            })
                            node_ids.add(rel["target_id"])
            
            return {
                "case_id": case_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get case relationship graph: {e}")
            return {"case_id": case_id, "nodes": [], "edges": [], "error": str(e)}
    
    async def delete_case_relationships(self, case_id: str) -> bool:
        """
        Delete all relationships for a case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if successful
        """
        try:
            query = """
            MATCH (d:Document {case_id: $case_id})-[r:RELATES_TO]-()
            DELETE r
            
            WITH count(*) as relationship_count
            
            MATCH (d:Document {case_id: $case_id})
            DETACH DELETE d
            
            WITH relationship_count, count(*) as node_count
            
            MATCH (c:Case {case_id: $case_id})
            DELETE c
            
            RETURN relationship_count, node_count
            """
            
            result = await self._execute_query(query, {"case_id": case_id})
            
            if result:
                logger.info(f"Deleted case relationships and nodes for case: {case_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete case relationships: {e}")
        
        return False
    
    async def _execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_all: bool = False,
        timeout: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """
        Execute a Cypher query with error handling and retry logic.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            fetch_all: Whether to fetch all results or just first
            timeout: Query timeout override
            
        Returns:
            Query results or None if failed
        """
        if not self.is_connected:
            await self.connect()
        
        parameters = parameters or {}
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        for attempt in range(self.max_retry_attempts):
            try:
                async with self.driver.session() as session:
                    result = await session.run(query, parameters, timeout=timeout)
                    
                    if fetch_all:
                        records = await result.data()
                        query_result = records
                    else:
                        record = await result.single()
                        query_result = dict(record) if record else None
                    
                    # Update performance metrics
                    execution_time = (time.time() - start_time) * 1000
                    self._performance_metrics["queries_executed"] += 1
                    self._performance_metrics["total_query_time_ms"] += execution_time
                    self._query_stats[query[:50]] += 1
                    
                    logger.debug(
                        f"Query executed successfully in {execution_time:.2f}ms",
                        query_preview=query[:100],
                        attempt=attempt + 1
                    )
                    
                    return query_result
                    
            except (TransientError, ServiceUnavailable) as e:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Query failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {self.max_retry_attempts} attempts: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Query execution failed: {e}", query=query[:200])
                self._performance_metrics["errors"] += 1
                raise
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform repository health check."""
        try:
            if not self.is_connected:
                return {"status": "disconnected", "message": "Not connected to Neo4j"}
            
            # Test basic connectivity
            result = await self._execute_query("RETURN 1 as test")
            
            if result and result.get("test") == 1:
                # Get database info
                db_info = await self._execute_query("""
                    CALL dbms.components() YIELD name, versions, edition
                    RETURN name, versions[0] as version, edition
                """)
                
                return {
                    "status": "healthy",
                    "connected": True,
                    "database_info": db_info,
                    "performance_metrics": self._performance_metrics.copy(),
                    "query_stats": dict(self._query_stats)
                }
            else:
                return {"status": "unhealthy", "message": "Query test failed"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get repository performance metrics."""
        metrics = self._performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics["queries_executed"] > 0:
            metrics["average_query_time_ms"] = (
                metrics["total_query_time_ms"] / metrics["queries_executed"]
            )
        else:
            metrics["average_query_time_ms"] = 0
        
        metrics["query_distribution"] = dict(self._query_stats)
        metrics["connection_status"] = "connected" if self.is_connected else "disconnected"
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._performance_metrics = {
            "queries_executed": 0,
            "total_query_time_ms": 0,
            "relationships_created": 0,
            "relationships_queried": 0,
            "nodes_created": 0,
            "errors": 0
        }
        self._query_stats.clear()
        logger.info("Performance metrics reset")