"""
Weaviate repository for vector storage and semantic search in Patexia Legal AI Chatbot.

This module provides data access layer for vector operations with:
- Per-case collection management for document isolation
- Vector storage and retrieval for document chunks
- Hybrid search combining semantic and keyword search
- Advanced filtering and ranking capabilities
- Batch operations for efficient processing
- Performance optimization and error handling
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

import weaviate
from weaviate.exceptions import WeaviateBaseError

from backend.app.core.database import get_weaviate_client
from backend.app.core.exceptions import (
    DatabaseError,
    SearchError,
    ErrorCode,
    raise_database_error,
    raise_search_error
)
from backend.app.models.domain.document import DocumentChunk, DocumentType
from backend.app.utils.logging import (
    get_logger,
    database_logger,
    performance_context
)

logger = get_logger(__name__)


class VectorRepository:
    """
    Weaviate repository for vector storage and semantic search operations.
    
    Provides comprehensive vector database operations including per-case collection
    management, hybrid search capabilities, and optimized batch operations for
    legal document processing and retrieval.
    """
    
    def __init__(self):
        """Initialize vector repository."""
        self._client: Optional[weaviate.Client] = None
        self._case_collections: Dict[str, str] = {}  # case_id -> collection_name mapping
    
    async def _get_client(self) -> weaviate.Client:
        """Get Weaviate client with lazy initialization."""
        if self._client is None:
            self._client = await get_weaviate_client()
        return self._client
    
    def _get_collection_name(self, case_id: str) -> str:
        """
        Get collection name for a case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Weaviate collection name
        """
        if case_id not in self._case_collections:
            # Convert case ID to valid Weaviate class name
            collection_name = f"LegalDocument_{case_id.replace('-', '_').replace('.', '_')}"
            self._case_collections[case_id] = collection_name
        
        return self._case_collections[case_id]
    
    async def create_case_collection(self, case_id: str) -> bool:
        """
        Create a new collection for a legal case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if collection was created, False if already exists
            
        Raises:
            DatabaseError: If collection creation fails
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_create_collection", case_id=case_id):
                # Check if collection already exists
                try:
                    existing_schema = client.schema.get(collection_name)
                    if existing_schema:
                        logger.info(f"Collection {collection_name} already exists")
                        return False
                except:
                    # Collection doesn't exist, proceed with creation
                    pass
                
                # Define schema for legal document chunks
                schema = {
                    "class": collection_name,
                    "description": f"Legal document chunks for case {case_id}",
                    "vectorizer": "none",  # We provide our own vectors
                    "moduleConfig": {
                        "generative-openai": {
                            "model": "gpt-3.5-turbo"
                        }
                    },
                    "properties": [
                        {
                            "name": "chunk_id",
                            "dataType": ["string"],
                            "description": "Unique chunk identifier",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "document_id",
                            "dataType": ["string"],
                            "description": "Parent document identifier",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "document_name",
                            "dataType": ["string"],
                            "description": "Document display name",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "case_id",
                            "dataType": ["string"],
                            "description": "Case identifier",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "user_id",
                            "dataType": ["string"],
                            "description": "User who uploaded the document",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": True
                                }
                            }
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Text content of the chunk",
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": False
                                }
                            }
                        },
                        {
                            "name": "chunk_index",
                            "dataType": ["int"],
                            "description": "Index of this chunk within the document"
                        },
                        {
                            "name": "start_char",
                            "dataType": ["int"],
                            "description": "Starting character position in document"
                        },
                        {
                            "name": "end_char",
                            "dataType": ["int"],
                            "description": "Ending character position in document"
                        },
                        {
                            "name": "chunk_size",
                            "dataType": ["int"],
                            "description": "Size of this text chunk in characters"
                        },
                        {
                            "name": "section_title",
                            "dataType": ["string"],
                            "description": "Section header if available"
                        },
                        {
                            "name": "page_number",
                            "dataType": ["int"],
                            "description": "Page number if available"
                        },
                        {
                            "name": "paragraph_number",
                            "dataType": ["int"],
                            "description": "Paragraph number if available"
                        },
                        {
                            "name": "legal_citations",
                            "dataType": ["string[]"],
                            "description": "Legal citations found in this chunk"
                        },
                        {
                            "name": "file_type",
                            "dataType": ["string"],
                            "description": "Original document file type"
                        },
                        {
                            "name": "created_at",
                            "dataType": ["date"],
                            "description": "Chunk creation timestamp"
                        },
                        {
                            "name": "processing_metadata",
                            "dataType": ["object"],
                            "description": "Processing metadata and context"
                        }
                    ]
                }
                
                # Create the collection
                client.schema.create_class(schema)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="create_collection",
                    collection=collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Weaviate collection created",
                    case_id=case_id,
                    collection_name=collection_name
                )
                
                return True
                
        except WeaviateBaseError as e:
            raise_database_error(
                f"Failed to create Weaviate collection for case {case_id}: {e}",
                database_type="weaviate",
                operation="create_collection",
                collection_name=collection_name
            )
        except Exception as e:
            raise_database_error(
                f"Unexpected error creating collection for case {case_id}: {e}",
                database_type="weaviate",
                operation="create_collection",
                collection_name=collection_name
            )
    
    async def delete_case_collection(self, case_id: str) -> bool:
        """
        Delete a collection for a legal case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if collection was deleted, False if not found
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_delete_collection", case_id=case_id):
                # Check if collection exists
                try:
                    client.schema.get(collection_name)
                except:
                    logger.info(f"Collection {collection_name} does not exist")
                    return False
                
                # Delete the collection
                client.schema.delete_class(collection_name)
                
                # Remove from cache
                self._case_collections.pop(case_id, None)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="delete_collection",
                    collection=collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Weaviate collection deleted",
                    case_id=case_id,
                    collection_name=collection_name
                )
                
                return True
                
        except Exception as e:
            logger.warning(f"Failed to delete collection for case {case_id}: {e}")
            return False
    
    async def add_chunks(
        self, 
        case_id: str, 
        chunks: List[DocumentChunk], 
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add document chunks with embeddings to a case collection.
        
        Args:
            case_id: Case identifier
            chunks: List of document chunks to add
            embeddings: Corresponding embedding vectors
            
        Returns:
            List of Weaviate object IDs
            
        Raises:
            DatabaseError: If adding chunks fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_add_chunks", case_id=case_id, count=len(chunks)):
                # Ensure collection exists
                await self.create_case_collection(case_id)
                
                object_ids = []
                
                # Add chunks in batches for better performance
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]
                    
                    # Prepare batch objects
                    with client.batch() as batch:
                        for chunk, embedding in zip(batch_chunks, batch_embeddings):
                            chunk_data = self._chunk_to_weaviate_object(chunk)
                            
                            # Add object with embedding
                            object_id = batch.add_data_object(
                                data_object=chunk_data,
                                class_name=collection_name,
                                vector=embedding
                            )
                            object_ids.append(object_id)
                    
                    # Small delay between batches
                    if i + batch_size < len(chunks):
                        await asyncio.sleep(0.1)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="add_chunks_batch",
                    collection=collection_name,
                    result_count=len(chunks)
                )
                
                logger.info(
                    "Chunks added to Weaviate",
                    case_id=case_id,
                    chunk_count=len(chunks),
                    collection_name=collection_name
                )
                
                return object_ids
                
        except WeaviateBaseError as e:
            raise_database_error(
                f"Failed to add chunks to Weaviate for case {case_id}: {e}",
                database_type="weaviate",
                operation="add_chunks",
                collection_name=collection_name
            )
        except Exception as e:
            raise_database_error(
                f"Unexpected error adding chunks for case {case_id}: {e}",
                database_type="weaviate",
                operation="add_chunks",
                collection_name=collection_name
            )
    
    async def semantic_search(
        self,
        case_id: str,
        query_embedding: List[float],
        limit: int = 15,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            case_id: Case identifier
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            min_score: Minimum similarity score threshold
            filters: Additional filters for the search
            
        Returns:
            List of search results with similarity scores
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_semantic_search", case_id=case_id):
                # Build the query
                query_builder = client.query.get(collection_name, [
                    "chunk_id", "document_id", "document_name", "content",
                    "chunk_index", "start_char", "end_char", "section_title",
                    "page_number", "legal_citations", "file_type", "created_at"
                ])
                
                # Add vector search
                query_builder = query_builder.with_near_vector({
                    "vector": query_embedding,
                    "certainty": min_score
                })
                
                # Add filters if provided
                if filters:
                    where_filter = self._build_where_filter(filters)
                    if where_filter:
                        query_builder = query_builder.with_where(where_filter)
                
                # Add limit
                query_builder = query_builder.with_limit(limit)
                
                # Add additional metadata
                query_builder = query_builder.with_additional(["certainty", "distance"])
                
                # Execute query
                result = query_builder.do()
                
                # Process results
                search_results = []
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        additional = obj.get("_additional", {})
                        search_result = {
                            "chunk_id": obj.get("chunk_id"),
                            "document_id": obj.get("document_id"),
                            "document_name": obj.get("document_name"),
                            "content": obj.get("content"),
                            "chunk_index": obj.get("chunk_index"),
                            "start_char": obj.get("start_char"),
                            "end_char": obj.get("end_char"),
                            "section_title": obj.get("section_title"),
                            "page_number": obj.get("page_number"),
                            "legal_citations": obj.get("legal_citations", []),
                            "file_type": obj.get("file_type"),
                            "created_at": obj.get("created_at"),
                            "similarity_score": additional.get("certainty", 0.0),
                            "distance": additional.get("distance", 1.0)
                        }
                        search_results.append(search_result)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="semantic_search",
                    collection=collection_name,
                    result_count=len(search_results)
                )
                
                logger.debug(
                    "Semantic search completed",
                    case_id=case_id,
                    results_count=len(search_results),
                    min_score=min_score
                )
                
                return search_results
                
        except WeaviateBaseError as e:
            raise_search_error(
                f"Semantic search failed for case {case_id}: {e}",
                case_id=case_id,
                search_type="semantic"
            )
        except Exception as e:
            raise_search_error(
                f"Unexpected error in semantic search for case {case_id}: {e}",
                case_id=case_id,
                search_type="semantic"
            )
    
    async def keyword_search(
        self,
        case_id: str,
        query_text: str,
        limit: int = 15,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search using BM25.
        
        Args:
            case_id: Case identifier
            query_text: Search query text
            limit: Maximum number of results
            filters: Additional filters for the search
            
        Returns:
            List of search results with BM25 scores
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_keyword_search", case_id=case_id):
                # Build the query
                query_builder = client.query.get(collection_name, [
                    "chunk_id", "document_id", "document_name", "content",
                    "chunk_index", "start_char", "end_char", "section_title",
                    "page_number", "legal_citations", "file_type", "created_at"
                ])
                
                # Add BM25 search
                query_builder = query_builder.with_bm25(
                    query=query_text,
                    properties=["content", "section_title"]
                )
                
                # Add filters if provided
                if filters:
                    where_filter = self._build_where_filter(filters)
                    if where_filter:
                        query_builder = query_builder.with_where(where_filter)
                
                # Add limit
                query_builder = query_builder.with_limit(limit)
                
                # Add additional metadata
                query_builder = query_builder.with_additional(["score"])
                
                # Execute query
                result = query_builder.do()
                
                # Process results
                search_results = []
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        additional = obj.get("_additional", {})
                        search_result = {
                            "chunk_id": obj.get("chunk_id"),
                            "document_id": obj.get("document_id"),
                            "document_name": obj.get("document_name"),
                            "content": obj.get("content"),
                            "chunk_index": obj.get("chunk_index"),
                            "start_char": obj.get("start_char"),
                            "end_char": obj.get("end_char"),
                            "section_title": obj.get("section_title"),
                            "page_number": obj.get("page_number"),
                            "legal_citations": obj.get("legal_citations", []),
                            "file_type": obj.get("file_type"),
                            "created_at": obj.get("created_at"),
                            "bm25_score": additional.get("score", 0.0)
                        }
                        search_results.append(search_result)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="keyword_search",
                    collection=collection_name,
                    result_count=len(search_results)
                )
                
                logger.debug(
                    "Keyword search completed",
                    case_id=case_id,
                    query_text=query_text[:50],
                    results_count=len(search_results)
                )
                
                return search_results
                
        except WeaviateBaseError as e:
            raise_search_error(
                f"Keyword search failed for case {case_id}: {e}",
                case_id=case_id,
                search_type="keyword",
                query=query_text
            )
        except Exception as e:
            raise_search_error(
                f"Unexpected error in keyword search for case {case_id}: {e}",
                case_id=case_id,
                search_type="keyword",
                query=query_text
            )
    
    async def hybrid_search(
        self,
        case_id: str,
        query_text: str,
        query_embedding: List[float],
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        limit: int = 15,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            case_id: Case identifier
            query_text: Search query text for keyword search
            query_embedding: Query vector for semantic search
            semantic_weight: Weight for semantic search results (0.0 to 1.0)
            keyword_weight: Weight for keyword search results (0.0 to 1.0)
            limit: Maximum number of results
            min_score: Minimum combined score threshold
            filters: Additional filters for the search
            
        Returns:
            List of search results with combined scores
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_hybrid_search", case_id=case_id):
                # Build the hybrid query
                query_builder = client.query.get(collection_name, [
                    "chunk_id", "document_id", "document_name", "content",
                    "chunk_index", "start_char", "end_char", "section_title",
                    "page_number", "legal_citations", "file_type", "created_at"
                ])
                
                # Add hybrid search with both vector and BM25
                query_builder = query_builder.with_hybrid(
                    query=query_text,
                    vector=query_embedding,
                    alpha=semantic_weight  # 0 = pure BM25, 1 = pure vector
                )
                
                # Add filters if provided
                if filters:
                    where_filter = self._build_where_filter(filters)
                    if where_filter:
                        query_builder = query_builder.with_where(where_filter)
                
                # Add limit
                query_builder = query_builder.with_limit(limit)
                
                # Add additional metadata
                query_builder = query_builder.with_additional([
                    "score", "explainScore"
                ])
                
                # Execute query
                result = query_builder.do()
                
                # Process results
                search_results = []
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        additional = obj.get("_additional", {})
                        combined_score = additional.get("score", 0.0)
                        
                        # Filter by minimum score
                        if combined_score < min_score:
                            continue
                        
                        search_result = {
                            "chunk_id": obj.get("chunk_id"),
                            "document_id": obj.get("document_id"),
                            "document_name": obj.get("document_name"),
                            "content": obj.get("content"),
                            "chunk_index": obj.get("chunk_index"),
                            "start_char": obj.get("start_char"),
                            "end_char": obj.get("end_char"),
                            "section_title": obj.get("section_title"),
                            "page_number": obj.get("page_number"),
                            "legal_citations": obj.get("legal_citations", []),
                            "file_type": obj.get("file_type"),
                            "created_at": obj.get("created_at"),
                            "hybrid_score": combined_score,
                            "score_explanation": additional.get("explainScore")
                        }
                        search_results.append(search_result)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="hybrid_search",
                    collection=collection_name,
                    result_count=len(search_results)
                )
                
                logger.debug(
                    "Hybrid search completed",
                    case_id=case_id,
                    query_text=query_text[:50],
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    results_count=len(search_results)
                )
                
                return search_results
                
        except WeaviateBaseError as e:
            raise_search_error(
                f"Hybrid search failed for case {case_id}: {e}",
                case_id=case_id,
                search_type="hybrid",
                query=query_text
            )
        except Exception as e:
            raise_search_error(
                f"Unexpected error in hybrid search for case {case_id}: {e}",
                case_id=case_id,
                search_type="hybrid",
                query=query_text
            )
    
    async def get_similar_chunks(
        self,
        case_id: str,
        reference_embedding: List[float],
        exclude_chunk_ids: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a reference embedding.
        
        Args:
            case_id: Case identifier
            reference_embedding: Reference vector for similarity
            exclude_chunk_ids: Chunk IDs to exclude from results
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of similar chunks
        """
        filters = {}
        if exclude_chunk_ids:
            filters["exclude_chunk_ids"] = exclude_chunk_ids
        
        return await self.semantic_search(
            case_id=case_id,
            query_embedding=reference_embedding,
            limit=limit,
            min_score=min_score,
            filters=filters
        )
    
    async def delete_chunks_by_document(self, case_id: str, document_id: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            case_id: Case identifier
            document_id: Document identifier
            
        Returns:
            Number of chunks deleted
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_delete_chunks", case_id=case_id, document_id=document_id):
                # First, find all chunks for the document
                query_result = client.query.get(collection_name, ["chunk_id"]).with_where({
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueString": document_id
                }).do()
                
                deleted_count = 0
                if "data" in query_result and "Get" in query_result["data"]:
                    chunks = query_result["data"]["Get"].get(collection_name, [])
                    
                    # Delete chunks in batches
                    with client.batch() as batch:
                        for chunk in chunks:
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id:
                                batch.delete_object(
                                    class_name=collection_name,
                                    where={
                                        "path": ["chunk_id"],
                                        "operator": "Equal",
                                        "valueString": chunk_id
                                    }
                                )
                                deleted_count += 1
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="delete_chunks_by_document",
                    collection=collection_name,
                    result_count=deleted_count
                )
                
                logger.info(
                    "Chunks deleted from Weaviate",
                    case_id=case_id,
                    document_id=document_id,
                    deleted_count=deleted_count
                )
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return 0
    
    async def get_collection_stats(self, case_id: str) -> Dict[str, Any]:
        """
        Get statistics for a case collection.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Dictionary containing collection statistics
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_collection_stats", case_id=case_id):
                # Get total object count
                result = client.query.aggregate(collection_name).with_meta_count().do()
                
                total_objects = 0
                if "data" in result and "Aggregate" in result["data"]:
                    agg_data = result["data"]["Aggregate"].get(collection_name, [])
                    if agg_data:
                        total_objects = agg_data[0].get("meta", {}).get("count", 0)
                
                # Get unique document count
                doc_result = client.query.aggregate(collection_name).with_group_by_filter(
                    ["document_id"]
                ).do()
                
                unique_documents = 0
                if "data" in doc_result and "Aggregate" in doc_result["data"]:
                    agg_data = doc_result["data"]["Aggregate"].get(collection_name, [])
                    unique_documents = len(agg_data)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="get_collection_stats",
                    collection=collection_name,
                    result_count=1
                )
                
                return {
                    "case_id": case_id,
                    "collection_name": collection_name,
                    "total_chunks": total_objects,
                    "unique_documents": unique_documents,
                    "avg_chunks_per_document": round(total_objects / unique_documents, 2) if unique_documents > 0 else 0
                }
                
        except Exception as e:
            logger.warning(f"Failed to get collection stats for case {case_id}: {e}")
            return {
                "case_id": case_id,
                "collection_name": collection_name,
                "total_chunks": 0,
                "unique_documents": 0,
                "avg_chunks_per_document": 0
            }
    
    def _chunk_to_weaviate_object(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Convert DocumentChunk to Weaviate object format.
        
        Args:
            chunk: DocumentChunk domain object
            
        Returns:
            Dictionary formatted for Weaviate storage
        """
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "chunk_size": chunk.chunk_size,
            "section_title": chunk.section_title,
            "page_number": chunk.page_number,
            "paragraph_number": chunk.paragraph_number,
            "legal_citations": chunk.legal_citations,
            "created_at": chunk.created_at.isoformat(),
            "processing_metadata": {
                "embedding_dimensions": chunk.embedding_dimensions,
                "has_embedding": chunk.has_embedding
            }
        }
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Weaviate where filter from filter dictionary.
        
        Args:
            filters: Filter criteria
            
        Returns:
            Weaviate where filter or None if no valid filters
        """
        where_conditions = []
        
        # Document ID filter
        if "document_id" in filters:
            where_conditions.append({
                "path": ["document_id"],
                "operator": "Equal",
                "valueString": filters["document_id"]
            })
        
        # Document IDs filter (multiple documents)
        if "document_ids" in filters and filters["document_ids"]:
            doc_conditions = []
            for doc_id in filters["document_ids"]:
                doc_conditions.append({
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueString": doc_id
                })
            
            if len(doc_conditions) == 1:
                where_conditions.append(doc_conditions[0])
            else:
                where_conditions.append({
                    "operator": "Or",
                    "operands": doc_conditions
                })
        
        # File type filter
        if "file_type" in filters:
            where_conditions.append({
                "path": ["file_type"],
                "operator": "Equal",
                "valueString": filters["file_type"]
            })
        
        # Page number filter
        if "page_number" in filters:
            where_conditions.append({
                "path": ["page_number"],
                "operator": "Equal",
                "valueInt": filters["page_number"]
            })
        
        # Section title filter
        if "section_title" in filters and filters["section_title"]:
            where_conditions.append({
                "path": ["section_title"],
                "operator": "Like",
                "valueString": f"*{filters['section_title']}*"
            })
        
        # Legal citations filter
        if "has_citations" in filters and filters["has_citations"]:
            where_conditions.append({
                "path": ["legal_citations"],
                "operator": "NotEqual",
                "valueString": ""
            })
        
        # Exclude chunk IDs
        if "exclude_chunk_ids" in filters and filters["exclude_chunk_ids"]:
            exclude_conditions = []
            for chunk_id in filters["exclude_chunk_ids"]:
                exclude_conditions.append({
                    "path": ["chunk_id"],
                    "operator": "NotEqual",
                    "valueString": chunk_id
                })
            
            if exclude_conditions:
                where_conditions.extend(exclude_conditions)
        
        # Date range filters
        if "created_after" in filters:
            where_conditions.append({
                "path": ["created_at"],
                "operator": "GreaterThan",
                "valueDate": filters["created_after"].isoformat()
            })
        
        if "created_before" in filters:
            where_conditions.append({
                "path": ["created_at"],
                "operator": "LessThan",
                "valueDate": filters["created_before"].isoformat()
            })
        
        # Combine conditions
        if not where_conditions:
            return None
        elif len(where_conditions) == 1:
            return where_conditions[0]
        else:
            return {
                "operator": "And",
                "operands": where_conditions
            }
    
    async def update_chunk_metadata(
        self,
        case_id: str,
        chunk_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific chunk.
        
        Args:
            case_id: Case identifier
            chunk_id: Chunk identifier
            metadata_updates: Metadata fields to update
            
        Returns:
            True if update successful, False if chunk not found
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_update_chunk_metadata", case_id=case_id, chunk_id=chunk_id):
                # Find the chunk first
                query_result = client.query.get(collection_name, ["chunk_id"]).with_where({
                    "path": ["chunk_id"],
                    "operator": "Equal",
                    "valueString": chunk_id
                }).with_additional(["id"]).do()
                
                if not ("data" in query_result and "Get" in query_result["data"]):
                    return False
                
                chunks = query_result["data"]["Get"].get(collection_name, [])
                if not chunks:
                    return False
                
                # Get the Weaviate object ID
                weaviate_id = chunks[0]["_additional"]["id"]
                
                # Update the object
                client.data_object.update(
                    data_object=metadata_updates,
                    class_name=collection_name,
                    uuid=weaviate_id
                )
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="update_chunk_metadata",
                    collection=collection_name,
                    result_count=1
                )
                
                logger.debug(
                    "Chunk metadata updated",
                    case_id=case_id,
                    chunk_id=chunk_id,
                    updates=list(metadata_updates.keys())
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update chunk metadata for {chunk_id}: {e}")
            return False
    
    async def get_chunks_by_document(
        self,
        case_id: str,
        document_id: str,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            case_id: Case identifier
            document_id: Document identifier
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of chunks for the document
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_get_chunks_by_document", case_id=case_id, document_id=document_id):
                # Build query
                properties = [
                    "chunk_id", "document_id", "content", "chunk_index",
                    "start_char", "end_char", "section_title", "page_number",
                    "legal_citations", "created_at"
                ]
                
                query_builder = client.query.get(collection_name, properties).with_where({
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueString": document_id
                }).with_sort([{
                    "path": ["chunk_index"],
                    "order": "asc"
                }])
                
                # Include embeddings if requested
                if include_embeddings:
                    query_builder = query_builder.with_additional(["vector"])
                
                # Execute query
                result = query_builder.do()
                
                chunks = []
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        chunk_data = {
                            "chunk_id": obj.get("chunk_id"),
                            "document_id": obj.get("document_id"),
                            "content": obj.get("content"),
                            "chunk_index": obj.get("chunk_index"),
                            "start_char": obj.get("start_char"),
                            "end_char": obj.get("end_char"),
                            "section_title": obj.get("section_title"),
                            "page_number": obj.get("page_number"),
                            "legal_citations": obj.get("legal_citations", []),
                            "created_at": obj.get("created_at")
                        }
                        
                        if include_embeddings:
                            additional = obj.get("_additional", {})
                            chunk_data["embedding"] = additional.get("vector", [])
                        
                        chunks.append(chunk_data)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="get_chunks_by_document",
                    collection=collection_name,
                    result_count=len(chunks)
                )
                
                logger.debug(
                    "Chunks retrieved for document",
                    case_id=case_id,
                    document_id=document_id,
                    chunk_count=len(chunks)
                )
                
                return chunks
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    async def search_citations(
        self,
        case_id: str,
        citation_query: str,
        exact_match: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for legal citations within chunks.
        
        Args:
            case_id: Case identifier
            citation_query: Citation text to search for
            exact_match: Whether to require exact citation match
            limit: Maximum number of results
            
        Returns:
            List of chunks containing the citation
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_search_citations", case_id=case_id):
                # Build where filter for citation search
                if exact_match:
                    where_filter = {
                        "path": ["legal_citations"],
                        "operator": "Equal",
                        "valueString": citation_query
                    }
                else:
                    where_filter = {
                        "path": ["legal_citations"],
                        "operator": "Like",
                        "valueString": f"*{citation_query}*"
                    }
                
                # Execute query
                query_builder = client.query.get(collection_name, [
                    "chunk_id", "document_id", "document_name", "content",
                    "chunk_index", "section_title", "page_number", "legal_citations"
                ]).with_where(where_filter).with_limit(limit)
                
                result = query_builder.do()
                
                citation_results = []
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        citation_result = {
                            "chunk_id": obj.get("chunk_id"),
                            "document_id": obj.get("document_id"),
                            "document_name": obj.get("document_name"),
                            "content": obj.get("content"),
                            "chunk_index": obj.get("chunk_index"),
                            "section_title": obj.get("section_title"),
                            "page_number": obj.get("page_number"),
                            "legal_citations": obj.get("legal_citations", []),
                            "matching_citations": [
                                cite for cite in obj.get("legal_citations", [])
                                if citation_query.lower() in cite.lower()
                            ]
                        }
                        citation_results.append(citation_result)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="search_citations",
                    collection=collection_name,
                    result_count=len(citation_results)
                )
                
                logger.debug(
                    "Citation search completed",
                    case_id=case_id,
                    citation_query=citation_query,
                    exact_match=exact_match,
                    results_count=len(citation_results)
                )
                
                return citation_results
                
        except Exception as e:
            logger.error(f"Failed to search citations in case {case_id}: {e}")
            return []
    
    async def backup_collection(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Create a backup of a case collection.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Backup data or None if failed
        """
        client = await self._get_client()
        collection_name = self._get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_backup_collection", case_id=case_id):
                # Get all objects from the collection
                query_builder = client.query.get(collection_name, [
                    "chunk_id", "document_id", "content", "chunk_index",
                    "start_char", "end_char", "section_title", "page_number",
                    "legal_citations", "file_type", "created_at", "processing_metadata"
                ]).with_additional(["vector", "id"]).with_limit(10000)  # Large limit for backup
                
                result = query_builder.do()
                
                backup_data = {
                    "case_id": case_id,
                    "collection_name": collection_name,
                    "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                    "objects": []
                }
                
                if "data" in result and "Get" in result["data"]:
                    objects = result["data"]["Get"].get(collection_name, [])
                    
                    for obj in objects:
                        additional = obj.get("_additional", {})
                        backup_obj = {
                            "properties": {k: v for k, v in obj.items() if k != "_additional"},
                            "vector": additional.get("vector", []),
                            "id": additional.get("id")
                        }
                        backup_data["objects"].append(backup_obj)
                
                logger.info(
                    "Collection backup created",
                    case_id=case_id,
                    object_count=len(backup_data["objects"])
                )
                
                return backup_data
                
        except Exception as e:
            logger.error(f"Failed to backup collection for case {case_id}: {e}")
            return None


# Singleton instance for dependency injection
_vector_repository: Optional[VectorRepository] = None


def get_vector_repository() -> VectorRepository:
    """
    Get the singleton vector repository instance.
    
    Returns:
        VectorRepository instance
    """
    global _vector_repository
    if _vector_repository is None:
        _vector_repository = VectorRepository()
    return _vector_repository