"""
MongoDB repository for legal document management in Patexia Legal AI Chatbot.

This module provides data access layer for legal documents with:
- CRUD operations for document entities with file content handling
- Complex querying and filtering capabilities for legal documents
- Document processing status tracking and metrics
- Batch operations for document management
- Text content and chunk management
- Performance optimization with proper indexing
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, WriteError
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection, AsyncIOMotorGridFSBucket

from backend.app.core.database import get_mongodb_database
from backend.app.core.exceptions import (
    DocumentProcessingError,
    DatabaseError,
    ErrorCode,
    raise_document_error,
    raise_database_error
)
from backend.app.models.domain.document import (
    LegalDocument,
    DocumentType,
    ProcessingStatus,
    DocumentPriority,
    DocumentChunk
)
from backend.app.utils.logging import (
    get_logger,
    database_logger,
    performance_context
)

logger = get_logger(__name__)


class DocumentRepository:
    """
    MongoDB repository for legal document data operations.
    
    Provides comprehensive data access layer for document management including
    CRUD operations, file storage via GridFS, processing status tracking,
    and performance-optimized queries for legal document workflows.
    """
    
    def __init__(self):
        """Initialize document repository."""
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._gridfs: Optional[AsyncIOMotorGridFSBucket] = None
        self._collection_name = "documents"
        self._gridfs_bucket = "document_files"
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection with lazy initialization."""
        if self._collection is None:
            self._db = await get_mongodb_database()
            self._collection = self._db[self._collection_name]
            self._gridfs = AsyncIOMotorGridFSBucket(self._db, bucket_name=self._gridfs_bucket)
            
            # Ensure indexes are created
            await self._ensure_indexes()
        
        return self._collection
    
    async def _get_gridfs(self) -> AsyncIOMotorGridFSBucket:
        """Get GridFS bucket for file storage."""
        if self._gridfs is None:
            await self._get_collection()  # This will initialize GridFS
        return self._gridfs
    
    async def _ensure_indexes(self) -> None:
        """Create necessary indexes for document collection."""
        if not self._collection:
            return
        
        try:
            # Unique index for document_id
            await self._collection.create_index([
                ("document_id", ASCENDING)
            ], unique=True, name="document_id_unique")
            
            # Compound index for case documents with status
            await self._collection.create_index([
                ("case_id", ASCENDING),
                ("status", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="case_status_updated")
            
            # Compound index for user documents
            await self._collection.create_index([
                ("user_id", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="user_documents_updated")
            
            # Index for processing status filtering
            await self._collection.create_index([
                ("status", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="status_updated")
            
            # Index for file type filtering
            await self._collection.create_index([
                ("file_type", ASCENDING)
            ], name="file_type_filter")
            
            # Index for priority filtering
            await self._collection.create_index([
                ("priority", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="priority_updated")
            
            # Text index for document name and content search
            await self._collection.create_index([
                ("document_name", TEXT),
                ("text_content", TEXT)
            ], name="document_text_search")
            
            # Index for creation date range queries
            await self._collection.create_index([
                ("created_at", DESCENDING)
            ], name="created_at_desc")
            
            # Index for file size filtering
            await self._collection.create_index([
                ("file_size", ASCENDING)
            ], name="file_size_filter")
            
            # Compound index for case and document name uniqueness
            await self._collection.create_index([
                ("case_id", ASCENDING),
                ("document_name", ASCENDING)
            ], name="case_document_name")
            
            logger.debug("Document collection indexes ensured")
            
        except Exception as e:
            logger.warning(f"Failed to create document indexes: {e}")
    
    async def create_document(self, document: LegalDocument) -> str:
        """
        Create a new document in the database with file storage.
        
        Args:
            document: LegalDocument domain object to create
            
        Returns:
            Document ID of the created document
            
        Raises:
            DocumentProcessingError: If document creation fails
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_create_document", document_id=document.document_id):
                # Store file content in GridFS
                file_id = await gridfs.upload_from_stream(
                    filename=document.original_filename,
                    source=document.file_content,
                    metadata={
                        "document_id": document.document_id,
                        "case_id": document.case_id,
                        "user_id": document.user_id,
                        "file_type": document.file_type.value,
                        "upload_timestamp": datetime.now(timezone.utc)
                    }
                )
                
                # Convert document to document format (without file content)
                doc_data = self._document_to_document_data(document)
                doc_data["file_id"] = file_id
                
                # Insert the document metadata
                result = await collection.insert_one(doc_data)
                
                if not result.inserted_id:
                    # Clean up GridFS file if document insert failed
                    await gridfs.delete(file_id)
                    raise_database_error(
                        "Failed to insert document metadata",
                        database_type="mongodb",
                        operation="create_document",
                        collection_name=self._collection_name
                    )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="insert_document_with_file",
                    collection=self._collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Document created successfully",
                    document_id=document.document_id,
                    case_id=document.case_id,
                    file_size=document.file_size,
                    file_id=str(file_id)
                )
                
                return document.document_id
                
        except DuplicateKeyError:
            raise_document_error(
                f"Document with ID {document.document_id} already exists",
                document_id=document.document_id,
                document_name=document.document_name
            )
        except Exception as e:
            raise_database_error(
                f"Failed to create document {document.document_id}: {e}",
                database_type="mongodb",
                operation="create_document",
                collection_name=self._collection_name
            )
    
    async def get_document_by_id(
        self, 
        document_id: str, 
        include_file_content: bool = False,
        include_chunks: bool = False
    ) -> Optional[LegalDocument]:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document identifier
            include_file_content: Whether to load file content from GridFS
            include_chunks: Whether to include text chunks
            
        Returns:
            LegalDocument object if found, None otherwise
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_get_document", document_id=document_id):
                # Find the document metadata
                doc_data = await collection.find_one({"document_id": document_id})
                
                if not doc_data:
                    return None
                
                # Load file content if requested
                file_content = b""
                if include_file_content and "file_id" in doc_data:
                    try:
                        file_stream = await gridfs.open_download_stream(doc_data["file_id"])
                        file_content = await file_stream.read()
                    except Exception as e:
                        logger.warning(f"Failed to load file content for {document_id}: {e}")
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_document_with_file" if include_file_content else "find_document",
                    collection=self._collection_name,
                    result_count=1
                )
                
                # Convert to domain object
                return self._document_data_to_document(doc_data, file_content)
                
        except Exception as e:
            raise_database_error(
                f"Failed to get document {document_id}: {e}",
                database_type="mongodb",
                operation="get_document_by_id",
                collection_name=self._collection_name
            )
    
    async def update_document(self, document: LegalDocument, update_file: bool = False) -> bool:
        """
        Update an existing document in the database.
        
        Args:
            document: Updated LegalDocument domain object
            update_file: Whether to update the file content in GridFS
            
        Returns:
            True if document was updated, False if not found
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_update_document", document_id=document.document_id):
                # Convert document to document format
                doc_data = self._document_to_document_data(document)
                
                # Remove fields that shouldn't be updated
                doc_data.pop("_id", None)
                doc_data.pop("file_id", None)  # Don't update file_id reference
                
                # Update file content if requested
                if update_file:
                    # Get current document to find file_id
                    current_doc = await collection.find_one({"document_id": document.document_id})
                    if current_doc and "file_id" in current_doc:
                        # Delete old file
                        await gridfs.delete(current_doc["file_id"])
                    
                    # Upload new file
                    file_id = await gridfs.upload_from_stream(
                        filename=document.original_filename,
                        source=document.file_content,
                        metadata={
                            "document_id": document.document_id,
                            "case_id": document.case_id,
                            "user_id": document.user_id,
                            "file_type": document.file_type.value,
                            "update_timestamp": datetime.now(timezone.utc)
                        }
                    )
                    doc_data["file_id"] = file_id
                
                # Update the document
                result = await collection.replace_one(
                    {"document_id": document.document_id},
                    doc_data
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="update_document_with_file" if update_file else "update_document",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                if result.matched_count == 0:
                    return False
                
                logger.info(
                    "Document updated successfully",
                    document_id=document.document_id,
                    modified_count=result.modified_count,
                    file_updated=update_file
                )
                
                return True
                
        except Exception as e:
            raise_database_error(
                f"Failed to update document {document.document_id}: {e}",
                database_type="mongodb",
                operation="update_document",
                collection_name=self._collection_name
            )
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the database including its file content.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if document was deleted, False if not found
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_delete_document", document_id=document_id):
                # Get document to find file_id
                doc_data = await collection.find_one({"document_id": document_id})
                if not doc_data:
                    return False
                
                # Delete file from GridFS if exists
                if "file_id" in doc_data:
                    try:
                        await gridfs.delete(doc_data["file_id"])
                    except Exception as e:
                        logger.warning(f"Failed to delete file for {document_id}: {e}")
                
                # Delete document metadata
                result = await collection.delete_one({"document_id": document_id})
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="delete_document_with_file",
                    collection=self._collection_name,
                    result_count=result.deleted_count
                )
                
                logger.info(
                    "Document deleted successfully",
                    document_id=document_id
                )
                
                return result.deleted_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to delete document {document_id}: {e}",
                database_type="mongodb",
                operation="delete_document",
                collection_name=self._collection_name
            )
    
    async def list_documents(
        self,
        case_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        file_type: Optional[DocumentType] = None,
        priority: Optional[DocumentPriority] = None,
        search_query: Optional[str] = None,
        has_failures: Optional[bool] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        min_file_size: Optional[int] = None,
        max_file_size: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        include_content: bool = False
    ) -> Tuple[List[LegalDocument], int]:
        """
        List documents with comprehensive filtering, sorting, and pagination.
        
        Args:
            case_id: Filter by case identifier
            user_id: Filter by user identifier
            status: Filter by processing status
            file_type: Filter by document type
            priority: Filter by processing priority
            search_query: Search in document names and content
            has_failures: Filter documents with/without processing failures
            created_after: Filter documents created after this date
            created_before: Filter documents created before this date
            min_file_size: Minimum file size in bytes
            max_file_size: Maximum file size in bytes
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            include_content: Whether to include text content
            
        Returns:
            Tuple of (documents_list, total_count)
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_list_documents"):
                # Build query filter
                query_filter = {}
                
                if case_id:
                    query_filter["case_id"] = case_id
                
                if user_id:
                    query_filter["user_id"] = user_id
                
                if status:
                    query_filter["status"] = status.value
                
                if file_type:
                    query_filter["file_type"] = file_type.value
                
                if priority:
                    query_filter["priority"] = priority.value
                
                if search_query:
                    query_filter["$text"] = {"$search": search_query}
                
                if has_failures is not None:
                    if has_failures:
                        query_filter["$or"] = [
                            {"status": "failed"},
                            {"processing_metadata.error_message": {"$ne": None}}
                        ]
                    else:
                        query_filter["status"] = {"$ne": "failed"}
                        query_filter["processing_metadata.error_message"] = None
                
                # Date range filtering
                if created_after or created_before:
                    date_filter = {}
                    if created_after:
                        date_filter["$gte"] = created_after.isoformat()
                    if created_before:
                        date_filter["$lte"] = created_before.isoformat()
                    query_filter["created_at"] = date_filter
                
                # File size filtering
                if min_file_size is not None or max_file_size is not None:
                    size_filter = {}
                    if min_file_size is not None:
                        size_filter["$gte"] = min_file_size
                    if max_file_size is not None:
                        size_filter["$lte"] = max_file_size
                    query_filter["file_size"] = size_filter
                
                # Build sort specification
                sort_direction = DESCENDING if sort_order == "desc" else ASCENDING
                sort_spec = [(sort_by, sort_direction)]
                
                # Add text search score sorting if search query is provided
                if search_query:
                    sort_spec.insert(0, ("score", {"$meta": "textScore"}))
                
                # Get total count
                total_count = await collection.count_documents(query_filter)
                
                # Build projection
                projection = {}
                if not include_content:
                    projection["text_content"] = 0  # Exclude large text content
                
                if search_query:
                    projection["score"] = {"$meta": "textScore"}
                
                # Get paginated results
                cursor = collection.find(query_filter, projection)
                cursor = cursor.sort(sort_spec).skip(offset).limit(limit)
                
                doc_data_list = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_documents_with_pagination",
                    collection=self._collection_name,
                    result_count=len(doc_data_list)
                )
                
                # Convert to domain objects (without file content)
                documents = [
                    self._document_data_to_document(doc_data, b"") 
                    for doc_data in doc_data_list
                ]
                
                logger.debug(
                    "Documents listed successfully",
                    total_count=total_count,
                    returned_count=len(documents),
                    offset=offset,
                    limit=limit
                )
                
                return documents, total_count
                
        except Exception as e:
            raise_database_error(
                f"Failed to list documents: {e}",
                database_type="mongodb",
                operation="list_documents",
                collection_name=self._collection_name
            )
    
    async def get_documents_by_case(self, case_id: str, include_content: bool = False) -> List[LegalDocument]:
        """
        Get all documents for a specific case.
        
        Args:
            case_id: Case identifier
            include_content: Whether to include text content
            
        Returns:
            List of documents in the case
        """
        documents, _ = await self.list_documents(
            case_id=case_id,
            include_content=include_content,
            limit=1000  # High limit for case documents
        )
        return documents
    
    async def get_documents_by_status(
        self, 
        status: ProcessingStatus, 
        case_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LegalDocument]:
        """
        Get documents with a specific processing status.
        
        Args:
            status: Processing status to filter by
            case_id: Optional case ID to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of documents with the specified status
        """
        documents, _ = await self.list_documents(
            case_id=case_id,
            status=status,
            limit=limit,
            sort_by="updated_at",
            sort_order="asc"  # Process oldest first
        )
        return documents
    
    async def update_processing_status(
        self, 
        document_id: str, 
        status: ProcessingStatus,
        error_message: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update document processing status and metadata.
        
        Args:
            document_id: Document identifier
            status: New processing status
            error_message: Error message if status is failed
            processing_metadata: Updated processing metadata
            
        Returns:
            True if update successful, False if document not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_update_processing_status", document_id=document_id):
                update_fields = {
                    "status": status.value,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                if error_message:
                    update_fields["processing_metadata.error_message"] = error_message
                    update_fields["processing_metadata.error_stage"] = status.value
                
                if processing_metadata:
                    for key, value in processing_metadata.items():
                        update_fields[f"processing_metadata.{key}"] = value
                
                result = await collection.update_one(
                    {"document_id": document_id},
                    {"$set": update_fields}
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="update_processing_status",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to update processing status for {document_id}: {e}",
                database_type="mongodb",
                operation="update_processing_status",
                collection_name=self._collection_name
            )
    
    async def batch_update_status(
        self, 
        document_ids: List[str], 
        status: ProcessingStatus
    ) -> Tuple[int, List[str]]:
        """
        Update processing status for multiple documents.
        
        Args:
            document_ids: List of document identifiers
            status: New processing status
            
        Returns:
            Tuple of (updated_count, failed_document_ids)
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_batch_update_status", count=len(document_ids)):
                result = await collection.update_many(
                    {"document_id": {"$in": document_ids}},
                    {
                        "$set": {
                            "status": status.value,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="batch_update_status",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                # Find which documents failed to update
                updated_docs = await collection.find(
                    {
                        "document_id": {"$in": document_ids},
                        "status": status.value
                    },
                    {"document_id": 1}
                ).to_list(length=None)
                
                updated_ids = {doc["document_id"] for doc in updated_docs}
                failed_ids = [doc_id for doc_id in document_ids if doc_id not in updated_ids]
                
                return result.matched_count, failed_ids
                
        except Exception as e:
            raise_database_error(
                f"Failed to batch update status: {e}",
                database_type="mongodb",
                operation="batch_update_status",
                collection_name=self._collection_name
            )
    
    async def get_document_statistics(
        self, 
        case_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get document statistics and analytics.
        
        Args:
            case_id: Optional case ID to filter statistics
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary containing document statistics
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_document_statistics"):
                # Build match stage for aggregation
                match_stage = {}
                if case_id:
                    match_stage["case_id"] = case_id
                if user_id:
                    match_stage["user_id"] = user_id
                
                # Aggregation pipeline for statistics
                pipeline = []
                
                if match_stage:
                    pipeline.append({"$match": match_stage})
                
                pipeline.extend([
                    {
                        "$group": {
                            "_id": None,
                            "total_documents": {"$sum": 1},
                            "status_counts": {"$push": "$status"},
                            "file_type_counts": {"$push": "$file_type"},
                            "priority_counts": {"$push": "$priority"},
                            "total_file_size": {"$sum": "$file_size"},
                            "total_chunks": {"$sum": "$processing_metadata.chunk_count"},
                            "avg_processing_time": {
                                "$avg": "$processing_metadata.processing_duration_seconds"
                            },
                            "total_processing_time": {
                                "$sum": "$processing_metadata.processing_duration_seconds"
                            }
                        }
                    }
                ])
                
                # Execute aggregation
                cursor = collection.aggregate(pipeline)
                results = await cursor.to_list(length=1)
                
                if not results:
                    return self._empty_statistics()
                
                stats = results[0]
                
                # Count status breakdown
                status_breakdown = {}
                for status in stats.get("status_counts", []):
                    status_breakdown[status] = status_breakdown.get(status, 0) + 1
                
                # Count file type breakdown
                file_type_breakdown = {}
                for file_type in stats.get("file_type_counts", []):
                    file_type_breakdown[file_type] = file_type_breakdown.get(file_type, 0) + 1
                
                # Count priority breakdown
                priority_breakdown = {}
                for priority in stats.get("priority_counts", []):
                    priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                
                # Calculate success rate
                total_docs = stats.get("total_documents", 0)
                completed_docs = status_breakdown.get("completed", 0)
                failed_docs = status_breakdown.get("failed", 0)
                processing_success_rate = completed_docs / total_docs if total_docs > 0 else 0.0
                
                # Get recent activity
                recent_cursor = collection.find(
                    match_stage,
                    {"document_id": 1, "document_name": 1, "updated_at": 1, "status": 1}
                ).sort("updated_at", DESCENDING).limit(10)
                
                recent_activity = await recent_cursor.to_list(length=10)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="aggregate_document_statistics",
                    collection=self._collection_name,
                    result_count=1
                )
                
                return {
                    "total_documents": total_docs,
                    "processed_documents": completed_docs,
                    "failed_documents": failed_docs,
                    "pending_documents": status_breakdown.get("pending", 0),
                    "processing_documents": status_breakdown.get("processing", 0) + 
                                          status_breakdown.get("extracting", 0) + 
                                          status_breakdown.get("chunking", 0) + 
                                          status_breakdown.get("embedding", 0) + 
                                          status_breakdown.get("indexing", 0),
                    "total_chunks": stats.get("total_chunks", 0),
                    "total_file_size_bytes": stats.get("total_file_size", 0),
                    "average_processing_time": round(stats.get("avg_processing_time", 0.0), 2),
                    "processing_success_rate": round(processing_success_rate, 3),
                    "status_breakdown": status_breakdown,
                    "file_type_breakdown": file_type_breakdown,
                    "priority_breakdown": priority_breakdown,
                    "recent_activity": [
                        {
                            "document_id": activity["document_id"],
                            "document_name": activity["document_name"],
                            "action": "status_updated",
                            "status": activity["status"],
                            "timestamp": activity["updated_at"]
                        }
                        for activity in recent_activity
                    ]
                }
                
        except Exception as e:
            raise_database_error(
                f"Failed to get document statistics: {e}",
                database_type="mongodb",
                operation="get_document_statistics",
                collection_name=self._collection_name
            )
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "pending_documents": 0,
            "processing_documents": 0,
            "total_chunks": 0,
            "total_file_size_bytes": 0,
            "average_processing_time": 0.0,
            "processing_success_rate": 0.0,
            "status_breakdown": {},
            "file_type_breakdown": {},
            "priority_breakdown": {},
            "recent_activity": []
        }
    
    def _document_to_document_data(self, document: LegalDocument) -> Dict[str, Any]:
        """Convert LegalDocument domain object to MongoDB document (without file content)."""
        return {
            "document_id": document.document_id,
            "user_id": document.user_id,
            "case_id": document.case_id,
            "document_name": document.document_name,
            "original_filename": document.original_filename,
            "file_type": document.file_type.value,
            "file_size": document.file_size,
            "file_hash": document.file_hash,
            "priority": document.priority.value,
            "status": document.status.value,
            "text_content": document.text_content,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "metadata": document.metadata,
            "processing_metadata": document.processing_metadata.to_dict(),
            "legal_citations": list(document.legal_citations),
            "section_headers": document.section_headers,
            "page_count": document.page_count,
            "chunks": [chunk.to_dict() for chunk in document.chunks]
        }
    
    def _document_data_to_document(self, doc_data: Dict[str, Any], file_content: bytes) -> LegalDocument:
        """Convert MongoDB document to LegalDocument domain object."""
        # Use the domain object's from_dict method for proper conversion
        return LegalDocument.from_dict(doc_data, file_content)
    
    async def search_documents_by_text(
        self, 
        search_query: str, 
        case_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[LegalDocument]:
        """
        Search documents using full-text search.
        
        Args:
            search_query: Text to search for
            case_id: Optional case ID for filtering
            user_id: Optional user ID for filtering
            limit: Maximum number of results
            
        Returns:
            List of matching documents ordered by relevance
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_search_documents", query=search_query):
                # Build query filter
                query_filter = {"$text": {"$search": search_query}}
                if case_id:
                    query_filter["case_id"] = case_id
                if user_id:
                    query_filter["user_id"] = user_id
                
                # Execute text search with score projection
                cursor = collection.find(
                    query_filter,
                    {"score": {"$meta": "textScore"}, "text_content": 0}  # Exclude large content
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
                
                doc_data_list = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="text_search_documents",
                    collection=self._collection_name,
                    result_count=len(doc_data_list)
                )
                
                return [self._document_data_to_document(doc_data, b"") for doc_data in doc_data_list]
                
        except Exception as e:
            raise_database_error(
                f"Failed to search documents: {e}",
                database_type="mongodb",
                operation="search_documents_by_text",
                collection_name=self._collection_name
            )
    
    async def get_documents_by_hash(self, file_hash: str) -> List[LegalDocument]:
        """
        Find documents with the same file hash (duplicates).
        
        Args:
            file_hash: File content hash to search for
            
        Returns:
            List of documents with matching hash
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_documents_by_hash", file_hash=file_hash):
                cursor = collection.find(
                    {"file_hash": file_hash},
                    {"text_content": 0}  # Exclude large content
                ).sort("created_at", ASCENDING)
                
                doc_data_list = await cursor.to_list(length=None)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_documents_by_hash",
                    collection=self._collection_name,
                    result_count=len(doc_data_list)
                )
                
                return [self._document_data_to_document(doc_data, b"") for doc_data in doc_data_list]
                
        except Exception as e:
            raise_database_error(
                f"Failed to get documents by hash: {e}",
                database_type="mongodb",
                operation="get_documents_by_hash",
                collection_name=self._collection_name
            )
    
    async def cleanup_orphaned_files(self) -> int:
        """
        Clean up GridFS files that don't have corresponding document records.
        
        Returns:
            Number of orphaned files cleaned up
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_cleanup_orphaned_files"):
                # Get all file IDs from documents
                document_file_ids = set()
                cursor = collection.find({}, {"file_id": 1})
                async for doc in cursor:
                    if "file_id" in doc:
                        document_file_ids.add(doc["file_id"])
                
                # Get all GridFS file IDs
                gridfs_file_ids = set()
                async for grid_file in gridfs.find():
                    gridfs_file_ids.add(grid_file._id)
                
                # Find orphaned files
                orphaned_files = gridfs_file_ids - document_file_ids
                
                # Delete orphaned files
                cleanup_count = 0
                for file_id in orphaned_files:
                    try:
                        await gridfs.delete(file_id)
                        cleanup_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete orphaned file {file_id}: {e}")
                
                logger.info(f"Cleaned up {cleanup_count} orphaned files")
                
                return cleanup_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return 0
    
    async def get_file_content(self, document_id: str) -> Optional[bytes]:
        """
        Get file content for a document from GridFS.
        
        Args:
            document_id: Document identifier
            
        Returns:
            File content bytes or None if not found
        """
        collection = await self._get_collection()
        gridfs = await self._get_gridfs()
        
        try:
            with performance_context("mongodb_get_file_content", document_id=document_id):
                # Get document to find file_id
                doc_data = await collection.find_one(
                    {"document_id": document_id},
                    {"file_id": 1}
                )
                
                if not doc_data or "file_id" not in doc_data:
                    return None
                
                # Download file content
                file_stream = await gridfs.open_download_stream(doc_data["file_id"])
                file_content = await file_stream.read()
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="download_file_content",
                    collection="gridfs",
                    result_count=1
                )
                
                return file_content
                
        except Exception as e:
            logger.warning(f"Failed to get file content for {document_id}: {e}")
            return None
    
    async def update_text_content(self, document_id: str, text_content: str) -> bool:
        """
        Update the extracted text content for a document.
        
        Args:
            document_id: Document identifier
            text_content: Extracted text content
            
        Returns:
            True if update successful, False if document not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_update_text_content", document_id=document_id):
                result = await collection.update_one(
                    {"document_id": document_id},
                    {
                        "$set": {
                            "text_content": text_content,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="update_text_content",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to update text content for {document_id}: {e}",
                database_type="mongodb",
                operation="update_text_content",
                collection_name=self._collection_name
            )
    
    async def add_chunks_to_document(self, document_id: str, chunks: List[DocumentChunk]) -> bool:
        """
        Add text chunks to a document.
        
        Args:
            document_id: Document identifier
            chunks: List of document chunks to add
            
        Returns:
            True if chunks were added successfully
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_add_chunks", document_id=document_id):
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                
                result = await collection.update_one(
                    {"document_id": document_id},
                    {
                        "$push": {"chunks": {"$each": chunk_dicts}},
                        "$set": {
                            "processing_metadata.chunk_count": len(chunk_dicts),
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="add_chunks_to_document",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to add chunks to document {document_id}: {e}",
                database_type="mongodb",
                operation="add_chunks_to_document",
                collection_name=self._collection_name
            )
    
    async def get_processing_queue(self, status: ProcessingStatus, limit: int = 10) -> List[str]:
        """
        Get document IDs that need processing in priority order.
        
        Args:
            status: Processing status to filter by
            limit: Maximum number of document IDs to return
            
        Returns:
            List of document IDs ordered by priority and creation time
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_processing_queue", status=status.value):
                # Define priority order
                priority_order = {
                    "urgent": 0,
                    "high": 1,
                    "normal": 2,
                    "low": 3
                }
                
                cursor = collection.find(
                    {"status": status.value},
                    {"document_id": 1, "priority": 1, "created_at": 1}
                ).sort([
                    ("priority", ASCENDING),  # Will be converted using priority_order
                    ("created_at", ASCENDING)  # Oldest first within same priority
                ]).limit(limit)
                
                docs = await cursor.to_list(length=limit)
                
                # Sort by actual priority order
                docs.sort(key=lambda x: (
                    priority_order.get(x.get("priority", "normal"), 2),
                    x.get("created_at", "")
                ))
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="get_processing_queue",
                    collection=self._collection_name,
                    result_count=len(docs)
                )
                
                return [doc["document_id"] for doc in docs]
                
        except Exception as e:
            raise_database_error(
                f"Failed to get processing queue: {e}",
                database_type="mongodb",
                operation="get_processing_queue",
                collection_name=self._collection_name
            )


# Singleton instance for dependency injection
_document_repository: Optional[DocumentRepository] = None


def get_document_repository() -> DocumentRepository:
    """
    Get the singleton document repository instance.
    
    Returns:
        DocumentRepository instance
    """
    global _document_repository
    if _document_repository is None:
        _document_repository = DocumentRepository()
    return _document_repository