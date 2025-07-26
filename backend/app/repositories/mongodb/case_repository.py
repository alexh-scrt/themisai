"""
MongoDB repository for legal case management in Patexia Legal AI Chatbot.

This module provides data access layer for legal cases with:
- CRUD operations for case entities
- Complex querying and filtering capabilities
- Case metrics and analytics aggregation
- Transaction support for data consistency
- Performance optimization with proper indexing
- Error handling and data validation
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, WriteError
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from backend.app.core.database import get_mongodb_database
from backend.app.core.exceptions import (
    CaseManagementError,
    DatabaseError,
    ErrorCode,
    raise_case_error,
    raise_database_error
)
from backend.app.models.domain.case import (
    LegalCase,
    CaseStatus,
    CasePriority,
    VisualMarker,
    CaseMetrics
)
from backend.app.utils.logging import (
    get_logger,
    database_logger,
    performance_context
)

logger = get_logger(__name__)


class CaseRepository:
    """
    MongoDB repository for legal case data operations.
    
    Provides comprehensive data access layer for case management including
    CRUD operations, complex queries, analytics, and performance optimization.
    """
    
    def __init__(self):
        """Initialize case repository."""
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = "cases"
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection with lazy initialization."""
        if self._collection is None:
            self._db = await get_mongodb_database()
            self._collection = self._db[self._collection_name]
            
            # Ensure indexes are created
            await self._ensure_indexes()
        
        return self._collection
    
    async def _ensure_indexes(self) -> None:
        """Create necessary indexes for case collection."""
        if not self._collection:
            return
        
        try:
            # Compound index for user cases with sorting
            await self._collection.create_index([
                ("user_id", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="user_cases_updated")
            
            # Unique index for case_id
            await self._collection.create_index([
                ("case_id", ASCENDING)
            ], unique=True, name="case_id_unique")
            
            # Compound index for user and case_id lookups
            await self._collection.create_index([
                ("user_id", ASCENDING),
                ("case_id", ASCENDING)
            ], unique=True, name="user_case_unique")
            
            # Index for status filtering
            await self._collection.create_index([
                ("status", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="status_updated")
            
            # Index for priority filtering
            await self._collection.create_index([
                ("priority", ASCENDING),
                ("updated_at", DESCENDING)
            ], name="priority_updated")
            
            # Text index for case name and summary search
            await self._collection.create_index([
                ("case_name", TEXT),
                ("initial_summary", TEXT),
                ("auto_summary", TEXT)
            ], name="case_text_search")
            
            # Index for creation date range queries
            await self._collection.create_index([
                ("created_at", DESCENDING)
            ], name="created_at_desc")
            
            # Index for tags filtering
            await self._collection.create_index([
                ("tags", ASCENDING)
            ], name="tags_filter")
            
            logger.debug("Case collection indexes ensured")
            
        except Exception as e:
            logger.warning(f"Failed to create case indexes: {e}")
    
    async def create_case(self, case: LegalCase) -> str:
        """
        Create a new case in the database.
        
        Args:
            case: LegalCase domain object to create
            
        Returns:
            Case ID of the created case
            
        Raises:
            CaseManagementError: If case creation fails
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_create_case", case_id=case.case_id):
                # Convert case to document format
                case_doc = self._case_to_document(case)
                
                # Insert the case document
                result = await collection.insert_one(case_doc)
                
                if not result.inserted_id:
                    raise_database_error(
                        "Failed to insert case document",
                        database_type="mongodb",
                        operation="create_case",
                        collection_name=self._collection_name
                    )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="insert_one",
                    collection=self._collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Case created successfully",
                    case_id=case.case_id,
                    user_id=case.user_id,
                    case_name=case.case_name
                )
                
                return case.case_id
                
        except DuplicateKeyError:
            raise_case_error(
                f"Case with ID {case.case_id} already exists",
                case_id=case.case_id,
                user_id=case.user_id,
                error_code=ErrorCode.CASE_DUPLICATE_NAME
            )
        except Exception as e:
            raise_database_error(
                f"Failed to create case {case.case_id}: {e}",
                database_type="mongodb",
                operation="create_case",
                collection_name=self._collection_name
            )
    
    async def get_case_by_id(self, case_id: str, user_id: Optional[str] = None) -> Optional[LegalCase]:
        """
        Get a case by its ID.
        
        Args:
            case_id: Case identifier
            user_id: Optional user ID for access control
            
        Returns:
            LegalCase object if found, None otherwise
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_case", case_id=case_id):
                # Build query filter
                query_filter = {"case_id": case_id}
                if user_id:
                    query_filter["user_id"] = user_id
                
                # Find the case document
                case_doc = await collection.find_one(query_filter)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_one",
                    collection=self._collection_name,
                    result_count=1 if case_doc else 0
                )
                
                if not case_doc:
                    return None
                
                # Convert document to domain object
                return self._document_to_case(case_doc)
                
        except Exception as e:
            raise_database_error(
                f"Failed to get case {case_id}: {e}",
                database_type="mongodb",
                operation="get_case_by_id",
                collection_name=self._collection_name
            )
    
    async def update_case(self, case: LegalCase) -> bool:
        """
        Update an existing case in the database.
        
        Args:
            case: Updated LegalCase domain object
            
        Returns:
            True if case was updated, False if not found
            
        Raises:
            CaseManagementError: If update fails
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_update_case", case_id=case.case_id):
                # Convert case to document format
                case_doc = self._case_to_document(case)
                
                # Remove _id field if present to avoid update conflicts
                case_doc.pop("_id", None)
                
                # Update the case document
                result = await collection.replace_one(
                    {"case_id": case.case_id, "user_id": case.user_id},
                    case_doc
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="replace_one",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                if result.matched_count == 0:
                    return False
                
                logger.info(
                    "Case updated successfully",
                    case_id=case.case_id,
                    modified_count=result.modified_count
                )
                
                return True
                
        except Exception as e:
            raise_database_error(
                f"Failed to update case {case.case_id}: {e}",
                database_type="mongodb",
                operation="update_case",
                collection_name=self._collection_name
            )
    
    async def delete_case(self, case_id: str, user_id: str) -> bool:
        """
        Delete a case from the database.
        
        Args:
            case_id: Case identifier
            user_id: User identifier for access control
            
        Returns:
            True if case was deleted, False if not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_delete_case", case_id=case_id):
                # Delete the case document
                result = await collection.delete_one({
                    "case_id": case_id,
                    "user_id": user_id
                })
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="delete_one",
                    collection=self._collection_name,
                    result_count=result.deleted_count
                )
                
                if result.deleted_count == 0:
                    return False
                
                logger.info(
                    "Case deleted successfully",
                    case_id=case_id,
                    user_id=user_id
                )
                
                return True
                
        except Exception as e:
            raise_database_error(
                f"Failed to delete case {case_id}: {e}",
                database_type="mongodb",
                operation="delete_case",
                collection_name=self._collection_name
            )
    
    async def list_cases(
        self,
        user_id: Optional[str] = None,
        status: Optional[CaseStatus] = None,
        priority: Optional[CasePriority] = None,
        tags: Optional[List[str]] = None,
        search_query: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc"
    ) -> Tuple[List[LegalCase], int]:
        """
        List cases with filtering, sorting, and pagination.
        
        Args:
            user_id: Filter by user ID
            status: Filter by case status
            priority: Filter by case priority
            tags: Filter by case tags (any match)
            search_query: Search in case names and summaries
            created_after: Filter cases created after this date
            created_before: Filter cases created before this date
            limit: Maximum number of cases to return
            offset: Number of cases to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Tuple of (cases_list, total_count)
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_list_cases"):
                # Build query filter
                query_filter = {}
                
                if user_id:
                    query_filter["user_id"] = user_id
                
                if status:
                    query_filter["status"] = status.value
                
                if priority:
                    query_filter["priority"] = priority.value
                
                if tags:
                    query_filter["tags"] = {"$in": tags}
                
                if search_query:
                    query_filter["$text"] = {"$search": search_query}
                
                # Date range filtering
                if created_after or created_before:
                    date_filter = {}
                    if created_after:
                        date_filter["$gte"] = created_after
                    if created_before:
                        date_filter["$lte"] = created_before
                    query_filter["created_at"] = date_filter
                
                # Build sort specification
                sort_direction = DESCENDING if sort_order == "desc" else ASCENDING
                sort_spec = [(sort_by, sort_direction)]
                
                # Add text search score sorting if search query is provided
                if search_query:
                    sort_spec.insert(0, ("score", {"$meta": "textScore"}))
                
                # Get total count
                total_count = await collection.count_documents(query_filter)
                
                # Get paginated results
                cursor = collection.find(query_filter)
                
                if search_query:
                    # Project text search score for sorting
                    cursor = cursor.project({"score": {"$meta": "textScore"}})
                
                cursor = cursor.sort(sort_spec).skip(offset).limit(limit)
                
                case_docs = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_with_pagination",
                    collection=self._collection_name,
                    result_count=len(case_docs)
                )
                
                # Convert documents to domain objects
                cases = [self._document_to_case(doc) for doc in case_docs]
                
                logger.debug(
                    "Cases listed successfully",
                    total_count=total_count,
                    returned_count=len(cases),
                    offset=offset,
                    limit=limit
                )
                
                return cases, total_count
                
        except Exception as e:
            raise_database_error(
                f"Failed to list cases: {e}",
                database_type="mongodb",
                operation="list_cases",
                collection_name=self._collection_name
            )
    
    async def get_cases_by_status(self, status: CaseStatus, user_id: Optional[str] = None) -> List[LegalCase]:
        """
        Get all cases with a specific status.
        
        Args:
            status: Case status to filter by
            user_id: Optional user ID for filtering
            
        Returns:
            List of cases with the specified status
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_cases_by_status", status=status.value):
                query_filter = {"status": status.value}
                if user_id:
                    query_filter["user_id"] = user_id
                
                cursor = collection.find(query_filter).sort("updated_at", DESCENDING)
                case_docs = await cursor.to_list(length=None)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="find_by_status",
                    collection=self._collection_name,
                    result_count=len(case_docs)
                )
                
                return [self._document_to_case(doc) for doc in case_docs]
                
        except Exception as e:
            raise_database_error(
                f"Failed to get cases by status {status.value}: {e}",
                database_type="mongodb",
                operation="get_cases_by_status",
                collection_name=self._collection_name
            )
    
    async def update_case_metrics(self, case_id: str, user_id: str, metrics: CaseMetrics) -> bool:
        """
        Update case metrics without affecting other fields.
        
        Args:
            case_id: Case identifier
            user_id: User identifier for access control
            metrics: Updated case metrics
            
        Returns:
            True if update successful, False if case not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_update_case_metrics", case_id=case_id):
                # Convert metrics to document format
                metrics_doc = metrics.to_dict()
                
                # Update only the metrics field
                result = await collection.update_one(
                    {"case_id": case_id, "user_id": user_id},
                    {
                        "$set": {
                            "metrics": metrics_doc,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="update_metrics",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to update case metrics for {case_id}: {e}",
                database_type="mongodb",
                operation="update_case_metrics",
                collection_name=self._collection_name
            )
    
    async def add_document_to_case(self, case_id: str, user_id: str, document_id: str) -> bool:
        """
        Add a document ID to a case's document list.
        
        Args:
            case_id: Case identifier
            user_id: User identifier for access control
            document_id: Document ID to add
            
        Returns:
            True if document was added, False if case not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_add_document_to_case", case_id=case_id):
                # Use $addToSet to avoid duplicates
                result = await collection.update_one(
                    {"case_id": case_id, "user_id": user_id},
                    {
                        "$addToSet": {"document_ids": document_id},
                        "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="add_document",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to add document {document_id} to case {case_id}: {e}",
                database_type="mongodb",
                operation="add_document_to_case",
                collection_name=self._collection_name
            )
    
    async def remove_document_from_case(self, case_id: str, user_id: str, document_id: str) -> bool:
        """
        Remove a document ID from a case's document list.
        
        Args:
            case_id: Case identifier
            user_id: User identifier for access control
            document_id: Document ID to remove
            
        Returns:
            True if document was removed, False if case not found
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_remove_document_from_case", case_id=case_id):
                result = await collection.update_one(
                    {"case_id": case_id, "user_id": user_id},
                    {
                        "$pull": {"document_ids": document_id},
                        "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}
                    }
                )
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="remove_document",
                    collection=self._collection_name,
                    result_count=result.matched_count
                )
                
                return result.matched_count > 0
                
        except Exception as e:
            raise_database_error(
                f"Failed to remove document {document_id} from case {case_id}: {e}",
                database_type="mongodb",
                operation="remove_document_from_case",
                collection_name=self._collection_name
            )
    
    async def get_case_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get case statistics and analytics.
        
        Args:
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary containing case statistics
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_case_statistics"):
                # Build match stage for aggregation
                match_stage = {}
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
                            "total_cases": {"$sum": 1},
                            "status_counts": {
                                "$push": "$status"
                            },
                            "priority_counts": {
                                "$push": "$priority"
                            },
                            "total_documents": {
                                "$sum": {"$size": {"$ifNull": ["$document_ids", []]}}
                            },
                            "avg_documents_per_case": {
                                "$avg": {"$size": {"$ifNull": ["$document_ids", []]}}
                            },
                            "total_processed_docs": {
                                "$sum": "$metrics.processed_documents"
                            },
                            "total_failed_docs": {
                                "$sum": "$metrics.failed_documents"
                            },
                            "total_search_queries": {
                                "$sum": "$metrics.total_search_queries"
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "processing_success_rate": {
                                "$cond": {
                                    "if": {"$gt": ["$total_documents", 0]},
                                    "then": {
                                        "$divide": ["$total_processed_docs", "$total_documents"]
                                    },
                                    "else": 0
                                }
                            }
                        }
                    }
                ])
                
                # Execute aggregation
                cursor = collection.aggregate(pipeline)
                results = await cursor.to_list(length=1)
                
                if not results:
                    return {
                        "total_cases": 0,
                        "active_cases": 0,
                        "completed_cases": 0,
                        "archived_cases": 0,
                        "cases_with_errors": 0,
                        "total_documents": 0,
                        "average_documents_per_case": 0.0,
                        "processing_success_rate": 0.0,
                        "status_breakdown": {},
                        "priority_breakdown": {}
                    }
                
                stats = results[0]
                
                # Count status breakdown
                status_breakdown = {}
                for status in stats.get("status_counts", []):
                    status_breakdown[status] = status_breakdown.get(status, 0) + 1
                
                # Count priority breakdown
                priority_breakdown = {}
                for priority in stats.get("priority_counts", []):
                    priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                
                # Get recent activity
                recent_cursor = collection.find(
                    match_stage,
                    {"case_id": 1, "case_name": 1, "updated_at": 1, "status": 1}
                ).sort("updated_at", DESCENDING).limit(10)
                
                recent_activity = await recent_cursor.to_list(length=10)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="aggregate_statistics",
                    collection=self._collection_name,
                    result_count=1
                )
                
                return {
                    "total_cases": stats.get("total_cases", 0),
                    "active_cases": status_breakdown.get("active", 0),
                    "completed_cases": status_breakdown.get("complete", 0),
                    "archived_cases": status_breakdown.get("archived", 0),
                    "cases_with_errors": status_breakdown.get("error", 0),
                    "total_documents": stats.get("total_documents", 0),
                    "total_processed_documents": stats.get("total_processed_docs", 0),
                    "total_failed_documents": stats.get("total_failed_docs", 0),
                    "total_search_queries": stats.get("total_search_queries", 0),
                    "average_documents_per_case": round(stats.get("avg_documents_per_case", 0.0), 2),
                    "processing_success_rate": round(stats.get("processing_success_rate", 0.0), 3),
                    "status_breakdown": status_breakdown,
                    "priority_breakdown": priority_breakdown,
                    "recent_activity": [
                        {
                            "case_id": activity["case_id"],
                            "case_name": activity["case_name"],
                            "action": "updated",
                            "timestamp": activity["updated_at"]
                        }
                        for activity in recent_activity
                    ]
                }
                
        except Exception as e:
            raise_database_error(
                f"Failed to get case statistics: {e}",
                database_type="mongodb",
                operation="get_case_statistics",
                collection_name=self._collection_name
            )
    
    async def search_cases_by_text(self, search_query: str, user_id: Optional[str] = None, limit: int = 20) -> List[LegalCase]:
        """
        Search cases using full-text search.
        
        Args:
            search_query: Text to search for
            user_id: Optional user ID for filtering
            limit: Maximum number of results
            
        Returns:
            List of matching cases ordered by relevance
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_search_cases", query=search_query):
                # Build query filter
                query_filter = {"$text": {"$search": search_query}}
                if user_id:
                    query_filter["user_id"] = user_id
                
                # Execute text search with score projection
                cursor = collection.find(
                    query_filter,
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
                
                case_docs = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="text_search",
                    collection=self._collection_name,
                    result_count=len(case_docs)
                )
                
                return [self._document_to_case(doc) for doc in case_docs]
                
        except Exception as e:
            raise_database_error(
                f"Failed to search cases: {e}",
                database_type="mongodb",
                operation="search_cases_by_text",
                collection_name=self._collection_name
            )
    
    def _case_to_document(self, case: LegalCase) -> Dict[str, Any]:
        """Convert LegalCase domain object to MongoDB document."""
        return {
            "case_id": case.case_id,
            "user_id": case.user_id,
            "case_name": case.case_name,
            "initial_summary": case.initial_summary,
            "auto_summary": case.auto_summary,
            "status": case.status.value,
            "priority": case.priority.value,
            "visual_marker": {
                "color": case.visual_marker.color,
                "icon": case.visual_marker.icon
            },
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
            "tags": list(case.tags),
            "metadata": case.metadata,
            "metrics": case.metrics.to_dict(),
            "document_ids": list(case.document_ids)
        }
    
    def _document_to_case(self, doc: Dict[str, Any]) -> LegalCase:
        """Convert MongoDB document to LegalCase domain object."""
        # Parse visual marker
        marker_data = doc.get("visual_marker", {})
        visual_marker = VisualMarker(
            color=marker_data.get("color", VisualMarker.COLORS[0]),
            icon=marker_data.get("icon", VisualMarker.ICONS[0])
        )
        
        # Parse timestamps
        created_at = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(doc["updated_at"].replace("Z", "+00:00"))
        
        # Create case object using from_dict method
        case_data = {
            "case_id": doc["case_id"],
            "user_id": doc["user_id"],
            "case_name": doc["case_name"],
            "initial_summary": doc["initial_summary"],
            "auto_summary": doc.get("auto_summary"),
            "status": doc.get("status", "draft"),
            "priority": doc.get("priority", "medium"),
            "visual_marker": marker_data,
            "created_at": doc["created_at"],
            "updated_at": doc["updated_at"],
            "tags": doc.get("tags", []),
            "metadata": doc.get("metadata", {}),
            "metrics": doc.get("metrics", {}),
            "document_ids": doc.get("document_ids", [])
        }
        
        return LegalCase.from_dict(case_data)


# Singleton instance for dependency injection
_case_repository: Optional[CaseRepository] = None


def get_case_repository() -> CaseRepository:
    """
    Get the singleton case repository instance.
    
    Returns:
        CaseRepository instance
    """
    global _case_repository
    if _case_repository is None:
        _case_repository = CaseRepository()
    return _case_repository