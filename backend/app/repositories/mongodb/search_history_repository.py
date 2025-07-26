"""
MongoDB repository for search history management in Patexia Legal AI Chatbot.

This module provides data access layer for search history and analytics with:
- CRUD operations for search query tracking
- Search analytics and pattern analysis
- Query suggestion generation based on history
- Performance metrics and timing analysis
- User search behavior tracking
- Popular query identification and trends
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import WriteError
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection

from backend.app.core.database import get_mongodb_database
from backend.app.core.exceptions import (
    DatabaseError,
    ErrorCode,
    raise_database_error
)
from backend.app.utils.logging import (
    get_logger,
    database_logger,
    performance_context
)

logger = get_logger(__name__)


class SearchHistoryRepository:
    """
    MongoDB repository for search history and analytics operations.
    
    Provides comprehensive data access layer for tracking search queries,
    analyzing search patterns, generating suggestions, and providing
    search analytics for legal document discovery optimization.
    """
    
    def __init__(self):
        """Initialize search history repository."""
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._collection_name = "search_history"
    
    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get MongoDB collection with lazy initialization."""
        if self._collection is None:
            self._db = await get_mongodb_database()
            self._collection = self._db[self._collection_name]
            
            # Ensure indexes are created
            await self._ensure_indexes()
        
        return self._collection
    
    async def _ensure_indexes(self) -> None:
        """Create necessary indexes for search history collection."""
        if not self._collection:
            return
        
        try:
            # Compound index for user search history
            await self._collection.create_index([
                ("user_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="user_history_timestamp")
            
            # Compound index for case search history
            await self._collection.create_index([
                ("case_id", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="case_history_timestamp")
            
            # Index for search type filtering
            await self._collection.create_index([
                ("search_type", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="search_type_timestamp")
            
            # Index for search scope filtering
            await self._collection.create_index([
                ("search_scope", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="search_scope_timestamp")
            
            # Text index for query content search
            await self._collection.create_index([
                ("query", TEXT)
            ], name="query_text_search")
            
            # Index for performance analysis
            await self._collection.create_index([
                ("execution_time_ms", ASCENDING)
            ], name="execution_time_performance")
            
            # Index for result count analysis
            await self._collection.create_index([
                ("results_count", ASCENDING)
            ], name="results_count_analysis")
            
            # Compound index for timestamp range queries
            await self._collection.create_index([
                ("timestamp", DESCENDING)
            ], name="timestamp_range")
            
            # Index for search ID lookups
            await self._collection.create_index([
                ("search_id", ASCENDING)
            ], unique=True, name="search_id_unique")
            
            # TTL index for automatic cleanup (optional)
            # Uncomment to enable automatic deletion of old search history
            # await self._collection.create_index([
            #     ("timestamp", ASCENDING)
            # ], expireAfterSeconds=365*24*60*60, name="search_history_ttl")  # 1 year
            
            logger.debug("Search history collection indexes ensured")
            
        except Exception as e:
            logger.warning(f"Failed to create search history indexes: {e}")
    
    async def record_search(
        self,
        search_id: str,
        user_id: str,
        query: str,
        search_type: str,
        search_scope: str,
        case_id: Optional[str] = None,
        results_count: int = 0,
        execution_time_ms: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a search query in the history.
        
        Args:
            search_id: Unique search identifier
            user_id: User who performed the search
            query: Search query text
            search_type: Type of search (semantic, keyword, hybrid)
            search_scope: Scope of search (case, document, global)
            case_id: Case ID if search was case-specific
            results_count: Number of results returned
            execution_time_ms: Search execution time in milliseconds
            filters: Search filters applied
            metadata: Additional search metadata
            
        Returns:
            True if search was recorded successfully
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_record_search", search_id=search_id):
                # Create search history entry
                search_entry = {
                    "search_id": search_id,
                    "user_id": user_id,
                    "query": query.strip(),
                    "query_normalized": self._normalize_query(query),
                    "search_type": search_type,
                    "search_scope": search_scope,
                    "case_id": case_id,
                    "results_count": results_count,
                    "execution_time_ms": execution_time_ms,
                    "filters": filters or {},
                    "metadata": metadata or {},
                    "timestamp": datetime.now(timezone.utc),
                    "query_length": len(query.strip()),
                    "has_results": results_count > 0
                }
                
                # Insert the search entry
                result = await collection.insert_one(search_entry)
                
                if not result.inserted_id:
                    return False
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="record_search",
                    collection=self._collection_name,
                    result_count=1
                )
                
                logger.debug(
                    "Search recorded successfully",
                    search_id=search_id,
                    user_id=user_id,
                    query_length=len(query),
                    results_count=results_count
                )
                
                return True
                
        except Exception as e:
            raise_database_error(
                f"Failed to record search {search_id}: {e}",
                database_type="mongodb",
                operation="record_search",
                collection_name=self._collection_name
            )
    
    async def get_user_search_history(
        self,
        user_id: str,
        case_id: Optional[str] = None,
        search_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        days_back: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get search history for a user.
        
        Args:
            user_id: User identifier
            case_id: Optional case ID filter
            search_type: Optional search type filter
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            days_back: Limit to searches within this many days
            
        Returns:
            Tuple of (search_history_list, total_count)
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_user_search_history", user_id=user_id):
                # Build query filter
                query_filter = {"user_id": user_id}
                
                if case_id:
                    query_filter["case_id"] = case_id
                
                if search_type:
                    query_filter["search_type"] = search_type
                
                if days_back:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                    query_filter["timestamp"] = {"$gte": cutoff_date}
                
                # Get total count
                total_count = await collection.count_documents(query_filter)
                
                # Get paginated results
                cursor = collection.find(query_filter, {
                    "search_id": 1,
                    "query": 1,
                    "search_type": 1,
                    "search_scope": 1,
                    "case_id": 1,
                    "results_count": 1,
                    "execution_time_ms": 1,
                    "timestamp": 1,
                    "has_results": 1
                }).sort("timestamp", DESCENDING).skip(offset).limit(limit)
                
                search_history = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="get_user_search_history",
                    collection=self._collection_name,
                    result_count=len(search_history)
                )
                
                logger.debug(
                    "User search history retrieved",
                    user_id=user_id,
                    total_count=total_count,
                    returned_count=len(search_history)
                )
                
                return search_history, total_count
                
        except Exception as e:
            raise_database_error(
                f"Failed to get user search history for {user_id}: {e}",
                database_type="mongodb",
                operation="get_user_search_history",
                collection_name=self._collection_name
            )
    
    async def get_popular_queries(
        self,
        case_id: Optional[str] = None,
        search_type: Optional[str] = None,
        days_back: int = 30,
        limit: int = 20,
        min_frequency: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get popular search queries based on frequency.
        
        Args:
            case_id: Optional case ID filter
            search_type: Optional search type filter
            days_back: Look at queries within this many days
            limit: Maximum number of popular queries
            min_frequency: Minimum frequency to be considered popular
            
        Returns:
            List of popular queries with frequency counts
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_popular_queries"):
                # Build match stage
                match_stage = {}
                
                if case_id:
                    match_stage["case_id"] = case_id
                
                if search_type:
                    match_stage["search_type"] = search_type
                
                # Add date filter
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                match_stage["timestamp"] = {"$gte": cutoff_date}
                
                # Aggregation pipeline
                pipeline = [
                    {"$match": match_stage},
                    {
                        "$group": {
                            "_id": "$query_normalized",
                            "query": {"$first": "$query"},
                            "frequency": {"$sum": 1},
                            "avg_results": {"$avg": "$results_count"},
                            "avg_execution_time": {"$avg": "$execution_time_ms"},
                            "success_rate": {
                                "$avg": {"$cond": [{"$gt": ["$results_count", 0]}, 1, 0]}
                            },
                            "last_searched": {"$max": "$timestamp"},
                            "unique_users": {"$addToSet": "$user_id"}
                        }
                    },
                    {
                        "$addFields": {
                            "unique_user_count": {"$size": "$unique_users"}
                        }
                    },
                    {
                        "$match": {
                            "frequency": {"$gte": min_frequency}
                        }
                    },
                    {
                        "$sort": {"frequency": DESCENDING}
                    },
                    {
                        "$limit": limit
                    },
                    {
                        "$project": {
                            "query": 1,
                            "frequency": 1,
                            "avg_results": {"$round": ["$avg_results", 1]},
                            "avg_execution_time": {"$round": ["$avg_execution_time", 1]},
                            "success_rate": {"$round": ["$success_rate", 3]},
                            "last_searched": 1,
                            "unique_user_count": 1
                        }
                    }
                ]
                
                # Execute aggregation
                cursor = collection.aggregate(pipeline)
                popular_queries = await cursor.to_list(length=limit)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="get_popular_queries",
                    collection=self._collection_name,
                    result_count=len(popular_queries)
                )
                
                logger.debug(
                    "Popular queries retrieved",
                    days_back=days_back,
                    query_count=len(popular_queries),
                    min_frequency=min_frequency
                )
                
                return popular_queries
                
        except Exception as e:
            raise_database_error(
                f"Failed to get popular queries: {e}",
                database_type="mongodb",
                operation="get_popular_queries",
                collection_name=self._collection_name
            )
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        limit: int = 5
    ) -> List[str]:
        """
        Get search query suggestions based on history.
        
        Args:
            partial_query: Partial query text for matching
            user_id: Optional user ID for personalized suggestions
            case_id: Optional case ID for context-specific suggestions
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested query strings
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_search_suggestions", partial_query=partial_query):
                partial_normalized = self._normalize_query(partial_query)
                
                # Build aggregation pipeline for suggestions
                match_stages = []
                
                # Text search for partial matches
                if len(partial_normalized) >= 2:
                    match_stages.append({
                        "$match": {
                            "query_normalized": {
                                "$regex": f"^{partial_normalized}",
                                "$options": "i"
                            }
                        }
                    })
                
                # Add user/case context if provided
                context_match = {}
                if user_id:
                    context_match["user_id"] = user_id
                if case_id:
                    context_match["case_id"] = case_id
                
                if context_match:
                    match_stages.append({"$match": context_match})
                
                # Build full pipeline
                pipeline = match_stages + [
                    {
                        "$group": {
                            "_id": "$query_normalized",
                            "query": {"$first": "$query"},
                            "frequency": {"$sum": 1},
                            "avg_results": {"$avg": "$results_count"},
                            "last_used": {"$max": "$timestamp"}
                        }
                    },
                    {
                        "$sort": {
                            "frequency": DESCENDING,
                            "avg_results": DESCENDING,
                            "last_used": DESCENDING
                        }
                    },
                    {
                        "$limit": limit
                    },
                    {
                        "$project": {
                            "query": 1
                        }
                    }
                ]
                
                # Execute aggregation
                cursor = collection.aggregate(pipeline)
                suggestion_docs = await cursor.to_list(length=limit)
                
                suggestions = [doc["query"] for doc in suggestion_docs]
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="get_search_suggestions",
                    collection=self._collection_name,
                    result_count=len(suggestions)
                )
                
                logger.debug(
                    "Search suggestions generated",
                    partial_query=partial_query,
                    suggestion_count=len(suggestions)
                )
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_search_analytics(
        self,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive search analytics.
        
        Args:
            user_id: Optional user ID filter
            case_id: Optional case ID filter
            days_back: Analyze searches within this many days
            
        Returns:
            Dictionary containing search analytics
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_search_analytics"):
                # Build match stage
                match_stage = {}
                
                if user_id:
                    match_stage["user_id"] = user_id
                
                if case_id:
                    match_stage["case_id"] = case_id
                
                # Add date filter
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                match_stage["timestamp"] = {"$gte": cutoff_date}
                
                # Multiple aggregation pipelines for different metrics
                
                # Overall statistics
                overall_pipeline = [
                    {"$match": match_stage},
                    {
                        "$group": {
                            "_id": None,
                            "total_searches": {"$sum": 1},
                            "unique_queries": {"$addToSet": "$query_normalized"},
                            "avg_execution_time": {"$avg": "$execution_time_ms"},
                            "avg_results_per_search": {"$avg": "$results_count"},
                            "successful_searches": {
                                "$sum": {"$cond": [{"$gt": ["$results_count", 0]}, 1, 0]}
                            },
                            "search_types": {"$push": "$search_type"},
                            "search_scopes": {"$push": "$search_scope"}
                        }
                    },
                    {
                        "$addFields": {
                            "unique_query_count": {"$size": "$unique_queries"},
                            "success_rate": {
                                "$divide": ["$successful_searches", "$total_searches"]
                            }
                        }
                    }
                ]
                
                overall_cursor = collection.aggregate(overall_pipeline)
                overall_stats = await overall_cursor.to_list(length=1)
                
                if not overall_stats:
                    return self._empty_analytics()
                
                stats = overall_stats[0]
                
                # Count search types
                search_type_counts = {}
                for search_type in stats.get("search_types", []):
                    search_type_counts[search_type] = search_type_counts.get(search_type, 0) + 1
                
                # Count search scopes
                search_scope_counts = {}
                for search_scope in stats.get("search_scopes", []):
                    search_scope_counts[search_scope] = search_scope_counts.get(search_scope, 0) + 1
                
                # Performance trends (daily breakdown)
                trends_pipeline = [
                    {"$match": match_stage},
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$timestamp"
                                }
                            },
                            "daily_searches": {"$sum": 1},
                            "daily_avg_time": {"$avg": "$execution_time_ms"},
                            "daily_avg_results": {"$avg": "$results_count"}
                        }
                    },
                    {
                        "$sort": {"_id": ASCENDING}
                    }
                ]
                
                trends_cursor = collection.aggregate(trends_pipeline)
                daily_trends = await trends_cursor.to_list(length=None)
                
                # Recent activity
                recent_cursor = collection.find(
                    match_stage,
                    {
                        "search_id": 1,
                        "query": 1,
                        "search_type": 1,
                        "results_count": 1,
                        "execution_time_ms": 1,
                        "timestamp": 1
                    }
                ).sort("timestamp", DESCENDING).limit(10)
                
                recent_searches = await recent_cursor.to_list(length=10)
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="get_search_analytics",
                    collection=self._collection_name,
                    result_count=1
                )
                
                analytics = {
                    "period_days": days_back,
                    "total_searches": stats.get("total_searches", 0),
                    "unique_queries": stats.get("unique_query_count", 0),
                    "average_execution_time_ms": round(stats.get("avg_execution_time", 0.0), 2),
                    "average_results_per_search": round(stats.get("avg_results_per_search", 0.0), 1),
                    "success_rate": round(stats.get("success_rate", 0.0), 3),
                    "search_type_breakdown": search_type_counts,
                    "search_scope_breakdown": search_scope_counts,
                    "daily_trends": [
                        {
                            "date": trend["_id"],
                            "searches": trend["daily_searches"],
                            "avg_time_ms": round(trend["daily_avg_time"], 2),
                            "avg_results": round(trend["daily_avg_results"], 1)
                        }
                        for trend in daily_trends
                    ],
                    "recent_searches": [
                        {
                            "search_id": search["search_id"],
                            "query": search["query"][:50] + "..." if len(search["query"]) > 50 else search["query"],
                            "search_type": search["search_type"],
                            "results_count": search["results_count"],
                            "execution_time_ms": search["execution_time_ms"],
                            "timestamp": search["timestamp"]
                        }
                        for search in recent_searches
                    ]
                }
                
                logger.debug(
                    "Search analytics generated",
                    days_back=days_back,
                    total_searches=analytics["total_searches"],
                    unique_queries=analytics["unique_queries"]
                )
                
                return analytics
                
        except Exception as e:
            raise_database_error(
                f"Failed to get search analytics: {e}",
                database_type="mongodb",
                operation="get_search_analytics",
                collection_name=self._collection_name
            )
    
    async def cleanup_old_searches(self, days_to_keep: int = 365) -> int:
        """
        Clean up old search history entries.
        
        Args:
            days_to_keep: Number of days of history to retain
            
        Returns:
            Number of entries deleted
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_cleanup_old_searches"):
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
                
                result = await collection.delete_many({
                    "timestamp": {"$lt": cutoff_date}
                })
                
                database_logger.query_executed(
                    database_type="mongodb",
                    operation="cleanup_old_searches",
                    collection=self._collection_name,
                    result_count=result.deleted_count
                )
                
                logger.info(
                    "Old search history cleaned up",
                    days_to_keep=days_to_keep,
                    deleted_count=result.deleted_count
                )
                
                return result.deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old searches: {e}")
            return 0
    
    async def get_search_performance_metrics(
        self,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Get search performance metrics for monitoring.
        
        Args:
            days_back: Analyze performance within this many days
            
        Returns:
            Performance metrics dictionary
        """
        collection = await self._get_collection()
        
        try:
            with performance_context("mongodb_get_performance_metrics"):
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                
                # Performance aggregation pipeline
                pipeline = [
                    {
                        "$match": {
                            "timestamp": {"$gte": cutoff_date}
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "total_searches": {"$sum": 1},
                            "avg_execution_time": {"$avg": "$execution_time_ms"},
                            "min_execution_time": {"$min": "$execution_time_ms"},
                            "max_execution_time": {"$max": "$execution_time_ms"},
                            "p50_execution_time": {
                                "$percentile": {
                                    "input": "$execution_time_ms",
                                    "p": [0.5],
                                    "method": "approximate"
                                }
                            },
                            "p95_execution_time": {
                                "$percentile": {
                                    "input": "$execution_time_ms",
                                    "p": [0.95],
                                    "method": "approximate"
                                }
                            },
                            "p99_execution_time": {
                                "$percentile": {
                                    "input": "$execution_time_ms",
                                    "p": [0.99],
                                    "method": "approximate"
                                }
                            },
                            "slow_queries": {
                                "$sum": {
                                    "$cond": [{"$gt": ["$execution_time_ms", 3000]}, 1, 0]
                                }
                            },
                            "zero_result_queries": {
                                "$sum": {
                                    "$cond": [{"$eq": ["$results_count", 0]}, 1, 0]
                                }
                            }
                        }
                    }
                ]
                
                cursor = collection.aggregate(pipeline)
                performance_data = await cursor.to_list(length=1)
                
                if not performance_data:
                    return {
                        "period_days": days_back,
                        "total_searches": 0,
                        "performance_metrics": {},
                        "quality_metrics": {}
                    }
                
                metrics = performance_data[0]
                
                return {
                    "period_days": days_back,
                    "total_searches": metrics.get("total_searches", 0),
                    "performance_metrics": {
                        "avg_execution_time_ms": round(metrics.get("avg_execution_time", 0.0), 2),
                        "min_execution_time_ms": round(metrics.get("min_execution_time", 0.0), 2),
                        "max_execution_time_ms": round(metrics.get("max_execution_time", 0.0), 2),
                        "p50_execution_time_ms": round(metrics.get("p50_execution_time", [0.0])[0], 2),
                        "p95_execution_time_ms": round(metrics.get("p95_execution_time", [0.0])[0], 2),
                        "p99_execution_time_ms": round(metrics.get("p99_execution_time", [0.0])[0], 2)
                    },
                    "quality_metrics": {
                        "slow_query_count": metrics.get("slow_queries", 0),
                        "slow_query_rate": round(
                            metrics.get("slow_queries", 0) / max(metrics.get("total_searches", 1), 1), 3
                        ),
                        "zero_result_count": metrics.get("zero_result_queries", 0),
                        "zero_result_rate": round(
                            metrics.get("zero_result_queries", 0) / max(metrics.get("total_searches", 1), 1), 3
                        )
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                "period_days": days_back,
                "total_searches": 0,
                "performance_metrics": {},
                "quality_metrics": {}
            }
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query for comparison and matching.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Remove common legal stopwords for better matching
        legal_stopwords = {"the", "and", "or", "of", "in", "to", "for", "with", "by", "a", "an"}
        words = normalized.split()
        words = [word for word in words if word not in legal_stopwords or len(words) <= 3]
        
        return " ".join(words)
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            "period_days": 0,
            "total_searches": 0,
            "unique_queries": 0,
            "average_execution_time_ms": 0.0,
            "average_results_per_search": 0.0,
            "success_rate": 0.0,
            "search_type_breakdown": {},
            "search_scope_breakdown": {},
            "daily_trends": [],
            "recent_searches": []
        }


# Singleton instance for dependency injection
_search_history_repository: Optional[SearchHistoryRepository] = None


def get_search_history_repository() -> SearchHistoryRepository:
    """
    Get the singleton search history repository instance.
    
    Returns:
        SearchHistoryRepository instance
    """
    global _search_history_repository
    if _search_history_repository is None:
        _search_history_repository = SearchHistoryRepository()
    return _search_history_repository