"""
Database connection and management system for Patexia Legal AI Chatbot.

This module provides:
- MongoDB connection management with async support
- Weaviate vector database client management
- Neo4j connection setup (Phase 2)
- Database health checking and monitoring
- Connection pooling and lifecycle management
- Graceful shutdown and error handling
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import motor.motor_asyncio
import pymongo
import weaviate
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from weaviate.exceptions import WeaviateBaseError

from .exceptions import (
    DatabaseError,
    ErrorCode,
    raise_database_error
)
from ..utils.logging import (
    database_logger,
    get_logger,
    performance_context
)
from config.settings import get_settings

logger = get_logger(__name__)


class MongoDBManager:
    """
    MongoDB connection and lifecycle management.
    
    Provides async MongoDB operations with connection pooling,
    health checking, and error handling for legal document storage.
    """
    
    def __init__(self):
        """Initialize MongoDB manager."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.is_connected: bool = False
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """
        Establish connection to MongoDB.
        
        Raises:
            DatabaseError: If connection fails
        """
        if self.is_connected:
            return
        
        async with self._connection_lock:
            if self.is_connected:
                return
            
            settings = get_settings()
            
            try:
                with performance_context("mongodb_connection"):
                    # Create MongoDB client with connection options
                    self.client = AsyncIOMotorClient(
                        settings.database.mongodb_uri,
                        serverSelectionTimeoutMS=5000,  # 5 second timeout
                        connectTimeoutMS=5000,
                        maxPoolSize=50,  # Connection pool size
                        minPoolSize=5,
                        maxIdleTimeMS=30000,  # 30 second idle timeout
                        retryWrites=True,
                        retryReads=True
                    )
                    
                    # Get database reference
                    self.database = self.client[settings.database.mongodb_database]
                    
                    # Test connection
                    await self.client.admin.command('ping')
                    
                    self.is_connected = True
                    
                    database_logger.connection_established(
                        database_type="mongodb",
                        database_name=settings.database.mongodb_database
                    )
                    
                    logger.info(
                        "MongoDB connection established",
                        database=settings.database.mongodb_database,
                        uri=settings.database.mongodb_uri.split('@')[-1]  # Hide credentials
                    )
                    
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                database_logger.connection_failed("mongodb", str(e))
                raise_database_error(
                    f"Failed to connect to MongoDB: {e}",
                    database_type="mongodb",
                    operation="connect"
                )
            except Exception as e:
                database_logger.connection_failed("mongodb", str(e))
                raise_database_error(
                    f"Unexpected error connecting to MongoDB: {e}",
                    database_type="mongodb",
                    operation="connect"
                )
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client and self.is_connected:
            try:
                self.client.close()
                self.is_connected = False
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform MongoDB health check.
        
        Returns:
            Health status information
        """
        if not self.is_connected or not self.client:
            return {
                "status": "disconnected",
                "error": "Not connected to MongoDB"
            }
        
        try:
            start_time = time.time()
            result = await self.client.admin.command('ping')
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "database_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
                "collections": stats.get("collections", 0),
                "indexes": stats.get("indexes", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def create_indexes(self) -> None:
        """Create necessary indexes for legal document collections."""
        if not self.database:
            raise_database_error(
                "Database not connected",
                database_type="mongodb",
                operation="create_indexes"
            )
        
        try:
            with performance_context("mongodb_create_indexes"):
                # Cases collection indexes
                await self.database.cases.create_index([
                    ("user_id", pymongo.ASCENDING),
                    ("case_id", pymongo.ASCENDING)
                ], unique=True)
                
                await self.database.cases.create_index([
                    ("user_id", pymongo.ASCENDING),
                    ("created_at", pymongo.DESCENDING)
                ])
                
                await self.database.cases.create_index([
                    ("case_name", pymongo.TEXT)
                ])
                
                # Documents collection indexes
                await self.database.documents.create_index([
                    ("user_id", pymongo.ASCENDING),
                    ("case_id", pymongo.ASCENDING)
                ])
                
                await self.database.documents.create_index([
                    ("case_id", pymongo.ASCENDING),
                    ("metadata.date_added", pymongo.DESCENDING)
                ])
                
                await self.database.documents.create_index([
                    ("metadata.document_name", pymongo.TEXT),
                    ("document", pymongo.TEXT)
                ])
                
                # Search history indexes
                await self.database.search_history.create_index([
                    ("user_id", pymongo.ASCENDING),
                    ("case_id", pymongo.ASCENDING)
                ])
                
                await self.database.search_history.create_index([
                    ("user_id", pymongo.ASCENDING),
                    ("search_queries.timestamp", pymongo.DESCENDING)
                ])
                
                logger.info("MongoDB indexes created successfully")
                
        except Exception as e:
            raise_database_error(
                f"Failed to create MongoDB indexes: {e}",
                database_type="mongodb",
                operation="create_indexes"
            )
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the MongoDB database instance.
        
        Returns:
            AsyncIOMotorDatabase instance
            
        Raises:
            DatabaseError: If not connected
        """
        if not self.database:
            raise_database_error(
                "MongoDB not connected",
                database_type="mongodb",
                operation="get_database"
            )
        return self.database


class WeaviateManager:
    """
    Weaviate vector database connection and management.
    
    Handles vector database operations, schema management,
    and per-case collection isolation for legal documents.
    """
    
    def __init__(self):
        """Initialize Weaviate manager."""
        self.client: Optional[weaviate.Client] = None
        self.is_connected: bool = False
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """
        Establish connection to Weaviate.
        
        Raises:
            DatabaseError: If connection fails
        """
        if self.is_connected:
            return
        
        async with self._connection_lock:
            if self.is_connected:
                return
            
            settings = get_settings()
            
            try:
                with performance_context("weaviate_connection"):
                    # Configure Weaviate client
                    auth_config = None
                    if settings.database.weaviate_api_key:
                        auth_config = weaviate.AuthApiKey(
                            api_key=settings.database.weaviate_api_key
                        )
                    
                    self.client = weaviate.Client(
                        url=settings.database.weaviate_url,
                        auth_client_secret=auth_config,
                        timeout_config=(5, 15),  # (connection, read) timeouts
                    )
                    
                    # Test connection
                    if not self.client.is_ready():
                        raise ConnectionError("Weaviate is not ready")
                    
                    self.is_connected = True
                    
                    database_logger.connection_established(
                        database_type="weaviate",
                        database_name="vector_store"
                    )
                    
                    logger.info(
                        "Weaviate connection established",
                        url=settings.database.weaviate_url
                    )
                    
            except WeaviateBaseError as e:
                database_logger.connection_failed("weaviate", str(e))
                raise_database_error(
                    f"Failed to connect to Weaviate: {e}",
                    database_type="weaviate",
                    operation="connect"
                )
            except Exception as e:
                database_logger.connection_failed("weaviate", str(e))
                raise_database_error(
                    f"Unexpected error connecting to Weaviate: {e}",
                    database_type="weaviate",
                    operation="connect"
                )
    
    async def disconnect(self) -> None:
        """Close Weaviate connection."""
        if self.client and self.is_connected:
            try:
                # Weaviate client doesn't have explicit close method
                self.client = None
                self.is_connected = False
                logger.info("Weaviate connection closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Weaviate health check.
        
        Returns:
            Health status information
        """
        if not self.is_connected or not self.client:
            return {
                "status": "disconnected",
                "error": "Not connected to Weaviate"
            }
        
        try:
            start_time = time.time()
            ready = self.client.is_ready()
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if ready:
                # Get cluster metadata
                meta = self.client.get_meta()
                
                return {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                    "version": meta.get("version", "unknown"),
                    "modules": list(meta.get("modules", {}).keys())
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Weaviate not ready"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def create_case_schema(self, case_id: str) -> None:
        """
        Create Weaviate schema for a specific legal case.
        
        Args:
            case_id: Unique case identifier
            
        Raises:
            DatabaseError: If schema creation fails
        """
        if not self.client:
            raise_database_error(
                "Weaviate not connected",
                database_type="weaviate",
                operation="create_schema"
            )
        
        try:
            with performance_context("weaviate_create_schema", case_id=case_id):
                class_name = f"LegalDocument_{case_id.replace('-', '_')}"
                
                # Check if class already exists
                try:
                    existing_schema = self.client.schema.get(class_name)
                    if existing_schema:
                        logger.info(f"Weaviate schema already exists for case {case_id}")
                        return
                except:
                    # Class doesn't exist, create it
                    pass
                
                # Define schema for legal documents
                schema = {
                    "class": class_name,
                    "description": f"Legal documents for case {case_id}",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "document_id",
                            "dataType": ["string"],
                            "description": "Unique document identifier"
                        },
                        {
                            "name": "document_name",
                            "dataType": ["string"],
                            "description": "Original document filename"
                        },
                        {
                            "name": "case_id",
                            "dataType": ["string"],
                            "description": "Case identifier"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["string"],
                            "description": "User who uploaded the document"
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Document text content"
                        },
                        {
                            "name": "chunk_index",
                            "dataType": ["int"],
                            "description": "Index of this chunk within the document"
                        },
                        {
                            "name": "chunk_size",
                            "dataType": ["int"],
                            "description": "Size of this text chunk"
                        },
                        {
                            "name": "file_type",
                            "dataType": ["string"],
                            "description": "Original file type (pdf, txt, etc.)"
                        },
                        {
                            "name": "date_added",
                            "dataType": ["date"],
                            "description": "Date when document was added"
                        },
                        {
                            "name": "processing_metadata",
                            "dataType": ["object"],
                            "description": "Document processing metadata"
                        }
                    ]
                }
                
                self.client.schema.create_class(schema)
                
                logger.info(
                    "Weaviate schema created for case",
                    case_id=case_id,
                    class_name=class_name
                )
                
        except Exception as e:
            raise_database_error(
                f"Failed to create Weaviate schema for case {case_id}: {e}",
                database_type="weaviate",
                operation="create_schema",
                collection_name=case_id
            )
    
    async def delete_case_schema(self, case_id: str) -> None:
        """
        Delete Weaviate schema for a specific case.
        
        Args:
            case_id: Case identifier to delete
        """
        if not self.client:
            raise_database_error(
                "Weaviate not connected",
                database_type="weaviate",
                operation="delete_schema"
            )
        
        try:
            class_name = f"LegalDocument_{case_id.replace('-', '_')}"
            self.client.schema.delete_class(class_name)
            
            logger.info(
                "Weaviate schema deleted for case",
                case_id=case_id,
                class_name=class_name
            )
            
        except Exception as e:
            logger.warning(
                f"Failed to delete Weaviate schema for case {case_id}: {e}"
            )
    
    def get_client(self) -> weaviate.Client:
        """
        Get the Weaviate client instance.
        
        Returns:
            Weaviate client instance
            
        Raises:
            DatabaseError: If not connected
        """
        if not self.client:
            raise_database_error(
                "Weaviate not connected",
                database_type="weaviate",
                operation="get_client"
            )
        return self.client


class Neo4jManager:
    """
    Neo4j graph database connection management (Phase 2).
    
    Handles document relationship storage and graph operations
    for complex legal document analysis.
    """
    
    def __init__(self):
        """Initialize Neo4j manager."""
        self.driver = None
        self.is_connected: bool = False
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """
        Establish connection to Neo4j (Phase 2).
        
        Note: This is a placeholder for Phase 2 implementation.
        """
        logger.info("Neo4j connection placeholder - Phase 2 feature")
        self.is_connected = False
    
    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        logger.info("Neo4j disconnect placeholder - Phase 2 feature")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Neo4j health check."""
        return {
            "status": "not_implemented",
            "message": "Neo4j integration planned for Phase 2"
        }


class DatabaseManager:
    """
    Central database management coordinator.
    
    Manages all database connections and provides unified interface
    for database operations across the legal AI system.
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.mongodb = MongoDBManager()
        self.weaviate = WeaviateManager()
        self.neo4j = Neo4jManager()
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize all database connections.
        
        Raises:
            DatabaseError: If any database connection fails
        """
        if self._initialized:
            return
        
        try:
            with performance_context("database_initialization"):
                # Connect to MongoDB
                await self.mongodb.connect()
                
                # Create MongoDB indexes
                await self.mongodb.create_indexes()
                
                # Connect to Weaviate
                await self.weaviate.connect()
                
                # Neo4j connection (Phase 2)
                await self.neo4j.connect()
                
                self._initialized = True
                
                logger.info("All database connections initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all database connections."""
        try:
            with performance_context("database_shutdown"):
                # Close connections in reverse order
                await self.neo4j.disconnect()
                await self.weaviate.disconnect()
                await self.mongodb.disconnect()
                
                self._initialized = False
                
                logger.info("All database connections closed")
                
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all databases.
        
        Returns:
            Health status for all database systems
        """
        try:
            with performance_context("database_health_check"):
                mongodb_health = await self.mongodb.health_check()
                weaviate_health = await self.weaviate.health_check()
                neo4j_health = await self.neo4j.health_check()
                
                all_healthy = all([
                    mongodb_health.get("status") == "healthy",
                    weaviate_health.get("status") == "healthy",
                    # neo4j_health.get("status") == "healthy"  # Phase 2
                ])
                
                return {
                    "overall_status": "healthy" if all_healthy else "degraded",
                    "databases": {
                        "mongodb": mongodb_health,
                        "weaviate": weaviate_health,
                        "neo4j": neo4j_health
                    }
                }
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e)
            }
    
    def get_mongodb(self) -> AsyncIOMotorDatabase:
        """Get MongoDB database instance."""
        return self.mongodb.get_database()
    
    def get_weaviate(self) -> weaviate.Client:
        """Get Weaviate client instance."""
        return self.weaviate.get_client()
    
    async def create_case_collections(self, case_id: str) -> None:
        """
        Create all necessary collections/schemas for a new case.
        
        Args:
            case_id: Unique case identifier
        """
        try:
            with performance_context("create_case_collections", case_id=case_id):
                # Create Weaviate schema for case
                await self.weaviate.create_case_schema(case_id)
                
                logger.info(
                    "Case collections created successfully",
                    case_id=case_id
                )
                
        except Exception as e:
            raise_database_error(
                f"Failed to create collections for case {case_id}: {e}",
                database_type="multiple",
                operation="create_case_collections"
            )
    
    async def delete_case_collections(self, case_id: str) -> None:
        """
        Delete all collections/schemas for a case.
        
        Args:
            case_id: Case identifier to delete
        """
        try:
            with performance_context("delete_case_collections", case_id=case_id):
                # Delete Weaviate schema
                await self.weaviate.delete_case_schema(case_id)
                
                # Delete MongoDB documents for the case
                db = self.get_mongodb()
                await db.documents.delete_many({"case_id": case_id})
                await db.search_history.delete_many({"case_id": case_id})
                
                logger.info(
                    "Case collections deleted successfully",
                    case_id=case_id
                )
                
        except Exception as e:
            logger.error(
                f"Failed to delete collections for case {case_id}: {e}"
            )


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def init_databases() -> None:
    """Initialize all databases for the application."""
    db_manager = get_database_manager()
    await db_manager.initialize()


async def close_databases() -> None:
    """Close all database connections."""
    global _db_manager
    if _db_manager:
        await _db_manager.shutdown()
        _db_manager = None


@asynccontextmanager
async def get_mongodb():
    """
    Async context manager for MongoDB operations.
    
    Usage:
        async with get_mongodb() as db:
            result = await db.cases.find_one({"case_id": "123"})
    """
    db_manager = get_database_manager()
    if not db_manager._initialized:
        await db_manager.initialize()
    
    try:
        yield db_manager.get_mongodb()
    except Exception as e:
        logger.error(f"MongoDB operation error: {e}")
        raise


@asynccontextmanager
async def get_weaviate():
    """
    Async context manager for Weaviate operations.
    
    Usage:
        async with get_weaviate() as client:
            result = client.query.get("LegalDocument_case_123").do()
    """
    db_manager = get_database_manager()
    if not db_manager._initialized:
        await db_manager.initialize()
    
    try:
        yield db_manager.get_weaviate()
    except Exception as e:
        logger.error(f"Weaviate operation error: {e}")
        raise


# FastAPI dependency functions
async def get_mongodb_database() -> AsyncIOMotorDatabase:
    """FastAPI dependency to get MongoDB database."""
    db_manager = get_database_manager()
    return db_manager.get_mongodb()


async def get_weaviate_client() -> weaviate.Client:
    """FastAPI dependency to get Weaviate client."""
    db_manager = get_database_manager()
    return db_manager.get_weaviate()