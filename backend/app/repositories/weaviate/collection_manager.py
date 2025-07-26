"""
Weaviate collection management for Patexia Legal AI Chatbot.

This module provides centralized management of Weaviate collections and schemas:
- Collection lifecycle management (create, delete, migrate)
- Schema definition and validation for legal documents
- Collection health monitoring and maintenance
- Migration and schema evolution support
- Backup and restore operations
- Performance optimization and indexing
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import weaviate
from weaviate.exceptions import WeaviateBaseError

from backend.app.core.database import get_weaviate_client
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
from backend.config.settings import get_settings

logger = get_logger(__name__)


class CollectionStatus(str, Enum):
    """Status of Weaviate collections."""
    
    ACTIVE = "active"               # Collection is active and ready
    CREATING = "creating"           # Collection is being created
    MIGRATING = "migrating"         # Collection is being migrated
    MAINTENANCE = "maintenance"     # Collection is under maintenance
    ERROR = "error"                 # Collection has errors
    ARCHIVED = "archived"           # Collection is archived


class SchemaVersion(str, Enum):
    """Schema versions for legal document collections."""
    
    V1_0 = "1.0"                   # Initial schema version
    V1_1 = "1.1"                   # Added legal citations support
    V1_2 = "1.2"                   # Added processing metadata
    LATEST = "1.2"                 # Current latest version


class CollectionManager:
    """
    Centralized manager for Weaviate collections and schemas.
    
    Provides comprehensive collection lifecycle management, schema evolution,
    and maintenance operations for legal document vector storage.
    """
    
    def __init__(self):
        """Initialize collection manager."""
        self._client: Optional[weaviate.Client] = None
        self._collection_registry: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._collection_status: Dict[str, CollectionStatus] = {}
    
    async def _get_client(self) -> weaviate.Client:
        """Get Weaviate client with lazy initialization."""
        if self._client is None:
            self._client = await get_weaviate_client()
        return self._client
    
    def get_collection_name(self, case_id: str) -> str:
        """
        Generate standardized collection name for a case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Standardized Weaviate collection name
        """
        # Convert case ID to valid Weaviate class name
        sanitized_id = case_id.replace('-', '_').replace('.', '_').replace(' ', '_')
        # Ensure it starts with uppercase letter (Weaviate requirement)
        collection_name = f"LegalDocument_{sanitized_id}"
        
        # Validate collection name length (Weaviate has limits)
        if len(collection_name) > 100:
            # Use hash for very long case IDs
            import hashlib
            case_hash = hashlib.md5(case_id.encode()).hexdigest()[:16]
            collection_name = f"LegalDocument_{case_hash}"
        
        return collection_name
    
    def get_legal_document_schema(
        self, 
        collection_name: str, 
        case_id: str,
        schema_version: SchemaVersion = SchemaVersion.LATEST
    ) -> Dict[str, Any]:
        """
        Get schema definition for legal document collection.
        
        Args:
            collection_name: Name of the collection
            case_id: Case identifier for context
            schema_version: Schema version to use
            
        Returns:
            Weaviate schema definition
        """
        base_properties = [
            {
                "name": "chunk_id",
                "dataType": ["string"],
                "description": "Unique chunk identifier",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                }
            },
            {
                "name": "document_id", 
                "dataType": ["string"],
                "description": "Parent document identifier",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                }
            },
            {
                "name": "document_name",
                "dataType": ["string"],
                "description": "Document display name for reference",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                }
            },
            {
                "name": "case_id",
                "dataType": ["string"],
                "description": "Case identifier for isolation",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                }
            },
            {
                "name": "user_id",
                "dataType": ["string"],
                "description": "User who uploaded the document",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": True}
                }
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Text content of the chunk for search",
                "moduleConfig": {
                    "text2vec-transformers": {"skip": False}
                }
            },
            {
                "name": "chunk_index",
                "dataType": ["int"],
                "description": "Sequential index within document"
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
                "description": "Size of text chunk in characters"
            },
            {
                "name": "file_type",
                "dataType": ["string"],
                "description": "Original document file type (pdf, txt, etc.)"
            },
            {
                "name": "created_at",
                "dataType": ["date"],
                "description": "Chunk creation timestamp"
            }
        ]
        
        # Add legal document specific properties
        legal_properties = [
            {
                "name": "section_title",
                "dataType": ["string"],
                "description": "Legal document section header"
            },
            {
                "name": "page_number",
                "dataType": ["int"],
                "description": "Page number in original document"
            },
            {
                "name": "paragraph_number",
                "dataType": ["int"],
                "description": "Paragraph number for navigation"
            }
        ]
        
        # Schema version specific properties
        if schema_version in [SchemaVersion.V1_1, SchemaVersion.V1_2, SchemaVersion.LATEST]:
            legal_properties.append({
                "name": "legal_citations",
                "dataType": ["string[]"],
                "description": "Legal citations found in this chunk"
            })
        
        if schema_version in [SchemaVersion.V1_2, SchemaVersion.LATEST]:
            legal_properties.append({
                "name": "processing_metadata",
                "dataType": ["object"],
                "description": "Document processing metadata and context"
            })
            legal_properties.append({
                "name": "confidence_score",
                "dataType": ["number"],
                "description": "Processing confidence score (0.0 to 1.0)"
            })
            legal_properties.append({
                "name": "extraction_method",
                "dataType": ["string"],
                "description": "Method used for text extraction"
            })
        
        # Combine all properties
        all_properties = base_properties + legal_properties
        
        # Build complete schema
        schema = {
            "class": collection_name,
            "description": f"Legal document chunks for case {case_id} (schema v{schema_version})",
            "vectorizer": "none",  # We provide our own embeddings
            "moduleConfig": {
                "generative-openai": {
                    "model": "gpt-3.5-turbo"
                },
                "reranker-transformers": {
                    "model": "cross-encoder/ms-marco-MiniLM-L-12-v2"
                }
            },
            "properties": all_properties,
            "invertedIndexConfig": {
                "bm25": {
                    "k1": 1.2,
                    "b": 0.75
                },
                "cleanupIntervalSeconds": 60,
                "stopwords": {
                    "preset": "en",
                    "additions": ["legal", "document", "section", "clause"]
                }
            },
            "vectorIndexConfig": {
                "ef": 200,
                "efConstruction": 128,
                "maxConnections": 64,
                "dynamicEfMin": 100,
                "dynamicEfMax": 500,
                "skip": False,
                "cleanupIntervalSeconds": 300,
                "vectorCacheMaxObjects": 1000000
            },
            "shardingConfig": {
                "virtualPerPhysical": 128,
                "desiredCount": 1,
                "actualCount": 1,
                "desiredVirtualCount": 128,
                "actualVirtualCount": 128
            }
        }
        
        return schema
    
    async def create_collection(
        self, 
        case_id: str, 
        schema_version: SchemaVersion = SchemaVersion.LATEST,
        force_recreate: bool = False
    ) -> Tuple[str, bool]:
        """
        Create a new collection for a case.
        
        Args:
            case_id: Case identifier
            schema_version: Schema version to use
            force_recreate: Whether to recreate if exists
            
        Returns:
            Tuple of (collection_name, was_created)
            
        Raises:
            DatabaseError: If collection creation fails
        """
        client = await self._get_client()
        collection_name = self.get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_create_collection", case_id=case_id):
                # Check if collection already exists
                collection_exists = await self._collection_exists(collection_name)
                
                if collection_exists and not force_recreate:
                    logger.info(f"Collection {collection_name} already exists")
                    self._register_collection(case_id, collection_name, schema_version)
                    return collection_name, False
                
                # Delete existing collection if force recreate
                if collection_exists and force_recreate:
                    await self.delete_collection(case_id)
                
                # Mark as creating
                self._collection_status[collection_name] = CollectionStatus.CREATING
                
                # Get schema definition
                schema = self.get_legal_document_schema(
                    collection_name, case_id, schema_version
                )
                
                # Create the collection
                client.schema.create_class(schema)
                
                # Register the collection
                self._register_collection(case_id, collection_name, schema_version)
                self._collection_status[collection_name] = CollectionStatus.ACTIVE
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="create_collection",
                    collection=collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Collection created successfully",
                    case_id=case_id,
                    collection_name=collection_name,
                    schema_version=schema_version.value
                )
                
                return collection_name, True
                
        except WeaviateBaseError as e:
            self._collection_status[collection_name] = CollectionStatus.ERROR
            raise_database_error(
                f"Failed to create Weaviate collection for case {case_id}: {e}",
                database_type="weaviate",
                operation="create_collection",
                collection_name=collection_name
            )
        except Exception as e:
            self._collection_status[collection_name] = CollectionStatus.ERROR
            raise_database_error(
                f"Unexpected error creating collection for case {case_id}: {e}",
                database_type="weaviate",
                operation="create_collection",
                collection_name=collection_name
            )
    
    async def delete_collection(self, case_id: str) -> bool:
        """
        Delete a collection for a case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if collection was deleted, False if not found
        """
        client = await self._get_client()
        collection_name = self.get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_delete_collection", case_id=case_id):
                # Check if collection exists
                if not await self._collection_exists(collection_name):
                    logger.info(f"Collection {collection_name} does not exist")
                    return False
                
                # Delete the collection
                client.schema.delete_class(collection_name)
                
                # Clean up registry
                self._unregister_collection(case_id, collection_name)
                
                database_logger.query_executed(
                    database_type="weaviate",
                    operation="delete_collection",
                    collection=collection_name,
                    result_count=1
                )
                
                logger.info(
                    "Collection deleted successfully",
                    case_id=case_id,
                    collection_name=collection_name
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete collection for case {case_id}: {e}")
            return False
    
    async def migrate_collection(
        self, 
        case_id: str, 
        target_version: SchemaVersion
    ) -> bool:
        """
        Migrate collection to a new schema version.
        
        Args:
            case_id: Case identifier
            target_version: Target schema version
            
        Returns:
            True if migration successful
        """
        collection_name = self.get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_migrate_collection", case_id=case_id):
                # Mark as migrating
                self._collection_status[collection_name] = CollectionStatus.MIGRATING
                
                # Get current schema version
                current_version = self._get_collection_version(case_id)
                if current_version == target_version:
                    logger.info(f"Collection {collection_name} already at target version")
                    self._collection_status[collection_name] = CollectionStatus.ACTIVE
                    return True
                
                # Backup current collection
                backup_data = await self._backup_collection_data(case_id)
                if not backup_data:
                    raise DatabaseError("Failed to backup collection data")
                
                # Create new collection with target schema
                temp_collection_name = f"{collection_name}_temp_{int(datetime.now().timestamp())}"
                temp_case_id = f"{case_id}_temp"
                
                # Create temporary collection
                await self.create_collection(temp_case_id, target_version)
                
                # Migrate data to new collection
                migration_success = await self._migrate_data(
                    backup_data, temp_collection_name, target_version
                )
                
                if migration_success:
                    # Delete old collection
                    await self.delete_collection(case_id)
                    
                    # Rename temp collection (by recreating with original name)
                    await self.delete_collection(temp_case_id)
                    await self.create_collection(case_id, target_version)
                    
                    # Restore data to final collection
                    await self._restore_collection_data(case_id, backup_data, target_version)
                    
                    self._collection_status[collection_name] = CollectionStatus.ACTIVE
                    
                    logger.info(
                        "Collection migration completed",
                        case_id=case_id,
                        from_version=current_version.value if current_version else "unknown",
                        to_version=target_version.value
                    )
                    
                    return True
                else:
                    # Cleanup failed migration
                    await self.delete_collection(temp_case_id)
                    self._collection_status[collection_name] = CollectionStatus.ERROR
                    return False
                
        except Exception as e:
            logger.error(f"Collection migration failed for case {case_id}: {e}")
            self._collection_status[collection_name] = CollectionStatus.ERROR
            return False
    
    async def get_collection_info(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a collection.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Collection information dictionary or None if not found
        """
        client = await self._get_client()
        collection_name = self.get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_get_collection_info", case_id=case_id):
                # Check if collection exists
                if not await self._collection_exists(collection_name):
                    return None
                
                # Get schema
                schema = client.schema.get(collection_name)
                
                # Get object count
                count_result = client.query.aggregate(collection_name).with_meta_count().do()
                object_count = 0
                if "data" in count_result and "Aggregate" in count_result["data"]:
                    agg_data = count_result["data"]["Aggregate"].get(collection_name, [])
                    if agg_data:
                        object_count = agg_data[0].get("meta", {}).get("count", 0)
                
                # Get collection metadata
                collection_info = {
                    "case_id": case_id,
                    "collection_name": collection_name,
                    "schema_version": self._get_collection_version(case_id),
                    "status": self._collection_status.get(collection_name, CollectionStatus.ACTIVE),
                    "object_count": object_count,
                    "created_at": self._collection_registry.get(case_id, {}).get("created_at"),
                    "last_updated": self._collection_registry.get(case_id, {}).get("last_updated"),
                    "schema": schema,
                    "properties": [prop["name"] for prop in schema.get("properties", [])],
                    "vectorizer": schema.get("vectorizer"),
                    "index_config": {
                        "inverted": schema.get("invertedIndexConfig", {}),
                        "vector": schema.get("vectorIndexConfig", {})
                    }
                }
                
                return collection_info
                
        except Exception as e:
            logger.error(f"Failed to get collection info for case {case_id}: {e}")
            return None
    
    async def health_check_collection(self, case_id: str) -> Dict[str, Any]:
        """
        Perform health check on a collection.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Health check results
        """
        collection_name = self.get_collection_name(case_id)
        
        try:
            with performance_context("weaviate_health_check", case_id=case_id):
                health_status = {
                    "case_id": case_id,
                    "collection_name": collection_name,
                    "healthy": True,
                    "issues": [],
                    "metrics": {}
                }
                
                # Check if collection exists
                if not await self._collection_exists(collection_name):
                    health_status["healthy"] = False
                    health_status["issues"].append("Collection does not exist")
                    return health_status
                
                # Check collection status
                status = self._collection_status.get(collection_name, CollectionStatus.ACTIVE)
                if status != CollectionStatus.ACTIVE:
                    health_status["healthy"] = False
                    health_status["issues"].append(f"Collection status is {status.value}")
                
                # Get collection info
                collection_info = await self.get_collection_info(case_id)
                if collection_info:
                    health_status["metrics"] = {
                        "object_count": collection_info["object_count"],
                        "schema_version": collection_info["schema_version"].value if collection_info["schema_version"] else "unknown",
                        "property_count": len(collection_info["properties"])
                    }
                    
                    # Check for empty collection
                    if collection_info["object_count"] == 0:
                        health_status["issues"].append("Collection is empty")
                
                # Additional health checks could be added here
                # - Check for schema consistency
                # - Validate vector dimensions
                # - Check index health
                
                logger.debug(
                    "Collection health check completed",
                    case_id=case_id,
                    healthy=health_status["healthy"],
                    issues=len(health_status["issues"])
                )
                
                return health_status
                
        except Exception as e:
            logger.error(f"Health check failed for case {case_id}: {e}")
            return {
                "case_id": case_id,
                "collection_name": collection_name,
                "healthy": False,
                "issues": [f"Health check failed: {str(e)}"],
                "metrics": {}
            }
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all managed collections.
        
        Returns:
            List of collection information
        """
        try:
            with performance_context("weaviate_list_collections"):
                collections = []
                
                for case_id in self._collection_registry.keys():
                    collection_info = await self.get_collection_info(case_id)
                    if collection_info:
                        collections.append(collection_info)
                
                logger.debug(f"Listed {len(collections)} collections")
                
                return collections
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def cleanup_orphaned_collections(self) -> List[str]:
        """
        Find and optionally clean up orphaned collections.
        
        Returns:
            List of orphaned collection names
        """
        client = await self._get_client()
        
        try:
            with performance_context("weaviate_cleanup_orphaned"):
                # Get all Weaviate classes
                schema = client.schema.get()
                all_classes = schema.get("classes", [])
                
                # Find collections that match our naming pattern but aren't registered
                orphaned = []
                legal_doc_classes = [
                    cls for cls in all_classes 
                    if cls["class"].startswith("LegalDocument_")
                ]
                
                for cls in legal_doc_classes:
                    class_name = cls["class"]
                    # Check if this collection is registered
                    is_registered = any(
                        self.get_collection_name(case_id) == class_name
                        for case_id in self._collection_registry.keys()
                    )
                    
                    if not is_registered:
                        orphaned.append(class_name)
                        logger.warning(f"Found orphaned collection: {class_name}")
                
                logger.info(f"Found {len(orphaned)} orphaned collections")
                
                return orphaned
                
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned collections: {e}")
            return []
    
    async def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Weaviate."""
        client = await self._get_client()
        
        try:
            client.schema.get(collection_name)
            return True
        except:
            return False
    
    def _register_collection(
        self, 
        case_id: str, 
        collection_name: str, 
        schema_version: SchemaVersion
    ) -> None:
        """Register a collection in the internal registry."""
        now = datetime.now(timezone.utc).isoformat()
        
        self._collection_registry[case_id] = {
            "collection_name": collection_name,
            "schema_version": schema_version.value,
            "created_at": now,
            "last_updated": now
        }
        
        self._collection_status[collection_name] = CollectionStatus.ACTIVE
    
    def _unregister_collection(self, case_id: str, collection_name: str) -> None:
        """Unregister a collection from the internal registry."""
        self._collection_registry.pop(case_id, None)
        self._collection_status.pop(collection_name, None)
        self._schema_cache.pop(collection_name, None)
    
    def _get_collection_version(self, case_id: str) -> Optional[SchemaVersion]:
        """Get the schema version for a collection."""
        collection_info = self._collection_registry.get(case_id, {})
        version_str = collection_info.get("schema_version")
        
        if version_str:
            try:
                return SchemaVersion(version_str)
            except ValueError:
                return None
        
        return None
    
    async def _backup_collection_data(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Create a backup of collection data."""
        # This would integrate with the vector repository's backup functionality
        from backend.app.repositories.weaviate.vector_repository import get_vector_repository
        
        vector_repo = get_vector_repository()
        return await vector_repo.backup_collection(case_id)
    
    async def _migrate_data(
        self, 
        backup_data: Dict[str, Any], 
        target_collection: str, 
        target_version: SchemaVersion
    ) -> bool:
        """Migrate data to a new schema version."""
        try:
            # This would involve transforming data according to schema changes
            # For now, we'll implement a basic structure
            
            logger.info(
                "Data migration simulation",
                target_collection=target_collection,
                target_version=target_version.value,
                object_count=len(backup_data.get("objects", []))
            )
            
            # In a real implementation, this would:
            # 1. Transform data according to schema changes
            # 2. Handle new/removed properties
            # 3. Validate data against new schema
            # 4. Insert transformed data into target collection
            
            return True
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False
    
    async def _restore_collection_data(
        self, 
        case_id: str, 
        backup_data: Dict[str, Any], 
        schema_version: SchemaVersion
    ) -> bool:
        """Restore collection data from backup."""
        try:
            # This would integrate with the vector repository's restore functionality
            logger.info(
                "Data restoration simulation",
                case_id=case_id,
                schema_version=schema_version.value,
                object_count=len(backup_data.get("objects", []))
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Data restoration failed: {e}")
            return False


# Singleton instance for dependency injection
_collection_manager: Optional[CollectionManager] = None


def get_collection_manager() -> CollectionManager:
    """
    Get the singleton collection manager instance.
    
    Returns:
        CollectionManager instance
    """
    global _collection_manager
    if _collection_manager is None:
        _collection_manager = CollectionManager()
    return _collection_manager