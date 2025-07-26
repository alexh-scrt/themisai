#!/usr/bin/env python3
"""
=============================================================================
Patexia Legal AI Chatbot - Database Initialization Script

This script initializes all databases required for the legal AI system:
- MongoDB: Document storage, case management, user data
- Weaviate: Vector embeddings with per-case collections
- Neo4j: Document relationships and graph data (Phase 2)
- GridFS: Legal document file storage
- Indexes: Optimized indexes for legal document queries
- Schemas: Proper data validation and structure
- Sample Data: Optional test data for development

Features:
- Comprehensive schema creation and validation
- Legal document-optimized indexes
- Per-case collection management
- Development and production modes
- Data migration support
- Health checks and validation
- Legal compliance configurations

Usage:
    python init_databases.py [--reset] [--sample-data] [--production]
    
Examples:
    python init_databases.py --reset --sample-data
    python init_databases.py --production
    python init_databases.py --check-only
=============================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

# Third-party imports
try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
    from pymongo.errors import CollectionInvalid, OperationFailure
    from gridfs import GridFS
    import weaviate
    from weaviate.exceptions import WeaviateException
    import neo4j
    from neo4j import GraphDatabase
    import requests
    from bson import ObjectId
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install pymongo weaviate-client neo4j requests")
    sys.exit(1)


class DatabaseType(Enum):
    """Types of databases to initialize."""
    MONGODB = "mongodb"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    ALL = "all"


class InitializationMode(Enum):
    """Database initialization modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # MongoDB settings
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "patexia_legal_ai"
    mongodb_auth_database: str = "admin"
    mongodb_username: Optional[str] = None
    mongodb_password: Optional[str] = None
    
    # Weaviate settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    
    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: Optional[str] = None
    neo4j_database: str = "legal_relationships"
    
    # Initialization settings
    reset_existing: bool = False
    create_sample_data: bool = False
    mode: InitializationMode = InitializationMode.DEVELOPMENT
    
    # Legal compliance settings
    enable_audit_logging: bool = True
    data_retention_days: int = 2555  # 7 years for legal compliance
    encryption_at_rest: bool = False


class DatabaseInitializer:
    """
    Main database initialization class for legal AI system.
    
    Handles setup of MongoDB, Weaviate, and Neo4j databases with
    proper schemas, indexes, and legal document optimizations.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration settings
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Database clients
        self.mongodb_client: Optional[MongoClient] = None
        self.weaviate_client: Optional[weaviate.Client] = None
        self.neo4j_driver: Optional[neo4j.Driver] = None
        
        # State tracking
        self.initialization_results = {
            "mongodb": {"success": False, "error": None},
            "weaviate": {"success": False, "error": None},
            "neo4j": {"success": False, "error": None}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("database_init")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def initialize_all_databases(self) -> bool:
        """
        Initialize all databases in the correct order.
        
        Returns:
            True if all databases initialized successfully
        """
        self.logger.info("Starting database initialization for Patexia Legal AI")
        
        success = True
        
        try:
            # Initialize MongoDB first (core data storage)
            if not await self.initialize_mongodb():
                success = False
            
            # Initialize Weaviate (vector storage)
            if not await self.initialize_weaviate():
                success = False
            
            # Initialize Neo4j (relationships - Phase 2)
            if not await self.initialize_neo4j():
                success = False
            
            if success:
                self.logger.info("All databases initialized successfully")
                
                # Run health checks
                await self.run_health_checks()
                
                # Create sample data if requested
                if self.config.create_sample_data:
                    await self.create_sample_data()
            else:
                self.logger.error("Some databases failed to initialize")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            success = False
        
        finally:
            await self.cleanup_connections()
        
        return success
    
    async def initialize_mongodb(self) -> bool:
        """
        Initialize MongoDB with collections, indexes, and GridFS.
        
        Returns:
            True if MongoDB initialized successfully
        """
        self.logger.info("Initializing MongoDB...")
        
        try:
            # Connect to MongoDB
            await self._connect_mongodb()
            
            db = self.mongodb_client[self.config.mongodb_database]
            
            # Reset database if requested
            if self.config.reset_existing:
                self.logger.warning("Resetting MongoDB database")
                await self._reset_mongodb(db)
            
            # Create collections with validation schemas
            await self._create_mongodb_collections(db)
            
            # Create indexes for performance
            await self._create_mongodb_indexes(db)
            
            # Initialize GridFS for file storage
            await self._initialize_gridfs(db)
            
            # Create database users and permissions
            if self.config.mode == InitializationMode.PRODUCTION:
                await self._setup_mongodb_security(db)
            
            self.initialization_results["mongodb"]["success"] = True
            self.logger.info("MongoDB initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"MongoDB initialization failed: {e}"
            self.logger.error(error_msg)
            self.initialization_results["mongodb"]["error"] = str(e)
            return False
    
    async def _connect_mongodb(self) -> None:
        """Connect to MongoDB with proper authentication."""
        self.logger.info(f"Connecting to MongoDB: {self.config.mongodb_uri}")
        
        # Build connection parameters
        connect_params = {
            "serverSelectionTimeoutMS": 30000,
            "connectTimeoutMS": 30000,
            "maxPoolSize": 50,
            "minPoolSize": 5
        }
        
        if self.config.mongodb_username and self.config.mongodb_password:
            connect_params.update({
                "username": self.config.mongodb_username,
                "password": self.config.mongodb_password,
                "authSource": self.config.mongodb_auth_database
            })
        
        self.mongodb_client = MongoClient(self.config.mongodb_uri, **connect_params)
        
        # Test connection
        await asyncio.get_event_loop().run_in_executor(
            None, self.mongodb_client.admin.command, "ping"
        )
        
        self.logger.info("MongoDB connection established")
    
    async def _reset_mongodb(self, db) -> None:
        """Reset MongoDB database (development mode only)."""
        if self.config.mode == InitializationMode.PRODUCTION:
            raise ValueError("Cannot reset database in production mode")
        
        # Drop all collections
        collections = await asyncio.get_event_loop().run_in_executor(
            None, db.list_collection_names
        )
        
        for collection_name in collections:
            await asyncio.get_event_loop().run_in_executor(
                None, db.drop_collection, collection_name
            )
            self.logger.info(f"Dropped collection: {collection_name}")
    
    async def _create_mongodb_collections(self, db) -> None:
        """Create MongoDB collections with validation schemas."""
        self.logger.info("Creating MongoDB collections with validation schemas")
        
        # Cases collection
        cases_schema = {
            "bsonType": "object",
            "required": ["case_id", "user_id", "case_name", "created_at", "updated_at", "status"],
            "properties": {
                "case_id": {
                    "bsonType": "string",
                    "pattern": "^CASE_[0-9]{4}_[0-9]{2}_[0-9]{2}_[A-Z0-9]{8}$"
                },
                "user_id": {"bsonType": "string"},
                "case_name": {"bsonType": "string", "minLength": 1, "maxLength": 200},
                "initial_summary": {"bsonType": "string", "maxLength": 1000},
                "visual_marker": {
                    "bsonType": "object",
                    "required": ["color", "icon"],
                    "properties": {
                        "color": {"bsonType": "string"},
                        "icon": {"bsonType": "string"}
                    }
                },
                "priority": {
                    "bsonType": "string",
                    "enum": ["low", "medium", "high", "urgent"]
                },
                "status": {
                    "bsonType": "string",
                    "enum": ["active", "archived", "closed", "on_hold"]
                },
                "tags": {"bsonType": "array", "items": {"bsonType": "string"}},
                "document_ids": {"bsonType": "array", "items": {"bsonType": "string"}},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "metadata": {"bsonType": "object"}
            }
        }
        
        await self._create_collection_with_validation(db, "cases", cases_schema)
        
        # Documents collection
        documents_schema = {
            "bsonType": "object",
            "required": ["document_id", "case_id", "user_id", "document_name", "file_type", "created_at"],
            "properties": {
                "document_id": {
                    "bsonType": "string",
                    "pattern": "^DOC_[0-9]{8}_[A-Z0-9]{8}$"
                },
                "case_id": {"bsonType": "string"},
                "user_id": {"bsonType": "string"},
                "document_name": {"bsonType": "string", "minLength": 1, "maxLength": 255},
                "original_filename": {"bsonType": "string"},
                "file_type": {
                    "bsonType": "string",
                    "enum": ["pdf", "txt", "docx", "doc"]
                },
                "file_size": {"bsonType": "long", "minimum": 0},
                "file_hash": {"bsonType": "string"},
                "gridfs_file_id": {"bsonType": "objectId"},
                "processing_status": {
                    "bsonType": "string",
                    "enum": ["pending", "processing", "completed", "failed"]
                },
                "processing_metadata": {"bsonType": "object"},
                "page_count": {"bsonType": "int", "minimum": 0},
                "legal_citations": {"bsonType": "array", "items": {"bsonType": "string"}},
                "section_headers": {"bsonType": "array", "items": {"bsonType": "string"}},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "metadata": {"bsonType": "object"}
            }
        }
        
        await self._create_collection_with_validation(db, "documents", documents_schema)
        
        # Document chunks collection
        chunks_schema = {
            "bsonType": "object",
            "required": ["chunk_id", "document_id", "case_id", "content", "chunk_index", "created_at"],
            "properties": {
                "chunk_id": {"bsonType": "string"},
                "document_id": {"bsonType": "string"},
                "case_id": {"bsonType": "string"},
                "content": {"bsonType": "string", "minLength": 1},
                "chunk_index": {"bsonType": "int", "minimum": 0},
                "start_char": {"bsonType": "int", "minimum": 0},
                "end_char": {"bsonType": "int", "minimum": 0},
                "chunk_size": {"bsonType": "int", "minimum": 1},
                "section_title": {"bsonType": "string"},
                "page_number": {"bsonType": "int", "minimum": 1},
                "paragraph_number": {"bsonType": "int", "minimum": 1},
                "legal_citations": {"bsonType": "array", "items": {"bsonType": "string"}},
                "has_embedding": {"bsonType": "bool"},
                "embedding_model": {"bsonType": "string"},
                "created_at": {"bsonType": "date"},
                "quality_score": {"bsonType": "double", "minimum": 0.0, "maximum": 1.0}
            }
        }
        
        await self._create_collection_with_validation(db, "document_chunks", chunks_schema)
        
        # Search history collection
        search_history_schema = {
            "bsonType": "object",
            "required": ["search_id", "user_id", "query", "timestamp"],
            "properties": {
                "search_id": {"bsonType": "string"},
                "user_id": {"bsonType": "string"},
                "case_id": {"bsonType": "string"},
                "query": {"bsonType": "string", "minLength": 1, "maxLength": 500},
                "search_type": {
                    "bsonType": "string",
                    "enum": ["semantic", "keyword", "hybrid", "citation"]
                },
                "search_scope": {
                    "bsonType": "string",
                    "enum": ["case", "document", "global"]
                },
                "results_count": {"bsonType": "int", "minimum": 0},
                "execution_time_ms": {"bsonType": "double", "minimum": 0},
                "timestamp": {"bsonType": "date"},
                "filters_applied": {"bsonType": "object"},
                "metadata": {"bsonType": "object"}
            }
        }
        
        await self._create_collection_with_validation(db, "search_history", search_history_schema)
        
        # Users collection
        users_schema = {
            "bsonType": "object",
            "required": ["user_id", "username", "email", "created_at"],
            "properties": {
                "user_id": {"bsonType": "string"},
                "username": {"bsonType": "string", "minLength": 3, "maxLength": 50},
                "email": {
                    "bsonType": "string",
                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                },
                "full_name": {"bsonType": "string", "maxLength": 100},
                "role": {
                    "bsonType": "string",
                    "enum": ["lawyer", "paralegal", "admin", "viewer"]
                },
                "permissions": {"bsonType": "array", "items": {"bsonType": "string"}},
                "last_login": {"bsonType": "date"},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "preferences": {"bsonType": "object"},
                "is_active": {"bsonType": "bool"}
            }
        }
        
        await self._create_collection_with_validation(db, "users", users_schema)
        
        # Audit logs collection (legal compliance)
        audit_logs_schema = {
            "bsonType": "object",
            "required": ["log_id", "user_id", "action", "timestamp"],
            "properties": {
                "log_id": {"bsonType": "string"},
                "user_id": {"bsonType": "string"},
                "action": {"bsonType": "string"},
                "resource_type": {"bsonType": "string"},
                "resource_id": {"bsonType": "string"},
                "case_id": {"bsonType": "string"},
                "document_id": {"bsonType": "string"},
                "details": {"bsonType": "object"},
                "ip_address": {"bsonType": "string"},
                "user_agent": {"bsonType": "string"},
                "timestamp": {"bsonType": "date"},
                "success": {"bsonType": "bool"},
                "error_message": {"bsonType": "string"}
            }
        }
        
        await self._create_collection_with_validation(db, "audit_logs", audit_logs_schema)
    
    async def _create_collection_with_validation(
        self,
        db,
        collection_name: str,
        schema: Dict[str, Any]
    ) -> None:
        """Create collection with JSON schema validation."""
        try:
            # Create collection with validation
            await asyncio.get_event_loop().run_in_executor(
                None,
                db.create_collection,
                collection_name,
                {
                    "validator": {"$jsonSchema": schema},
                    "validationLevel": "strict",
                    "validationAction": "error"
                }
            )
            self.logger.info(f"Created collection with validation: {collection_name}")
            
        except CollectionInvalid:
            # Collection already exists, update validation if needed
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    db.command,
                    {
                        "collMod": collection_name,
                        "validator": {"$jsonSchema": schema},
                        "validationLevel": "strict"
                    }
                )
                self.logger.info(f"Updated validation for existing collection: {collection_name}")
            except OperationFailure as e:
                self.logger.warning(f"Could not update validation for {collection_name}: {e}")
    
    async def _create_mongodb_indexes(self, db) -> None:
        """Create optimized indexes for legal document queries."""
        self.logger.info("Creating MongoDB indexes for optimal performance")
        
        # Cases collection indexes
        cases = db.cases
        await asyncio.get_event_loop().run_in_executor(
            None, cases.create_index, [("case_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, cases.create_index, [("user_id", ASCENDING), ("status", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, cases.create_index, [("created_at", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, cases.create_index, [("updated_at", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, cases.create_index, [("tags", ASCENDING)]
        )
        
        # Documents collection indexes
        documents = db.documents
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("document_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("case_id", ASCENDING), ("processing_status", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("user_id", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("file_type", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("created_at", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("file_hash", ASCENDING)]
        )
        
        # Text index for document search
        await asyncio.get_event_loop().run_in_executor(
            None, documents.create_index, [("document_name", TEXT), ("legal_citations", TEXT)]
        )
        
        # Document chunks collection indexes
        chunks = db.document_chunks
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("chunk_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("document_id", ASCENDING), ("chunk_index", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("case_id", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("has_embedding", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("page_number", ASCENDING)]
        )
        
        # Text index for content search
        await asyncio.get_event_loop().run_in_executor(
            None, chunks.create_index, [("content", TEXT), ("legal_citations", TEXT)]
        )
        
        # Search history indexes
        search_history = db.search_history
        await asyncio.get_event_loop().run_in_executor(
            None, search_history.create_index, [("search_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, search_history.create_index, [("user_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, search_history.create_index, [("case_id", ASCENDING)]
        )
        
        # Users collection indexes
        users = db.users
        await asyncio.get_event_loop().run_in_executor(
            None, users.create_index, [("user_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, users.create_index, [("username", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, users.create_index, [("email", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, users.create_index, [("role", ASCENDING)]
        )
        
        # Audit logs indexes (legal compliance)
        audit_logs = db.audit_logs
        await asyncio.get_event_loop().run_in_executor(
            None, audit_logs.create_index, [("log_id", ASCENDING)], {"unique": True}
        )
        await asyncio.get_event_loop().run_in_executor(
            None, audit_logs.create_index, [("user_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, audit_logs.create_index, [("case_id", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, audit_logs.create_index, [("action", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, audit_logs.create_index, [("timestamp", DESCENDING)]
        )
        
        # TTL index for audit logs (legal retention)
        if self.config.data_retention_days > 0:
            await asyncio.get_event_loop().run_in_executor(
                None,
                audit_logs.create_index,
                [("timestamp", ASCENDING)],
                {"expireAfterSeconds": self.config.data_retention_days * 24 * 3600}
            )
        
        self.logger.info("MongoDB indexes created successfully")
    
    async def _initialize_gridfs(self, db) -> None:
        """Initialize GridFS for legal document file storage."""
        self.logger.info("Initializing GridFS for document storage")
        
        # Create GridFS instance
        fs = GridFS(db)
        
        # Create indexes for GridFS collections
        fs_files = db.fs.files
        fs_chunks = db.fs.chunks
        
        # Additional indexes for legal document metadata
        await asyncio.get_event_loop().run_in_executor(
            None, fs_files.create_index, [("filename", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, fs_files.create_index, [("uploadDate", DESCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, fs_files.create_index, [("metadata.case_id", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, fs_files.create_index, [("metadata.document_id", ASCENDING)]
        )
        await asyncio.get_event_loop().run_in_executor(
            None, fs_files.create_index, [("metadata.file_type", ASCENDING)]
        )
        
        self.logger.info("GridFS initialized successfully")
    
    async def _setup_mongodb_security(self, db) -> None:
        """Setup MongoDB security for production."""
        if self.config.mode != InitializationMode.PRODUCTION:
            return
        
        self.logger.info("Setting up MongoDB security for production")
        
        # This would typically involve:
        # - Creating database users with specific roles
        # - Setting up authentication
        # - Configuring SSL/TLS
        # - Setting up field-level encryption for sensitive data
        
        # Example: Create application user with limited permissions
        try:
            admin_db = self.mongodb_client.admin
            
            # Check if user already exists
            users = await asyncio.get_event_loop().run_in_executor(
                None, admin_db.command, "usersInfo"
            )
            
            app_user_exists = any(
                user["user"] == "legal_ai_app" 
                for user in users.get("users", [])
            )
            
            if not app_user_exists:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    db.command,
                    {
                        "createUser": "legal_ai_app",
                        "pwd": os.getenv("MONGODB_APP_PASSWORD", "secure_password"),
                        "roles": [
                            {"role": "readWrite", "db": self.config.mongodb_database},
                            {"role": "dbAdmin", "db": self.config.mongodb_database}
                        ]
                    }
                )
                self.logger.info("Created application database user")
        
        except Exception as e:
            self.logger.warning(f"Could not setup MongoDB security: {e}")
    
    async def initialize_weaviate(self) -> bool:
        """
        Initialize Weaviate vector database with legal document schema.
        
        Returns:
            True if Weaviate initialized successfully
        """
        self.logger.info("Initializing Weaviate vector database...")
        
        try:
            # Connect to Weaviate
            await self._connect_weaviate()
            
            # Reset schema if requested
            if self.config.reset_existing:
                self.logger.warning("Resetting Weaviate schema")
                await self._reset_weaviate()
            
            # Create legal document schema
            await self._create_weaviate_schema()
            
            # Create per-case collections
            if self.config.create_sample_data:
                await self._create_sample_case_collections()
            
            self.initialization_results["weaviate"]["success"] = True
            self.logger.info("Weaviate initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Weaviate initialization failed: {e}"
            self.logger.error(error_msg)
            self.initialization_results["weaviate"]["error"] = str(e)
            return False
    
    async def _connect_weaviate(self) -> None:
        """Connect to Weaviate with proper authentication."""
        self.logger.info(f"Connecting to Weaviate: {self.config.weaviate_url}")
        
        auth_config = None
        if self.config.weaviate_api_key:
            auth_config = weaviate.AuthApiKey(api_key=self.config.weaviate_api_key)
        
        self.weaviate_client = weaviate.Client(
            url=self.config.weaviate_url,
            auth_client_secret=auth_config,
            timeout_config=(30, 120)
        )
        
        # Test connection
        if not self.weaviate_client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        
        self.logger.info("Weaviate connection established")
    
    async def _reset_weaviate(self) -> None:
        """Reset Weaviate schema (development mode only)."""
        if self.config.mode == InitializationMode.PRODUCTION:
            raise ValueError("Cannot reset Weaviate in production mode")
        
        # Delete all existing classes
        schema = self.weaviate_client.schema.get()
        for class_obj in schema.get("classes", []):
            class_name = class_obj["class"]
            self.weaviate_client.schema.delete_class(class_name)
            self.logger.info(f"Deleted Weaviate class: {class_name}")
    
    async def _create_weaviate_schema(self) -> None:
        """Create Weaviate schema for legal documents."""
        self.logger.info("Creating Weaviate schema for legal documents")
        
        # Legal Document Chunk class (base template)
        legal_chunk_class = {
            "class": "LegalDocumentChunk",
            "description": "Legal document chunk with semantic content and metadata",
            "vectorizer": "none",  # We'll handle embeddings manually
            "properties": [
                {
                    "name": "chunkId",
                    "dataType": ["string"],
                    "description": "Unique chunk identifier",
                    "moduleConfig": {
                        "text2vec-transformers": {"skip": True}
                    }
                },
                {
                    "name": "documentId", 
                    "dataType": ["string"],
                    "description": "Parent document identifier",
                    "moduleConfig": {
                        "text2vec-transformers": {"skip": True}
                    }
                },
                {
                    "name": "caseId",
                    "dataType": ["string"],
                    "description": "Associated legal case identifier",
                    "moduleConfig": {
                        "text2vec-transformers": {"skip": True}
                    }
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Text content of the chunk"
                },
                {
                    "name": "chunkIndex",
                    "dataType": ["int"],
                    "description": "Position of chunk within document"
                },
                {
                    "name": "startChar",
                    "dataType": ["int"], 
                    "description": "Starting character position"
                },
                {
                    "name": "endChar",
                    "dataType": ["int"],
                    "description": "Ending character position"
                },
                {
                    "name": "sectionTitle",
                    "dataType": ["string"],
                    "description": "Section header if available"
                },
                {
                    "name": "pageNumber",
                    "dataType": ["int"],
                    "description": "Page number if available"
                },
                {
                    "name": "paragraphNumber",
                    "dataType": ["int"],
                    "description": "Paragraph number if available"
                },
                {
                    "name": "legalCitations",
                    "dataType": ["string[]"],
                    "description": "Legal citations found in chunk"
                },
                {
                    "name": "embeddingModel",
                    "dataType": ["string"],
                    "description": "Model used for embedding generation",
                    "moduleConfig": {
                        "text2vec-transformers": {"skip": True}
                    }
                },
                {
                    "name": "createdAt",
                    "dataType": ["date"],
                    "description": "Chunk creation timestamp"
                },
                {
                    "name": "qualityScore",
                    "dataType": ["number"],
                    "description": "Content quality score (0.0-1.0)"
                }
            ]
        }
        
        # Create the base class
        try:
            self.weaviate_client.schema.create_class(legal_chunk_class)
            self.logger.info("Created Weaviate class: LegalDocumentChunk")
        except WeaviateException as e:
            if "already exists" in str(e):
                self.logger.info("LegalDocumentChunk class already exists")
            else:
                raise
        
        # Legal Case Summary class
        case_summary_class = {
            "class": "LegalCaseSummary",
            "description": "AI-generated summaries of legal cases",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "caseId",
                    "dataType": ["string"],
                    "description": "Legal case identifier",
                    "moduleConfig": {
                        "text2vec-transformers": {"skip": True}
                    }
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "description": "AI-generated case summary"
                },
                {
                    "name": "keyTopics",
                    "dataType": ["string[]"],
                    "description": "Key legal topics identified"
                },
                {
                    "name": "documentCount",
                    "dataType": ["int"],
                    "description": "Number of documents in case"
                },
                {
                    "name": "confidence",
                    "dataType": ["number"],
                    "description": "Summary confidence score"
                },
                {
                    "name": "lastUpdated",
                    "dataType": ["date"],
                    "description": "Summary last updated timestamp"
                }
            ]
        }
        
        try:
            self.weaviate_client.schema.create_class(case_summary_class)
            self.logger.info("Created Weaviate class: LegalCaseSummary")
        except WeaviateException as e:
            if "already exists" in str(e):
                self.logger.info("LegalCaseSummary class already exists")
            else:
                raise
    
    async def _create_sample_case_collections(self) -> None:
        """Create sample per-case collections for development."""
        self.logger.info("Creating sample case collections")
        
        sample_cases = [
            "CASE_2025_01_15_ABC123XY",
            "CASE_2025_01_16_DEF456ZW"
        ]
        
        for case_id in sample_cases:
            # Create case-specific class
            case_class_name = f"Case_{case_id.replace('_', '')}"
            
            case_class = {
                "class": case_class_name,
                "description": f"Legal document chunks for case {case_id}",
                "vectorizer": "none",
                "properties": [
                    {
                        "name": "chunkId",
                        "dataType": ["string"],
                        "description": "Unique chunk identifier"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Text content of the chunk"
                    },
                    {
                        "name": "documentName",
                        "dataType": ["string"],
                        "description": "Source document name"
                    },
                    {
                        "name": "relevanceScore",
                        "dataType": ["number"],
                        "description": "Relevance score for searches"
                    }
                ]
            }
            
            try:
                self.weaviate_client.schema.create_class(case_class)
                self.logger.info(f"Created case-specific class: {case_class_name}")
            except WeaviateException as e:
                if "already exists" in str(e):
                    self.logger.info(f"Class {case_class_name} already exists")
                else:
                    self.logger.warning(f"Could not create class {case_class_name}: {e}")
    
    async def initialize_neo4j(self) -> bool:
        """
        Initialize Neo4j graph database for document relationships.
        
        Returns:
            True if Neo4j initialized successfully
        """
        self.logger.info("Initializing Neo4j graph database...")
        
        try:
            # Connect to Neo4j
            await self._connect_neo4j()
            
            # Reset database if requested
            if self.config.reset_existing:
                self.logger.warning("Resetting Neo4j database")
                await self._reset_neo4j()
            
            # Create constraints and indexes
            await self._create_neo4j_constraints()
            
            # Create node labels and relationship types
            await self._create_neo4j_schema()
            
            self.initialization_results["neo4j"]["success"] = True
            self.logger.info("Neo4j initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Neo4j initialization failed: {e}"
            self.logger.error(error_msg)
            self.initialization_results["neo4j"]["error"] = str(e)
            
            # Neo4j is optional (Phase 2), so don't fail completely
            self.logger.warning("Neo4j initialization failed, but continuing (Phase 2 feature)")
            return True
    
    async def _connect_neo4j(self) -> None:
        """Connect to Neo4j with proper authentication."""
        self.logger.info(f"Connecting to Neo4j: {self.config.neo4j_uri}")
        
        auth = None
        if self.config.neo4j_username and self.config.neo4j_password:
            auth = (self.config.neo4j_username, self.config.neo4j_password)
        
        self.neo4j_driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=auth,
            max_connection_lifetime=3600,
            max_connection_pool_size=50
        )
        
        # Test connection
        async def test_connection():
            with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        
        if not await asyncio.get_event_loop().run_in_executor(None, lambda: test_connection()):
            raise ConnectionError("Neo4j connection test failed")
        
        self.logger.info("Neo4j connection established")
    
    async def _reset_neo4j(self) -> None:
        """Reset Neo4j database (development mode only)."""
        if self.config.mode == InitializationMode.PRODUCTION:
            raise ValueError("Cannot reset Neo4j in production mode")
        
        async def delete_all():
            with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                
                # Drop all constraints
                constraints = session.run("SHOW CONSTRAINTS").data()
                for constraint in constraints:
                    session.run(f"DROP CONSTRAINT {constraint['name']}")
                
                # Drop all indexes
                indexes = session.run("SHOW INDEXES").data()
                for index in indexes:
                    if index['type'] != 'LOOKUP':  # Don't drop lookup indexes
                        session.run(f"DROP INDEX {index['name']}")
        
        await asyncio.get_event_loop().run_in_executor(None, delete_all)
        self.logger.info("Neo4j database reset completed")
    
    async def _create_neo4j_constraints(self) -> None:
        """Create Neo4j constraints for data integrity."""
        self.logger.info("Creating Neo4j constraints")
        
        constraints = [
            "CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.documentId IS UNIQUE",
            "CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:LegalCase) REQUIRE c.caseId IS UNIQUE", 
            "CREATE CONSTRAINT legal_chunk_id IF NOT EXISTS FOR (ch:DocumentChunk) REQUIRE ch.chunkId IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE"
        ]
        
        async def create_constraints():
            with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        self.logger.info(f"Created constraint: {constraint.split('FOR')[0].split('CONSTRAINT')[1].strip()}")
                    except Exception as e:
                        if "already exists" in str(e) or "equivalent constraint already exists" in str(e):
                            continue
                        else:
                            self.logger.warning(f"Could not create constraint: {e}")
        
        await asyncio.get_event_loop().run_in_executor(None, create_constraints)
    
    async def _create_neo4j_schema(self) -> None:
        """Create Neo4j schema with indexes for legal relationships."""
        self.logger.info("Creating Neo4j indexes for performance")
        
        indexes = [
            "CREATE INDEX legal_document_case_id IF NOT EXISTS FOR (d:LegalDocument) ON (d.caseId)",
            "CREATE INDEX legal_document_type IF NOT EXISTS FOR (d:LegalDocument) ON (d.documentType)",
            "CREATE INDEX legal_document_created_at IF NOT EXISTS FOR (d:LegalDocument) ON (d.createdAt)",
            "CREATE INDEX legal_case_status IF NOT EXISTS FOR (c:LegalCase) ON (c.status)",
            "CREATE INDEX legal_case_created_at IF NOT EXISTS FOR (c:LegalCase) ON (c.createdAt)",
            "CREATE INDEX document_chunk_document_id IF NOT EXISTS FOR (ch:DocumentChunk) ON (ch.documentId)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relationshipType)",
            "CREATE INDEX relationship_confidence IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence)"
        ]
        
        async def create_indexes():
            with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                for index in indexes:
                    try:
                        session.run(index)
                        index_name = index.split('INDEX')[1].split('IF')[0].strip()
                        self.logger.info(f"Created index: {index_name}")
                    except Exception as e:
                        if "already exists" in str(e) or "equivalent index already exists" in str(e):
                            continue
                        else:
                            self.logger.warning(f"Could not create index: {e}")
        
        await asyncio.get_event_loop().run_in_executor(None, create_indexes)
    
    async def run_health_checks(self) -> Dict[str, bool]:
        """Run health checks on all initialized databases."""
        self.logger.info("Running database health checks")
        
        health_status = {}
        
        # MongoDB health check
        try:
            if self.mongodb_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.mongodb_client.admin.command, "ping"
                )
                health_status["mongodb"] = True
                self.logger.info("✓ MongoDB health check passed")
            else:
                health_status["mongodb"] = False
        except Exception as e:
            health_status["mongodb"] = False
            self.logger.error(f"✗ MongoDB health check failed: {e}")
        
        # Weaviate health check
        try:
            if self.weaviate_client and self.weaviate_client.is_ready():
                health_status["weaviate"] = True
                self.logger.info("✓ Weaviate health check passed")
            else:
                health_status["weaviate"] = False
        except Exception as e:
            health_status["weaviate"] = False
            self.logger.error(f"✗ Weaviate health check failed: {e}")
        
        # Neo4j health check
        try:
            if self.neo4j_driver:
                async def test_neo4j():
                    with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
                        result = session.run("RETURN 1 as test")
                        return result.single()["test"] == 1
                
                if await asyncio.get_event_loop().run_in_executor(None, test_neo4j):
                    health_status["neo4j"] = True
                    self.logger.info("✓ Neo4j health check passed")
                else:
                    health_status["neo4j"] = False
            else:
                health_status["neo4j"] = False
        except Exception as e:
            health_status["neo4j"] = False
            self.logger.error(f"✗ Neo4j health check failed: {e}")
        
        return health_status
    
    async def create_sample_data(self) -> None:
        """Create sample data for development and testing."""
        self.logger.info("Creating sample data for development")
        
        if self.config.mode == InitializationMode.PRODUCTION:
            self.logger.warning("Skipping sample data creation in production mode")
            return
        
        try:
            await self._create_sample_users()
            await self._create_sample_cases()
            await self._create_sample_documents()
            
            self.logger.info("Sample data created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create sample data: {e}")
    
    async def _create_sample_users(self) -> None:
        """Create sample users for development."""
        if not self.mongodb_client:
            return
        
        db = self.mongodb_client[self.config.mongodb_database]
        users = db.users
        
        sample_users = [
            {
                "user_id": "user_001",
                "username": "john_lawyer",
                "email": "john@lawfirm.com",
                "full_name": "John Doe",
                "role": "lawyer",
                "permissions": ["create_case", "upload_document", "search", "admin"],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "preferences": {
                    "search_type": "hybrid",
                    "results_per_page": 15
                },
                "is_active": True
            },
            {
                "user_id": "user_002", 
                "username": "jane_paralegal",
                "email": "jane@lawfirm.com",
                "full_name": "Jane Smith",
                "role": "paralegal",
                "permissions": ["upload_document", "search"],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "preferences": {
                    "search_type": "keyword",
                    "results_per_page": 10
                },
                "is_active": True
            }
        ]
        
        for user in sample_users:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, users.insert_one, user
                )
                self.logger.info(f"Created sample user: {user['username']}")
            except Exception as e:
                if "duplicate key" in str(e):
                    continue
                else:
                    self.logger.warning(f"Could not create sample user {user['username']}: {e}")
    
    async def _create_sample_cases(self) -> None:
        """Create sample legal cases for development.""" 
        if not self.mongodb_client:
            return
        
        db = self.mongodb_client[self.config.mongodb_database]
        cases = db.cases
        
        sample_cases = [
            {
                "case_id": "CASE_2025_01_15_ABC123XY",
                "user_id": "user_001",
                "case_name": "Patent Infringement Analysis - WiFi Technology",
                "initial_summary": "Analysis of patent infringement claims related to WiFi 6 technology implementation.",
                "visual_marker": {
                    "color": "#3B82F6",
                    "icon": "📄"
                },
                "priority": "high",
                "status": "active",
                "tags": ["patent", "wifi", "technology", "infringement"],
                "document_ids": [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "metadata": {
                    "estimated_hours": 120,
                    "practice_area": "intellectual_property"
                }
            },
            {
                "case_id": "CASE_2025_01_16_DEF456ZW",
                "user_id": "user_001", 
                "case_name": "Contract Review - Software Licensing Agreement",
                "initial_summary": "Review and analysis of software licensing agreement terms and conditions.",
                "visual_marker": {
                    "color": "#10B981",
                    "icon": "📋"
                },
                "priority": "medium",
                "status": "active", 
                "tags": ["contract", "software", "licensing"],
                "document_ids": [],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "metadata": {
                    "estimated_hours": 40,
                    "practice_area": "contract_law"
                }
            }
        ]
        
        for case in sample_cases:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, cases.insert_one, case
                )
                self.logger.info(f"Created sample case: {case['case_name']}")
            except Exception as e:
                if "duplicate key" in str(e):
                    continue
                else:
                    self.logger.warning(f"Could not create sample case {case['case_id']}: {e}")
    
    async def _create_sample_documents(self) -> None:
        """Create sample document metadata for development."""
        if not self.mongodb_client:
            return
        
        db = self.mongodb_client[self.config.mongodb_database]
        documents = db.documents
        
        sample_documents = [
            {
                "document_id": "DOC_20250115_ABC123XY",
                "case_id": "CASE_2025_01_15_ABC123XY",
                "user_id": "user_001",
                "document_name": "WiFi6_Patent_Application_2024.pdf",
                "original_filename": "WiFi6_Patent_Application_2024.pdf",
                "file_type": "pdf",
                "file_size": 2048000,
                "processing_status": "completed",
                "processing_metadata": {
                    "processing_time_ms": 15000,
                    "extraction_method": "LlamaIndex_PDF",
                    "chunk_count": 45
                },
                "page_count": 25,
                "legal_citations": ["35 U.S.C. § 101", "IEEE 802.11ax"],
                "section_headers": ["Technical Field", "Background", "Summary", "Claims"],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "metadata": {
                    "language": "en",
                    "confidence_score": 0.95
                }
            }
        ]
        
        for document in sample_documents:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, documents.insert_one, document
                )
                self.logger.info(f"Created sample document: {document['document_name']}")
            except Exception as e:
                if "duplicate key" in str(e):
                    continue
                else:
                    self.logger.warning(f"Could not create sample document {document['document_id']}: {e}")
    
    async def cleanup_connections(self) -> None:
        """Clean up database connections."""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
                self.logger.info("MongoDB connection closed")
            
            if self.neo4j_driver:
                self.neo4j_driver.close()
                self.logger.info("Neo4j connection closed")
            
            # Weaviate client doesn't need explicit cleanup
            
        except Exception as e:
            self.logger.warning(f"Error during connection cleanup: {e}")
    
    def print_initialization_summary(self) -> None:
        """Print summary of initialization results."""
        print("\n" + "="*60)
        print("DATABASE INITIALIZATION SUMMARY")
        print("="*60)
        
        for db_name, result in self.initialization_results.items():
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            print(f"{db_name.upper():<12}: {status}")
            
            if not result["success"] and result["error"]:
                print(f"             Error: {result['error']}")
        
        print("="*60)
        
        # Print next steps
        all_success = all(r["success"] for r in self.initialization_results.values())
        
        if all_success:
            print("🎉 All databases initialized successfully!")
            print("\nNext steps:")
            print("1. Start the backend application")
            print("2. Upload sample legal documents")
            print("3. Test search functionality")
            print("4. Configure Ollama models")
        else:
            print("⚠️  Some databases failed to initialize.")
            print("Please check the error messages above and retry.")


async def main():
    """Main entry point for database initialization script."""
    parser = argparse.ArgumentParser(description="Patexia Legal AI Database Initialization")
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing databases (WARNING: This will delete all data)"
    )
    
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create sample data for development"
    )
    
    parser.add_argument(
        "--production",
        action="store_true",
        help="Initialize for production mode"
    )
    
    parser.add_argument(
        "--database",
        choices=["mongodb", "weaviate", "neo4j", "all"],
        default="all",
        help="Specific database to initialize"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true", 
        help="Only run health checks, don't initialize"
    )
    
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to database configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DatabaseConfig()
    
    if args.config_file and args.config_file.exists():
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Apply command line overrides
    config.reset_existing = args.reset
    config.create_sample_data = args.sample_data
    
    if args.production:
        config.mode = InitializationMode.PRODUCTION
        config.create_sample_data = False  # Never create sample data in production
    
    # Confirm reset operation
    if args.reset and not args.check_only:
        if config.mode == InitializationMode.PRODUCTION:
            print("ERROR: Cannot reset databases in production mode")
            sys.exit(1)
        
        print("⚠️  WARNING: This will delete all existing data!")
        confirmation = input("Type 'yes' to confirm database reset: ")
        if confirmation.lower() != 'yes':
            print("Database reset cancelled")
            sys.exit(0)
    
    # Initialize databases
    initializer = DatabaseInitializer(config)
    
    try:
        if args.check_only:
            # Run health checks only
            print("Running database health checks...")
            await initializer.initialize_clients()
            health_status = await initializer.run_health_checks()
            
            all_healthy = all(health_status.values())
            print(f"\nOverall health status: {'✓ HEALTHY' if all_healthy else '✗ ISSUES FOUND'}")
            
            sys.exit(0 if all_healthy else 1)
        
        # Full initialization
        if args.database == "all":
            success = await initializer.initialize_all_databases()
        elif args.database == "mongodb":
            success = await initializer.initialize_mongodb()
        elif args.database == "weaviate":
            success = await initializer.initialize_weaviate()
        elif args.database == "neo4j":
            success = await initializer.initialize_neo4j()
        
        # Print summary
        initializer.print_initialization_summary()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nInitialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error during initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())