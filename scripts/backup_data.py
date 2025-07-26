#!/usr/bin/env python3
"""
=============================================================================
Patexia Legal AI Chatbot - Data Backup Script

This script provides comprehensive backup functionality for the legal AI system:
- MongoDB document and metadata backup
- Weaviate vector database backup
- Configuration files backup
- GridFS file storage backup
- Encrypted backup support for legal compliance
- Incremental and full backup modes
- Backup verification and integrity checks
- Automated cleanup and retention management
- Legal audit trail and compliance logging

Usage:
    python backup_data.py [--type TYPE] [--encrypt] [--verify] [--cleanup]
    
Examples:
    python backup_data.py --type full --encrypt --verify
    python backup_data.py --type incremental --cleanup
    python backup_data.py --type mongodb --encrypt
=============================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import zipfile
from dataclasses import dataclass, field
from enum import Enum

# Third-party imports
try:
    import pymongo
    from pymongo import MongoClient
    from gridfs import GridFS
    import weaviate
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import psutil
    import requests
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install pymongo weaviate-client cryptography psutil requests")
    sys.exit(1)


class BackupType(Enum):
    """Types of backup operations."""
    FULL = "full"
    INCREMENTAL = "incremental"
    MONGODB = "mongodb"
    WEAVIATE = "weaviate"
    CONFIG = "config"
    FILES = "files"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    
    # MongoDB settings
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "patexia_legal_ai"
    mongodb_auth_database: str = "admin"
    
    # Weaviate settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    
    # Backup paths
    backup_root: Path = Path("backups")
    temp_dir: Path = Path("temp")
    config_dir: Path = Path("../backend/config")
    
    # Retention settings
    retention_days: int = 30
    max_backup_files: int = 100
    
    # Compression settings
    compression_level: int = 6
    enable_compression: bool = True
    
    # Encryption settings
    encryption_key_file: Optional[Path] = None
    enable_encryption: bool = False
    
    # Performance settings
    chunk_size: int = 64 * 1024 * 1024  # 64MB chunks
    max_memory_usage: float = 0.8  # 80% of available memory
    
    # Legal compliance
    audit_log_file: Path = Path("backup_audit.log")
    compliance_mode: bool = True
    anonymize_data: bool = False


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    status: BackupStatus
    file_path: Optional[Path] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # Content metadata
    mongodb_collections: List[str] = field(default_factory=list)
    weaviate_classes: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    total_documents: int = 0
    total_vectors: int = 0
    
    # Legal compliance
    encrypted: bool = False
    anonymized: bool = False
    compliance_notes: Optional[str] = None


class LegalDataBackup:
    """
    Main backup management class for legal AI system.
    
    Handles comprehensive backup operations with legal compliance,
    encryption, verification, and audit trail functionality.
    """
    
    def __init__(self, config: BackupConfig):
        """
        Initialize backup manager with configuration.
        
        Args:
            config: Backup configuration settings
        """
        self.config = config
        self.logger = self._setup_logging()
        self.audit_logger = self._setup_audit_logging()
        
        # Initialize clients
        self.mongodb_client: Optional[MongoClient] = None
        self.weaviate_client: Optional[weaviate.Client] = None
        
        # Runtime state
        self.current_backup_id: Optional[str] = None
        self.backup_metadata: Dict[str, BackupMetadata] = {}
        
        # Ensure directories exist
        self._create_directories()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup main logging configuration."""
        logger = logging.getLogger("backup_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup audit logging for legal compliance."""
        audit_logger = logging.getLogger("backup_audit")
        audit_logger.setLevel(logging.INFO)
        
        if not audit_logger.handlers:
            handler = logging.FileHandler(self.config.audit_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(message)s'
            )
            handler.setFormatter(formatter)
            audit_logger.addHandler(handler)
            
        return audit_logger
    
    def _create_directories(self) -> None:
        """Create necessary backup directories."""
        directories = [
            self.config.backup_root,
            self.config.temp_dir,
            self.config.backup_root / "mongodb",
            self.config.backup_root / "weaviate",
            self.config.backup_root / "config",
            self.config.backup_root / "files",
            self.config.backup_root / "full"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup identifier."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}_{int(time.time())}"
    
    async def initialize_clients(self) -> None:
        """Initialize database clients."""
        try:
            # Initialize MongoDB client
            self.mongodb_client = MongoClient(
                self.config.mongodb_uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000
            )
            
            # Test MongoDB connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.mongodb_client.admin.command, "ping"
            )
            self.logger.info("MongoDB connection established")
            
            # Initialize Weaviate client
            auth_config = None
            if self.config.weaviate_api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.config.weaviate_api_key)
            
            self.weaviate_client = weaviate.Client(
                url=self.config.weaviate_url,
                auth_client_secret=auth_config,
                timeout_config=(30, 120)
            )
            
            # Test Weaviate connection
            if self.weaviate_client.is_ready():
                self.logger.info("Weaviate connection established")
            else:
                raise ConnectionError("Weaviate not ready")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize clients: {e}")
            raise
    
    async def create_backup(
        self,
        backup_type: BackupType,
        encrypt: bool = False,
        verify: bool = True
    ) -> BackupMetadata:
        """
        Create a backup of specified type.
        
        Args:
            backup_type: Type of backup to create
            encrypt: Whether to encrypt the backup
            verify: Whether to verify backup integrity
            
        Returns:
            Backup metadata
        """
        backup_id = self._generate_backup_id()
        self.current_backup_id = backup_id
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=datetime.now(timezone.utc),
            status=BackupStatus.PENDING,
            encrypted=encrypt
        )
        
        self.backup_metadata[backup_id] = metadata
        
        try:
            self.logger.info(f"Starting {backup_type.value} backup: {backup_id}")
            self._log_audit_event("BACKUP_STARTED", backup_id, backup_type.value)
            
            start_time = time.time()
            metadata.status = BackupStatus.IN_PROGRESS
            
            # Initialize clients if needed
            await self.initialize_clients()
            
            # Perform backup based on type
            if backup_type == BackupType.FULL:
                await self._create_full_backup(metadata)
            elif backup_type == BackupType.INCREMENTAL:
                await self._create_incremental_backup(metadata)
            elif backup_type == BackupType.MONGODB:
                await self._backup_mongodb(metadata)
            elif backup_type == BackupType.WEAVIATE:
                await self._backup_weaviate(metadata)
            elif backup_type == BackupType.CONFIG:
                await self._backup_config(metadata)
            elif backup_type == BackupType.FILES:
                await self._backup_files(metadata)
            else:
                raise ValueError(f"Unsupported backup type: {backup_type}")
            
            # Calculate duration
            metadata.duration_seconds = time.time() - start_time
            
            # Encrypt backup if requested
            if encrypt:
                await self._encrypt_backup(metadata)
            
            # Verify backup integrity
            if verify:
                await self._verify_backup(metadata)
            
            metadata.status = BackupStatus.COMPLETED
            self.logger.info(f"Backup completed: {backup_id}")
            self._log_audit_event("BACKUP_COMPLETED", backup_id, 
                               f"Duration: {metadata.duration_seconds:.2f}s")
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self.logger.error(f"Backup failed: {backup_id} - {e}")
            self._log_audit_event("BACKUP_FAILED", backup_id, str(e))
            raise
        
        return metadata
    
    async def _create_full_backup(self, metadata: BackupMetadata) -> None:
        """Create a complete system backup."""
        self.logger.info("Creating full system backup")
        
        # Create temporary directory for full backup
        temp_backup_dir = self.config.temp_dir / f"full_{metadata.backup_id}"
        temp_backup_dir.mkdir(exist_ok=True)
        
        try:
            # Backup MongoDB
            mongodb_metadata = BackupMetadata(
                backup_id=f"{metadata.backup_id}_mongodb",
                backup_type=BackupType.MONGODB,
                timestamp=metadata.timestamp,
                status=BackupStatus.PENDING
            )
            await self._backup_mongodb(mongodb_metadata)
            metadata.mongodb_collections.extend(mongodb_metadata.mongodb_collections)
            metadata.total_documents += mongodb_metadata.total_documents
            
            # Backup Weaviate
            weaviate_metadata = BackupMetadata(
                backup_id=f"{metadata.backup_id}_weaviate",
                backup_type=BackupType.WEAVIATE,
                timestamp=metadata.timestamp,
                status=BackupStatus.PENDING
            )
            await self._backup_weaviate(weaviate_metadata)
            metadata.weaviate_classes.extend(weaviate_metadata.weaviate_classes)
            metadata.total_vectors += weaviate_metadata.total_vectors
            
            # Backup configuration
            config_metadata = BackupMetadata(
                backup_id=f"{metadata.backup_id}_config",
                backup_type=BackupType.CONFIG,
                timestamp=metadata.timestamp,
                status=BackupStatus.PENDING
            )
            await self._backup_config(config_metadata)
            metadata.config_files.extend(config_metadata.config_files)
            
            # Create combined archive
            backup_path = self.config.backup_root / "full" / f"{metadata.backup_id}.tar.gz"
            await self._create_compressed_archive(temp_backup_dir, backup_path)
            
            metadata.file_path = backup_path
            metadata.file_size = backup_path.stat().st_size
            metadata.checksum = await self._calculate_checksum(backup_path)
            
        finally:
            # Cleanup temporary directory
            if temp_backup_dir.exists():
                shutil.rmtree(temp_backup_dir)
    
    async def _create_incremental_backup(self, metadata: BackupMetadata) -> None:
        """Create an incremental backup."""
        self.logger.info("Creating incremental backup")
        
        # Find last full backup
        last_backup = await self._find_last_backup(BackupType.FULL)
        if not last_backup:
            self.logger.warning("No previous full backup found, creating full backup")
            await self._create_full_backup(metadata)
            return
        
        # Get changes since last backup
        since_timestamp = last_backup.timestamp
        
        # Backup only changed MongoDB documents
        await self._backup_mongodb_incremental(metadata, since_timestamp)
        
        # Backup only changed Weaviate vectors
        await self._backup_weaviate_incremental(metadata, since_timestamp)
        
        # Backup only changed configuration files
        await self._backup_config_incremental(metadata, since_timestamp)
    
    async def _backup_mongodb(self, metadata: BackupMetadata) -> None:
        """Backup MongoDB database."""
        self.logger.info("Starting MongoDB backup")
        
        if not self.mongodb_client:
            raise RuntimeError("MongoDB client not initialized")
        
        db = self.mongodb_client[self.config.mongodb_database]
        backup_dir = self.config.backup_root / "mongodb"
        
        # Get list of collections
        collections = db.list_collection_names()
        metadata.mongodb_collections = collections
        
        total_documents = 0
        
        for collection_name in collections:
            self.logger.info(f"Backing up collection: {collection_name}")
            
            collection = db[collection_name]
            doc_count = collection.count_documents({})
            total_documents += doc_count
            
            # Export collection to JSON
            collection_file = backup_dir / f"{metadata.backup_id}_{collection_name}.json"
            
            with open(collection_file, 'w', encoding='utf-8') as f:
                documents = []
                cursor = collection.find()
                
                for doc in cursor:
                    # Convert ObjectId to string for JSON serialization
                    doc['_id'] = str(doc['_id'])
                    documents.append(doc)
                
                json.dump(documents, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Exported {doc_count} documents from {collection_name}")
        
        # Backup GridFS files if they exist
        if 'fs.files' in collections:
            await self._backup_gridfs(metadata)
        
        metadata.total_documents = total_documents
        
        # Create MongoDB dump using mongodump for binary backup
        await self._create_mongodump(metadata)
    
    async def _backup_gridfs(self, metadata: BackupMetadata) -> None:
        """Backup GridFS files."""
        self.logger.info("Backing up GridFS files")
        
        if not self.mongodb_client:
            raise RuntimeError("MongoDB client not initialized")
        
        db = self.mongodb_client[self.config.mongodb_database]
        fs = GridFS(db)
        
        gridfs_dir = self.config.backup_root / "files" / metadata.backup_id
        gridfs_dir.mkdir(exist_ok=True)
        
        # Export all GridFS files
        for grid_file in fs.find():
            file_path = gridfs_dir / f"{grid_file._id}_{grid_file.filename}"
            
            with open(file_path, 'wb') as f:
                f.write(grid_file.read())
            
            # Save file metadata
            metadata_file = gridfs_dir / f"{grid_file._id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'filename': grid_file.filename,
                    'length': grid_file.length,
                    'upload_date': grid_file.upload_date.isoformat(),
                    'md5': grid_file.md5,
                    'metadata': grid_file.metadata
                }, f, indent=2, default=str)
    
    async def _create_mongodump(self, metadata: BackupMetadata) -> None:
        """Create MongoDB dump using mongodump utility."""
        dump_dir = self.config.backup_root / "mongodb" / f"{metadata.backup_id}_dump"
        
        try:
            # Build mongodump command
            cmd = [
                'mongodump',
                '--uri', self.config.mongodb_uri,
                '--db', self.config.mongodb_database,
                '--out', str(dump_dir),
                '--gzip'
            ]
            
            # Run mongodump
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown mongodump error"
                raise RuntimeError(f"mongodump failed: {error_msg}")
            
            self.logger.info("MongoDB dump created successfully")
            
        except FileNotFoundError:
            self.logger.warning("mongodump utility not found, skipping binary dump")
    
    async def _backup_weaviate(self, metadata: BackupMetadata) -> None:
        """Backup Weaviate vector database."""
        self.logger.info("Starting Weaviate backup")
        
        if not self.weaviate_client:
            raise RuntimeError("Weaviate client not initialized")
        
        backup_dir = self.config.backup_root / "weaviate"
        
        # Get schema information
        schema = self.weaviate_client.schema.get()
        schema_file = backup_dir / f"{metadata.backup_id}_schema.json"
        
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Get list of classes
        classes = [cls['class'] for cls in schema.get('classes', [])]
        metadata.weaviate_classes = classes
        
        total_vectors = 0
        
        # Backup each class
        for class_name in classes:
            self.logger.info(f"Backing up Weaviate class: {class_name}")
            
            # Get all objects from the class
            result = self.weaviate_client.data_object.get(
                class_name=class_name,
                with_vector=True
            )
            
            objects = result.get('objects', [])
            total_vectors += len(objects)
            
            # Save objects to file
            class_file = backup_dir / f"{metadata.backup_id}_{class_name}.json"
            with open(class_file, 'w') as f:
                json.dump(objects, f, indent=2)
            
            self.logger.info(f"Exported {len(objects)} vectors from {class_name}")
        
        metadata.total_vectors = total_vectors
        
        # Try to create Weaviate backup using API if available
        await self._create_weaviate_backup(metadata)
    
    async def _create_weaviate_backup(self, metadata: BackupMetadata) -> None:
        """Create Weaviate backup using backup API."""
        try:
            # Create backup using Weaviate backup API
            backup_config = {
                "id": metadata.backup_id,
                "backend": "filesystem",
                "config": {
                    "path": str(self.config.backup_root / "weaviate")
                }
            }
            
            # Start backup
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.config.weaviate_url}/v1/backups/filesystem",
                    json=backup_config,
                    headers={"Content-Type": "application/json"}
                )
            )
            
            if response.status_code == 200:
                self.logger.info("Weaviate backup API called successfully")
            else:
                self.logger.warning(f"Weaviate backup API failed: {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"Weaviate backup API not available: {e}")
    
    async def _backup_config(self, metadata: BackupMetadata) -> None:
        """Backup configuration files."""
        self.logger.info("Starting configuration backup")
        
        backup_dir = self.config.backup_root / "config"
        config_backup_dir = backup_dir / metadata.backup_id
        config_backup_dir.mkdir(exist_ok=True)
        
        config_files = []
        
        # Copy configuration files
        if self.config.config_dir.exists():
            for config_file in self.config.config_dir.rglob("*.json"):
                relative_path = config_file.relative_to(self.config.config_dir)
                target_path = config_backup_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(config_file, target_path)
                config_files.append(str(relative_path))
            
            # Copy Python config files
            for config_file in self.config.config_dir.rglob("*.py"):
                relative_path = config_file.relative_to(self.config.config_dir)
                target_path = config_backup_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(config_file, target_path)
                config_files.append(str(relative_path))
        
        metadata.config_files = config_files
        
        # Create configuration archive
        config_archive = backup_dir / f"{metadata.backup_id}_config.tar.gz"
        await self._create_compressed_archive(config_backup_dir, config_archive)
        
        # Cleanup temporary directory
        shutil.rmtree(config_backup_dir)
    
    async def _backup_files(self, metadata: BackupMetadata) -> None:
        """Backup additional files and documents."""
        self.logger.info("Starting files backup")
        
        # This can be extended to backup additional file storage
        # For now, GridFS files are handled in MongoDB backup
        pass
    
    async def _backup_mongodb_incremental(
        self,
        metadata: BackupMetadata,
        since_timestamp: datetime
    ) -> None:
        """Create incremental MongoDB backup."""
        self.logger.info(f"Creating incremental MongoDB backup since {since_timestamp}")
        
        if not self.mongodb_client:
            raise RuntimeError("MongoDB client not initialized")
        
        db = self.mongodb_client[self.config.mongodb_database]
        backup_dir = self.config.backup_root / "mongodb"
        
        collections = db.list_collection_names()
        metadata.mongodb_collections = collections
        
        total_documents = 0
        
        for collection_name in collections:
            collection = db[collection_name]
            
            # Query for documents modified since last backup
            query = {
                "$or": [
                    {"created_at": {"$gte": since_timestamp}},
                    {"updated_at": {"$gte": since_timestamp}}
                ]
            }
            
            changed_docs = list(collection.find(query))
            total_documents += len(changed_docs)
            
            if changed_docs:
                # Export changed documents
                collection_file = backup_dir / f"{metadata.backup_id}_{collection_name}_incremental.json"
                
                with open(collection_file, 'w', encoding='utf-8') as f:
                    for doc in changed_docs:
                        doc['_id'] = str(doc['_id'])
                    
                    json.dump(changed_docs, f, indent=2, default=str, ensure_ascii=False)
                
                self.logger.info(f"Exported {len(changed_docs)} changed documents from {collection_name}")
        
        metadata.total_documents = total_documents
    
    async def _backup_weaviate_incremental(
        self,
        metadata: BackupMetadata,
        since_timestamp: datetime
    ) -> None:
        """Create incremental Weaviate backup."""
        self.logger.info(f"Creating incremental Weaviate backup since {since_timestamp}")
        
        # Weaviate doesn't have built-in change tracking
        # This would require custom timestamp tracking in metadata
        # For now, perform full backup
        await self._backup_weaviate(metadata)
    
    async def _backup_config_incremental(
        self,
        metadata: BackupMetadata,
        since_timestamp: datetime
    ) -> None:
        """Create incremental configuration backup."""
        self.logger.info(f"Creating incremental config backup since {since_timestamp}")
        
        backup_dir = self.config.backup_root / "config"
        config_backup_dir = backup_dir / metadata.backup_id
        config_backup_dir.mkdir(exist_ok=True)
        
        config_files = []
        
        if self.config.config_dir.exists():
            for config_file in self.config.config_dir.rglob("*"):
                if config_file.is_file():
                    # Check if file was modified since last backup
                    file_mtime = datetime.fromtimestamp(
                        config_file.stat().st_mtime, 
                        tz=timezone.utc
                    )
                    
                    if file_mtime >= since_timestamp:
                        relative_path = config_file.relative_to(self.config.config_dir)
                        target_path = config_backup_dir / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(config_file, target_path)
                        config_files.append(str(relative_path))
        
        metadata.config_files = config_files
        
        if config_files:
            # Create configuration archive
            config_archive = backup_dir / f"{metadata.backup_id}_config_incremental.tar.gz"
            await self._create_compressed_archive(config_backup_dir, config_archive)
        
        # Cleanup temporary directory
        if config_backup_dir.exists():
            shutil.rmtree(config_backup_dir)
    
    async def _encrypt_backup(self, metadata: BackupMetadata) -> None:
        """Encrypt backup file for legal compliance."""
        if not metadata.file_path or not metadata.file_path.exists():
            raise ValueError("No backup file to encrypt")
        
        self.logger.info(f"Encrypting backup: {metadata.backup_id}")
        
        # Generate encryption key if not provided
        encryption_key = await self._get_encryption_key()
        
        # Read original file
        with open(metadata.file_path, 'rb') as f:
            data = f.read()
        
        # Encrypt data
        fernet = Fernet(encryption_key)
        encrypted_data = fernet.encrypt(data)
        
        # Write encrypted file
        encrypted_path = metadata.file_path.with_suffix(metadata.file_path.suffix + '.enc')
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Remove original file
        metadata.file_path.unlink()
        
        # Update metadata
        metadata.file_path = encrypted_path
        metadata.file_size = len(encrypted_data)
        metadata.encrypted = True
        metadata.checksum = await self._calculate_checksum(encrypted_path)
        
        self._log_audit_event("BACKUP_ENCRYPTED", metadata.backup_id, str(encrypted_path))
    
    async def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        if self.config.encryption_key_file and self.config.encryption_key_file.exists():
            with open(self.config.encryption_key_file, 'rb') as f:
                return f.read()
        
        # Generate new key
        key = Fernet.generate_key()
        
        if self.config.encryption_key_file:
            # Save key for future use
            self.config.encryption_key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.encryption_key_file, 'wb') as f:
                f.write(key)
            
            # Secure key file permissions
            os.chmod(self.config.encryption_key_file, 0o600)
        
        return key
    
    async def _verify_backup(self, metadata: BackupMetadata) -> None:
        """Verify backup integrity."""
        if not metadata.file_path or not metadata.file_path.exists():
            raise ValueError("No backup file to verify")
        
        self.logger.info(f"Verifying backup: {metadata.backup_id}")
        
        # Calculate checksum
        calculated_checksum = await self._calculate_checksum(metadata.file_path)
        
        if metadata.checksum and calculated_checksum != metadata.checksum:
            metadata.status = BackupStatus.CORRUPTED
            raise ValueError("Backup checksum verification failed")
        
        # Additional verification for different backup types
        if metadata.backup_type in [BackupType.FULL, BackupType.MONGODB]:
            await self._verify_mongodb_backup(metadata)
        
        if metadata.backup_type in [BackupType.FULL, BackupType.WEAVIATE]:
            await self._verify_weaviate_backup(metadata)
        
        metadata.status = BackupStatus.VERIFIED
        self._log_audit_event("BACKUP_VERIFIED", metadata.backup_id, "Integrity check passed")
    
    async def _verify_mongodb_backup(self, metadata: BackupMetadata) -> None:
        """Verify MongoDB backup integrity."""
        # Basic verification - check if backup files exist and are readable
        backup_dir = self.config.backup_root / "mongodb"
        
        for collection in metadata.mongodb_collections:
            collection_file = backup_dir / f"{metadata.backup_id}_{collection}.json"
            if collection_file.exists():
                try:
                    with open(collection_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Corrupted backup file: {collection_file}")
    
    async def _verify_weaviate_backup(self, metadata: BackupMetadata) -> None:
        """Verify Weaviate backup integrity."""
        # Basic verification - check if backup files exist and are readable
        backup_dir = self.config.backup_root / "weaviate"
        
        schema_file = backup_dir / f"{metadata.backup_id}_schema.json"
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Corrupted schema file: {schema_file}")
        
        for class_name in metadata.weaviate_classes:
            class_file = backup_dir / f"{metadata.backup_id}_{class_name}.json"
            if class_file.exists():
                try:
                    with open(class_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Corrupted class file: {class_file}")
    
    async def _create_compressed_archive(self, source_dir: Path, target_path: Path) -> None:
        """Create compressed archive of directory."""
        self.logger.info(f"Creating compressed archive: {target_path}")
        
        def create_archive():
            with tarfile.open(target_path, 'w:gz', compresslevel=self.config.compression_level) as tar:
                tar.add(source_dir, arcname=source_dir.name)
        
        await asyncio.get_event_loop().run_in_executor(None, create_archive)
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        def calculate():
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        
        return await asyncio.get_event_loop().run_in_executor(None, calculate)
    
    async def _find_last_backup(self, backup_type: BackupType) -> Optional[BackupMetadata]:
        """Find the most recent backup of specified type."""
        matching_backups = [
            metadata for metadata in self.backup_metadata.values()
            if metadata.backup_type == backup_type and metadata.status == BackupStatus.COMPLETED
        ]
        
        if not matching_backups:
            return None
        
        return max(matching_backups, key=lambda x: x.timestamp)
    
    def _log_audit_event(self, event_type: str, backup_id: str, details: str) -> None:
        """Log audit event for legal compliance."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "backup_id": backup_id,
            "details": details,
            "user": os.getenv("USER", "system"),
            "hostname": os.getenv("HOSTNAME", "localhost")
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    async def cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        self.logger.info("Starting backup cleanup")
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
        
        # Clean up backup files
        for backup_dir in [
            self.config.backup_root / "mongodb",
            self.config.backup_root / "weaviate", 
            self.config.backup_root / "config",
            self.config.backup_root / "files",
            self.config.backup_root / "full"
        ]:
            if backup_dir.exists():
                for backup_file in backup_dir.iterdir():
                    if backup_file.is_file():
                        file_mtime = datetime.fromtimestamp(
                            backup_file.stat().st_mtime,
                            tz=timezone.utc
                        )
                        
                        if file_mtime < cutoff_date:
                            backup_file.unlink()
                            self.logger.info(f"Deleted old backup: {backup_file}")
                            self._log_audit_event("BACKUP_DELETED", str(backup_file), "Retention policy cleanup")
        
        # Clean up metadata
        old_metadata_keys = [
            backup_id for backup_id, metadata in self.backup_metadata.items()
            if metadata.timestamp < cutoff_date
        ]
        
        for backup_id in old_metadata_keys:
            del self.backup_metadata[backup_id]
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List all available backups."""
        return list(self.backup_metadata.values())
    
    async def restore_backup(self, backup_id: str) -> None:
        """Restore from backup (placeholder for future implementation)."""
        self.logger.info(f"Restore functionality for backup {backup_id} not yet implemented")
        self._log_audit_event("RESTORE_REQUESTED", backup_id, "Restore not implemented")
        
        # This would be implemented in a separate restoration script
        # to avoid accidental data overwrites during backup operations


async def main():
    """Main entry point for backup script."""
    parser = argparse.ArgumentParser(description="Patexia Legal AI Data Backup Tool")
    
    parser.add_argument(
        "--type",
        choices=["full", "incremental", "mongodb", "weaviate", "config", "files"],
        default="full",
        help="Type of backup to create"
    )
    
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt backup for legal compliance"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify backup integrity"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up old backups after creating new one"
    )
    
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to backup configuration file"
    )
    
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Number of days to retain backups"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = BackupConfig()
    
    if args.config_file and args.config_file.exists():
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    config.retention_days = args.retention_days
    config.enable_encryption = args.encrypt
    
    # Initialize backup manager
    backup_manager = LegalDataBackup(config)
    
    try:
        # Create backup
        backup_type = BackupType(args.type)
        metadata = await backup_manager.create_backup(
            backup_type=backup_type,
            encrypt=args.encrypt,
            verify=args.verify
        )
        
        print(f"Backup completed successfully:")
        print(f"  Backup ID: {metadata.backup_id}")
        print(f"  Type: {metadata.backup_type.value}")
        print(f"  Duration: {metadata.duration_seconds:.2f} seconds")
        print(f"  File: {metadata.file_path}")
        print(f"  Size: {metadata.file_size:,} bytes")
        print(f"  Encrypted: {metadata.encrypted}")
        
        if metadata.total_documents > 0:
            print(f"  Documents: {metadata.total_documents:,}")
        
        if metadata.total_vectors > 0:
            print(f"  Vectors: {metadata.total_vectors:,}")
        
        # Cleanup old backups if requested
        if args.cleanup:
            await backup_manager.cleanup_old_backups()
            print("Old backups cleaned up")
    
    except Exception as e:
        print(f"Backup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())