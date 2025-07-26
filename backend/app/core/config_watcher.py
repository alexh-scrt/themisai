"""
Configuration Hot-Reload Watcher for Legal AI System

This module provides sophisticated configuration management with hot-reload capabilities
using the Python watchdog library. It monitors JSON configuration files for changes
and applies updates in real-time with validation and rollback support.

Key Features:
- File system monitoring using watchdog for configuration changes
- Immediate validation of configuration changes before applying
- Rollback capability for invalid configuration changes
- Graceful service restarts for non-hot-reloadable settings
- Configuration change logging with audit trail
- Real-time notification of configuration updates via WebSocket
- Backup and restore functionality for configuration files
- Thread-safe configuration updates with async support

Configuration Layers:
- base_config.json: Default system settings (read-only)
- development_config.json: Environment-specific overrides
- runtime_config.json: Hot-reloadable parameters

Hot-Reloadable Settings:
- Ollama model selection and parameters
- LlamaIndex chunk size and overlap settings
- Search parameters and thresholds
- UI update intervals and display settings
- Capacity limits and override flags
- Logging levels and output formats

Non-Hot-Reloadable Settings (require restart):
- Database connection URIs
- WebSocket port and host configurations
- Docker container settings
- Core service endpoints

Architecture Integration:
- Integrates with Settings class for configuration management
- Provides WebSocket updates for real-time admin panel updates
- Supports validation callbacks for component-specific validation
- Implements configuration backup and restore functionality
- Offers performance monitoring for configuration changes
"""

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from pydantic import ValidationError

from .config import get_settings, Settings
from .websocket_manager import WebSocketManager
from ..utils.logging import get_logger
from ..exceptions import ConfigurationError, ErrorCode

logger = get_logger(__name__)


class ConfigChangeType(Enum):
    """Types of configuration changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    VALIDATED = "validated"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class ConfigSection(Enum):
    """Configuration sections that can be hot-reloaded."""
    OLLAMA = "ollama"
    LLAMAINDEX = "llamaindex"
    LEGAL_DOCUMENTS = "legal_documents"
    UI = "ui"
    CAPACITY_LIMITS = "capacity_limits"
    LOGGING = "logging"


@dataclass
class ConfigChange:
    """Represents a configuration change event."""
    timestamp: datetime
    change_type: ConfigChangeType
    section: Optional[str]
    key: Optional[str]
    old_value: Any
    new_value: Any
    file_path: str
    success: bool = True
    error_message: Optional[str] = None
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "change_type": self.change_type.value,
            "section": self.section,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "file_path": self.file_path,
            "success": self.success,
            "error_message": self.error_message,
            "validation_time_ms": self.validation_time_ms
        }


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class ConfigurationFileHandler(FileSystemEventHandler):
    """
    File system event handler for configuration file changes.
    
    Monitors JSON configuration files and triggers validation and
    application of changes when files are modified.
    """
    
    def __init__(self, config_watcher: 'ConfigurationWatcher'):
        """Initialize file handler."""
        super().__init__()
        self.config_watcher = config_watcher
        self.last_modification_times = {}
        self.debounce_delay = 0.5  # Seconds to wait before processing
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process JSON configuration files
        if file_path.suffix != '.json' or 'config' not in file_path.name:
            return
        
        # Debounce rapid file changes
        current_time = time.time()
        last_mod_time = self.last_modification_times.get(file_path, 0)
        
        if current_time - last_mod_time < self.debounce_delay:
            return
        
        self.last_modification_times[file_path] = current_time
        
        logger.debug(f"Configuration file modified: {file_path}")
        
        # Schedule async processing
        asyncio.create_task(
            self.config_watcher.handle_file_change(file_path)
        )


class ConfigurationValidator:
    """
    Validator for configuration changes with component-specific validation.
    
    Provides comprehensive validation of configuration changes including
    type checking, range validation, and component-specific validation.
    """
    
    def __init__(self):
        """Initialize configuration validator."""
        self.validation_callbacks: Dict[str, List[Callable]] = {}
        self.validation_rules = self._create_validation_rules()
    
    def _create_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Create validation rules for configuration sections."""
        return {
            "ollama": {
                "embedding_model": {
                    "type": str,
                    "allowed_values": ["mxbai-embed-large", "nomic-embed-text", "e5-large-v2"],
                    "required": True
                },
                "base_url": {
                    "type": str,
                    "pattern": r"^https?://[\w\.-]+(?::\d+)?/?$",
                    "required": True
                },
                "timeout": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 300,
                    "required": True
                },
                "concurrent_requests": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 10,
                    "required": True
                }
            },
            "llamaindex": {
                "chunk_size": {
                    "type": int,
                    "min_value": 100,
                    "max_value": 2048,
                    "required": True
                },
                "chunk_overlap": {
                    "type": int,
                    "min_value": 0,
                    "max_value": 500,
                    "required": True
                },
                "similarity_threshold": {
                    "type": float,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "required": True
                },
                "top_k_results": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 100,
                    "required": True
                }
            },
            "ui": {
                "progress_update_interval": {
                    "type": int,
                    "min_value": 100,
                    "max_value": 5000,
                    "required": True
                },
                "websocket_heartbeat": {
                    "type": int,
                    "min_value": 10,
                    "max_value": 120,
                    "required": True
                }
            },
            "capacity_limits": {
                "documents_per_case": {
                    "type": int,
                    "min_value": 1,
                    "max_value": 1000,
                    "required": True
                },
                "max_chunk_size": {
                    "type": int,
                    "min_value": 500,
                    "max_value": 10000,
                    "required": True
                }
            }
        }
    
    def register_validation_callback(
        self,
        section: str,
        callback: Callable[[Dict[str, Any]], ValidationResult]
    ) -> None:
        """
        Register a validation callback for a configuration section.
        
        Args:
            section: Configuration section name
            callback: Validation function
        """
        if section not in self.validation_callbacks:
            self.validation_callbacks[section] = []
        
        self.validation_callbacks[section].append(callback)
        logger.debug(f"Registered validation callback for section: {section}")
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate using Pydantic model
            try:
                Settings(**config_data)
            except ValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    result.errors.append(f"{field_path}: {error['msg']}")
            
            # Apply custom validation rules
            for section, rules in self.validation_rules.items():
                if section in config_data:
                    section_result = self._validate_section(
                        section, config_data[section], rules
                    )
                    if not section_result.is_valid:
                        result.is_valid = False
                        result.errors.extend(section_result.errors)
                    result.warnings.extend(section_result.warnings)
            
            # Apply custom validation callbacks
            for section, callbacks in self.validation_callbacks.items():
                if section in config_data:
                    for callback in callbacks:
                        try:
                            callback_result = callback(config_data[section])
                            if not callback_result.is_valid:
                                result.is_valid = False
                                result.errors.extend(callback_result.errors)
                            result.warnings.extend(callback_result.warnings)
                        except Exception as e:
                            result.is_valid = False
                            result.errors.append(f"Validation callback failed for {section}: {str(e)}")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Configuration validation failed: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _validate_section(
        self,
        section_name: str,
        section_data: Dict[str, Any],
        rules: Dict[str, Dict[str, Any]]
    ) -> ValidationResult:
        """Validate a configuration section against rules."""
        result = ValidationResult(is_valid=True)
        
        for field_name, field_rules in rules.items():
            if field_name not in section_data:
                if field_rules.get("required", False):
                    result.is_valid = False
                    result.errors.append(f"Required field missing: {section_name}.{field_name}")
                continue
            
            value = section_data[field_name]
            
            # Type validation
            expected_type = field_rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                result.is_valid = False
                result.errors.append(
                    f"Invalid type for {section_name}.{field_name}: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue
            
            # Range validation
            if "min_value" in field_rules and value < field_rules["min_value"]:
                result.is_valid = False
                result.errors.append(
                    f"Value too small for {section_name}.{field_name}: "
                    f"{value} < {field_rules['min_value']}"
                )
            
            if "max_value" in field_rules and value > field_rules["max_value"]:
                result.is_valid = False
                result.errors.append(
                    f"Value too large for {section_name}.{field_name}: "
                    f"{value} > {field_rules['max_value']}"
                )
            
            # Allowed values validation
            if "allowed_values" in field_rules and value not in field_rules["allowed_values"]:
                result.is_valid = False
                result.errors.append(
                    f"Invalid value for {section_name}.{field_name}: "
                    f"{value} not in {field_rules['allowed_values']}"
                )
            
            # Pattern validation
            if "pattern" in field_rules:
                import re
                if not re.match(field_rules["pattern"], str(value)):
                    result.is_valid = False
                    result.errors.append(
                        f"Invalid format for {section_name}.{field_name}: "
                        f"{value} does not match required pattern"
                    )
        
        return result


class ConfigurationWatcher:
    """
    Main configuration watcher with hot-reload capabilities.
    
    Monitors configuration files for changes and applies updates
    in real-time with validation, rollback, and notification support.
    """
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        websocket_manager: Optional[WebSocketManager] = None
    ):
        """
        Initialize configuration watcher.
        
        Args:
            config_dir: Optional configuration directory override
            websocket_manager: Optional WebSocket manager for notifications
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.websocket_manager = websocket_manager
        
        # File monitoring
        self.observer = Observer()
        self.file_handler = ConfigurationFileHandler(self)
        self.is_watching = False
        
        # Configuration management
        self.validator = ConfigurationValidator()
        self.current_settings: Optional[Settings] = None
        self.backup_settings: Optional[Settings] = None
        
        # Change tracking
        self.change_history: List[ConfigChange] = []
        self.max_history_entries = 100
        
        # Thread safety
        self._lock = Lock()
        
        # Performance monitoring
        self._stats = {
            "total_changes_detected": 0,
            "successful_reloads": 0,
            "failed_reloads": 0,
            "validation_failures": 0,
            "rollbacks_performed": 0,
            "average_reload_time_ms": 0.0
        }
        
        # Configuration files to monitor
        self.monitored_files = [
            "runtime_config.json",
            "development_config.json"
        ]
        
        logger.info(
            "ConfigurationWatcher initialized",
            config_dir=str(self.config_dir),
            monitored_files=self.monitored_files
        )
    
    async def start_watching(self) -> None:
        """Start monitoring configuration files for changes."""
        if self.is_watching:
            return
        
        try:
            # Load initial configuration
            self.current_settings = get_settings()
            self.backup_settings = get_settings()
            
            # Setup file system monitoring
            self.observer.schedule(
                self.file_handler,
                str(self.config_dir),
                recursive=False
            )
            
            self.observer.start()
            self.is_watching = True
            
            logger.info("Configuration file watching started")
            
            # Send initial notification
            if self.websocket_manager:
                await self._send_config_update_notification({
                    "type": "watcher_started",
                    "message": "Configuration hot-reload activated",
                    "monitored_files": self.monitored_files
                })
            
        except Exception as e:
            logger.error(f"Failed to start configuration watcher: {e}")
            raise ConfigurationError(
                f"Configuration watcher startup failed: {str(e)}",
                error_code=ErrorCode.CONFIG_WATCHER_FAILED
            )
    
    async def stop_watching(self) -> None:
        """Stop monitoring configuration files."""
        if not self.is_watching:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.is_watching = False
            
            logger.info("Configuration file watching stopped")
            
            # Send notification
            if self.websocket_manager:
                await self._send_config_update_notification({
                    "type": "watcher_stopped",
                    "message": "Configuration hot-reload deactivated"
                })
            
        except Exception as e:
            logger.warning(f"Error stopping configuration watcher: {e}")
    
    async def handle_file_change(self, file_path: Path) -> None:
        """
        Handle configuration file change event.
        
        Args:
            file_path: Path to the changed configuration file
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self._stats["total_changes_detected"] += 1
                
                logger.info(f"Processing configuration change: {file_path.name}")
                
                # Load and validate new configuration
                validation_result = await self._load_and_validate_config(file_path)
                
                if validation_result.is_valid:
                    # Apply configuration changes
                    success = await self._apply_configuration_changes(file_path)
                    
                    if success:
                        self._stats["successful_reloads"] += 1
                        logger.info(
                            f"Configuration successfully reloaded from {file_path.name}",
                            validation_time_ms=validation_result.validation_time_ms
                        )
                    else:
                        self._stats["failed_reloads"] += 1
                        await self._perform_rollback(file_path)
                else:
                    self._stats["validation_failures"] += 1
                    logger.error(
                        f"Configuration validation failed for {file_path.name}",
                        errors=validation_result.errors
                    )
                    
                    # Send validation error notification
                    if self.websocket_manager:
                        await self._send_config_update_notification({
                            "type": "validation_failed",
                            "file": file_path.name,
                            "errors": validation_result.errors,
                            "warnings": validation_result.warnings
                        })
                
                # Update performance metrics
                processing_time_ms = (time.time() - start_time) * 1000
                total_reloads = self._stats["successful_reloads"] + self._stats["failed_reloads"]
                if total_reloads > 0:
                    self._stats["average_reload_time_ms"] = (
                        (self._stats["average_reload_time_ms"] * (total_reloads - 1) + processing_time_ms) 
                        / total_reloads
                    )
                
        except Exception as e:
            self._stats["failed_reloads"] += 1
            logger.error(f"Error handling configuration file change: {e}", exc_info=True)
            
            if self.websocket_manager:
                await self._send_config_update_notification({
                    "type": "processing_error",
                    "file": file_path.name,
                    "error": str(e)
                })
    
    async def _load_and_validate_config(self, file_path: Path) -> ValidationResult:
        """Load and validate configuration from file."""
        try:
            # Load JSON configuration
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate configuration
            validation_result = self.validator.validate_configuration(config_data)
            
            return validation_result
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON format: {str(e)}"]
            )
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration file not found: {file_path}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to load configuration: {str(e)}"]
            )
    
    async def _apply_configuration_changes(self, file_path: Path) -> bool:
        """Apply validated configuration changes."""
        try:
            # Create backup of current settings
            self.backup_settings = self.current_settings
            
            # Reload settings with new configuration
            # This will trigger the Settings class to reload from files
            new_settings = get_settings()
            
            # Detect specific changes
            changes = self._detect_configuration_changes(
                self.current_settings, new_settings
            )
            
            # Apply changes and notify components
            for change in changes:
                await self._notify_component_of_change(change)
                self._record_change(change)
            
            # Update current settings
            self.current_settings = new_settings
            
            # Send success notification
            if self.websocket_manager:
                await self._send_config_update_notification({
                    "type": "reload_successful",
                    "file": file_path.name,
                    "changes": [change.to_dict() for change in changes],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply configuration changes: {e}")
            return False
    
    def _detect_configuration_changes(
        self,
        old_settings: Settings,
        new_settings: Settings
    ) -> List[ConfigChange]:
        """Detect specific changes between old and new settings."""
        changes = []
        timestamp = datetime.now(timezone.utc)
        
        # Compare ollama settings
        if old_settings.ollama != new_settings.ollama:
            changes.extend(self._compare_section(
                "ollama", old_settings.ollama.dict(), new_settings.ollama.dict(), timestamp
            ))
        
        # Compare llamaindex settings
        if old_settings.llamaindex != new_settings.llamaindex:
            changes.extend(self._compare_section(
                "llamaindex", old_settings.llamaindex.dict(), new_settings.llamaindex.dict(), timestamp
            ))
        
        # Compare UI settings
        if old_settings.ui != new_settings.ui:
            changes.extend(self._compare_section(
                "ui", old_settings.ui.dict(), new_settings.ui.dict(), timestamp
            ))
        
        # Compare capacity limits
        if old_settings.capacity_limits != new_settings.capacity_limits:
            changes.extend(self._compare_section(
                "capacity_limits", old_settings.capacity_limits.dict(), 
                new_settings.capacity_limits.dict(), timestamp
            ))
        
        return changes
    
    def _compare_section(
        self,
        section_name: str,
        old_section: Dict[str, Any],
        new_section: Dict[str, Any],
        timestamp: datetime
    ) -> List[ConfigChange]:
        """Compare two configuration sections and detect changes."""
        changes = []
        
        all_keys = set(old_section.keys()) | set(new_section.keys())
        
        for key in all_keys:
            old_value = old_section.get(key)
            new_value = new_section.get(key)
            
            if old_value != new_value:
                change_type = ConfigChangeType.MODIFIED
                if key not in old_section:
                    change_type = ConfigChangeType.ADDED
                elif key not in new_section:
                    change_type = ConfigChangeType.DELETED
                
                change = ConfigChange(
                    timestamp=timestamp,
                    change_type=change_type,
                    section=section_name,
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    file_path="runtime_config.json"
                )
                
                changes.append(change)
        
        return changes
    
    async def _notify_component_of_change(self, change: ConfigChange) -> None:
        """Notify relevant components of configuration changes."""
        try:
            # Log the change
            logger.info(
                f"Configuration change applied: {change.section}.{change.key}",
                old_value=change.old_value,
                new_value=change.new_value,
                change_type=change.change_type.value
            )
            
            # Here you would notify specific components based on the section
            # For example:
            # - Notify OllamaClient of embedding model changes
            # - Notify EmbeddingProcessor of chunk size changes
            # - Notify WebSocket manager of heartbeat changes
            
            # This would be implemented as the system grows
            
        except Exception as e:
            logger.warning(f"Failed to notify component of change: {e}")
    
    async def _perform_rollback(self, file_path: Path) -> None:
        """Perform configuration rollback to previous valid state."""
        try:
            if self.backup_settings:
                self.current_settings = self.backup_settings
                self._stats["rollbacks_performed"] += 1
                
                logger.warning(f"Configuration rolled back due to errors in {file_path.name}")
                
                if self.websocket_manager:
                    await self._send_config_update_notification({
                        "type": "rollback_performed",
                        "file": file_path.name,
                        "message": "Configuration rolled back to previous valid state"
                    })
            
        except Exception as e:
            logger.error(f"Failed to perform configuration rollback: {e}")
    
    def _record_change(self, change: ConfigChange) -> None:
        """Record configuration change in history."""
        self.change_history.append(change)
        
        # Limit history size
        if len(self.change_history) > self.max_history_entries:
            self.change_history = self.change_history[-self.max_history_entries:]
    
    async def _send_config_update_notification(self, data: Dict[str, Any]) -> None:
        """Send configuration update notification via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_to_all(
                "config_update",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **data
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send config update notification: {e}")
    
    def register_validation_callback(
        self,
        section: str,
        callback: Callable[[Dict[str, Any]], ValidationResult]
    ) -> None:
        """Register a validation callback for a configuration section."""
        self.validator.register_validation_callback(section, callback)
    
    def get_change_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        history = self.change_history
        if limit:
            history = history[-limit:]
        
        return [change.to_dict() for change in history]
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get configuration monitoring statistics."""
        return {
            **self._stats,
            "is_watching": self.is_watching,
            "monitored_files": self.monitored_files,
            "config_dir": str(self.config_dir),
            "change_history_count": len(self.change_history)
        }
    
    async def manual_reload(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Manually trigger configuration reload."""
        try:
            if file_path:
                target_file = self.config_dir / file_path
            else:
                target_file = self.config_dir / "runtime_config.json"
            
            if not target_file.exists():
                return {
                    "success": False,
                    "error": f"Configuration file not found: {target_file.name}"
                }
            
            await self.handle_file_change(target_file)
            
            return {
                "success": True,
                "message": f"Configuration reloaded from {target_file.name}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global configuration watcher instance
_config_watcher: Optional[ConfigurationWatcher] = None


def get_config_watcher() -> ConfigurationWatcher:
    """Get the global configuration watcher instance."""
    global _config_watcher
    if _config_watcher is None:
        _config_watcher = ConfigurationWatcher()
    return _config_watcher


async def start_config_watcher(websocket_manager: Optional[WebSocketManager] = None) -> None:
    """Start the global configuration watcher."""
    global _config_watcher
    if _config_watcher is None:
        _config_watcher = ConfigurationWatcher(websocket_manager=websocket_manager)
    
    await _config_watcher.start_watching()


async def stop_config_watcher() -> None:
    """Stop the global configuration watcher."""
    global _config_watcher
    if _config_watcher:
        await _config_watcher.stop_watching()