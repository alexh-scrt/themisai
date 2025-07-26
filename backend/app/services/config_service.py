"""
Configuration Management Service - Hot-Reload Business Logic

This module provides the business logic layer for configuration management in the
Patexia Legal AI Chatbot. It orchestrates hot-reload configuration changes, validates
settings, manages component dependencies, and coordinates real-time updates.

Key Features:
- Hot-reload configuration management with validation
- Component-specific configuration validation and callbacks
- Real-time configuration updates via WebSocket notifications
- Configuration backup and rollback mechanisms
- Performance monitoring and change tracking
- Business rule enforcement for configuration changes
- Service dependency management and restart coordination
- Configuration templates and presets management

Business Rules:
- Ollama model switching with availability validation
- LlamaIndex parameter constraints and optimization
- Capacity limit enforcement with override mechanisms
- UI parameter validation for user experience
- Service restart coordination for non-hot-reloadable changes
- Configuration change audit trail and compliance

Configuration Layers:
- base_config.json: Default system settings (read-only)
- development_config.json: Environment-specific overrides
- runtime_config.json: Hot-reloadable parameters
- user_config.json: User-specific preferences

Service Integration:
- Coordinates with ConfigurationWatcher for file monitoring
- Integrates with component services for setting updates
- Manages Ollama client configuration and model switching
- Coordinates with DocumentProcessor for parameter updates
- Provides WebSocket notifications for real-time admin updates

Architecture Features:
- Service-oriented configuration management
- Dependency injection for component configuration
- Event-driven configuration updates
- Transactional configuration changes with rollback
- Performance optimization for configuration operations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import copy

from ..core.config import get_settings, Settings
from ..core.config_watcher import (
    ConfigurationWatcher, ConfigChange, ConfigChangeType, 
    ConfigSection, ValidationResult
)
from ..core.websocket_manager import WebSocketManager
from ..core.ollama_client import get_ollama_client
from ..exceptions import (
    ConfigurationError, ValidationError, ServiceError,
    ErrorCode, raise_configuration_error, raise_validation_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class ConfigurationScope(str, Enum):
    """Scope of configuration changes."""
    SYSTEM = "system"
    USER = "user"
    SESSION = "session"
    COMPONENT = "component"


class ConfigurationPriority(str, Enum):
    """Priority levels for configuration changes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ServiceRestartRequirement(str, Enum):
    """Service restart requirements for configuration changes."""
    NONE = "none"
    HOT_RELOAD = "hot_reload"
    GRACEFUL_RESTART = "graceful_restart"
    FULL_RESTART = "full_restart"


@dataclass
class ConfigurationTemplate:
    """Template for configuration presets."""
    name: str
    description: str
    scope: ConfigurationScope
    settings: Dict[str, Any]
    restart_required: ServiceRestartRequirement = ServiceRestartRequirement.NONE
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def apply_to(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template to current configuration."""
        updated_config = copy.deepcopy(current_config)
        
        for key, value in self.settings.items():
            self._set_nested_value(updated_config, key, value)
        
        return updated_config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value


@dataclass
class ConfigurationChangeRequest:
    """Request for configuration change."""
    changes: Dict[str, Any]
    scope: ConfigurationScope = ConfigurationScope.SYSTEM
    priority: ConfigurationPriority = ConfigurationPriority.NORMAL
    user_id: Optional[str] = None
    reason: Optional[str] = None
    validate_only: bool = False
    force_restart: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "changes": self.changes,
            "scope": self.scope.value,
            "priority": self.priority.value,
            "user_id": self.user_id,
            "reason": self.reason,
            "validate_only": self.validate_only,
            "force_restart": self.force_restart
        }


@dataclass
class ConfigurationChangeResult:
    """Result of configuration change operation."""
    success: bool
    changes_applied: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    restart_required: ServiceRestartRequirement = ServiceRestartRequirement.NONE
    processing_time_ms: float = 0.0
    rollback_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        """Check if result has validation errors."""
        return len(self.validation_errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if result has warnings."""
        return len(self.warnings) > 0


class ComponentConfigurationManager:
    """Manages component-specific configuration handlers."""
    
    def __init__(self):
        """Initialize component configuration manager."""
        self.component_handlers: Dict[str, Callable] = {}
        self.validation_callbacks: Dict[str, List[Callable]] = {}
        self.restart_requirements: Dict[str, ServiceRestartRequirement] = {}
        
        # Register default component handlers
        self._register_default_handlers()
    
    def register_component_handler(
        self,
        component_name: str,
        handler: Callable[[Dict[str, Any]], None],
        restart_required: ServiceRestartRequirement = ServiceRestartRequirement.HOT_RELOAD
    ) -> None:
        """Register configuration handler for a component."""
        self.component_handlers[component_name] = handler
        self.restart_requirements[component_name] = restart_required
        
        logger.debug(f"Registered configuration handler for {component_name}")
    
    def register_validation_callback(
        self,
        section: str,
        callback: Callable[[Dict[str, Any]], ValidationResult]
    ) -> None:
        """Register validation callback for configuration section."""
        if section not in self.validation_callbacks:
            self.validation_callbacks[section] = []
        
        self.validation_callbacks[section].append(callback)
        logger.debug(f"Registered validation callback for {section}")
    
    async def handle_component_update(
        self,
        component_name: str,
        new_config: Dict[str, Any]
    ) -> bool:
        """Handle configuration update for a specific component."""
        if component_name not in self.component_handlers:
            logger.warning(f"No handler registered for component: {component_name}")
            return False
        
        try:
            handler = self.component_handlers[component_name]
            if asyncio.iscoroutinefunction(handler):
                await handler(new_config)
            else:
                handler(new_config)
            
            logger.info(f"Configuration updated for component: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration for {component_name}: {e}")
            return False
    
    async def validate_section(
        self,
        section: str,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate configuration section using registered callbacks."""
        if section not in self.validation_callbacks:
            return ValidationResult(is_valid=True)
        
        start_time = time.time()
        errors = []
        warnings = []
        
        for callback in self.validation_callbacks[section]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(config)
                else:
                    result = callback(config)
                
                if not result.is_valid:
                    errors.extend(result.errors)
                
                warnings.extend(result.warnings)
                
            except Exception as e:
                errors.append(f"Validation callback error: {str(e)}")
        
        validation_time = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time_ms=validation_time
        )
    
    def get_restart_requirement(self, component_name: str) -> ServiceRestartRequirement:
        """Get restart requirement for component."""
        return self.restart_requirements.get(component_name, ServiceRestartRequirement.HOT_RELOAD)
    
    def _register_default_handlers(self) -> None:
        """Register default configuration handlers for core components."""
        # Ollama configuration handler
        async def ollama_handler(config: Dict[str, Any]) -> None:
            client = get_ollama_client()
            
            # Update base URL if changed
            if 'base_url' in config:
                await client.disconnect()
                client._base_url = config['base_url']
                await client.connect()
            
            # Switch embedding model if changed
            if 'embedding_model' in config:
                model_name = config['embedding_model']
                if await client.validate_model_availability(model_name, auto_pull=True):
                    logger.info(f"Switched to embedding model: {model_name}")
                else:
                    raise ConfigurationError(f"Model {model_name} not available")
        
        self.register_component_handler('ollama', ollama_handler)
        
        # LlamaIndex configuration handler
        def llamaindex_handler(config: Dict[str, Any]) -> None:
            from llama_index.core import Settings
            
            if 'chunk_size' in config:
                Settings.chunk_size = config['chunk_size']
            
            if 'chunk_overlap' in config:
                Settings.chunk_overlap = config['chunk_overlap']
            
            logger.info("LlamaIndex settings updated")
        
        self.register_component_handler('llamaindex', llamaindex_handler)


class ConfigurationService:
    """
    Business logic service for configuration management with hot-reload capabilities.
    
    Orchestrates configuration changes, validates settings, manages component
    dependencies, and provides real-time configuration updates.
    """
    
    def __init__(
        self,
        websocket_manager: Optional[WebSocketManager] = None,
        config_watcher: Optional[ConfigurationWatcher] = None
    ):
        """
        Initialize configuration service.
        
        Args:
            websocket_manager: WebSocket manager for real-time notifications
            config_watcher: Configuration file watcher
        """
        self.websocket_manager = websocket_manager
        self.config_watcher = config_watcher
        self.component_manager = ComponentConfigurationManager()
        
        # Configuration state
        self.current_settings = get_settings()
        self.change_history: List[ConfigChange] = []
        self.configuration_templates: Dict[str, ConfigurationTemplate] = {}
        self.backup_configurations: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_changes": 0,
            "successful_changes": 0,
            "failed_changes": 0,
            "average_processing_time_ms": 0.0,
            "rollbacks_performed": 0
        }
        
        # Initialize templates
        self._initialize_configuration_templates()
        
        # Register validation callbacks
        self._register_validation_callbacks()
        
        logger.info("ConfigurationService initialized")
    
    async def apply_configuration_changes(
        self,
        request: ConfigurationChangeRequest
    ) -> ConfigurationChangeResult:
        """
        Apply configuration changes with validation and component updates.
        
        Args:
            request: Configuration change request
            
        Returns:
            ConfigurationChangeResult with operation details
        """
        start_time = time.time()
        
        try:
            with performance_context("config_service_apply_changes"):
                logger.info(
                    "Processing configuration change request",
                    changes=request.to_dict()
                )
                
                # Validate request
                validation_result = await self._validate_configuration_changes(request)
                
                if not validation_result.is_valid:
                    return ConfigurationChangeResult(
                        success=False,
                        validation_errors=validation_result.errors,
                        warnings=validation_result.warnings,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                
                # Return early if validation only
                if request.validate_only:
                    return ConfigurationChangeResult(
                        success=True,
                        warnings=validation_result.warnings,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                
                # Create backup before applying changes
                backup_key = await self._create_configuration_backup()
                
                # Apply changes
                result = await self._apply_changes_to_components(request)
                
                if result.success:
                    # Update configuration files
                    await self._persist_configuration_changes(request.changes)
                    
                    # Track successful change
                    self._track_configuration_change(request, result.changes_applied)
                    
                    # Send notifications
                    await self._send_configuration_notifications(
                        request, result.changes_applied
                    )
                    
                    result.rollback_available = True
                    result.metadata["backup_key"] = backup_key
                    
                else:
                    # Rollback on failure
                    await self._restore_configuration_backup(backup_key)
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                result.processing_time_ms = processing_time
                self._update_performance_metrics(result)
                
                return result
                
        except Exception as e:
            logger.error(f"Configuration change failed: {e}", exc_info=True)
            
            return ConfigurationChangeResult(
                success=False,
                validation_errors=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def get_current_configuration(
        self,
        section: Optional[str] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Get current configuration with optional filtering.
        
        Args:
            section: Optional section to filter (e.g., 'ollama', 'llamaindex')
            include_metadata: Whether to include configuration metadata
            
        Returns:
            Current configuration dictionary
        """
        try:
            with performance_context("config_service_get_current"):
                # Refresh settings
                self.current_settings = get_settings()
                config = self.current_settings.dict()
                
                # Filter by section if requested
                if section:
                    config = config.get(section, {})
                
                # Add metadata if requested
                if include_metadata:
                    config["_metadata"] = {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "total_changes": len(self.change_history),
                        "performance_metrics": self.performance_metrics.copy()
                    }
                
                return config
                
        except Exception as e:
            logger.error(f"Failed to get current configuration: {e}")
            return {}
    
    async def apply_configuration_template(
        self,
        template_name: str,
        user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> ConfigurationChangeResult:
        """
        Apply a predefined configuration template.
        
        Args:
            template_name: Name of the template to apply
            user_id: User applying the template
            reason: Reason for applying template
            
        Returns:
            ConfigurationChangeResult with template application details
        """
        try:
            if template_name not in self.configuration_templates:
                return ConfigurationChangeResult(
                    success=False,
                    validation_errors=[f"Template '{template_name}' not found"]
                )
            
            template = self.configuration_templates[template_name]
            current_config = await self.get_current_configuration()
            
            # Apply template to current configuration
            updated_config = template.apply_to(current_config)
            
            # Create change request
            request = ConfigurationChangeRequest(
                changes=template.settings,
                scope=template.scope,
                user_id=user_id,
                reason=reason or f"Applied template: {template_name}"
            )
            
            # Apply changes
            result = await self.apply_configuration_changes(request)
            
            if result.success:
                result.metadata["template_applied"] = template_name
                logger.info(f"Configuration template '{template_name}' applied successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply configuration template: {e}")
            return ConfigurationChangeResult(
                success=False,
                validation_errors=[str(e)]
            )
    
    async def rollback_configuration_changes(
        self,
        backup_key: str,
        user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> ConfigurationChangeResult:
        """
        Rollback configuration to a previous backup.
        
        Args:
            backup_key: Backup identifier to restore
            user_id: User performing rollback
            reason: Reason for rollback
            
        Returns:
            ConfigurationChangeResult with rollback details
        """
        try:
            with performance_context("config_service_rollback"):
                if backup_key not in self.backup_configurations:
                    return ConfigurationChangeResult(
                        success=False,
                        validation_errors=[f"Backup '{backup_key}' not found"]
                    )
                
                backup_config = self.backup_configurations[backup_key]
                current_config = await self.get_current_configuration()
                
                # Calculate changes needed for rollback
                rollback_changes = self._calculate_rollback_changes(
                    current_config, backup_config
                )
                
                # Create rollback request
                request = ConfigurationChangeRequest(
                    changes=rollback_changes,
                    scope=ConfigurationScope.SYSTEM,
                    priority=ConfigurationPriority.HIGH,
                    user_id=user_id,
                    reason=reason or f"Rollback to backup: {backup_key}"
                )
                
                # Apply rollback
                result = await self.apply_configuration_changes(request)
                
                if result.success:
                    self.performance_metrics["rollbacks_performed"] += 1
                    result.metadata["rollback_from"] = backup_key
                    
                    logger.info(f"Configuration rolled back to backup: {backup_key}")
                
                return result
                
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return ConfigurationChangeResult(
                success=False,
                validation_errors=[str(e)]
            )
    
    async def get_configuration_history(
        self,
        limit: int = 50,
        section: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get configuration change history with filtering.
        
        Args:
            limit: Maximum number of changes to return
            section: Optional section filter
            user_id: Optional user filter
            
        Returns:
            List of configuration changes
        """
        try:
            filtered_changes = []
            
            for change in reversed(self.change_history[-limit:]):
                # Apply filters
                if section and change.section != section:
                    continue
                
                if user_id and change.metadata.get("user_id") != user_id:
                    continue
                
                filtered_changes.append(change.to_dict())
            
            return filtered_changes
            
        except Exception as e:
            logger.error(f"Failed to get configuration history: {e}")
            return []
    
    async def validate_configuration(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate configuration without applying changes.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation details
        """
        try:
            request = ConfigurationChangeRequest(
                changes=config,
                validate_only=True
            )
            
            return await self._validate_configuration_changes(request)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[str(e)]
            )
    
    def register_component_handler(
        self,
        component_name: str,
        handler: Callable[[Dict[str, Any]], None],
        restart_required: ServiceRestartRequirement = ServiceRestartRequirement.HOT_RELOAD
    ) -> None:
        """Register configuration handler for a component."""
        self.component_manager.register_component_handler(
            component_name, handler, restart_required
        )
    
    def register_validation_callback(
        self,
        section: str,
        callback: Callable[[Dict[str, Any]], ValidationResult]
    ) -> None:
        """Register validation callback for configuration section."""
        self.component_manager.register_validation_callback(section, callback)
    
    # Private helper methods
    
    async def _validate_configuration_changes(
        self,
        request: ConfigurationChangeRequest
    ) -> ValidationResult:
        """Validate configuration changes using registered validators."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Validate each section
        for section_name, section_config in request.changes.items():
            if not isinstance(section_config, dict):
                continue
            
            section_result = await self.component_manager.validate_section(
                section_name, section_config
            )
            
            errors.extend(section_result.errors)
            warnings.extend(section_result.warnings)
        
        # Additional business rule validation
        business_validation = await self._validate_business_rules(request)
        errors.extend(business_validation.errors)
        warnings.extend(business_validation.warnings)
        
        validation_time = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time_ms=validation_time
        )
    
    async def _validate_business_rules(
        self,
        request: ConfigurationChangeRequest
    ) -> ValidationResult:
        """Validate configuration changes against business rules."""
        errors = []
        warnings = []
        
        # Validate Ollama model availability
        if 'ollama' in request.changes:
            ollama_config = request.changes['ollama']
            if 'embedding_model' in ollama_config:
                model_name = ollama_config['embedding_model']
                
                try:
                    client = get_ollama_client()
                    if not await client.validate_model_availability(model_name):
                        errors.append(f"Embedding model '{model_name}' is not available")
                except Exception as e:
                    warnings.append(f"Could not validate model availability: {e}")
        
        # Validate LlamaIndex parameter constraints
        if 'llamaindex' in request.changes:
            llamaindex_config = request.changes['llamaindex']
            
            chunk_size = llamaindex_config.get('chunk_size')
            chunk_overlap = llamaindex_config.get('chunk_overlap')
            
            if chunk_size and chunk_overlap:
                if chunk_overlap >= chunk_size:
                    errors.append("Chunk overlap must be less than chunk size")
        
        # Validate capacity limits
        if 'capacity_limits' in request.changes:
            capacity_config = request.changes['capacity_limits']
            
            docs_per_case = capacity_config.get('documents_per_case')
            if docs_per_case and docs_per_case > 100:
                warnings.append(
                    f"High document limit ({docs_per_case}) may impact performance"
                )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def _apply_changes_to_components(
        self,
        request: ConfigurationChangeRequest
    ) -> ConfigurationChangeResult:
        """Apply configuration changes to relevant components."""
        changes_applied = {}
        restart_required = ServiceRestartRequirement.NONE
        
        try:
            for section_name, section_config in request.changes.items():
                if not isinstance(section_config, dict):
                    continue
                
                # Apply to component
                success = await self.component_manager.handle_component_update(
                    section_name, section_config
                )
                
                if success:
                    changes_applied[section_name] = section_config
                    
                    # Check restart requirement
                    component_restart = self.component_manager.get_restart_requirement(
                        section_name
                    )
                    
                    if component_restart.value > restart_required.value:
                        restart_required = component_restart
                else:
                    return ConfigurationChangeResult(
                        success=False,
                        validation_errors=[f"Failed to update component: {section_name}"]
                    )
            
            return ConfigurationChangeResult(
                success=True,
                changes_applied=changes_applied,
                restart_required=restart_required
            )
            
        except Exception as e:
            return ConfigurationChangeResult(
                success=False,
                validation_errors=[str(e)]
            )
    
    async def _persist_configuration_changes(
        self,
        changes: Dict[str, Any]
    ) -> None:
        """Persist configuration changes to runtime config file."""
        try:
            # Get current runtime config
            runtime_config_path = Path("config/runtime_config.json")
            
            if runtime_config_path.exists():
                with open(runtime_config_path, 'r') as f:
                    runtime_config = json.load(f)
            else:
                runtime_config = {}
            
            # Apply changes
            for section, config in changes.items():
                if section not in runtime_config:
                    runtime_config[section] = {}
                runtime_config[section].update(config)
            
            # Write back to file
            with open(runtime_config_path, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            
            logger.debug("Configuration changes persisted to runtime config")
            
        except Exception as e:
            logger.error(f"Failed to persist configuration changes: {e}")
    
    async def _create_configuration_backup(self) -> str:
        """Create backup of current configuration."""
        backup_key = f"backup_{int(time.time())}"
        current_config = await self.get_current_configuration()
        
        self.backup_configurations[backup_key] = current_config
        
        # Keep only last 10 backups
        if len(self.backup_configurations) > 10:
            oldest_key = min(self.backup_configurations.keys())
            del self.backup_configurations[oldest_key]
        
        logger.debug(f"Configuration backup created: {backup_key}")
        return backup_key
    
    async def _restore_configuration_backup(self, backup_key: str) -> bool:
        """Restore configuration from backup."""
        if backup_key not in self.backup_configurations:
            logger.error(f"Backup not found: {backup_key}")
            return False
        
        try:
            backup_config = self.backup_configurations[backup_key]
            
            # Apply backup configuration
            request = ConfigurationChangeRequest(
                changes=backup_config,
                scope=ConfigurationScope.SYSTEM,
                reason=f"Restore backup: {backup_key}"
            )
            
            result = await self._apply_changes_to_components(request)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_key}: {e}")
            return False
    
    def _calculate_rollback_changes(
        self,
        current_config: Dict[str, Any],
        target_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate changes needed to rollback to target configuration."""
        changes = {}
        
        # Find differences between current and target
        for section, target_section in target_config.items():
            if section in current_config:
                current_section = current_config[section]
                section_changes = {}
                
                for key, target_value in target_section.items():
                    current_value = current_section.get(key)
                    if current_value != target_value:
                        section_changes[key] = target_value
                
                if section_changes:
                    changes[section] = section_changes
        
        return changes
    
    def _track_configuration_change(
        self,
        request: ConfigurationChangeRequest,
        changes_applied: Dict[str, Any]
    ) -> None:
        """Track configuration change in history."""
        timestamp = datetime.now(timezone.utc)
        
        for section, section_changes in changes_applied.items():
            for key, new_value in section_changes.items():
                change = ConfigChange(
                    timestamp=timestamp,
                    change_type=ConfigChangeType.MODIFIED,
                    section=section,
                    key=key,
                    old_value=None,  # Would need to track previous values
                    new_value=new_value,
                    file_path="runtime_config.json",
                    success=True
                )
                
                # Add metadata
                change.metadata = {
                    "user_id": request.user_id,
                    "reason": request.reason,
                    "scope": request.scope.value,
                    "priority": request.priority.value
                }
                
                self.change_history.append(change)
        
        # Keep only last 1000 changes
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]
    
    async def _send_configuration_notifications(
        self,
        request: ConfigurationChangeRequest,
        changes_applied: Dict[str, Any]
    ) -> None:
        """Send WebSocket notifications for configuration changes."""
        if not self.websocket_manager:
            return
        
        try:
            notification_data = {
                "type": "configuration_updated",
                "changes": changes_applied,
                "user_id": request.user_id,
                "reason": request.reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Broadcast to all admin users
            await self.websocket_manager.broadcast_to_all(
                "config_update", notification_data
            )
            
        except Exception as e:
            logger.warning(f"Failed to send configuration notifications: {e}")
    
    def _update_performance_metrics(self, result: ConfigurationChangeResult) -> None:
        """Update performance metrics based on operation result."""
        self.performance_metrics["total_changes"] += 1
        
        if result.success:
            self.performance_metrics["successful_changes"] += 1
        else:
            self.performance_metrics["failed_changes"] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics["average_processing_time_ms"]
        total_changes = self.performance_metrics["total_changes"]
        
        self.performance_metrics["average_processing_time_ms"] = (
            (current_avg * (total_changes - 1) + result.processing_time_ms) / total_changes
        )
    
    def _initialize_configuration_templates(self) -> None:
        """Initialize predefined configuration templates."""
        # Performance optimization template
        self.configuration_templates["performance"] = ConfigurationTemplate(
            name="performance",
            description="Optimized settings for maximum performance",
            scope=ConfigurationScope.SYSTEM,
            settings={
                "ollama.concurrent_requests": 6,
                "llamaindex.chunk_size": 1024,
                "llamaindex.chunk_overlap": 200,
                "llamaindex.similarity_threshold": 0.8
            }
        )
        
        # Memory optimization template
        self.configuration_templates["memory_optimized"] = ConfigurationTemplate(
            name="memory_optimized",
            description="Settings optimized for lower memory usage",
            scope=ConfigurationScope.SYSTEM,
            settings={
                "ollama.concurrent_requests": 2,
                "llamaindex.chunk_size": 512,
                "llamaindex.chunk_overlap": 50,
                "capacity_limits.documents_per_case": 15
            }
        )
        
        # Development template
        self.configuration_templates["development"] = ConfigurationTemplate(
            name="development",
            description="Settings for development and testing",
            scope=ConfigurationScope.SYSTEM,
            settings={
                "ui.progress_update_interval": 500,
                "ui.websocket_heartbeat": 15,
                "capacity_limits.documents_per_case": 5
            }
        )
    
    def _register_validation_callbacks(self) -> None:
        """Register validation callbacks for configuration sections."""
        # Ollama validation
        def validate_ollama(config: Dict[str, Any]) -> ValidationResult:
            errors = []
            warnings = []
            
            if 'timeout' in config:
                timeout = config['timeout']
                if timeout < 1 or timeout > 300:
                    errors.append("Ollama timeout must be between 1 and 300 seconds")
            
            if 'concurrent_requests' in config:
                concurrent = config['concurrent_requests']
                if concurrent < 1 or concurrent > 10:
                    errors.append("Concurrent requests must be between 1 and 10")
                elif concurrent > 6:
                    warnings.append("High concurrent requests may impact performance")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
        
        self.component_manager.register_validation_callback('ollama', validate_ollama)
        
        # LlamaIndex validation
        def validate_llamaindex(config: Dict[str, Any]) -> ValidationResult:
            errors = []
            warnings = []
            
            if 'chunk_size' in config:
                chunk_size = config['chunk_size']
                if chunk_size < 100 or chunk_size > 2048:
                    errors.append("Chunk size must be between 100 and 2048")
            
            if 'similarity_threshold' in config:
                threshold = config['similarity_threshold']
                if threshold < 0.0 or threshold > 1.0:
                    errors.append("Similarity threshold must be between 0.0 and 1.0")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
        
        self.component_manager.register_validation_callback('llamaindex', validate_llamaindex)


# Factory function for dependency injection
def create_config_service(
    websocket_manager: Optional[WebSocketManager] = None
) -> ConfigurationService:
    """
    Factory function to create ConfigurationService with dependencies.
    
    Args:
        websocket_manager: Optional WebSocket manager for notifications
        
    Returns:
        Configured ConfigurationService instance
    """
    return ConfigurationService(websocket_manager=websocket_manager)