"""
Component Configuration Manager for Legal AI System

This module provides centralized management of configuration for different system
components. It handles component-specific validation, dependency management,
restart requirements, and configuration distribution.

Key Features:
- Component-specific configuration validation and updates
- Dependency tracking and impact analysis
- Hot-reload capability assessment and execution
- Component health monitoring and status tracking
- Configuration rollback and recovery mechanisms
- Performance impact analysis for configuration changes

Supported Components:
- Database connections (MongoDB, Weaviate, Neo4j)
- AI models and embedding services (Ollama integration)
- Search engines and indexing services
- WebSocket and real-time notification systems
- Document processing pipeline components
- Security and authentication services

Architecture Integration:
- Used by ConfigurationService for component-specific operations
- Integrates with system monitoring for health checks
- Supports graceful configuration updates with minimal downtime
- Provides configuration validation specific to each component type
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
import threading
from abc import ABC, abstractmethod

from ..utils.logging import get_logger, performance_context
from ..models.domain.configuration import ValidationResult, ValidationIssue, ValidationSeverity

logger = get_logger(__name__)


class ComponentType(str, Enum):
    """Types of system components that can be managed."""
    DATABASE = "database"
    AI_MODEL = "ai_model"
    SEARCH_ENGINE = "search_engine"
    WEBSOCKET = "websocket"
    DOCUMENT_PROCESSOR = "document_processor"
    SECURITY = "security"
    MONITORING = "monitoring"
    CACHE = "cache"
    QUEUE = "queue"
    FILE_STORAGE = "file_storage"


class ComponentStatus(str, Enum):
    """Current status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class RestartStrategy(str, Enum):
    """Strategies for restarting components."""
    HOT_RELOAD = "hot_reload"          # Update without restart
    GRACEFUL_RESTART = "graceful"      # Graceful shutdown and restart
    FORCE_RESTART = "force"            # Immediate restart
    ROLLING_RESTART = "rolling"        # Restart instances one by one
    FULL_SYSTEM_RESTART = "full"       # Restart entire system


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""
    name: str
    component_type: ComponentType
    required: bool = True
    restart_on_change: bool = False
    validation_order: int = 100  # Lower numbers validate first


@dataclass
class ComponentInfo:
    """Information about a system component."""
    name: str
    component_type: ComponentType
    status: ComponentStatus = ComponentStatus.UNKNOWN
    
    # Configuration
    current_config: Dict[str, Any] = field(default_factory=dict)
    config_schema: Optional[Dict[str, Any]] = None
    config_path: str = ""  # Dot-notation path in main config
    
    # Dependencies
    dependencies: List[ComponentDependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # Components that depend on this one
    
    # Restart behavior
    restart_strategy: RestartStrategy = RestartStrategy.GRACEFUL_RESTART
    supports_hot_reload: bool = False
    restart_timeout_seconds: int = 30
    
    # Health monitoring
    last_health_check: Optional[datetime] = None
    health_check_interval_seconds: int = 60
    consecutive_failures: int = 0
    
    # Performance tracking
    config_update_count: int = 0
    last_config_update: Optional[datetime] = None
    average_restart_time_seconds: float = 0.0
    
    # Metadata
    description: str = ""
    version: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentValidator(ABC):
    """Abstract base class for component-specific validators."""
    
    @abstractmethod
    async def validate_config(
        self,
        component: ComponentInfo,
        new_config: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate configuration for a specific component.
        
        Args:
            component: Component information
            new_config: Proposed new configuration
            current_config: Current configuration
            
        Returns:
            ValidationResult with any issues found
        """
        pass
    
    @abstractmethod
    async def apply_config(
        self,
        component: ComponentInfo,
        new_config: Dict[str, Any]
    ) -> bool:
        """
        Apply new configuration to the component.
        
        Args:
            component: Component information
            new_config: Configuration to apply
            
        Returns:
            True if successfully applied, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self, component: ComponentInfo) -> ComponentStatus:
        """
        Check the health status of the component.
        
        Args:
            component: Component to check
            
        Returns:
            Current component status
        """
        pass


class DatabaseValidator(ComponentValidator):
    """Validator for database components."""
    
    async def validate_config(
        self,
        component: ComponentInfo,
        new_config: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate database configuration."""
        result = ValidationResult(configuration_scope=f"component.{component.name}")
        
        # Check required fields
        required_fields = ["host", "port"]
        for field in required_fields:
            if field not in new_config:
                result.add_error(
                    code="DB_MISSING_FIELD",
                    message=f"Required field '{field}' missing",
                    path=f"{component.config_path}.{field}",
                    category="configuration"
                )
        
        # Validate connection parameters
        if "timeout" in new_config:
            timeout = new_config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                result.add_error(
                    code="DB_INVALID_TIMEOUT",
                    message="Timeout must be a positive number",
                    path=f"{component.config_path}.timeout",
                    current_value=timeout,
                    expected_value="positive number",
                    category="performance"
                )
        
        # Check for security issues
        if new_config.get("ssl_enabled", True) is False:
            result.add_warning(
                code="DB_SSL_DISABLED",
                message="SSL is disabled for database connection",
                path=f"{component.config_path}.ssl_enabled",
                category="security",
                suggestion="Enable SSL for secure database connections"
            )
        
        return result
    
    async def apply_config(self, component: ComponentInfo, new_config: Dict[str, Any]) -> bool:
        """Apply database configuration."""
        try:
            # In a real implementation, this would update the database connection
            component.current_config.update(new_config)
            component.last_config_update = datetime.now(timezone.utc)
            component.config_update_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to apply database config for {component.name}: {e}")
            return False
    
    async def health_check(self, component: ComponentInfo) -> ComponentStatus:
        """Check database health."""
        # In a real implementation, this would test the database connection
        # For now, return a simulated status
        return ComponentStatus.HEALTHY


class AIModelValidator(ComponentValidator):
    """Validator for AI model components."""
    
    async def validate_config(
        self,
        component: ComponentInfo,
        new_config: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate AI model configuration."""
        result = ValidationResult(configuration_scope=f"component.{component.name}")
        
        # Check model name
        if "model_name" not in new_config:
            result.add_error(
                code="MODEL_MISSING_NAME",
                message="Model name is required",
                path=f"{component.config_path}.model_name",
                category="configuration"
            )
        
        # Validate GPU settings
        if "gpu_enabled" in new_config and new_config["gpu_enabled"]:
            if "gpu_memory_limit" not in new_config:
                result.add_warning(
                    code="MODEL_NO_GPU_LIMIT",
                    message="GPU memory limit not specified",
                    path=f"{component.config_path}.gpu_memory_limit",
                    category="performance",
                    suggestion="Set GPU memory limit to prevent out-of-memory errors"
                )
        
        # Check batch size
        if "batch_size" in new_config:
            batch_size = new_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                result.add_error(
                    code="MODEL_INVALID_BATCH_SIZE",
                    message="Batch size must be a positive integer",
                    path=f"{component.config_path}.batch_size",
                    current_value=batch_size,
                    category="performance"
                )
            elif batch_size > 100:
                result.add_warning(
                    code="MODEL_LARGE_BATCH_SIZE",
                    message="Large batch size may cause memory issues",
                    path=f"{component.config_path}.batch_size",
                    current_value=batch_size,
                    category="performance"
                )
        
        return result
    
    async def apply_config(self, component: ComponentInfo, new_config: Dict[str, Any]) -> bool:
        """Apply AI model configuration."""
        try:
            # In a real implementation, this would update the model configuration
            component.current_config.update(new_config)
            component.last_config_update = datetime.now(timezone.utc)
            component.config_update_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to apply AI model config for {component.name}: {e}")
            return False
    
    async def health_check(self, component: ComponentInfo) -> ComponentStatus:
        """Check AI model health."""
        # In a real implementation, this would test model availability
        return ComponentStatus.HEALTHY


class ComponentConfigurationManager:
    """
    Centralized manager for system component configurations.
    
    Provides component registration, validation, configuration updates,
    and health monitoring for all system components.
    """
    
    def __init__(self):
        """Initialize component configuration manager."""
        self.components: Dict[str, ComponentInfo] = {}
        self.validators: Dict[ComponentType, ComponentValidator] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.update_history: deque = deque(maxlen=1000)
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        
        # Register default validators
        self._register_default_validators()
        
        # Initialize core components
        self._initialize_core_components()
        
        logger.info("ComponentConfigurationManager initialized")
    
    def _register_default_validators(self) -> None:
        """Register default validators for component types."""
        self.validators[ComponentType.DATABASE] = DatabaseValidator()
        self.validators[ComponentType.AI_MODEL] = AIModelValidator()
        # Additional validators would be registered here
    
    def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        # MongoDB component
        self.register_component(ComponentInfo(
            name="mongodb",
            component_type=ComponentType.DATABASE,
            config_path="database.mongodb",
            supports_hot_reload=False,
            restart_strategy=RestartStrategy.GRACEFUL_RESTART,
            description="MongoDB document database"
        ))
        
        # Weaviate component
        self.register_component(ComponentInfo(
            name="weaviate",
            component_type=ComponentType.DATABASE,
            config_path="database.weaviate",
            supports_hot_reload=False,
            restart_strategy=RestartStrategy.GRACEFUL_RESTART,
            description="Weaviate vector database"
        ))
        
        # Ollama embedding service
        self.register_component(ComponentInfo(
            name="ollama_embedding",
            component_type=ComponentType.AI_MODEL,
            config_path="ai.embedding",
            supports_hot_reload=True,
            restart_strategy=RestartStrategy.HOT_RELOAD,
            description="Ollama embedding model service"
        ))
    
    def register_component(self, component: ComponentInfo) -> None:
        """
        Register a new component with the manager.
        
        Args:
            component: Component information to register
        """
        with self._lock:
            self.components[component.name] = component
            
            # Build dependency graph
            for dep in component.dependencies:
                self.dependency_graph[component.name].add(dep.name)
                
                # Add reverse dependency
                if dep.name in self.components:
                    self.components[dep.name].dependents.append(component.name)
            
            logger.info(f"Registered component: {component.name} ({component.component_type.value})")
    
    def get_component(self, name: str) -> Optional[ComponentInfo]:
        """
        Get component information by name.
        
        Args:
            name: Component name
            
        Returns:
            ComponentInfo if found, None otherwise
        """
        return self.components.get(name)
    
    def list_components(self, component_type: Optional[ComponentType] = None) -> List[ComponentInfo]:
        """
        List all registered components, optionally filtered by type.
        
        Args:
            component_type: Optional component type filter
            
        Returns:
            List of component information
        """
        components = list(self.components.values())
        
        if component_type:
            components = [c for c in components if c.component_type == component_type]
        
        return components
    
    async def validate_component_config(
        self,
        component_name: str,
        new_config: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate configuration for a specific component.
        
        Args:
            component_name: Name of component to validate
            new_config: Proposed new configuration
            
        Returns:
            ValidationResult with validation details
        """
        component = self.get_component(component_name)
        if not component:
            result = ValidationResult(configuration_scope=f"component.{component_name}")
            result.add_error(
                code="COMPONENT_NOT_FOUND",
                message=f"Component '{component_name}' not found",
                category="configuration"
            )
            return result
        
        # Get validator for component type
        validator = self.validators.get(component.component_type)
        if not validator:
            result = ValidationResult(configuration_scope=f"component.{component_name}")
            result.add_warning(
                code="NO_VALIDATOR",
                message=f"No validator available for component type {component.component_type.value}",
                category="validation"
            )
            return result
        
        # Perform validation
        with performance_context(f"validate_component_{component_name}"):
            result = await validator.validate_config(
                component=component,
                new_config=new_config,
                current_config=component.current_config
            )
        
        return result
    
    async def apply_component_config(
        self,
        component_name: str,
        new_config: Dict[str, Any],
        validate_first: bool = True
    ) -> Tuple[bool, ValidationResult]:
        """
        Apply new configuration to a component.
        
        Args:
            component_name: Name of component to update
            new_config: New configuration to apply
            validate_first: Whether to validate before applying
            
        Returns:
            Tuple of (success, validation_result)
        """
        component = self.get_component(component_name)
        if not component:
            result = ValidationResult(configuration_scope=f"component.{component_name}")
            result.add_error(
                code="COMPONENT_NOT_FOUND",
                message=f"Component '{component_name}' not found"
            )
            return False, result
        
        # Validate first if requested
        if validate_first:
            validation_result = await self.validate_component_config(component_name, new_config)
            if not validation_result.is_valid:
                return False, validation_result
        else:
            validation_result = ValidationResult.create_success(f"component.{component_name}")
        
        # Get validator and apply configuration
        validator = self.validators.get(component.component_type)
        if not validator:
            validation_result.add_error(
                code="NO_VALIDATOR",
                message=f"No validator available for component type {component.component_type.value}"
            )
            return False, validation_result
        
        try:
            success = await validator.apply_config(component, new_config)
            
            if success:
                # Record update in history
                self.update_history.append({
                    "component_name": component_name,
                    "timestamp": datetime.now(timezone.utc),
                    "config_changes": new_config,
                    "success": True
                })
                
                logger.info(f"Successfully applied configuration to component: {component_name}")
            else:
                validation_result.add_error(
                    code="CONFIG_APPLY_FAILED",
                    message=f"Failed to apply configuration to component {component_name}"
                )
            
            return success, validation_result
            
        except Exception as e:
            logger.error(f"Error applying configuration to {component_name}: {e}")
            validation_result.add_error(
                code="CONFIG_APPLY_ERROR",
                message=f"Error applying configuration: {str(e)}"
            )
            return False, validation_result
    
    async def check_component_health(self, component_name: str) -> ComponentStatus:
        """
        Check the health status of a component.
        
        Args:
            component_name: Name of component to check
            
        Returns:
            Current component status
        """
        component = self.get_component(component_name)
        if not component:
            return ComponentStatus.UNKNOWN
        
        validator = self.validators.get(component.component_type)
        if not validator:
            return ComponentStatus.UNKNOWN
        
        try:
            status = await validator.health_check(component)
            component.status = status
            component.last_health_check = datetime.now(timezone.utc)
            
            if status != ComponentStatus.HEALTHY:
                component.consecutive_failures += 1
            else:
                component.consecutive_failures = 0
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            component.status = ComponentStatus.UNHEALTHY
            component.consecutive_failures += 1
            return ComponentStatus.UNHEALTHY
    
    async def check_all_components_health(self) -> Dict[str, ComponentStatus]:
        """
        Check health status of all registered components.
        
        Returns:
            Dictionary mapping component names to their status
        """
        health_results = {}
        
        # Check all components concurrently
        tasks = {
            name: self.check_component_health(name)
            for name in self.components.keys()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                health_results[name] = ComponentStatus.UNHEALTHY
                logger.error(f"Health check exception for {name}: {result}")
            else:
                health_results[name] = result
        
        return health_results
    
    def get_component_dependencies(self, component_name: str) -> List[str]:
        """
        Get list of components that the specified component depends on.
        
        Args:
            component_name: Name of component
            
        Returns:
            List of dependency component names
        """
        return list(self.dependency_graph.get(component_name, set()))
    
    def get_component_dependents(self, component_name: str) -> List[str]:
        """
        Get list of components that depend on the specified component.
        
        Args:
            component_name: Name of component
            
        Returns:
            List of dependent component names
        """
        component = self.get_component(component_name)
        return component.dependents if component else []
    
    def get_update_history(self, component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get configuration update history.
        
        Args:
            component_name: Optional component name filter
            
        Returns:
            List of update history entries
        """
        history = list(self.update_history)
        
        if component_name:
            history = [h for h in history if h.get("component_name") == component_name]
        
        return history