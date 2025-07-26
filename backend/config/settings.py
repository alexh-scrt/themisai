"""
Configuration management with hot-reload capabilities for Patexia Legal AI Chatbot.

This module provides a centralized configuration system that supports:
- JSON-based configuration files with layered overrides
- Hot-reload functionality for development velocity
- Environment variable integration
- Type validation with Pydantic
- Runtime configuration updates without service restart
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


logger = logging.getLogger(__name__)


class OllamaSettings(BaseSettings):
    """Ollama model management configuration."""
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama service base URL"
    )
    
    embedding_model: str = Field(
        default="mxbai-embed-large",
        description="Primary embedding model for legal documents"
    )
    
    fallback_model: str = Field(
        default="nomic-embed-text",
        description="Fallback embedding model for reliability"
    )
    
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Text generation model for future features"
    )
    
    timeout: int = Field(
        default=45,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    
    concurrent_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent requests to Ollama"
    )
    
    gpu_memory_limit: str = Field(
        default="8GB",
        description="GPU memory allocation limit"
    )
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra='ignore')


class LlamaIndexSettings(BaseSettings):
    """LlamaIndex document processing configuration."""
    
    chunk_size: int = Field(
        default=768,
        ge=100,
        le=4096,
        description="Text chunk size for legal documents"
    )
    
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between chunks for context preservation"
    )
    
    hybrid_search_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Balance between semantic and keyword search (0=keyword, 1=semantic)"
    )
    
    top_k_results: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum number of search results to return"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for search results"
    )
    
    model_config = SettingsConfigDict(env_prefix="LLAMAINDEX_", extra='ignore')


class LegalDocumentSettings(BaseSettings):
    """Legal document processing specific configuration."""
    
    preserve_legal_structure: bool = Field(
        default=True,
        description="Maintain legal document hierarchy during processing"
    )
    
    section_aware_chunking: bool = Field(
        default=True,
        description="Respect legal section boundaries when chunking"
    )
    
    citation_extraction: bool = Field(
        default=True,
        description="Extract and preserve legal citations"
    )
    
    metadata_enhancement: bool = Field(
        default=True,
        description="Enhance document metadata with legal-specific information"
    )
    
    model_config = SettingsConfigDict(env_prefix="LEGAL_", extra='ignore')


class UISettings(BaseSettings):
    """User interface and WebSocket configuration."""
    
    progress_update_interval: int = Field(
        default=250,
        ge=100,
        le=5000,
        description="WebSocket progress update interval in milliseconds"
    )
    
    max_search_history: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of search queries to store per case"
    )
    
    websocket_heartbeat: int = Field(
        default=25,
        ge=5,
        le=120,
        description="WebSocket heartbeat interval in seconds"
    )
    
    result_highlight_context: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of surrounding sentences for search result highlighting"
    )
    
    model_config = SettingsConfigDict(env_prefix="UI_", extra='ignore')


class CapacityLimitSettings(BaseSettings):
    """System capacity and resource limits."""
    
    documents_per_case: int = Field(
        default=25,
        ge=1,
        le=1000,
        description="Maximum documents per legal case"
    )
    
    manual_override_enabled: bool = Field(
        default=True,
        description="Allow manual override of capacity limits"
    )
    
    max_chunk_size: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Maximum size for individual text chunks"
    )
    
    embedding_cache_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of embeddings to cache in memory"
    )
    
    model_config = SettingsConfigDict(env_prefix="CAPACITY_", extra='ignore')


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""
    
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    
    mongodb_database: str = Field(
        default="patexia_legal_ai",
        description="MongoDB database name"
    )
    
    weaviate_url: str = Field(
        default="http://localhost:8080",
        description="Weaviate instance URL"
    )
    
    weaviate_api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key (if authentication enabled)"
    )
    
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI for Phase 2"
    )
    
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    
    neo4j_password: Optional[str] = Field(
        default=None,
        description="Neo4j password"
    )
    
    model_config = SettingsConfigDict(env_prefix="DB_", extra='ignore')


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(
        default="DEBUG",
        description="Logging level for POC development"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    enable_correlation_ids: bool = Field(
        default=True,
        description="Include correlation IDs for request tracking"
    )
    
    log_sql_queries: bool = Field(
        default=False,
        description="Log database queries for debugging"
    )
    
    model_config = SettingsConfigDict(env_prefix="LOG_", extra='ignore')


def json_config_settings_source() -> Dict[str, Any]:
    """
    Load configuration from JSON files with layered override support.
    
    Priority order (highest to lowest):
    1. runtime_config.json (hot-reloadable)
    2. development_config.json (environment-specific)
    3. base_config.json (defaults)
    """
    config_dir = Path(__file__).parent
    config_data = {}
    
    # Load base configuration
    base_config_path = config_dir / "base_config.json"
    if base_config_path.exists():
        try:
            with open(base_config_path, 'r') as f:
                base_config = json.load(f)
                config_data.update(base_config)
                logger.debug(f"Loaded base configuration from {base_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load base config: {e}")
    
    # Load environment-specific configuration
    env_config_path = config_dir / "development_config.json"
    if env_config_path.exists():
        try:
            with open(env_config_path, 'r') as f:
                env_config = json.load(f)
                config_data.update(env_config)
                logger.debug(f"Loaded environment configuration from {env_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load environment config: {e}")
    
    # Load runtime configuration (hot-reloadable)
    runtime_config_path = config_dir / "runtime_config.json"
    if runtime_config_path.exists():
        try:
            with open(runtime_config_path, 'r') as f:
                runtime_config = json.load(f)
                config_data.update(runtime_config)
                logger.debug(f"Loaded runtime configuration from {runtime_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load runtime config: {e}")
    
    return config_data


class Settings(BaseSettings):
    """
    Main application settings with hot-reload support and JSON configuration.
    
    This class aggregates all configuration sections and provides a single
    point of access for application configuration with hot-reload capabilities.
    """
    
    # Application metadata
    app_name: str = Field(
        default="Patexia Legal AI Chatbot",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode for POC development"
    )
    
    # Configuration sections
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    llamaindex: LlamaIndexSettings = Field(default_factory=LlamaIndexSettings)
    legal_documents: LegalDocumentSettings = Field(default_factory=LegalDocumentSettings)
    ui: UISettings = Field(default_factory=UISettings)
    capacity_limits: CapacityLimitSettings = Field(default_factory=CapacityLimitSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    @field_validator('ollama', mode='before')
    @classmethod
    def parse_ollama_settings(cls, v):
        """Parse ollama settings from nested configuration."""
        if isinstance(v, dict):
            return OllamaSettings(**v)
        return v
    
    @field_validator('llamaindex', mode='before')
    @classmethod
    def parse_llamaindex_settings(cls, v):
        """Parse llamaindex settings from nested configuration."""
        if isinstance(v, dict):
            return LlamaIndexSettings(**v)
        return v
    
    @field_validator('legal_documents', mode='before')
    @classmethod
    def parse_legal_documents_settings(cls, v):
        """Parse legal documents settings from nested configuration."""
        if isinstance(v, dict):
            return LegalDocumentSettings(**v)
        return v
    
    @field_validator('ui', mode='before')
    @classmethod
    def parse_ui_settings(cls, v):
        """Parse UI settings from nested configuration."""
        if isinstance(v, dict):
            return UISettings(**v)
        return v
    
    @field_validator('capacity_limits', mode='before')
    @classmethod
    def parse_capacity_limits_settings(cls, v):
        """Parse capacity limits settings from nested configuration."""
        if isinstance(v, dict):
            return CapacityLimitSettings(**v)
        return v
    
    @field_validator('database', mode='before')
    @classmethod
    def parse_database_settings(cls, v):
        """Parse database settings from nested configuration."""
        if isinstance(v, dict):
            return DatabaseSettings(**v)
        return v
    
    @field_validator('logging', mode='before')
    @classmethod
    def parse_logging_settings(cls, v):
        """Parse logging settings from nested configuration."""
        if isinstance(v, dict):
            return LoggingSettings(**v)
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for JSON serialization."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "ollama": self.ollama.model_dump(),
            "llamaindex": self.llamaindex.model_dump(),
            "legal_documents": self.legal_documents.model_dump(),
            "ui": self.ui.model_dump(),
            "capacity_limits": self.capacity_limits.model_dump(),
            "database": self.database.model_dump(),
            "logging": self.logging.model_dump(),
        }
    
    def validate_ollama_model_availability(self) -> bool:
        """
        Validate that configured Ollama models are available.
        This will be used by the hot-reload system for immediate validation.
        """
        # This method will be implemented when we create the Ollama client
        # For now, return True to allow configuration loading
        return True
    
    def get_runtime_config_path(self) -> Path:
        """Get the path to the runtime configuration file for hot-reload."""
        return Path(__file__).parent / "runtime_config.json"
    
    def save_runtime_config(self) -> bool:
        """
        Save current configuration to runtime_config.json for persistence.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            runtime_config_path = self.get_runtime_config_path()
            with open(runtime_config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Runtime configuration saved to {runtime_config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save runtime configuration: {e}")
            return False
    
    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra='ignore'
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Define the order of configuration sources.
        
        Priority order (highest to lowest):
        1. Environment variables
        2. JSON configuration files (layered)
        3. Default values
        """
        return (
            env_settings,
            json_config_settings_source,
            init_settings,
        )


# Global settings instance for application use
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This function provides dependency injection support for FastAPI
    and ensures a single settings instance across the application.
    
    Returns:
        Settings: The global settings instance
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from configuration files.
    
    This function supports hot-reload functionality by creating a new
    settings instance with the latest configuration values.
    
    Returns:
        Settings: New settings instance with reloaded configuration
    """
    global settings
    try:
        # Create new settings instance to reload from files
        new_settings = Settings()
        
        # Validate the new configuration
        if new_settings.validate_ollama_model_availability():
            settings = new_settings
            logger.info("Settings reloaded successfully")
        else:
            logger.error("Settings validation failed, keeping current configuration")
            
    except Exception as e:
        logger.error(f"Failed to reload settings: {e}")
    
    return settings


def update_runtime_setting(section: str, key: str, value: Any) -> bool:
    """
    Update a single runtime setting and persist to file.
    
    Args:
        section: Configuration section name (e.g., 'ollama', 'llamaindex')
        key: Setting key within the section
        value: New value for the setting
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    global settings
    try:
        # Get current settings as dict
        current_config = settings.to_dict()
        
        # Update the specific setting
        if section in current_config:
            current_config[section][key] = value
        else:
            logger.error(f"Configuration section '{section}' not found")
            return False
        
        # Create new settings instance with updated values
        new_settings = Settings(**current_config)
        
        # Validate the new configuration
        if new_settings.validate_ollama_model_availability():
            # Save to runtime config file
            if new_settings.save_runtime_config():
                # Update global settings
                settings = new_settings
                logger.info(f"Updated {section}.{key} = {value}")
                return True
        else:
            logger.error("Configuration validation failed")
            
    except Exception as e:
        logger.error(f"Failed to update runtime setting: {e}")
    
    return False