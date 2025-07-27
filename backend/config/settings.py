# Example of how to refactor FieldInfo objects to proper Field definitions
# in backend/config/settings.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# First, define proper nested model classes
class OllamaSettings(BaseModel):
    """Ollama configuration settings."""
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    embedding_model: str = Field(
        default="mxbai-embed-large",
        description="Default embedding model"
    )
    fallback_model: str = Field(
        default="nomic-embed-text", 
        description="Fallback embedding model"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )


class LlamaIndexSettings(BaseModel):
    """LlamaIndex configuration settings."""
    chunk_size: int = Field(
        default=512,
        description="Text chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between text chunks"
    )
    similarity_top_k: int = Field(
        default=5,
        description="Number of similar chunks to retrieve"
    )


class LegalDocumentSettings(BaseModel):
    """Legal document processing settings."""
    supported_formats: list[str] = Field(
        default=["pdf", "docx", "txt"],
        description="Supported document formats"
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size in MB"
    )
    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for scanned documents"
    )


class UISettings(BaseModel):
    """User interface configuration settings."""
    theme: str = Field(
        default="light",
        description="UI theme (light/dark)"
    )
    items_per_page: int = Field(
        default=20,
        description="Items per page in lists"
    )
    enable_animations: bool = Field(
        default=True,
        description="Enable UI animations"
    )


class CapacityLimits(BaseModel):
    """System capacity and resource limits."""
    max_documents_per_case: int = Field(
        default=25,
        description="Maximum documents per case"
    )
    max_concurrent_uploads: int = Field(
        default=5,
        description="Maximum concurrent file uploads"
    )
    max_search_results: int = Field(
        default=100,
        description="Maximum search results to return"
    )


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URL"
    )
    mongodb_database: str = Field(
        default="legal_ai",
        description="MongoDB database name"
    )
    weaviate_url: str = Field(
        default="http://localhost:8080",
        description="Weaviate server URL"
    )
    neo4j_url: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j server URL"
    )


class LoggingSettings(BaseModel):
    """Logging configuration settings."""
    
    # Basic logging configuration
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    format: str = Field(
        default="json",
        description="Log format (json/text)"
    )
    
    enable_correlation_ids: bool = Field(
        default=True,
        description="Enable correlation ID tracking"
    )
    
    # Request/Response logging configuration
    log_request_body: bool = Field(
        default=False,
        description="Enable request body logging (use carefully in production)"
    )
    
    log_response_body: bool = Field(
        default=False,
        description="Enable response body logging (use carefully in production)"
    )
    
    log_query_params: bool = Field(
        default=True,
        description="Include query parameters in request logs"
    )
    
    log_headers: bool = Field(
        default=True,
        description="Include headers in request logs"
    )
    
    log_user_agent: bool = Field(
        default=True,
        description="Include user agent in request logs"
    )
    
    # Security and privacy settings
    mask_sensitive_data: bool = Field(
        default=True,
        description="Mask sensitive data in logs (passwords, tokens, etc.)"
    )
    
    # Performance and size limits
    max_body_size: int = Field(
        default=1024,
        description="Maximum request/response body size to log (bytes)"
    )
    
    slow_request_threshold_ms: float = Field(
        default=1000.0,
        description="Threshold for slow request warnings (milliseconds)"
    )
    
    large_request_threshold_bytes: int = Field(
        default=1048576,  # 1MB
        description="Threshold for large request warnings (bytes)"
    )
    
    # Path exclusions
    excluded_paths: list[str] = Field(
        default_factory=lambda: ["/health", "/healthz", "/ping", "/metrics", "/static", "/docs", "/redoc"],
        description="Paths to exclude from request logging"
    )
    
    excluded_headers: list[str] = Field(
        default_factory=lambda: ["authorization", "cookie", "x-api-key", "x-auth-token"],
        description="Headers to exclude from logging"
    )
    
    # File logging configuration
    enable_file_logging: bool = Field(
        default=False,
        description="Enable logging to file"
    )
    
    log_file_path: str = Field(
        default="logs/legal_ai.log",
        description="Path to log file"
    )
    
    log_file_max_size_mb: int = Field(
        default=100,
        description="Maximum log file size in MB before rotation"
    )
    
    log_file_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring and metrics"
    )
    
    performance_metrics_interval: int = Field(
        default=1000,
        description="Interval for performance metrics logging (requests)"
    )
    
    # Business event logging
    enable_business_event_logging: bool = Field(
        default=True,
        description="Enable business event logging"
    )
    
    business_event_log_level: str = Field(
        default="INFO",
        description="Log level for business events"
    )
    
    # Development settings
    enable_debug_logging: bool = Field(
        default=False,
        description="Enable verbose debug logging"
    )
    
    log_sql_queries: bool = Field(
        default=False,
        description="Log database SQL queries (development only)"
    )
    
    def to_request_logging_config(self) -> "RequestLoggingConfig":
        """
        Convert LoggingSettings to RequestLoggingConfig for middleware.
        
        Returns:
            RequestLoggingConfig instance
        """
        from backend.app.api.middleware.logging import RequestLoggingConfig
        
        return RequestLoggingConfig(
            log_request_body=self.log_request_body,
            log_response_body=self.log_response_body,
            max_body_size=self.max_body_size,
            excluded_paths=set(self.excluded_paths),
            excluded_headers=set(self.excluded_headers),
            log_query_params=self.log_query_params,
            log_user_agent=self.log_user_agent,
            mask_sensitive_data=self.mask_sensitive_data
        )
    
    def get_log_level_numeric(self) -> int:
        """
        Get numeric log level for Python logging.
        
        Returns:
            Numeric log level
        """
        import logging
        return getattr(logging, self.level.upper(), logging.INFO)
    
    def is_development_mode(self) -> bool:
        """
        Check if logging is in development mode.
        
        Returns:
            True if in development mode
        """
        return self.enable_debug_logging or self.log_sql_queries


# Now the main Settings class with proper field definitions
class Settings(BaseModel):
    """
    Application configuration settings.
    
    This class uses Pydantic BaseSettings to automatically load configuration
    from environment variables, .env files, and default values.
    """
    
    # Basic application settings - these are primitive types
    app_name: str = Field(
        default="Patexia Legal AI Chatbot",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        description="Debug mode enabled"
    )
    
    environment: str = Field(
        default="development",
        description="Environment (development/staging/production)"
    )
    
    # Nested settings - these are Pydantic model instances
    ollama: OllamaSettings = Field(
        default_factory=OllamaSettings,
        description="Ollama configuration"
    )
    
    llamaindex: LlamaIndexSettings = Field(
        default_factory=LlamaIndexSettings,
        description="LlamaIndex configuration"
    )
    
    legal_documents: LegalDocumentSettings = Field(
        default_factory=LegalDocumentSettings,
        description="Legal document processing settings"
    )
    
    ui: UISettings = Field(
        default_factory=UISettings,
        description="User interface settings"
    )
    
    capacity_limits: CapacityLimits = Field(
        default_factory=CapacityLimits,
        description="System capacity limits"
    )
    
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="Database configuration"
    )
    
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging configuration"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # Allows OLLAMA__BASE_URL environment variables
        case_sensitive = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for JSON serialization."""
        # Now this works perfectly because all fields are proper Pydantic models or primitives
        return self.model_dump(
            exclude_unset=False,
            exclude_none=False
        )
    
    def get_nested_setting(self, path: str, default: Any = None) -> Any:
        """
        Get a nested setting using dot notation.
        
        Args:
            path: Dot-separated path (e.g., "ollama.base_url")
            default: Default value if path not found
            
        Returns:
            Setting value or default
        """
        try:
            current = self
            for part in path.split('.'):
                current = getattr(current, part)
            return current
        except AttributeError:
            return default
    
    def update_nested_setting(self, path: str, value: Any) -> None:
        """
        Update a nested setting using dot notation.
        
        Args:
            path: Dot-separated path (e.g., "ollama.base_url")
            value: New value to set
        """
        parts = path.split('.')
        current = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Set the final value
        setattr(current, parts[-1], value)
    
    def validate_configuration(self) -> Dict[str, list[str]]:
        """
        Validate the entire configuration.
        
        Returns:
            Dictionary with validation errors by section
        """
        errors = {}
        
        # Validate URLs
        url_fields = [
            ("ollama.base_url", self.ollama.base_url),
            ("database.mongodb_url", self.database.mongodb_url),
            ("database.weaviate_url", self.database.weaviate_url),
            ("database.neo4j_url", self.database.neo4j_url),
        ]
        
        for field_path, url in url_fields:
            if not self._is_valid_url(url):
                section = field_path.split('.')[0]
                if section not in errors:
                    errors[section] = []
                errors[section].append(f"Invalid URL: {field_path} = {url}")
        
        # Validate positive integers
        positive_int_fields = [
            ("capacity_limits.max_documents_per_case", self.capacity_limits.max_documents_per_case),
            ("capacity_limits.max_concurrent_uploads", self.capacity_limits.max_concurrent_uploads),
            ("legal_documents.max_file_size_mb", self.legal_documents.max_file_size_mb),
        ]
        
        for field_path, value in positive_int_fields:
            if not isinstance(value, int) or value <= 0:
                section = field_path.split('.')[0]
                if section not in errors:
                    errors[section] = []
                errors[section].append(f"Must be positive integer: {field_path} = {value}")
        
        return errors
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False


# Environment variable mapping examples:
# OLLAMA__BASE_URL=http://ollama-server:11434
# DATABASE__MONGODB_URL=mongodb://mongo:27017
# CAPACITY_LIMITS__MAX_DOCUMENTS_PER_CASE=50
# LOGGING__LEVEL=DEBUG

# Usage example:
def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


# Example of how environment variables would map:
"""
Environment Variable Examples:
=============================

# Basic settings
APP_NAME="My Legal AI"
APP_VERSION="2.0.0" 
DEBUG=true
ENVIRONMENT=production

# Nested Ollama settings
OLLAMA__BASE_URL=http://remote-ollama:11434
OLLAMA__EMBEDDING_MODEL=custom-model
OLLAMA__TIMEOUT_SECONDS=60

# Database settings
DATABASE__MONGODB_URL=mongodb://user:pass@cluster.mongodb.net/
DATABASE__WEAVIATE_URL=https://my-cluster.weaviate.network

# Capacity limits
CAPACITY_LIMITS__MAX_DOCUMENTS_PER_CASE=100
CAPACITY_LIMITS__MAX_CONCURRENT_UPLOADS=10

# Logging
LOGGING__LEVEL=ERROR
LOGGING__FORMAT=text
"""