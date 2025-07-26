"""
Pydantic API Schemas for Administrative Operations in LegalAI System

This module defines the request/response schemas for administrative API endpoints:
- System configuration management and hot-reload operations
- Resource monitoring and performance metrics
- WebSocket connection management and monitoring
- System health checks and service status
- User activity tracking and management
- Alert and notification management
- Performance analytics and optimization

Key Features:
- Configuration validation and change tracking
- Real-time system resource monitoring
- WebSocket connection pool management
- Administrative alerts and notifications
- Performance metrics and trend analysis
- System health monitoring and diagnostics
- User session and activity management
- Service status and component health

Administrative Operations:
- Configuration: Manage system settings with hot-reload capabilities
- Monitoring: Track system resources and performance metrics
- Alerts: Configure and manage system alerts and notifications
- Users: Monitor and manage user sessions and activities
- Services: Monitor service health and component status
- WebSocket: Manage real-time connections and messaging

Architecture Integration:
- Supports hot-reload configuration management
- Integrates with resource monitoring systems
- Enables real-time WebSocket communication
- Provides comprehensive system health monitoring
- Implements structured logging and analytics
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from ..domain.document import DocumentType, ProcessingStatus
from .common_schemas import ApiResponse, ErrorResponse


# Enums for Admin Operations

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


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class ResourceType(str, Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    APPLICATION = "application"


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


# Configuration Management Schemas

class ConfigurationUpdateRequest(BaseModel):
    """Schema for configuration update requests."""
    
    changes: Dict[str, Any] = Field(
        ...,
        description="Configuration changes to apply",
        example={
            "ollama": {
                "embedding_model": "mxbai-embed-large",
                "base_url": "http://localhost:11434"
            },
            "llamaindex": {
                "chunk_size": 1024,
                "chunk_overlap": 200
            }
        }
    )
    
    scope: ConfigurationScope = Field(
        ConfigurationScope.SYSTEM,
        description="Scope of configuration changes"
    )
    
    priority: ConfigurationPriority = Field(
        ConfigurationPriority.NORMAL,
        description="Priority level for changes"
    )
    
    reason: Optional[str] = Field(
        None,
        description="Reason for configuration change",
        max_length=500
    )
    
    validate_only: bool = Field(
        False,
        description="Only validate changes without applying"
    )
    
    force_restart: bool = Field(
        False,
        description="Force service restart even if not required"
    )


class ConfigurationValidationRequest(BaseModel):
    """Schema for configuration validation requests."""
    
    configuration: Dict[str, Any] = Field(
        ...,
        description="Configuration to validate"
    )
    
    section: Optional[str] = Field(
        None,
        description="Specific section to validate (optional)"
    )


class ConfigurationTemplateRequest(BaseModel):
    """Schema for applying configuration templates."""
    
    template_name: str = Field(
        ...,
        description="Name of the configuration template to apply",
        example="performance_optimized"
    )
    
    reason: Optional[str] = Field(
        None,
        description="Reason for applying template",
        max_length=500
    )


class ConfigurationChangeResponse(BaseModel):
    """Schema for configuration change response."""
    
    success: bool = Field(..., description="Whether changes were applied successfully")
    changes_applied: Dict[str, Any] = Field(..., description="Changes that were applied")
    validation_errors: List[str] = Field(..., description="Validation errors encountered")
    warnings: List[str] = Field(..., description="Warnings generated")
    restart_required: ServiceRestartRequirement = Field(..., description="Restart requirement")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    rollback_available: bool = Field(..., description="Whether rollback is available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConfigurationHistoryEntry(BaseModel):
    """Schema for configuration history entries."""
    
    timestamp: datetime = Field(..., description="Change timestamp")
    change_type: str = Field(..., description="Type of change")
    section: Optional[str] = Field(None, description="Configuration section")
    key: Optional[str] = Field(None, description="Configuration key")
    old_value: Optional[Any] = Field(None, description="Previous value")
    new_value: Optional[Any] = Field(None, description="New value")
    user_id: Optional[str] = Field(None, description="User who made the change")
    reason: Optional[str] = Field(None, description="Reason for change")
    success: bool = Field(..., description="Whether change was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ConfigurationTemplate(BaseModel):
    """Schema for configuration templates."""
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    scope: ConfigurationScope = Field(..., description="Template scope")
    settings: Dict[str, Any] = Field(..., description="Template settings")
    restart_required: ServiceRestartRequirement = Field(..., description="Restart requirement")
    tags: List[str] = Field(default_factory=list, description="Template tags")


# System Monitoring Schemas

class SystemMetrics(BaseModel):
    """Schema for system resource metrics."""
    
    cpu_percent: float = Field(..., description="CPU usage percentage", ge=0.0, le=100.0)
    cpu_count: int = Field(..., description="Number of CPU cores", ge=1)
    cpu_per_core: List[float] = Field(..., description="Per-core CPU usage")
    
    memory_total: int = Field(..., description="Total memory in bytes", ge=0)
    memory_used: int = Field(..., description="Used memory in bytes", ge=0)
    memory_available: int = Field(..., description="Available memory in bytes", ge=0)
    memory_percent: float = Field(..., description="Memory usage percentage", ge=0.0, le=100.0)
    
    disk_total: int = Field(..., description="Total disk space in bytes", ge=0)
    disk_used: int = Field(..., description="Used disk space in bytes", ge=0)
    disk_free: int = Field(..., description="Free disk space in bytes", ge=0)
    disk_percent: float = Field(..., description="Disk usage percentage", ge=0.0, le=100.0)
    
    network_bytes_sent: int = Field(..., description="Network bytes sent", ge=0)
    network_bytes_recv: int = Field(..., description="Network bytes received", ge=0)
    network_connections: int = Field(..., description="Active network connections", ge=0)
    
    load_average: List[float] = Field(..., description="System load average (1, 5, 15 min)")
    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0.0)
    
    timestamp: datetime = Field(..., description="Metrics timestamp")


class GPUMetrics(BaseModel):
    """Schema for GPU metrics."""
    
    gpu_count: int = Field(..., description="Number of GPUs", ge=0)
    gpus: List[Dict[str, Any]] = Field(..., description="Per-GPU metrics")
    
    total_memory: int = Field(..., description="Total GPU memory in bytes", ge=0)
    used_memory: int = Field(..., description="Used GPU memory in bytes", ge=0)
    free_memory: int = Field(..., description="Free GPU memory in bytes", ge=0)
    memory_percent: float = Field(..., description="GPU memory usage percentage", ge=0.0, le=100.0)
    
    average_utilization: float = Field(..., description="Average GPU utilization", ge=0.0, le=100.0)
    average_temperature: float = Field(..., description="Average GPU temperature in Celsius")
    
    timestamp: datetime = Field(..., description="Metrics timestamp")


class ApplicationMetrics(BaseModel):
    """Schema for application-specific metrics."""
    
    # Document processing metrics
    documents_processed_total: int = Field(..., description="Total documents processed", ge=0)
    documents_processing: int = Field(..., description="Documents currently processing", ge=0)
    documents_failed: int = Field(..., description="Documents that failed processing", ge=0)
    average_processing_time: float = Field(..., description="Average processing time in seconds", ge=0.0)
    
    # Search metrics
    searches_total: int = Field(..., description="Total search queries executed", ge=0)
    average_search_time: float = Field(..., description="Average search time in seconds", ge=0.0)
    
    # Model metrics
    models_loaded: int = Field(..., description="Number of models loaded", ge=0)
    embedding_requests_total: int = Field(..., description="Total embedding requests", ge=0)
    average_embedding_time: float = Field(..., description="Average embedding time in seconds", ge=0.0)
    
    # Cache metrics
    cache_hits: int = Field(..., description="Cache hits", ge=0)
    cache_misses: int = Field(..., description="Cache misses", ge=0)
    cache_hit_rate: float = Field(..., description="Cache hit rate", ge=0.0, le=1.0)
    
    timestamp: datetime = Field(..., description="Metrics timestamp")


class AlertConfiguration(BaseModel):
    """Schema for alert configuration."""
    
    resource_type: ResourceType = Field(..., description="Type of resource to monitor")
    metric_name: str = Field(..., description="Metric name to monitor")
    threshold_value: float = Field(..., description="Threshold value for alert")
    alert_level: AlertLevel = Field(..., description="Alert severity level")
    enabled: bool = Field(True, description="Whether alert is enabled")
    cooldown_minutes: int = Field(5, description="Cooldown period in minutes", ge=0)
    message_template: str = Field(..., description="Alert message template")


class Alert(BaseModel):
    """Schema for system alerts."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(..., description="Alert timestamp")
    resource_type: ResourceType = Field(..., description="Resource type")
    metric_name: str = Field(..., description="Metric name")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold value")
    alert_level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    acknowledged: bool = Field(False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved: bool = Field(False, description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")


# WebSocket Management Schemas

class ConnectionInfo(BaseModel):
    """Schema for WebSocket connection information."""
    
    connection_id: str = Field(..., description="Unique connection identifier")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    client_ip: str = Field(..., description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    connected_at: datetime = Field(..., description="Connection timestamp")
    last_ping: Optional[datetime] = Field(None, description="Last ping timestamp")
    last_pong: Optional[datetime] = Field(None, description="Last pong timestamp")
    state: ConnectionState = Field(..., description="Connection state")
    subscriptions: List[str] = Field(default_factory=list, description="Subscribed channels")
    message_count: int = Field(0, description="Messages sent/received", ge=0)
    bytes_sent: int = Field(0, description="Bytes sent", ge=0)
    bytes_received: int = Field(0, description="Bytes received", ge=0)


class WebSocketMetrics(BaseModel):
    """Schema for WebSocket connection metrics."""
    
    total_connections: int = Field(0, description="Total active connections", ge=0)
    authenticated_connections: int = Field(0, description="Authenticated connections", ge=0)
    active_users: int = Field(0, description="Number of active users", ge=0)
    messages_sent_total: int = Field(0, description="Total messages sent", ge=0)
    messages_received_total: int = Field(0, description="Total messages received", ge=0)
    bytes_sent_total: int = Field(0, description="Total bytes sent", ge=0)
    bytes_received_total: int = Field(0, description="Total bytes received", ge=0)
    average_connection_duration_seconds: float = Field(0.0, description="Average connection duration", ge=0.0)
    peak_concurrent_connections: int = Field(0, description="Peak concurrent connections", ge=0)
    connection_errors_total: int = Field(0, description="Total connection errors", ge=0)
    message_queue_size: int = Field(0, description="Queued messages", ge=0)
    timestamp: datetime = Field(..., description="Metrics timestamp")


class BroadcastRequest(BaseModel):
    """Schema for broadcasting messages to WebSocket connections."""
    
    message_type: str = Field(..., description="Type of message to broadcast")
    data: Dict[str, Any] = Field(..., description="Message data")
    target_users: Optional[List[str]] = Field(None, description="Specific users to target")
    priority: str = Field("normal", description="Message priority level")
    expires_at: Optional[datetime] = Field(None, description="Message expiration time")


class ConnectionDisconnectRequest(BaseModel):
    """Schema for disconnecting WebSocket connections."""
    
    connection_id: str = Field(..., description="Connection ID to disconnect")
    reason: Optional[str] = Field(None, description="Reason for disconnection")
    notify_user: bool = Field(True, description="Whether to notify the user")


# Service Health Schemas

class ServiceHealthCheck(BaseModel):
    """Schema for individual service health check."""
    
    service_name: str = Field(..., description="Service name")
    status: ServiceStatus = Field(..., description="Service health status")
    message: str = Field(..., description="Health check message")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: float = Field(..., description="Health check response time", ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional health data")


class SystemHealthResponse(BaseModel):
    """Schema for overall system health response."""
    
    status: ServiceStatus = Field(..., description="Overall system status")
    components: Dict[str, ServiceHealthCheck] = Field(..., description="Component health status")
    resources: SystemMetrics = Field(..., description="Resource utilization")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    alerts: List[Alert] = Field(default_factory=list, description="Active alerts")
    timestamp: datetime = Field(..., description="Health check timestamp")


# User Management Schemas

class UserSession(BaseModel):
    """Schema for user session information."""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    started_at: datetime = Field(..., description="Session start time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    ip_address: str = Field(..., description="User IP address")
    user_agent: str = Field(..., description="User agent string")
    active_case_id: Optional[str] = Field(None, description="Currently active case")
    connection_count: int = Field(0, description="Active WebSocket connections", ge=0)
    operations_count: int = Field(0, description="Operations performed", ge=0)


class UserActivity(BaseModel):
    """Schema for user activity tracking."""
    
    user_id: str = Field(..., description="User identifier")
    activity_type: str = Field(..., description="Type of activity")
    timestamp: datetime = Field(..., description="Activity timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Activity details")
    session_id: Optional[str] = Field(None, description="Associated session")
    case_id: Optional[str] = Field(None, description="Associated case")
    ip_address: Optional[str] = Field(None, description="User IP address")


class UserStatistics(BaseModel):
    """Schema for user usage statistics."""
    
    user_id: str = Field(..., description="User identifier")
    total_sessions: int = Field(0, description="Total sessions", ge=0)
    total_cases: int = Field(0, description="Total cases created", ge=0)
    total_documents: int = Field(0, description="Total documents uploaded", ge=0)
    total_searches: int = Field(0, description="Total searches performed", ge=0)
    average_session_duration: float = Field(0.0, description="Average session duration in minutes", ge=0.0)
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    created_at: datetime = Field(..., description="User creation timestamp")


# Performance Analytics Schemas

class PerformanceTrend(BaseModel):
    """Schema for performance trend data."""
    
    metric_name: str = Field(..., description="Metric name")
    timestamps: List[datetime] = Field(..., description="Data timestamps")
    values: List[float] = Field(..., description="Metric values")
    trend_direction: str = Field(..., description="Trend direction (up/down/stable)")
    average_value: float = Field(..., description="Average value")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")


class PerformanceReport(BaseModel):
    """Schema for comprehensive performance reports."""
    
    report_id: str = Field(..., description="Report identifier")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    
    system_summary: Dict[str, Any] = Field(..., description="System performance summary")
    resource_trends: List[PerformanceTrend] = Field(..., description="Resource usage trends")
    application_metrics: ApplicationMetrics = Field(..., description="Application performance")
    alerts_summary: Dict[str, int] = Field(..., description="Alerts summary")
    recommendations: List[str] = Field(default_factory=list, description="Performance recommendations")


# Maintenance Schemas

class MaintenanceWindow(BaseModel):
    """Schema for system maintenance windows."""
    
    window_id: str = Field(..., description="Maintenance window identifier")
    title: str = Field(..., description="Maintenance title")
    description: str = Field(..., description="Maintenance description")
    scheduled_start: datetime = Field(..., description="Scheduled start time")
    scheduled_end: datetime = Field(..., description="Scheduled end time")
    actual_start: Optional[datetime] = Field(None, description="Actual start time")
    actual_end: Optional[datetime] = Field(None, description="Actual end time")
    status: str = Field(..., description="Maintenance status")
    affected_services: List[str] = Field(..., description="Affected services")
    notification_sent: bool = Field(False, description="Whether notifications were sent")
    created_by: str = Field(..., description="User who created the maintenance window")


class ServiceRestartRequest(BaseModel):
    """Schema for service restart requests."""
    
    services: List[str] = Field(..., description="Services to restart")
    reason: Optional[str] = Field(None, description="Reason for restart")
    force: bool = Field(False, description="Force restart without graceful shutdown")
    notify_users: bool = Field(True, description="Whether to notify users")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled restart time")


# Response Schemas

class AdminDashboardResponse(BaseModel):
    """Schema for admin dashboard data."""
    
    system_health: SystemHealthResponse = Field(..., description="System health status")
    websocket_metrics: WebSocketMetrics = Field(..., description="WebSocket metrics")
    active_sessions: List[UserSession] = Field(..., description="Active user sessions")
    recent_alerts: List[Alert] = Field(..., description="Recent alerts")
    performance_summary: Dict[str, Any] = Field(..., description="Performance summary")
    configuration_status: Dict[str, Any] = Field(..., description="Configuration status")
    timestamp: datetime = Field(..., description="Dashboard data timestamp")


# Export commonly used response aliases
AdminHealthResponse = SystemHealthResponse
AdminMetricsResponse = WebSocketMetrics
AdminConfigResponse = ConfigurationChangeResponse