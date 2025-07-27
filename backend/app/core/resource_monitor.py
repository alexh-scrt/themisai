"""
System Resource Monitor for Legal AI Application

This module provides comprehensive system resource monitoring including GPU, CPU, memory,
disk usage, and application-specific metrics. It offers real-time monitoring capabilities
with alerting, threshold management, and performance optimization recommendations.

Key Features:
- GPU utilization and memory monitoring (NVIDIA CUDA)
- CPU usage tracking with per-core breakdown
- System and application memory monitoring
- Disk usage and I/O performance tracking
- Network connection and WebSocket monitoring
- Database connection pool monitoring
- Model loading and caching metrics
- Performance trend analysis and alerting
- Resource optimization recommendations

GPU Monitoring:
- CUDA memory allocation and usage
- GPU utilization percentage
- Temperature and power consumption
- Model loading efficiency
- Batch processing optimization

System Monitoring:
- CPU usage per core and overall
- RAM usage and available memory
- Disk space and I/O operations
- Network traffic and connection counts
- Process monitoring for containerized services

Application Metrics:
- Document processing throughput
- Embedding generation performance
- Search query response times
- WebSocket connection health
- Database query performance

Architecture Integration:
- Provides data for admin panel resource display
- Integrates with WebSocket manager for real-time updates
- Supports alerting and threshold-based notifications
- Offers performance optimization recommendations
- Enables proactive resource management
"""

import asyncio
import logging
import platform
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import json

import psutil
from pydantic import BaseModel, Field

try:
    import GPUtil
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GPU monitoring libraries not available. Install nvidia-ml-py3 and GPUtil for GPU monitoring.")

from .websocket_manager import WebSocketManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    APPLICATION = "application"


@dataclass
class ResourceThreshold:
    """Resource monitoring threshold configuration."""
    resource_type: ResourceType
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    alert_cooldown_minutes: int = 5
    last_alert_time: Optional[datetime] = None
    
    def should_alert(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last alert."""
        if not self.enabled or self.last_alert_time is None:
            return True
        
        cooldown_delta = timedelta(minutes=self.alert_cooldown_minutes)
        return current_time - self.last_alert_time > cooldown_delta


@dataclass
class ResourceAlert:
    """Resource monitoring alert."""
    timestamp: datetime
    resource_type: ResourceType
    metric_name: str
    current_value: float
    threshold_value: float
    alert_level: AlertLevel
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: Tuple[float, float, float]


@dataclass
class GPUMetrics:
    """GPU resource metrics snapshot."""
    timestamp: datetime
    gpu_count: int
    gpu_utilization: List[float]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_memory_percent: List[float]
    gpu_temperature: List[float]
    gpu_power_draw: List[float]
    gpu_name: List[str]


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_websocket_connections: int
    document_processing_queue_size: int
    embedding_cache_size: int
    embedding_cache_hit_rate: float
    average_query_response_time_ms: float
    documents_processed_last_hour: int
    active_database_connections: int
    model_memory_usage_mb: float
    background_tasks_count: int


class GPUMonitor:
    """GPU monitoring using NVIDIA Management Library."""
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.available = GPU_MONITORING_AVAILABLE
        self.gpu_count = 0
        
        if self.available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring initialized with {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.available = False
    
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        if not self.available:
            return None
        
        try:
            timestamp = datetime.now(timezone.utc)
            utilization = []
            memory_used = []
            memory_total = []
            memory_percent = []
            temperature = []
            power_draw = []
            gpu_names = []
            
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization.append(util.gpu)
                
                # Memory information
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = mem_info.used / 1024 / 1024
                memory_total_mb = mem_info.total / 1024 / 1024
                memory_used.append(memory_used_mb)
                memory_total.append(memory_total_mb)
                memory_percent.append((memory_used_mb / memory_total_mb) * 100)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature.append(temp)
                except:
                    temperature.append(0.0)
                
                # Power draw
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_draw.append(power)
                except:
                    power_draw.append(0.0)
                
                # GPU name
                try:
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    gpu_names.append(name)
                except:
                    gpu_names.append(f"GPU {i}")
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_count=self.gpu_count,
                gpu_utilization=utilization,
                gpu_memory_used=memory_used,
                gpu_memory_total=memory_total,
                gpu_memory_percent=memory_percent,
                gpu_temperature=temperature,
                gpu_power_draw=power_draw,
                gpu_name=gpu_names
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return None


class SystemMonitor:
    """System resource monitoring using psutil."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        self.last_measurement_time = time.time()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        current_time = time.time()
        time_delta = current_time - self.last_measurement_time
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / 1024 / 1024 / 1024
        memory_total_gb = memory.total / 1024 / 1024 / 1024
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = disk_usage.percent
        disk_used_gb = disk_usage.used / 1024 / 1024 / 1024
        disk_total_gb = disk_usage.total / 1024 / 1024 / 1024
        
        # Disk I/O metrics
        current_disk_io = psutil.disk_io_counters()
        if self.last_disk_io and time_delta > 0:
            disk_io_read_mb = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / 1024 / 1024 / time_delta
            disk_io_write_mb = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / 1024 / 1024 / time_delta
        else:
            disk_io_read_mb = 0.0
            disk_io_write_mb = 0.0
        
        # Network metrics
        current_network_io = psutil.net_io_counters()
        if self.last_network_io and time_delta > 0:
            network_sent_mb = (current_network_io.bytes_sent - self.last_network_io.bytes_sent) / 1024 / 1024 / time_delta
            network_recv_mb = (current_network_io.bytes_recv - self.last_network_io.bytes_recv) / 1024 / 1024 / time_delta
        else:
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        
        # Process metrics
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems)
        try:
            load_average = psutil.getloadavg()
        except AttributeError:
            load_average = (0.0, 0.0, 0.0)
        
        # Update last measurements
        self.last_disk_io = current_disk_io
        self.last_network_io = current_network_io
        self.last_measurement_time = current_time
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_usage_percent=disk_usage_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            load_average=load_average
        )


class ResourceMonitor:
    """
    Comprehensive system resource monitor for the legal AI application.
    
    Monitors GPU, CPU, memory, disk, network, and application-specific metrics
    with alerting, threshold management, and real-time reporting capabilities.
    """
    
    def __init__(
        self,
        websocket_manager: Optional[WebSocketManager] = None,
        monitoring_interval: int = 5,
        history_retention_minutes: int = 60
    ):
        """
        Initialize resource monitor.
        
        Args:
            websocket_manager: Optional WebSocket manager for real-time updates
            monitoring_interval: Monitoring interval in seconds
            history_retention_minutes: How long to retain metrics history
        """
        self.websocket_manager = websocket_manager
        self.monitoring_interval = monitoring_interval
        self.history_retention = timedelta(minutes=history_retention_minutes)
        
        # Initialize monitors
        self.gpu_monitor = GPUMonitor()
        self.system_monitor = SystemMonitor()
        
        # Metrics history
        self.system_metrics_history = deque(maxlen=history_retention_minutes * 60 // monitoring_interval)
        self.gpu_metrics_history = deque(maxlen=history_retention_minutes * 60 // monitoring_interval)
        self.application_metrics_history = deque(maxlen=history_retention_minutes * 60 // monitoring_interval)
        
        # Alerting system
        self.thresholds = self._create_default_thresholds()
        self.active_alerts: List[ResourceAlert] = []
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        
        # Application metrics tracking
        self.app_metrics = {
            "websocket_connections": 0,
            "processing_queue_size": 0,
            "embedding_cache_size": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "query_response_times": deque(maxlen=100),
            "documents_processed": deque(maxlen=3600),  # Last hour
            "database_connections": 0,
            "model_memory_usage": 0.0,
            "background_tasks": 0
        }
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Performance optimization tracking
        self.optimization_recommendations = []
        self.last_optimization_check = datetime.now(timezone.utc)
        
        logger.info(
            "ResourceMonitor initialized",
            gpu_available=self.gpu_monitor.available,
            gpu_count=self.gpu_monitor.gpu_count,
            monitoring_interval=monitoring_interval,
            history_retention_minutes=history_retention_minutes
        )
    
    def _create_default_thresholds(self) -> Dict[str, ResourceThreshold]:
        """Create default resource monitoring thresholds."""
        return {
            "cpu_usage": ResourceThreshold(
                resource_type=ResourceType.CPU,
                metric_name="cpu_percent",
                warning_threshold=70.0,
                critical_threshold=90.0
            ),
            "memory_usage": ResourceThreshold(
                resource_type=ResourceType.MEMORY,
                metric_name="memory_percent",
                warning_threshold=80.0,
                critical_threshold=95.0
            ),
            "disk_usage": ResourceThreshold(
                resource_type=ResourceType.DISK,
                metric_name="disk_usage_percent",
                warning_threshold=85.0,
                critical_threshold=95.0
            ),
            "gpu_utilization": ResourceThreshold(
                resource_type=ResourceType.GPU,
                metric_name="gpu_utilization",
                warning_threshold=90.0,
                critical_threshold=98.0
            ),
            "gpu_memory": ResourceThreshold(
                resource_type=ResourceType.GPU,
                metric_name="gpu_memory_percent",
                warning_threshold=85.0,
                critical_threshold=95.0
            ),
            "gpu_temperature": ResourceThreshold(
                resource_type=ResourceType.GPU,
                metric_name="gpu_temperature",
                warning_threshold=80.0,
                critical_threshold=90.0
            )
        }
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Resource monitoring started")
        
        # Send initial notification
        if self.websocket_manager:
            await self._send_monitoring_update({
                "type": "monitoring_started",
                "message": "Resource monitoring activated",
                "interval": self.monitoring_interval
            })
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitoring stopped")
        
        if self.websocket_manager:
            await self._send_monitoring_update({
                "type": "monitoring_stopped",
                "message": "Resource monitoring deactivated"
            })
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Collect metrics
                system_metrics = self.system_monitor.get_system_metrics()
                gpu_metrics = self.gpu_monitor.get_gpu_metrics()
                app_metrics = self._get_application_metrics()
                
                # Store metrics history
                self.system_metrics_history.append(system_metrics)
                if gpu_metrics:
                    self.gpu_metrics_history.append(gpu_metrics)
                self.application_metrics_history.append(app_metrics)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(system_metrics, gpu_metrics, app_metrics)
                
                # Send real-time updates
                if self.websocket_manager:
                    await self._send_realtime_metrics(system_metrics, gpu_metrics, app_metrics)
                
                # Check for optimization opportunities
                await self._check_optimization_opportunities()
                
                # Clean up old data
                self._cleanup_old_data()
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
    
    def _get_application_metrics(self) -> ApplicationMetrics:
        """Get current application-specific metrics."""
        # Calculate cache hit rate
        total_cache_requests = self.app_metrics["embedding_cache_hits"] + self.app_metrics["embedding_cache_misses"]
        cache_hit_rate = (
            self.app_metrics["embedding_cache_hits"] / total_cache_requests
            if total_cache_requests > 0 else 0.0
        )
        
        # Calculate average query response time
        response_times = list(self.app_metrics["query_response_times"])
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Count documents processed in last hour
        current_time = datetime.now(timezone.utc)
        one_hour_ago = current_time - timedelta(hours=1)
        docs_last_hour = sum(
            1 for timestamp in self.app_metrics["documents_processed"]
            if timestamp > one_hour_ago
        )
        
        return ApplicationMetrics(
            timestamp=current_time,
            active_websocket_connections=self.app_metrics["websocket_connections"],
            document_processing_queue_size=self.app_metrics["processing_queue_size"],
            embedding_cache_size=self.app_metrics["embedding_cache_size"],
            embedding_cache_hit_rate=cache_hit_rate,
            average_query_response_time_ms=avg_response_time,
            documents_processed_last_hour=docs_last_hour,
            active_database_connections=self.app_metrics["database_connections"],
            model_memory_usage_mb=self.app_metrics["model_memory_usage"],
            background_tasks_count=self.app_metrics["background_tasks"]
        )
    
    async def _check_thresholds(
        self,
        system_metrics: SystemMetrics,
        gpu_metrics: Optional[GPUMetrics],
        app_metrics: ApplicationMetrics
    ) -> None:
        """Check resource thresholds and generate alerts."""
        current_time = datetime.now(timezone.utc)
        
        # Check system thresholds
        await self._check_metric_threshold("cpu_usage", system_metrics.cpu_percent, current_time)
        await self._check_metric_threshold("memory_usage", system_metrics.memory_percent, current_time)
        await self._check_metric_threshold("disk_usage", system_metrics.disk_usage_percent, current_time)
        
        # Check GPU thresholds
        if gpu_metrics:
            for i, (util, mem_pct, temp) in enumerate(zip(
                gpu_metrics.gpu_utilization,
                gpu_metrics.gpu_memory_percent,
                gpu_metrics.gpu_temperature
            )):
                await self._check_metric_threshold(f"gpu_{i}_utilization", util, current_time)
                await self._check_metric_threshold(f"gpu_{i}_memory", mem_pct, current_time)
                await self._check_metric_threshold(f"gpu_{i}_temperature", temp, current_time)
    
    async def _check_metric_threshold(
        self,
        metric_name: str,
        current_value: float,
        current_time: datetime
    ) -> None:
        """Check a specific metric against its threshold."""
        # Find applicable threshold
        threshold_key = metric_name
        if metric_name.startswith("gpu_") and "_" in metric_name:
            # Handle per-GPU metrics
            parts = metric_name.split("_")
            if len(parts) >= 3:
                threshold_key = f"gpu_{parts[2]}"
        
        threshold = self.thresholds.get(threshold_key)
        if not threshold or not threshold.enabled:
            return
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        if current_value >= threshold.critical_threshold:
            alert_level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_threshold
        elif current_value >= threshold.warning_threshold:
            alert_level = AlertLevel.WARNING
            threshold_value = threshold.warning_threshold
        
        if alert_level and threshold.should_alert(current_time):
            # Create alert
            alert = ResourceAlert(
                timestamp=current_time,
                resource_type=threshold.resource_type,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                alert_level=alert_level,
                message=f"{metric_name} is {current_value:.1f}% (threshold: {threshold_value:.1f}%)"
            )
            
            await self._trigger_alert(alert)
            threshold.last_alert_time = current_time
    
    async def _trigger_alert(self, alert: ResourceAlert) -> None:
        """Trigger a resource alert."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        logger.warning(
            f"Resource alert: {alert.message}",
            resource_type=alert.resource_type.value,
            metric=alert.metric_name,
            level=alert.alert_level.value
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Send WebSocket notification
        if self.websocket_manager:
            await self._send_monitoring_update({
                "type": "resource_alert",
                "alert": {
                    "timestamp": alert.timestamp.isoformat(),
                    "resource_type": alert.resource_type.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "alert_level": alert.alert_level.value,
                    "message": alert.message
                }
            })
    
    async def _send_realtime_metrics(
        self,
        system_metrics: SystemMetrics,
        gpu_metrics: Optional[GPUMetrics],
        app_metrics: ApplicationMetrics
    ) -> None:
        """Send real-time metrics via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            metrics_data = {
                "timestamp": system_metrics.timestamp.isoformat(),
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "memory_used_gb": system_metrics.memory_used_gb,
                    "memory_total_gb": system_metrics.memory_total_gb,
                    "disk_usage_percent": system_metrics.disk_usage_percent,
                    "disk_used_gb": system_metrics.disk_used_gb,
                    "disk_total_gb": system_metrics.disk_total_gb,
                    "process_count": system_metrics.process_count,
                    "load_average": system_metrics.load_average
                },
                "application": {
                    "websocket_connections": app_metrics.active_websocket_connections,
                    "processing_queue_size": app_metrics.document_processing_queue_size,
                    "cache_hit_rate": app_metrics.embedding_cache_hit_rate,
                    "avg_response_time_ms": app_metrics.average_query_response_time_ms,
                    "docs_processed_last_hour": app_metrics.documents_processed_last_hour,
                    "database_connections": app_metrics.active_database_connections,
                    "background_tasks": app_metrics.background_tasks_count
                }
            }
            
            if gpu_metrics:
                metrics_data["gpu"] = {
                    "count": gpu_metrics.gpu_count,
                    "utilization": gpu_metrics.gpu_utilization,
                    "memory_used": gpu_metrics.gpu_memory_used,
                    "memory_total": gpu_metrics.gpu_memory_total,
                    "memory_percent": gpu_metrics.gpu_memory_percent,
                    "temperature": gpu_metrics.gpu_temperature,
                    "power_draw": gpu_metrics.gpu_power_draw,
                    "names": gpu_metrics.gpu_name
                }
            
            await self.websocket_manager.broadcast_to_all("resource_metrics", metrics_data)
            
        except Exception as e:
            logger.warning(f"Failed to send real-time metrics: {e}")
    
    async def _check_optimization_opportunities(self) -> None:
        """Check for performance optimization opportunities."""
        current_time = datetime.now(timezone.utc)
        
        # Only check every 5 minutes
        if current_time - self.last_optimization_check < timedelta(minutes=5):
            return
        
        self.last_optimization_check = current_time
        recommendations = []
        
        # Check recent system metrics
        if len(self.system_metrics_history) >= 10:
            recent_metrics = list(self.system_metrics_history)[-10:]
            
            # High CPU usage recommendations
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            if avg_cpu > 80:
                recommendations.append({
                    "type": "cpu_optimization",
                    "message": "High CPU usage detected. Consider reducing concurrent processing tasks.",
                    "severity": "warning"
                })
            
            # High memory usage recommendations
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            if avg_memory > 85:
                recommendations.append({
                    "type": "memory_optimization",
                    "message": "High memory usage detected. Consider reducing embedding cache size.",
                    "severity": "warning"
                })
        
        # Check GPU metrics
        if len(self.gpu_metrics_history) >= 10:
            recent_gpu = list(self.gpu_metrics_history)[-10:]
            
            for gpu_idx in range(self.gpu_monitor.gpu_count):
                avg_util = sum(m.gpu_utilization[gpu_idx] for m in recent_gpu) / len(recent_gpu)
                avg_mem = sum(m.gpu_memory_percent[gpu_idx] for m in recent_gpu) / len(recent_gpu)
                
                if avg_util < 30 and avg_mem > 70:
                    recommendations.append({
                        "type": "gpu_optimization",
                        "message": f"GPU {gpu_idx} has low utilization but high memory usage. Consider model optimization.",
                        "severity": "info"
                    })
        
        # Update recommendations
        self.optimization_recommendations = recommendations
        
        # Send notifications for new recommendations
        if recommendations and self.websocket_manager:
            await self._send_monitoring_update({
                "type": "optimization_recommendations",
                "recommendations": recommendations
            })
    
    async def _send_monitoring_update(self, data: Dict[str, Any]) -> None:
        """Send monitoring update via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_to_all("resource_monitoring", data)
        except Exception as e:
            logger.warning(f"Failed to send monitoring update: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data and resolve stale alerts."""
        current_time = datetime.now(timezone.utc)
        
        # Resolve old alerts (if metric has improved)
        for alert in self.active_alerts[:]:
            if not alert.resolved:
                # Check if we should auto-resolve this alert
                # This would require checking current values against thresholds
                # For now, we'll just mark very old alerts as resolved
                if current_time - alert.timestamp > timedelta(minutes=30):
                    alert.resolved = True
                    alert.resolved_at = current_time
                    self.active_alerts.remove(alert)
    
    # Public API methods
    
    def update_app_metric(self, metric_name: str, value: Any) -> None:
        """Update an application metric."""
        if metric_name in self.app_metrics:
            if metric_name == "documents_processed":
                # Add timestamp for documents processed
                self.app_metrics[metric_name].append(datetime.now(timezone.utc))
            elif metric_name in ["query_response_times"]:
                # Add to deque for time-based metrics
                self.app_metrics[metric_name].append(value)
            else:
                # Direct value update
                self.app_metrics[metric_name] = value
    
    def increment_app_metric(self, metric_name: str, amount: int = 1) -> None:
        """Increment an application metric."""
        if metric_name in self.app_metrics and isinstance(self.app_metrics[metric_name], (int, float)):
            self.app_metrics[metric_name] += amount
    
    def register_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Register a callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics snapshot."""
        system_metrics = self.system_monitor.get_system_metrics()
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        app_metrics = self._get_application_metrics()
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "process_count": system_metrics.process_count
            },
            "application": {
                "websocket_connections": app_metrics.active_websocket_connections,
                "processing_queue_size": app_metrics.document_processing_queue_size,
                "cache_hit_rate": app_metrics.embedding_cache_hit_rate,
                "database_connections": app_metrics.active_database_connections
            },
            "alerts": {
                "active_count": len([a for a in self.active_alerts if not a.resolved]),
                "total_count": len(self.alert_history)
            }
        }
        
        if gpu_metrics:
            result["gpu"] = {
                "count": gpu_metrics.gpu_count,
                "utilization": gpu_metrics.gpu_utilization,
                "memory_percent": gpu_metrics.gpu_memory_percent,
                "temperature": gpu_metrics.gpu_temperature
            }
        
        return result
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "gpu_monitoring_available": self.gpu_monitor.available,
            "gpu_count": self.gpu_monitor.gpu_count,
            "metrics_history_size": {
                "system": len(self.system_metrics_history),
                "gpu": len(self.gpu_metrics_history),
                "application": len(self.application_metrics_history)
            },
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "total_alerts": len(self.alert_history),
            "optimization_recommendations": len(self.optimization_recommendations)
        }


# Global resource monitor instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


async def start_resource_monitoring(websocket_manager: Optional[WebSocketManager] = None) -> None:
    """Start the global resource monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor(websocket_manager=websocket_manager)
    
    await _resource_monitor.start_monitoring()


async def stop_resource_monitoring() -> None:
    """Stop the global resource monitor."""
    global _resource_monitor
    if _resource_monitor:
        await _resource_monitor.stop_monitoring()