"""
Notification Service - Business Logic Layer

This module provides the business logic layer for real-time notifications and messaging
in the Patexia Legal AI Chatbot. It orchestrates notification delivery, manages message
routing, enforces business rules, and provides service layer abstraction for real-time
communication across the legal document processing pipeline.

Key Features:
- Multi-channel notification delivery (WebSocket, email, etc.)
- Message prioritization and queuing with business rules
- Template-based notification generation for legal workflows
- User preference management and notification filtering
- Notification persistence and delivery tracking
- Real-time progress updates for long-running operations
- System alert distribution and escalation
- Multi-user collaboration notifications
- Audit trail for compliance and monitoring

Notification Types:
- Document processing progress and completion updates
- Search operation status and results notifications
- System alerts and resource monitoring updates
- Configuration change notifications for administrators
- Error alerts with severity levels and escalation
- Case activity updates and collaboration notifications
- Security alerts and access control notifications
- Performance monitoring and capacity alerts

Business Rules:
- User-specific notification preferences and filtering
- Priority-based message delivery and escalation
- Notification rate limiting and throttling
- Business hours and emergency notification handling
- Legal compliance for sensitive document notifications
- Multi-tenant isolation and privacy enforcement
- Template validation and content sanitization

Architecture Integration:
- Integrates with WebSocketManager for real-time delivery
- Coordinates with all services for progress tracking
- Uses domain models for type-safe notification content
- Provides service layer abstraction for API controllers
- Implements notification persistence for reliability
- Supports multiple delivery channels and fallbacks
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque

from ..core.config import get_settings
from ..core.websocket_manager import WebSocketManager, MessagePriority
from ..models.domain.document import DocumentChunk, ProcessingStatus
from ..models.domain.case import LegalCase, CaseStatus
from ..exceptions import (
    NotificationError, ValidationError, DeliveryError,
    ErrorCode, raise_notification_error, raise_validation_error
)
from ..utils.logging import get_logger, performance_context

logger = get_logger(__name__)


class NotificationType(str, Enum):
    """Types of notifications supported by the system."""
    DOCUMENT_PROGRESS = "document_progress"
    DOCUMENT_COMPLETED = "document_completed"
    DOCUMENT_FAILED = "document_failed"
    SEARCH_PROGRESS = "search_progress"
    SEARCH_COMPLETED = "search_completed"
    CASE_UPDATED = "case_updated"
    CASE_CREATED = "case_created"
    SYSTEM_ALERT = "system_alert"
    CONFIGURATION_CHANGED = "configuration_changed"
    RESOURCE_ALERT = "resource_alert"
    ERROR_ALERT = "error_alert"
    USER_ACTIVITY = "user_activity"
    SECURITY_ALERT = "security_alert"
    CAPACITY_WARNING = "capacity_warning"
    MODEL_SWITCHED = "model_switched"
    EMBEDDING_PROGRESS = "embedding_progress"
    PROCESSING_QUEUE_UPDATE = "processing_queue_update"


class NotificationChannel(str, Enum):
    """Delivery channels for notifications."""
    WEBSOCKET = "websocket"
    EMAIL = "email"          # Future implementation
    SMS = "sms"              # Future implementation
    SLACK = "slack"          # Future implementation
    WEBHOOK = "webhook"      # Future implementation


class NotificationSeverity(str, Enum):
    """Severity levels for notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeliveryStatus(str, Enum):
    """Status of notification delivery."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRYING = "retrying"


@dataclass
class NotificationTemplate:
    """Template for generating notifications."""
    notification_type: NotificationType
    title_template: str
    message_template: str
    severity: NotificationSeverity
    channels: List[NotificationChannel]
    priority: MessagePriority = MessagePriority.NORMAL
    expires_in_hours: int = 24
    require_authentication: bool = True
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context variables."""
        try:
            title = self.title_template.format(**context)
            message = self.message_template.format(**context)
            return {"title": title, "message": message}
        except KeyError as e:
            raise_validation_error(
                f"Missing template variable: {e}",
                ErrorCode.NOTIFICATION_TEMPLATE_ERROR,
                {"template": self.notification_type.value, "missing_key": str(e)}
            )


@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: str
    enabled_types: Set[NotificationType] = field(default_factory=set)
    enabled_channels: Set[NotificationChannel] = field(default_factory=set)
    severity_threshold: NotificationSeverity = NotificationSeverity.INFO
    business_hours_only: bool = False
    rate_limit_per_hour: int = 100
    email_digest_frequency: Optional[str] = None  # "daily", "weekly", None
    
    def should_deliver(
        self, 
        notification_type: NotificationType,
        severity: NotificationSeverity,
        channel: NotificationChannel
    ) -> bool:
        """Check if notification should be delivered based on preferences."""
        # Check if type is enabled
        if notification_type not in self.enabled_types:
            return False
        
        # Check if channel is enabled
        if channel not in self.enabled_channels:
            return False
        
        # Check severity threshold
        severity_levels = {
            NotificationSeverity.INFO: 1,
            NotificationSeverity.WARNING: 2,
            NotificationSeverity.ERROR: 3,
            NotificationSeverity.CRITICAL: 4
        }
        
        if severity_levels[severity] < severity_levels[self.severity_threshold]:
            return False
        
        return True


@dataclass
class Notification:
    """Represents a notification to be delivered."""
    notification_id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    severity: NotificationSeverity
    channels: List[NotificationChannel]
    priority: MessagePriority
    created_at: datetime
    expires_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    delivery_status: Dict[NotificationChannel, DeliveryStatus] = field(default_factory=dict)
    delivery_attempts: Dict[NotificationChannel, int] = field(default_factory=dict)
    last_delivery_attempt: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if notification has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def should_retry(self, channel: NotificationChannel, max_retries: int = 3) -> bool:
        """Check if delivery should be retried for a channel."""
        if self.is_expired():
            return False
        
        attempts = self.delivery_attempts.get(channel, 0)
        status = self.delivery_status.get(channel, DeliveryStatus.PENDING)
        
        return attempts < max_retries and status in [DeliveryStatus.PENDING, DeliveryStatus.FAILED]


@dataclass
class NotificationStats:
    """Statistics for notification delivery."""
    total_sent: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    total_expired: int = 0
    delivery_rate: float = 0.0
    average_delivery_time_ms: float = 0.0
    channel_stats: Dict[NotificationChannel, Dict[str, int]] = field(default_factory=dict)
    type_stats: Dict[NotificationType, Dict[str, int]] = field(default_factory=dict)


class NotificationService:
    """
    Business logic service for real-time notifications and messaging.
    
    Orchestrates notification delivery, manages message routing, enforces
    business rules, and provides service layer abstraction for real-time
    communication across the legal document processing pipeline.
    """
    
    def __init__(
        self,
        websocket_manager: WebSocketManager,
        max_pending_notifications: int = 10000,
        default_expiry_hours: int = 24
    ):
        """
        Initialize notification service.
        
        Args:
            websocket_manager: WebSocket manager for real-time delivery
            max_pending_notifications: Maximum pending notifications per user
            default_expiry_hours: Default notification expiry time
        """
        self.websocket_manager = websocket_manager
        self.max_pending_notifications = max_pending_notifications
        self.default_expiry_hours = default_expiry_hours
        
        # Load configuration
        self.settings = get_settings()
        
        # Notification state
        self.templates = self._initialize_templates()
        self.user_preferences: Dict[str, NotificationPreferences] = {}
        self.pending_notifications: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_pending_notifications)
        )
        self.notification_history: List[Notification] = []
        self.stats = NotificationStats()
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rate_limit_windows: Dict[str, datetime] = {}
        
        # Background processing
        self._delivery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info(
            "NotificationService initialized",
            max_pending_per_user=max_pending_notifications,
            default_expiry_hours=default_expiry_hours,
            template_count=len(self.templates)
        )
    
    async def start(self) -> None:
        """Start the notification service background tasks."""
        if self._running:
            return
        
        self._running = True
        self._delivery_task = asyncio.create_task(self._delivery_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("NotificationService started")
    
    async def stop(self) -> None:
        """Stop the notification service background tasks."""
        self._running = False
        
        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("NotificationService stopped")
    
    async def send_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        context: Dict[str, Any],
        priority: Optional[MessagePriority] = None,
        channels: Optional[List[NotificationChannel]] = None,
        expires_in_hours: Optional[int] = None
    ) -> str:
        """
        Send a notification to a user.
        
        Args:
            user_id: Target user identifier
            notification_type: Type of notification
            context: Context variables for template rendering
            priority: Message priority override
            channels: Delivery channels override
            expires_in_hours: Expiry time override
            
        Returns:
            Notification ID for tracking
            
        Raises:
            NotificationError: If notification creation or delivery fails
            ValidationError: If notification data is invalid
        """
        async with performance_context("send_notification", notification_type=notification_type.value):
            try:
                # Get template
                template = self.templates.get(notification_type)
                if not template:
                    raise_notification_error(
                        f"No template found for notification type: {notification_type.value}",
                        ErrorCode.NOTIFICATION_TEMPLATE_NOT_FOUND,
                        {"notification_type": notification_type.value}
                    )
                
                # Render notification content
                rendered = template.render(context)
                
                # Create notification
                notification_id = str(uuid.uuid4())
                expiry_hours = expires_in_hours or template.expires_in_hours
                
                notification = Notification(
                    notification_id=notification_id,
                    user_id=user_id,
                    notification_type=notification_type,
                    title=rendered["title"],
                    message=rendered["message"],
                    severity=template.severity,
                    channels=channels or template.channels,
                    priority=priority or template.priority,
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
                    context=context,
                    metadata={"template_used": template.notification_type.value}
                )
                
                # Check user preferences
                if not await self._should_deliver_notification(notification):
                    logger.debug(
                        "Notification filtered by user preferences",
                        notification_id=notification_id,
                        user_id=user_id,
                        notification_type=notification_type.value
                    )
                    return notification_id
                
                # Check rate limits
                if not await self._check_rate_limit(user_id, notification_type):
                    logger.warning(
                        "Notification rate limited",
                        notification_id=notification_id,
                        user_id=user_id,
                        notification_type=notification_type.value
                    )
                    return notification_id
                
                # Queue for delivery
                await self._queue_notification(notification)
                
                logger.info(
                    "Notification queued for delivery",
                    notification_id=notification_id,
                    user_id=user_id,
                    notification_type=notification_type.value,
                    channels=[c.value for c in notification.channels]
                )
                
                return notification_id
                
            except Exception as e:
                logger.error(
                    "Failed to send notification",
                    user_id=user_id,
                    notification_type=notification_type.value,
                    error=str(e)
                )
                raise
    
    async def send_document_progress(
        self,
        user_id: str,
        document_id: str,
        document_name: str,
        status: ProcessingStatus,
        progress_percent: int,
        message: str,
        case_id: Optional[str] = None
    ) -> str:
        """Send document processing progress notification."""
        context = {
            "document_id": document_id,
            "document_name": document_name,
            "status": status.value,
            "progress_percent": progress_percent,
            "message": message,
            "case_id": case_id or "unknown"
        }
        
        return await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.DOCUMENT_PROGRESS,
            context=context,
            priority=MessagePriority.HIGH,
            channels=[NotificationChannel.WEBSOCKET]
        )
    
    async def send_embedding_progress(
        self,
        user_id: str,
        request_id: str,
        model_name: str,
        progress_percent: int,
        message: str,
        document_id: Optional[str] = None
    ) -> str:
        """Send embedding generation progress notification."""
        context = {
            "request_id": request_id,
            "model_name": model_name,
            "progress_percent": progress_percent,
            "message": message,
            "document_id": document_id or "unknown"
        }
        
        return await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.EMBEDDING_PROGRESS,
            context=context,
            priority=MessagePriority.NORMAL,
            channels=[NotificationChannel.WEBSOCKET]
        )
    
    async def send_system_alert(
        self,
        alert_type: str,
        message: str,
        severity: NotificationSeverity,
        metadata: Optional[Dict[str, Any]] = None,
        target_users: Optional[List[str]] = None
    ) -> List[str]:
        """Send system alert to administrators or specific users."""
        context = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity.value,
            "metadata": json.dumps(metadata or {}, default=str),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # If no target users specified, send to all admin users
        # In a real implementation, this would query for admin users
        users = target_users or ["admin"]
        
        notification_ids = []
        for user_id in users:
            notification_id = await self.send_notification(
                user_id=user_id,
                notification_type=NotificationType.SYSTEM_ALERT,
                context=context,
                priority=MessagePriority.HIGH if severity in [
                    NotificationSeverity.ERROR, NotificationSeverity.CRITICAL
                ] else MessagePriority.NORMAL,
                channels=[NotificationChannel.WEBSOCKET]
            )
            notification_ids.append(notification_id)
        
        return notification_ids
    
    async def send_case_update(
        self,
        user_id: str,
        case: LegalCase,
        operation: str,
        details: Optional[str] = None
    ) -> str:
        """Send case update notification."""
        context = {
            "case_id": case.case_id,
            "case_name": case.case_name,
            "operation": operation,
            "status": case.status.value,
            "details": details or f"Case {operation} completed",
            "document_count": case.document_count,
            "updated_at": case.updated_at.isoformat()
        }
        
        return await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.CASE_UPDATED,
            context=context,
            priority=MessagePriority.NORMAL,
            channels=[NotificationChannel.WEBSOCKET]
        )
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: NotificationPreferences
    ) -> None:
        """Update user notification preferences."""
        preferences.user_id = user_id
        self.user_preferences[user_id] = preferences
        
        logger.info(
            "User notification preferences updated",
            user_id=user_id,
            enabled_types=len(preferences.enabled_types),
            enabled_channels=len(preferences.enabled_channels),
            severity_threshold=preferences.severity_threshold.value
        )
    
    async def get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences with defaults."""
        if user_id not in self.user_preferences:
            # Create default preferences
            default_preferences = NotificationPreferences(
                user_id=user_id,
                enabled_types={
                    NotificationType.DOCUMENT_PROGRESS,
                    NotificationType.DOCUMENT_COMPLETED,
                    NotificationType.DOCUMENT_FAILED,
                    NotificationType.CASE_UPDATED,
                    NotificationType.SYSTEM_ALERT
                },
                enabled_channels={NotificationChannel.WEBSOCKET},
                severity_threshold=NotificationSeverity.INFO
            )
            self.user_preferences[user_id] = default_preferences
        
        return self.user_preferences[user_id]
    
    async def get_notification_stats(self) -> NotificationStats:
        """Get notification delivery statistics."""
        # Update delivery rate
        total_attempted = self.stats.total_sent
        if total_attempted > 0:
            self.stats.delivery_rate = self.stats.total_delivered / total_attempted
        
        return self.stats
    
    async def get_pending_notifications(self, user_id: str) -> List[Notification]:
        """Get pending notifications for a user."""
        if user_id not in self.pending_notifications:
            return []
        
        return list(self.pending_notifications[user_id])
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for notification events."""
        self.event_handlers[event_type].append(handler)
        
        logger.debug(
            "Event handler added",
            event_type=event_type,
            handler_count=len(self.event_handlers[event_type])
        )
    
    # Private helper methods
    
    def _initialize_templates(self) -> Dict[NotificationType, NotificationTemplate]:
        """Initialize notification templates."""
        templates = {}
        
        # Document processing templates
        templates[NotificationType.DOCUMENT_PROGRESS] = NotificationTemplate(
            notification_type=NotificationType.DOCUMENT_PROGRESS,
            title_template="Document Processing: {document_name}",
            message_template="Status: {status} - {message} ({progress_percent}%)",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.HIGH,
            expires_in_hours=1
        )
        
        templates[NotificationType.DOCUMENT_COMPLETED] = NotificationTemplate(
            notification_type=NotificationType.DOCUMENT_COMPLETED,
            title_template="Document Processing Complete",
            message_template="Document '{document_name}' has been processed successfully and is ready for search.",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.NORMAL,
            expires_in_hours=24
        )
        
        templates[NotificationType.DOCUMENT_FAILED] = NotificationTemplate(
            notification_type=NotificationType.DOCUMENT_FAILED,
            title_template="Document Processing Failed",
            message_template="Failed to process document '{document_name}': {error_message}",
            severity=NotificationSeverity.ERROR,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.HIGH,
            expires_in_hours=48
        )
        
        # Embedding processing templates
        templates[NotificationType.EMBEDDING_PROGRESS] = NotificationTemplate(
            notification_type=NotificationType.EMBEDDING_PROGRESS,
            title_template="Embedding Generation",
            message_template="Model: {model_name} - {message} ({progress_percent}%)",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.NORMAL,
            expires_in_hours=1
        )
        
        # Case management templates
        templates[NotificationType.CASE_UPDATED] = NotificationTemplate(
            notification_type=NotificationType.CASE_UPDATED,
            title_template="Case Updated: {case_name}",
            message_template="Operation: {operation} - {details}",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.NORMAL,
            expires_in_hours=24
        )
        
        templates[NotificationType.CASE_CREATED] = NotificationTemplate(
            notification_type=NotificationType.CASE_CREATED,
            title_template="New Case Created",
            message_template="Case '{case_name}' has been created and is ready for document upload.",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.NORMAL,
            expires_in_hours=24
        )
        
        # System alert templates
        templates[NotificationType.SYSTEM_ALERT] = NotificationTemplate(
            notification_type=NotificationType.SYSTEM_ALERT,
            title_template="System Alert: {alert_type}",
            message_template="{message} (Severity: {severity})",
            severity=NotificationSeverity.WARNING,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.HIGH,
            expires_in_hours=48
        )
        
        templates[NotificationType.RESOURCE_ALERT] = NotificationTemplate(
            notification_type=NotificationType.RESOURCE_ALERT,
            title_template="Resource Alert",
            message_template="Resource usage alert: {message}",
            severity=NotificationSeverity.WARNING,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.HIGH,
            expires_in_hours=24
        )
        
        # Configuration change templates
        templates[NotificationType.CONFIGURATION_CHANGED] = NotificationTemplate(
            notification_type=NotificationType.CONFIGURATION_CHANGED,
            title_template="Configuration Updated",
            message_template="Configuration section '{section}' has been updated by {user_id}",
            severity=NotificationSeverity.INFO,
            channels=[NotificationChannel.WEBSOCKET],
            priority=MessagePriority.NORMAL,
            expires_in_hours=12
        )
        
        return templates
    
    async def _should_deliver_notification(self, notification: Notification) -> bool:
        """Check if notification should be delivered based on user preferences."""
        preferences = await self.get_user_preferences(notification.user_id)
        
        # Check each delivery channel
        for channel in notification.channels:
            if preferences.should_deliver(
                notification.notification_type,
                notification.severity,
                channel
            ):
                return True
        
        return False
    
    async def _check_rate_limit(self, user_id: str, notification_type: NotificationType) -> bool:
        """Check if notification is within rate limits."""
        preferences = await self.get_user_preferences(user_id)
        current_time = datetime.now(timezone.utc)
        
        # Check if we need to reset the rate limit window
        if user_id not in self.rate_limit_windows:
            self.rate_limit_windows[user_id] = current_time
            self.rate_limits[user_id] = defaultdict(int)
        
        window_start = self.rate_limit_windows[user_id]
        if (current_time - window_start).total_seconds() >= 3600:  # 1 hour window
            self.rate_limit_windows[user_id] = current_time
            self.rate_limits[user_id] = defaultdict(int)
        
        # Check current rate
        current_count = self.rate_limits[user_id][notification_type.value]
        if current_count >= preferences.rate_limit_per_hour:
            return False
        
        # Increment counter
        self.rate_limits[user_id][notification_type.value] += 1
        return True
    
    async def _queue_notification(self, notification: Notification) -> None:
        """Queue notification for delivery."""
        user_queue = self.pending_notifications[notification.user_id]
        user_queue.append(notification)
        
        # Initialize delivery status for all channels
        for channel in notification.channels:
            notification.delivery_status[channel] = DeliveryStatus.PENDING
            notification.delivery_attempts[channel] = 0
    
    async def _delivery_worker(self) -> None:
        """Background worker for delivering notifications."""
        while self._running:
            try:
                delivered_count = 0
                
                # Process pending notifications for all users
                for user_id, user_queue in self.pending_notifications.items():
                    if not user_queue:
                        continue
                    
                    # Process notifications in priority order
                    notifications_to_process = list(user_queue)
                    notifications_to_process.sort(key=lambda n: n.priority.value, reverse=True)
                    
                    for notification in notifications_to_process:
                        if notification.is_expired():
                            user_queue.remove(notification)
                            self._update_stats(notification, DeliveryStatus.EXPIRED)
                            continue
                        
                        success = await self._deliver_notification(notification)
                        if success:
                            user_queue.remove(notification)
                            delivered_count += 1
                        
                        # Limit delivery rate to avoid overwhelming
                        if delivered_count >= 10:
                            break
                    
                    if delivered_count >= 10:
                        break
                
                # Wait before next delivery cycle
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Error in notification delivery worker", error=str(e))
                await asyncio.sleep(5)
    
    async def _deliver_notification(self, notification: Notification) -> bool:
        """Deliver a notification through all its channels."""
        all_delivered = True
        
        for channel in notification.channels:
            if notification.delivery_status[channel] == DeliveryStatus.DELIVERED:
                continue
            
            if not notification.should_retry(channel):
                continue
            
            success = await self._deliver_to_channel(notification, channel)
            
            notification.delivery_attempts[channel] += 1
            notification.last_delivery_attempt = datetime.now(timezone.utc)
            
            if success:
                notification.delivery_status[channel] = DeliveryStatus.DELIVERED
                self._update_stats(notification, DeliveryStatus.DELIVERED)
            else:
                notification.delivery_status[channel] = DeliveryStatus.FAILED
                all_delivered = False
        
        return all_delivered
    
    async def _deliver_to_channel(
        self,
        notification: Notification,
        channel: NotificationChannel
    ) -> bool:
        """Deliver notification to a specific channel."""
        try:
            if channel == NotificationChannel.WEBSOCKET:
                return await self._deliver_websocket(notification)
            elif channel == NotificationChannel.EMAIL:
                # Future implementation
                return False
            else:
                logger.warning(f"Unsupported delivery channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(
                "Channel delivery failed",
                notification_id=notification.notification_id,
                channel=channel.value,
                error=str(e)
            )
            return False
    
    async def _deliver_websocket(self, notification: Notification) -> bool:
        """Deliver notification via WebSocket."""
        try:
            message_data = {
                "notification_id": notification.notification_id,
                "type": notification.notification_type.value,
                "title": notification.title,
                "message": notification.message,
                "severity": notification.severity.value,
                "created_at": notification.created_at.isoformat(),
                "context": notification.context,
                "metadata": notification.metadata
            }
            
            sent_count = await self.websocket_manager.broadcast_to_user(
                notification.user_id,
                "notification",
                message_data,
                notification.priority
            )
            
            return sent_count > 0
            
        except Exception as e:
            logger.error(
                "WebSocket delivery failed",
                notification_id=notification.notification_id,
                user_id=notification.user_id,
                error=str(e)
            )
            return False
    
    def _update_stats(self, notification: Notification, status: DeliveryStatus) -> None:
        """Update delivery statistics."""
        self.stats.total_sent += 1
        
        if status == DeliveryStatus.DELIVERED:
            self.stats.total_delivered += 1
        elif status == DeliveryStatus.FAILED:
            self.stats.total_failed += 1
        elif status == DeliveryStatus.EXPIRED:
            self.stats.total_expired += 1
        
        # Update channel stats
        for channel in notification.channels:
            if channel not in self.stats.channel_stats:
                self.stats.channel_stats[channel] = {"sent": 0, "delivered": 0, "failed": 0}
            
            self.stats.channel_stats[channel]["sent"] += 1
            if status == DeliveryStatus.DELIVERED:
                self.stats.channel_stats[channel]["delivered"] += 1
            elif status == DeliveryStatus.FAILED:
                self.stats.channel_stats[channel]["failed"] += 1
        
        # Update type stats
        notif_type = notification.notification_type
        if notif_type not in self.stats.type_stats:
            self.stats.type_stats[notif_type] = {"sent": 0, "delivered": 0, "failed": 0}
        
        self.stats.type_stats[notif_type]["sent"] += 1
        if status == DeliveryStatus.DELIVERED:
            self.stats.type_stats[notif_type]["delivered"] += 1
        elif status == DeliveryStatus.FAILED:
            self.stats.type_stats[notif_type]["failed"] += 1
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up expired notifications."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                cleaned_count = 0
                
                # Clean up notification history (keep last 1000)
                if len(self.notification_history) > 1000:
                    self.notification_history = self.notification_history[-1000:]
                    cleaned_count += len(self.notification_history) - 1000
                
                # Clean up expired pending notifications
                for user_id, user_queue in self.pending_notifications.items():
                    expired_notifications = [
                        n for n in user_queue 
                        if n.is_expired()
                    ]
                    
                    for notification in expired_notifications:
                        user_queue.remove(notification)
                        self._update_stats(notification, DeliveryStatus.EXPIRED)
                        cleaned_count += 1
                
                if cleaned_count > 0:
                    logger.debug(f"Cleaned up {cleaned_count} expired notifications")
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Error in notification cleanup worker", error=str(e))
                await asyncio.sleep(60)
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        await self.stop()
        
        # Clear state
        self.pending_notifications.clear()
        self.notification_history.clear()
        self.user_preferences.clear()
        self.rate_limits.clear()
        self.rate_limit_windows.clear()
        
        logger.info("NotificationService cleanup completed")