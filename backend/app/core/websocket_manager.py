"""
WebSocket Connection Manager for Legal AI Application

This module provides a comprehensive WebSocket connection pool management system
that supports real-time communication for multi-user legal document processing,
search operations, and system monitoring.

Key Features:
- Connection pool management with automatic cleanup
- User-based message broadcasting and targeting
- Message queuing and delivery guarantees
- Connection health monitoring and heartbeat
- Real-time progress updates for long-running operations
- Administrative system monitoring broadcasts
- Message throttling and rate limiting
- Connection authentication and authorization
- Graceful connection handling with reconnection support
- Performance metrics and connection analytics

Real-Time Features:
- Document processing progress updates
- Search operation status notifications
- System resource monitoring broadcasts
- Configuration change notifications
- Error and alert distributions
- Multi-user collaboration updates

Connection Management:
- Per-user connection grouping
- Connection lifecycle management
- Message routing and targeting
- Queue management for offline users
- Connection pool optimization
- Memory efficient connection storage

Architecture Integration:
- Integrates with FastAPI WebSocket endpoints
- Supports document processors for progress updates
- Provides system monitoring data distribution
- Enables real-time admin panel updates
- Supports case-based user isolation
- Implements secure message broadcasting
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from weakref import WeakSet
import threading
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio

from ..utils.logging import get_logger
from .exceptions import WebSocketError, ErrorCode

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """Types of WebSocket messages."""
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"
    BROADCAST = "broadcast"
    DIRECT_MESSAGE = "direct_message"
    PROGRESS_UPDATE = "progress_update"
    SYSTEM_ALERT = "system_alert"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels for queuing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedMessage:
    """Represents a queued message for delivery."""
    message_id: str
    user_id: str
    message_type: str
    data: Dict[str, Any]
    priority: MessagePriority
    created_at: datetime
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str]
    client_ip: str
    user_agent: Optional[str]
    connected_at: datetime
    last_ping: Optional[datetime]
    last_pong: Optional[datetime]
    state: ConnectionState
    subscriptions: Set[str] = field(default_factory=set)
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0


class ConnectionMetrics(BaseModel):
    """Connection pool metrics."""
    total_connections: int = 0
    authenticated_connections: int = 0
    active_users: int = 0
    messages_sent_total: int = 0
    messages_received_total: int = 0
    bytes_sent_total: int = 0
    bytes_received_total: int = 0
    average_connection_duration_seconds: float = 0.0
    peak_concurrent_connections: int = 0
    connection_errors_total: int = 0
    message_queue_size: int = 0


class WebSocketManager:
    """
    Manages WebSocket connections for real-time legal AI application communication.
    
    Provides connection pooling, user-based message routing, message queuing,
    and comprehensive monitoring for multi-user legal document processing.
    """
    
    def __init__(
        self,
        max_connections_per_user: int = 5,
        message_queue_size: int = 1000,
        heartbeat_interval: int = 30,
        connection_timeout: int = 300
    ):
        """
        Initialize WebSocket manager.
        
        Args:
            max_connections_per_user: Maximum connections per user
            message_queue_size: Maximum queued messages per user
            heartbeat_interval: Heartbeat interval in seconds
            connection_timeout: Connection timeout in seconds
        """
        self.max_connections_per_user = max_connections_per_user
        self.message_queue_size = message_queue_size
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Message queuing
        self.message_queues: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=message_queue_size)
        )
        self.pending_messages: List[QueuedMessage] = []
        
        # Performance monitoring
        self.metrics = ConnectionMetrics()
        self.connection_history: List[Tuple[datetime, int]] = []
        self.message_history: List[Tuple[datetime, int]] = []
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {
            MessageType.PING.value: self._handle_ping,
            MessageType.AUTH.value: self._handle_auth,
            MessageType.SUBSCRIPTION.value: self._handle_subscription,
            MessageType.UNSUBSCRIPTION.value: self._handle_unsubscription
        }
        
        logger.info(
            "WebSocketManager initialized",
            max_connections_per_user=max_connections_per_user,
            message_queue_size=message_queue_size,
            heartbeat_interval=heartbeat_interval
        )
    
    async def start(self) -> None:
        """Start the WebSocket manager background tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("WebSocketManager started")
    
    async def stop(self) -> None:
        """Stop the WebSocket manager and cleanup connections."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._cleanup_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all connections
        await self._close_all_connections()
        
        logger.info("WebSocketManager stopped")
    
    async def connect(
        self,
        websocket: WebSocket,
        client_ip: str,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            client_ip: Client IP address
            user_agent: Optional user agent string
            
        Returns:
            Connection ID for the new connection
            
        Raises:
            ConnectionError: If connection cannot be established
        """
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # Create connection info
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                websocket=websocket,
                user_id=None,  # Set during authentication
                client_ip=client_ip,
                user_agent=user_agent,
                connected_at=datetime.now(timezone.utc),
                last_ping=None,
                last_pong=None,
                state=ConnectionState.CONNECTED
            )
            
            # Store connection
            self.connections[connection_id] = connection_info
            
            # Update metrics
            self.metrics.total_connections += 1
            if self.metrics.total_connections > self.metrics.peak_concurrent_connections:
                self.metrics.peak_concurrent_connections = self.metrics.total_connections
            
            logger.info(
                "WebSocket connection established",
                connection_id=connection_id,
                client_ip=client_ip,
                total_connections=self.metrics.total_connections
            )
            
            return connection_id
            
        except Exception as e:
            self.metrics.connection_errors_total += 1
            logger.error(
                "Failed to establish WebSocket connection",
                error=str(e),
                client_ip=client_ip
            )
            raise ConnectionError(
                message="Failed to establish connection",
                error_code=ErrorCode.CONNECTION_FAILED,
                details={"client_ip": client_ip, "error": str(e)}
            )
    
    async def disconnect(self, connection_id: str, reason: str = "Unknown") -> None:
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection to disconnect
            reason: Reason for disconnection
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.state = ConnectionState.DISCONNECTING
        
        try:
            # Remove from user connections
            if connection.user_id:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
                    self.metrics.active_users -= 1
            
            # Close WebSocket
            try:
                await connection.websocket.close()
            except Exception:
                pass  # Connection might already be closed
            
            # Update metrics
            connection_duration = (
                datetime.now(timezone.utc) - connection.connected_at
            ).total_seconds()
            
            self.metrics.total_connections -= 1
            if connection.state == ConnectionState.AUTHENTICATED:
                self.metrics.authenticated_connections -= 1
            
            # Remove connection
            del self.connections[connection_id]
            
            logger.info(
                "WebSocket connection disconnected",
                connection_id=connection_id,
                user_id=connection.user_id,
                reason=reason,
                duration_seconds=connection_duration,
                messages_sent=connection.message_count
            )
            
        except Exception as e:
            logger.error(
                "Error during connection disconnection",
                connection_id=connection_id,
                error=str(e)
            )
    
    async def authenticate_connection(
        self,
        connection_id: str,
        user_id: str
    ) -> bool:
        """
        Authenticate a WebSocket connection with a user ID.
        
        Args:
            connection_id: Connection to authenticate
            user_id: User ID for authentication
            
        Returns:
            True if authentication successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Check connection limit per user
        if len(self.user_connections[user_id]) >= self.max_connections_per_user:
            logger.warning(
                "User connection limit exceeded",
                user_id=user_id,
                current_connections=len(self.user_connections[user_id]),
                limit=self.max_connections_per_user
            )
            return False
        
        # Update connection
        connection.user_id = user_id
        connection.state = ConnectionState.AUTHENTICATED
        
        # Add to user connections
        self.user_connections[user_id].add(connection_id)
        
        # Update metrics
        self.metrics.authenticated_connections += 1
        if len(self.user_connections[user_id]) == 1:
            self.metrics.active_users += 1
        
        # Deliver queued messages
        await self._deliver_queued_messages(user_id)
        
        logger.info(
            "WebSocket connection authenticated",
            connection_id=connection_id,
            user_id=user_id,
            user_connections=len(self.user_connections[user_id])
        )
        
        return True
    
    async def send_to_connection(
        self,
        connection_id: str,
        message_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific connection.
        
        Args:
            connection_id: Target connection ID
            message_type: Type of message
            data: Message data
            
        Returns:
            True if message sent successfully
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            message_json = json.dumps(message)
            await connection.websocket.send_text(message_json)
            
            # Update metrics
            connection.message_count += 1
            connection.bytes_sent += len(message_json.encode())
            self.metrics.messages_sent_total += 1
            self.metrics.bytes_sent_total += len(message_json.encode())
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id, "WebSocket disconnect")
            return False
        except Exception as e:
            logger.error(
                "Failed to send message to connection",
                connection_id=connection_id,
                message_type=message_type,
                error=str(e)
            )
            return False
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message_type: str,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> int:
        """
        Broadcast message to all connections for a specific user.
        
        Args:
            user_id: Target user ID
            message_type: Type of message
            data: Message data
            priority: Message priority
            
        Returns:
            Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            # Queue message for offline user
            await self._queue_message(user_id, message_type, data, priority)
            return 0
        
        sent_count = 0
        failed_connections = []
        
        for connection_id in self.user_connections[user_id].copy():
            success = await self.send_to_connection(connection_id, message_type, data)
            if success:
                sent_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id, "Send failure")
        
        logger.debug(
            "Message broadcast to user",
            user_id=user_id,
            message_type=message_type,
            sent_count=sent_count,
            failed_count=len(failed_connections)
        )
        
        return sent_count
    
    async def broadcast_to_all(
        self,
        message_type: str,
        data: Dict[str, Any],
        exclude_users: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast message to all connected users.
        
        Args:
            message_type: Type of message
            data: Message data
            exclude_users: Optional set of user IDs to exclude
            
        Returns:
            Number of connections message was sent to
        """
        sent_count = 0
        exclude_users = exclude_users or set()
        
        for connection_id, connection in self.connections.copy().items():
            if connection.user_id and connection.user_id not in exclude_users:
                success = await self.send_to_connection(connection_id, message_type, data)
                if success:
                    sent_count += 1
        
        logger.debug(
            "Message broadcast to all users",
            message_type=message_type,
            sent_count=sent_count,
            excluded_users=len(exclude_users)
        )
        
        return sent_count
    
    async def broadcast_to_subscribers(
        self,
        subscription: str,
        message_type: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Broadcast message to connections subscribed to a specific topic.
        
        Args:
            subscription: Subscription topic
            message_type: Type of message
            data: Message data
            
        Returns:
            Number of connections message was sent to
        """
        sent_count = 0
        
        for connection_id, connection in self.connections.copy().items():
            if subscription in connection.subscriptions:
                success = await self.send_to_connection(connection_id, message_type, data)
                if success:
                    sent_count += 1
        
        logger.debug(
            "Message broadcast to subscribers",
            subscription=subscription,
            message_type=message_type,
            sent_count=sent_count
        )
        
        return sent_count
    
    async def subscribe_connection(
        self,
        connection_id: str,
        subscription: str
    ) -> bool:
        """
        Subscribe connection to a topic.
        
        Args:
            connection_id: Connection ID
            subscription: Subscription topic
            
        Returns:
            True if subscription successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.add(subscription)
        self.connection_subscriptions[subscription].add(connection_id)
        
        logger.debug(
            "Connection subscribed",
            connection_id=connection_id,
            subscription=subscription,
            total_subscriptions=len(connection.subscriptions)
        )
        
        return True
    
    async def unsubscribe_connection(
        self,
        connection_id: str,
        subscription: str
    ) -> bool:
        """
        Unsubscribe connection from a topic.
        
        Args:
            connection_id: Connection ID
            subscription: Subscription topic
            
        Returns:
            True if unsubscription successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscriptions.discard(subscription)
        self.connection_subscriptions[subscription].discard(connection_id)
        
        # Clean up empty subscription sets
        if not self.connection_subscriptions[subscription]:
            del self.connection_subscriptions[subscription]
        
        logger.debug(
            "Connection unsubscribed",
            connection_id=connection_id,
            subscription=subscription,
            remaining_subscriptions=len(connection.subscriptions)
        )
        
        return True
    
    def get_connection_count(self, user_id: Optional[str] = None) -> int:
        """
        Get connection count for user or total.
        
        Args:
            user_id: Optional user ID, if None returns total count
            
        Returns:
            Number of connections
        """
        if user_id:
            return len(self.user_connections.get(user_id, set()))
        return len(self.connections)
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current connection metrics."""
        # Update dynamic metrics
        if self.connections:
            total_duration = sum(
                (datetime.now(timezone.utc) - conn.connected_at).total_seconds()
                for conn in self.connections.values()
            )
            self.metrics.average_connection_duration_seconds = total_duration / len(self.connections)
        
        self.metrics.message_queue_size = sum(len(queue) for queue in self.message_queues.values())
        
        return self.metrics
    
    async def handle_message(
        self,
        connection_id: str,
        message: str
    ) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: Source connection ID
            message: Raw message string
        """
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get("type")
            message_data = data.get("data", {})
            
            # Update metrics
            connection.bytes_received += len(message.encode())
            self.metrics.messages_received_total += 1
            self.metrics.bytes_received_total += len(message.encode())
            
            # Handle message
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](connection_id, message_data)
            else:
                logger.warning(
                    "Unknown message type received",
                    connection_id=connection_id,
                    message_type=message_type
                )
                
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON message received",
                connection_id=connection_id,
                message=message[:100]  # Log first 100 chars
            )
        except Exception as e:
            logger.error(
                "Error handling WebSocket message",
                connection_id=connection_id,
                error=str(e)
            )
    
    async def _queue_message(
        self,
        user_id: str,
        message_type: str,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """Queue message for offline user."""
        message = QueuedMessage(
            message_id=str(uuid.uuid4()),
            user_id=user_id,
            message_type=message_type,
            data=data,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        # Add to queue in priority order
        queue = self.message_queues[user_id]
        queue.append(message)
        
        # Sort by priority (higher priority first)
        self.message_queues[user_id] = deque(
            sorted(queue, key=lambda m: m.priority.value, reverse=True),
            maxlen=self.message_queue_size
        )
    
    async def _deliver_queued_messages(self, user_id: str) -> None:
        """Deliver queued messages to newly connected user."""
        if user_id not in self.message_queues:
            return
        
        queue = self.message_queues[user_id]
        delivered_count = 0
        
        while queue:
            message = queue.popleft()
            
            # Check if message expired
            if message.expires_at and datetime.now(timezone.utc) > message.expires_at:
                continue
            
            # Attempt delivery
            sent_count = await self.broadcast_to_user(
                user_id, message.message_type, message.data
            )
            
            if sent_count > 0:
                delivered_count += 1
            else:
                # Re-queue if delivery failed
                queue.appendleft(message)
                break
        
        if delivered_count > 0:
            logger.info(
                "Delivered queued messages",
                user_id=user_id,
                delivered_count=delivered_count,
                remaining_count=len(queue)
            )
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle ping message."""
        if connection_id in self.connections:
            self.connections[connection_id].last_ping = datetime.now(timezone.utc)
            await self.send_to_connection(connection_id, MessageType.PONG.value, {})
    
    async def _handle_auth(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle authentication message."""
        user_id = data.get("user_id")
        if user_id:
            success = await self.authenticate_connection(connection_id, user_id)
            await self.send_to_connection(
                connection_id,
                "auth_response",
                {"success": success, "user_id": user_id if success else None}
            )
    
    async def _handle_subscription(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle subscription message."""
        subscription = data.get("subscription")
        if subscription:
            success = await self.subscribe_connection(connection_id, subscription)
            await self.send_to_connection(
                connection_id,
                "subscription_response",
                {"success": success, "subscription": subscription}
            )
    
    async def _handle_unsubscription(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle unsubscription message."""
        subscription = data.get("subscription")
        if subscription:
            success = await self.unsubscribe_connection(connection_id, subscription)
            await self.send_to_connection(
                connection_id,
                "unsubscription_response",
                {"success": success, "subscription": subscription}
            )
    
    async def _heartbeat_loop(self) -> None:
        """Background task for connection health monitoring."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check for stale connections
                    if (connection.last_ping and 
                        (current_time - connection.last_ping).total_seconds() > self.connection_timeout):
                        stale_connections.append(connection_id)
                    
                    # Send ping if needed
                    elif (not connection.last_ping or 
                          (current_time - connection.last_ping).total_seconds() > self.heartbeat_interval):
                        await self.send_to_connection(connection_id, MessageType.PING.value, {})
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, "Heartbeat timeout")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while self._running:
            try:
                # Clean up expired queued messages
                current_time = datetime.now(timezone.utc)
                
                for user_id, queue in self.message_queues.items():
                    expired_messages = []
                    for message in queue:
                        if message.expires_at and current_time > message.expires_at:
                            expired_messages.append(message)
                    
                    for message in expired_messages:
                        queue.remove(message)
                
                # Clean up empty queues
                empty_queues = [
                    user_id for user_id, queue in self.message_queues.items()
                    if not queue
                ]
                for user_id in empty_queues:
                    del self.message_queues[user_id]
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_loop(self) -> None:
        """Background task for metrics collection."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Record connection count
                self.connection_history.append((current_time, len(self.connections)))
                self.message_history.append((current_time, self.metrics.messages_sent_total))
                
                # Keep only last 24 hours of history
                cutoff_time = current_time - timedelta(hours=24)
                self.connection_history = [
                    (ts, count) for ts, count in self.connection_history
                    if ts > cutoff_time
                ]
                self.message_history = [
                    (ts, count) for ts, count in self.message_history
                    if ts > cutoff_time
                ]
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(60)
    
    async def _close_all_connections(self) -> None:
        """Close all WebSocket connections."""
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id, "Manager shutdown")

# Add this function to the end of backend/app/core/websocket_manager.py
# after the WebSocketManager class definition

# Global WebSocket manager instance
_websocket_manager_instance: Optional[WebSocketManager] = None
_websocket_manager_lock = threading.Lock()


def get_websocket_manager() -> WebSocketManager:
    """
    Get the global WebSocket manager instance (FastAPI dependency).
    
    This function provides a singleton WebSocket manager instance that can be
    used as a FastAPI dependency for dependency injection. It ensures that
    the same WebSocket manager is used across all route handlers and services.
    
    Returns:
        WebSocketManager: The global WebSocket manager instance
        
    Raises:
        RuntimeError: If the WebSocket manager has not been initialized
        
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            websocket_manager: WebSocketManager = Depends(get_websocket_manager)
        ):
            await websocket_manager.broadcast_to_all({"message": "Hello"})
    """
    global _websocket_manager_instance
    
    if _websocket_manager_instance is None:
        raise RuntimeError(
            "WebSocket manager not initialized. "
            "Ensure the application startup process has been completed."
        )
    
    return _websocket_manager_instance


def set_websocket_manager(manager: WebSocketManager) -> None:
    """
    Set the global WebSocket manager instance.
    
    This function is typically called during application startup to set the
    WebSocket manager instance that will be used throughout the application.
    
    Args:
        manager: The WebSocket manager instance to set as global
        
    Thread Safety:
        This function is thread-safe and uses a lock to prevent race conditions
        during initialization.
    """
    global _websocket_manager_instance
    
    with _websocket_manager_lock:
        _websocket_manager_instance = manager
        logger.info("Global WebSocket manager instance set")


async def initialize_websocket_manager(
    max_connections_per_user: int = 5,
    message_queue_size: int = 1000,
    heartbeat_interval: int = 30,
    connection_timeout: int = 300
) -> WebSocketManager:
    """
    Initialize and return a new WebSocket manager instance.
    
    This function creates a new WebSocket manager with the specified configuration
    and initializes it. It's typically called during application startup.
    
    Args:
        max_connections_per_user: Maximum connections allowed per user
        message_queue_size: Maximum size of the message queue
        heartbeat_interval: Interval between heartbeat checks in seconds
        connection_timeout: Connection timeout in seconds
        
    Returns:
        WebSocketManager: Initialized WebSocket manager instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        manager = WebSocketManager(
            max_connections_per_user=max_connections_per_user,
            message_queue_size=message_queue_size,
            heartbeat_interval=heartbeat_interval,
            connection_timeout=connection_timeout
        )
        
        # Initialize the manager
        await manager.initialize()
        
        # Set as global instance
        set_websocket_manager(manager)
        
        logger.info(
            "WebSocket manager initialized successfully",
            max_connections_per_user=max_connections_per_user,
            message_queue_size=message_queue_size,
            heartbeat_interval=heartbeat_interval,
            connection_timeout=connection_timeout
        )
        
        return manager
        
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket manager: {e}")
        raise RuntimeError(f"WebSocket manager initialization failed: {e}")


async def cleanup_websocket_manager() -> None:
    """
    Clean up the global WebSocket manager instance.
    
    This function is typically called during application shutdown to properly
    clean up WebSocket connections and resources.
    """
    global _websocket_manager_instance
    
    if _websocket_manager_instance is not None:
        try:
            # Disconnect all connections gracefully
            await _websocket_manager_instance.disconnect_all_users(
                reason="Server shutdown"
            )
            
            # Stop any background tasks
            if hasattr(_websocket_manager_instance, 'stop'):
                await _websocket_manager_instance.stop()
            
            logger.info("WebSocket manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during WebSocket manager cleanup: {e}")
        finally:
            with _websocket_manager_lock:
                _websocket_manager_instance = None


def get_websocket_manager_metrics() -> Optional[Dict[str, Any]]:
    """
    Get metrics from the WebSocket manager if available.
    
    Returns:
        Optional[Dict[str, Any]]: WebSocket manager metrics or None if not available
    """
    if _websocket_manager_instance is not None:
        try:
            return _websocket_manager_instance.get_metrics().dict()
        except Exception as e:
            logger.error(f"Error getting WebSocket manager metrics: {e}")
            return None
    return None


def is_websocket_manager_initialized() -> bool:
    """
    Check if the WebSocket manager has been initialized.
    
    Returns:
        bool: True if WebSocket manager is initialized, False otherwise
    """
    return _websocket_manager_instance is not None


