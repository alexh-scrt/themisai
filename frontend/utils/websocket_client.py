"""
WebSocket Client for Patexia Legal AI Chatbot Frontend

This module provides a comprehensive WebSocket client for real-time communication
with the FastAPI backend. It handles document processing progress updates, 
system notifications, case management events, and admin panel monitoring.

Key Features:
- Async WebSocket connection management with auto-reconnection
- Message routing and handler registration system
- Real-time progress tracking for document processing
- System alert and notification handling
- Connection health monitoring with heartbeat
- Message queuing for offline scenarios
- Event-driven architecture with callback support
- Error handling and connection recovery
- Authentication and session management
- Performance monitoring and metrics collection

Architecture Integration:
- Communicates with backend WebSocketManager via FastAPI WebSocket endpoints
- Provides real-time updates for all frontend components
- Supports case-scoped message filtering and routing
- Integrates with authentication system for secure connections
- Handles document processing pipeline events
- Enables admin panel real-time monitoring

Real-Time Features:
- Document upload and processing progress
- Search operation status updates
- Case creation and modification events
- System resource monitoring broadcasts
- Configuration change notifications
- Error and alert distribution
- Multi-user collaboration updates

Connection Management:
- Automatic connection establishment and recovery
- Heartbeat monitoring for connection health
- Graceful degradation when WebSocket unavailable
- Message delivery guarantees with retry logic
- Connection pooling and resource optimization
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import ssl

import websockets
from websockets.exceptions import (
    ConnectionClosed, WebSocketException, InvalidMessage,
    ConnectionClosedError, ConnectionClosedOK
)


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class MessageType(str, Enum):
    """Types of WebSocket messages."""
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    AUTH_RESPONSE = "auth_response"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DOCUMENT_PROGRESS = "document_progress"
    DOCUMENT_COMPLETED = "document_completed"
    DOCUMENT_ERROR = "document_error"
    SEARCH_PROGRESS = "search_progress"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_ERROR = "search_error"
    CASE_CREATED = "case_created"
    CASE_UPDATED = "case_updated"
    CASE_DELETED = "case_deleted"
    SYSTEM_ALERT = "system_alert"
    CONFIG_UPDATED = "config_updated"
    RESOURCE_UPDATE = "resource_update"
    ERROR = "error"


class ReconnectStrategy(str, Enum):
    """Reconnection strategies."""
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket client."""
    base_url: str = "ws://localhost:8000"
    path: str = "/api/v1/ws"
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_reconnect_attempts: int = 10
    reconnect_strategy: ReconnectStrategy = ReconnectStrategy.EXPONENTIAL_BACKOFF
    reconnect_backoff_base: float = 2.0
    reconnect_max_delay: float = 60.0
    message_timeout: float = 30.0
    enable_logging: bool = True
    auth_timeout: float = 10.0


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    message_id: str
    message_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    case_id: Optional[str] = None


@dataclass
class ConnectionMetrics:
    """WebSocket connection metrics."""
    connection_count: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_connection_time: Optional[datetime] = None
    last_message_time: Optional[datetime] = None
    average_latency_ms: float = 0.0
    connection_uptime_seconds: float = 0.0


class WebSocketError(Exception):
    """Base exception for WebSocket client errors."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ConnectionError(WebSocketError):
    """Exception for connection-related errors."""
    pass


class AuthenticationError(WebSocketError):
    """Exception for authentication errors."""
    pass


class MessageError(WebSocketError):
    """Exception for message-related errors."""
    pass


class WebSocketClient:
    """
    Comprehensive WebSocket client for real-time legal AI application communication.
    
    Provides async connection management, message routing, progress tracking,
    and real-time updates for legal document processing workflows.
    """
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        """
        Initialize WebSocket client.
        
        Args:
            config: Optional WebSocket client configuration
        """
        self.config = config or WebSocketConfig()
        self.logger = logging.getLogger(f"{__name__}.WebSocketClient")
        
        # Connection state
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self._session_id: str = str(uuid.uuid4())
        self._user_id: Optional[str] = None
        self._auth_token: Optional[str] = None
        
        # Message handling
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._subscriptions: Set[str] = set()
        
        # Connection management
        self._reconnect_attempts: int = 0
        self._last_ping_time: Optional[datetime] = None
        self._last_pong_time: Optional[datetime] = None
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_task: Optional[asyncio.Task] = None
        
        # Message queuing for offline scenarios
        self._message_queue: List[WebSocketMessage] = []
        self._max_queue_size: int = 1000
        
        # Performance metrics
        self._metrics = ConnectionMetrics()
        self._latency_samples: List[float] = []
        
        # Event callbacks
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable[[Exception], None]] = []
        
        if self.config.enable_logging:
            self.logger.info(f"WebSocket client initialized with session: {self._session_id}")
    
    async def connect(self, user_id: Optional[str] = None, auth_token: Optional[str] = None) -> None:
        """
        Establish WebSocket connection to the backend.
        
        Args:
            user_id: Optional user ID for authentication
            auth_token: Optional authentication token
            
        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
            self.logger.warning("WebSocket already connected")
            return
        
        self._user_id = user_id
        self._auth_token = auth_token
        self._connection_state = ConnectionState.CONNECTING
        
        try:
            # Build WebSocket URL
            ws_url = urljoin(self.config.base_url, self.config.path)
            
            self.logger.info(f"Connecting to WebSocket: {ws_url}")
            
            # Establish connection
            self._websocket = await websockets.connect(
                ws_url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=10.0,
                max_size=10**7,  # 10MB max message size
                max_queue=32
            )
            
            self._connection_state = ConnectionState.CONNECTED
            self._metrics.connection_count += 1
            self._metrics.successful_connections += 1
            self._metrics.last_connection_time = datetime.now(timezone.utc)
            self._reconnect_attempts = 0
            
            self.logger.info("WebSocket connection established")
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._message_task = asyncio.create_task(self._message_loop())
            
            # Authenticate if credentials provided
            if self._user_id and self._auth_token:
                await self._authenticate()
            
            # Process queued messages
            await self._process_queued_messages()
            
            # Notify connection callbacks
            for callback in self._on_connect_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error(f"Error in connect callback: {e}")
                    
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            self._metrics.failed_connections += 1
            error_msg = f"Failed to connect to WebSocket: {e}"
            self.logger.error(error_msg)
            
            # Notify error callbacks
            for callback in self._on_error_callbacks:
                try:
                    callback(e)
                except Exception as cb_error:
                    self.logger.error(f"Error in error callback: {cb_error}")
            
            raise ConnectionError(error_msg)
    
    async def disconnect(self, reason: str = "Client requested") -> None:
        """
        Disconnect WebSocket connection gracefully.
        
        Args:
            reason: Reason for disconnection
        """
        if self._connection_state == ConnectionState.DISCONNECTED:
            return
        
        self.logger.info(f"Disconnecting WebSocket: {reason}")
        self._connection_state = ConnectionState.DISCONNECTED
        
        # Cancel background tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
        
        # Close WebSocket connection
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        # Notify disconnect callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error(f"Error in disconnect callback: {e}")
    
    async def send_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> None:
        """
        Send message to WebSocket server.
        
        Args:
            message_type: Type of message to send
            data: Message data
            correlation_id: Optional correlation ID for tracking
            timeout: Optional timeout for response
            
        Raises:
            MessageError: If message cannot be sent
        """
        if self._connection_state not in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
            # Queue message for later delivery
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                data=data,
                correlation_id=correlation_id,
                user_id=self._user_id
            )
            
            if len(self._message_queue) < self._max_queue_size:
                self._message_queue.append(message)
                self.logger.info(f"Queued message: {message_type}")
            else:
                self.logger.warning("Message queue full, dropping message")
            
            return
        
        try:
            message = {
                "message_id": str(uuid.uuid4()),
                "type": message_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": self._user_id,
                "session_id": self._session_id
            }
            
            if correlation_id:
                message["correlation_id"] = correlation_id
            
            message_json = json.dumps(message)
            
            if self._websocket:
                await self._websocket.send(message_json)
                
                self._metrics.messages_sent += 1
                self._metrics.bytes_sent += len(message_json.encode())
                
                self.logger.debug(f"Sent message: {message_type}")
            else:
                raise MessageError("WebSocket connection not available")
                
        except Exception as e:
            error_msg = f"Failed to send message {message_type}: {e}"
            self.logger.error(error_msg)
            raise MessageError(error_msg)
    
    def add_handler(self, message_type: str, handler: Callable[[WebSocketMessage], None]) -> None:
        """
        Add message handler for specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function (can be async)
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        
        self._message_handlers[message_type].append(handler)
        self.logger.debug(f"Added handler for {message_type}")
    
    def remove_handler(self, message_type: str, handler: Callable) -> None:
        """
        Remove message handler for specific message type.
        
        Args:
            message_type: Type of message
            handler: Handler function to remove
        """
        if message_type in self._message_handlers:
            try:
                self._message_handlers[message_type].remove(handler)
                self.logger.debug(f"Removed handler for {message_type}")
            except ValueError:
                self.logger.warning(f"Handler not found for {message_type}")
    
    async def subscribe(self, topic: str) -> None:
        """
        Subscribe to a specific topic for targeted messages.
        
        Args:
            topic: Topic to subscribe to (e.g., case_id, user_id)
        """
        if topic not in self._subscriptions:
            self._subscriptions.add(topic)
            await self.send_message(MessageType.SUBSCRIBE, {"topic": topic})
            self.logger.info(f"Subscribed to topic: {topic}")
    
    async def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a specific topic.
        
        Args:
            topic: Topic to unsubscribe from
        """
        if topic in self._subscriptions:
            self._subscriptions.remove(topic)
            await self.send_message(MessageType.UNSUBSCRIBE, {"topic": topic})
            self.logger.info(f"Unsubscribed from topic: {topic}")
    
    def on_connect(self, callback: Callable) -> None:
        """Register callback for connection events."""
        self._on_connect_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable) -> None:
        """Register callback for disconnection events."""
        self._on_disconnect_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for error events."""
        self._on_error_callbacks.append(callback)
    
    async def wait_for_response(
        self,
        correlation_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Wait for a response message with specific correlation ID.
        
        Args:
            correlation_id: Correlation ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Response message data
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        future = asyncio.Future()
        self._pending_responses[correlation_id] = future
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_responses.pop(correlation_id, None)
    
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        if self._metrics.last_connection_time:
            uptime = (datetime.now(timezone.utc) - self._metrics.last_connection_time).total_seconds()
            self._metrics.connection_uptime_seconds = uptime
        
        if self._latency_samples:
            self._metrics.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
        
        return self._metrics
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]
    
    async def _authenticate(self) -> None:
        """Authenticate with the WebSocket server."""
        try:
            correlation_id = str(uuid.uuid4())
            
            await self.send_message(
                MessageType.AUTH,
                {
                    "user_id": self._user_id,
                    "auth_token": self._auth_token
                },
                correlation_id=correlation_id
            )
            
            # Wait for authentication response
            response = await self.wait_for_response(correlation_id, self.config.auth_timeout)
            
            if response.get("success", False):
                self._connection_state = ConnectionState.AUTHENTICATED
                self.logger.info("WebSocket authentication successful")
            else:
                error = response.get("error", "Authentication failed")
                raise AuthenticationError(error)
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop to monitor connection health."""
        while self._connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
            try:
                if self._websocket:
                    # Send ping
                    ping_time = time.time()
                    await self.send_message(MessageType.PING, {"timestamp": ping_time})
                    self._last_ping_time = datetime.now(timezone.utc)
                    
                    # Wait for heartbeat interval
                    await asyncio.sleep(self.config.ping_interval)
                    
                    # Check if pong was received recently
                    if (self._last_pong_time and self._last_ping_time and
                        self._last_pong_time < self._last_ping_time):
                        time_since_pong = (datetime.now(timezone.utc) - self._last_pong_time).total_seconds()
                        if time_since_pong > self.config.ping_timeout:
                            self.logger.warning("Heartbeat timeout, connection may be stale")
                            await self._handle_connection_error("Heartbeat timeout")
                            break
                            
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await self._handle_connection_error(f"Heartbeat error: {e}")
                break
    
    async def _message_loop(self) -> None:
        """Background message loop to receive and process messages."""
        while self._connection_state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
            try:
                if self._websocket:
                    # Receive message
                    message_json = await self._websocket.recv()
                    
                    self._metrics.messages_received += 1
                    self._metrics.bytes_received += len(message_json.encode())
                    self._metrics.last_message_time = datetime.now(timezone.utc)
                    
                    # Parse message
                    try:
                        message_data = json.loads(message_json)
                        await self._handle_message(message_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse message: {e}")
                        
            except ConnectionClosed:
                self.logger.info("WebSocket connection closed by server")
                await self._handle_connection_error("Connection closed by server")
                break
            except Exception as e:
                self.logger.error(f"Message loop error: {e}")
                await self._handle_connection_error(f"Message loop error: {e}")
                break
    
    async def _handle_message(self, message_data: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message_data: Parsed message data
        """
        try:
            message = WebSocketMessage(
                message_id=message_data.get("message_id", str(uuid.uuid4())),
                message_type=message_data.get("type", "unknown"),
                data=message_data.get("data", {}),
                correlation_id=message_data.get("correlation_id"),
                user_id=message_data.get("user_id"),
                case_id=message_data.get("case_id")
            )
            
            # Handle special message types
            if message.message_type == MessageType.PONG:
                self._last_pong_time = datetime.now(timezone.utc)
                # Calculate latency
                ping_timestamp = message.data.get("timestamp")
                if ping_timestamp:
                    latency = (time.time() - ping_timestamp) * 1000  # Convert to ms
                    self._latency_samples.append(latency)
                    # Keep only recent samples
                    if len(self._latency_samples) > 100:
                        self._latency_samples = self._latency_samples[-50:]
                return
            
            # Handle response messages
            if message.correlation_id and message.correlation_id in self._pending_responses:
                future = self._pending_responses[message.correlation_id]
                if not future.done():
                    future.set_result(message.data)
                return
            
            # Route message to handlers
            handlers = self._message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler for {message.message_type}: {e}")
            
            self.logger.debug(f"Processed message: {message.message_type}")
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _handle_connection_error(self, error_message: str) -> None:
        """
        Handle connection errors and attempt reconnection.
        
        Args:
            error_message: Description of the error
        """
        self.logger.warning(f"Connection error: {error_message}")
        self._connection_state = ConnectionState.ERROR
        
        # Cancel background tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
        
        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        # Attempt reconnection
        if (self.config.reconnect_strategy != ReconnectStrategy.NONE and
            self._reconnect_attempts < self.config.max_reconnect_attempts):
            
            await self._attempt_reconnection()
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect with backoff strategy."""
        self._reconnect_attempts += 1
        self._connection_state = ConnectionState.RECONNECTING
        self._metrics.reconnection_count += 1
        
        # Calculate delay based on strategy
        if self.config.reconnect_strategy == ReconnectStrategy.IMMEDIATE:
            delay = 0
        elif self.config.reconnect_strategy == ReconnectStrategy.FIXED_INTERVAL:
            delay = 5.0
        else:  # EXPONENTIAL_BACKOFF
            delay = min(
                self.config.reconnect_backoff_base ** self._reconnect_attempts,
                self.config.reconnect_max_delay
            )
        
        self.logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            await self.connect(self._user_id, self._auth_token)
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
            
            if self._reconnect_attempts >= self.config.max_reconnect_attempts:
                self.logger.error("Max reconnection attempts reached, giving up")
                self._connection_state = ConnectionState.DISCONNECTED
            else:
                # Schedule next reconnection attempt
                asyncio.create_task(self._attempt_reconnection())
    
    async def _process_queued_messages(self) -> None:
        """Process messages that were queued while disconnected."""
        if not self._message_queue:
            return
        
        self.logger.info(f"Processing {len(self._message_queue)} queued messages")
        
        messages_to_send = self._message_queue.copy()
        self._message_queue.clear()
        
        for message in messages_to_send:
            try:
                await self.send_message(
                    message.message_type,
                    message.data,
                    message.correlation_id
                )
            except Exception as e:
                self.logger.warning(f"Failed to send queued message: {e}")
                # Re-queue the message
                if len(self._message_queue) < self._max_queue_size:
                    self._message_queue.append(message)


# Convenience functions for common WebSocket operations

def create_websocket_client(
    base_url: str = "ws://localhost:8000",
    **config_kwargs
) -> WebSocketClient:
    """
    Create a WebSocket client with configuration.
    
    Args:
        base_url: WebSocket server base URL
        **config_kwargs: Additional configuration options
        
    Returns:
        Configured WebSocket client
    """
    config = WebSocketConfig(base_url=base_url, **config_kwargs)
    return WebSocketClient(config)


async def test_websocket_connection(base_url: str = "ws://localhost:8000") -> bool:
    """
    Test WebSocket connection health.
    
    Args:
        base_url: WebSocket server base URL
        
    Returns:
        True if connection is healthy
    """
    try:
        client = create_websocket_client(base_url)
        await client.connect()
        
        # Send a test ping
        await client.send_message(MessageType.PING, {"test": True})
        
        # Wait a moment for response
        await asyncio.sleep(1)
        
        is_connected = client.is_connected()
        await client.disconnect()
        
        return is_connected
    except Exception:
        return False


# Export public interface
__all__ = [
    "WebSocketClient",
    "WebSocketConfig",
    "WebSocketMessage",
    "ConnectionState",
    "MessageType",
    "ReconnectStrategy",
    "WebSocketError",
    "ConnectionError",
    "AuthenticationError",
    "MessageError",
    "ConnectionMetrics",
    "create_websocket_client",
    "test_websocket_connection"
]