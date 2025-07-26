"""
WebSocket API Routes for LegalAI Document Processing System

This module provides WebSocket endpoints for real-time communication in the Patexia 
Legal AI Chatbot. It enables live progress tracking, instant notifications, system 
monitoring, and multi-user collaboration features.

Key Features:
- Real-time document processing progress updates
- Live search operation status notifications
- System resource monitoring and alerts
- Configuration change notifications
- Multi-user collaboration and case updates
- Connection authentication and management
- Message queuing and delivery guarantees
- Connection health monitoring and heartbeat

WebSocket Operations:
- Connect: Establish WebSocket connection with authentication
- Subscribe: Subscribe to specific event channels
- Broadcast: Send messages to user groups or all users
- Progress: Real-time progress tracking for long operations
- Notifications: System alerts and user notifications
- Monitoring: Administrative system monitoring data

Real-Time Features:
- Document Processing: Live updates during PDF extraction, chunking, embedding
- Search Operations: Real-time search progress and result streaming
- Case Management: Live case updates and collaboration
- System Health: Resource monitoring and performance alerts
- Configuration: Hot-reload configuration change notifications
- Error Handling: Real-time error reporting and recovery

Connection Management:
- User-based connection pooling
- Connection state management and lifecycle
- Message routing and targeting
- Queue management for offline users
- Performance monitoring and analytics
- Graceful connection handling and reconnection

Architecture Integration:
- Uses WebSocketManager for connection pooling and management
- Integrates with document processors for progress updates
- Connects to system monitoring for resource alerts
- Supports admin panel real-time updates
- Implements secure authentication and authorization
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    Query,
    Header,
    status
)
from fastapi.responses import JSONResponse

from config.settings import get_settings
from ...core.websocket_manager import (
    WebSocketManager,
    ConnectionState,
    MessageType,
    MessagePriority,
    get_websocket_manager
)
from ...core.resource_monitor import ResourceMonitor, get_resource_monitor
from ...services.case_service import CaseService
from ...services.document_service import DocumentService
from ...services.search_service import SearchService
from ...utils.logging import (
    get_logger,
    performance_context,
    websocket_logger,
    set_correlation_id
)
from ...core.exceptions import (
    ConnectionError,
    ValidationError,
    ErrorCode
)
from ..deps import get_case_service, get_document_service, get_search_service


logger = get_logger(__name__)
router = APIRouter()


# WebSocket Connection Endpoints

@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    case_id: Optional[str] = Query(None, description="Case ID for context"),
    client_type: str = Query("web", description="Client type (web, mobile, admin)"),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Main WebSocket connection endpoint for real-time communication.
    
    Supports:
    - Document processing progress updates
    - Search operation notifications
    - Case collaboration updates
    - System monitoring alerts
    - Configuration change notifications
    """
    connection_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    # Extract client information
    client_ip = websocket.client.host if websocket.client else "unknown"
    user_agent = websocket.headers.get("user-agent", "unknown")
    
    logger.info(
        "WebSocket connection attempt",
        connection_id=connection_id,
        user_id=user_id,
        case_id=case_id,
        client_type=client_type,
        client_ip=client_ip,
        correlation_id=correlation_id
    )
    
    try:
        # Accept WebSocket connection
        await websocket.accept()
        
        # Register connection with manager
        await websocket_manager.connect(
            websocket=websocket,
            connection_id=connection_id,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Send connection confirmation
        await _send_connection_message(
            websocket,
            connection_id,
            "connection_established",
            {
                "connection_id": connection_id,
                "server_time": datetime.now(timezone.utc).isoformat(),
                "features": [
                    "document_progress",
                    "search_notifications", 
                    "case_updates",
                    "system_monitoring"
                ]
            }
        )
        
        # Authenticate if user_id provided
        if user_id:
            success = await websocket_manager.authenticate_connection(connection_id, user_id)
            if success:
                await _send_connection_message(
                    websocket,
                    connection_id,
                    "authentication_success",
                    {
                        "user_id": user_id,
                        "case_id": case_id,
                        "authenticated_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Auto-subscribe to user-specific events
                await _auto_subscribe_user_events(
                    websocket_manager,
                    connection_id,
                    user_id,
                    case_id
                )
            else:
                await _send_connection_message(
                    websocket,
                    connection_id,
                    "authentication_failed",
                    {"error": "Authentication failed or connection limit exceeded"}
                )
        
        # Start message handling loop
        await _handle_websocket_messages(
            websocket,
            websocket_manager,
            connection_id,
            user_id,
            case_id
        )
        
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            connection_id=connection_id,
            user_id=user_id,
            correlation_id=correlation_id
        )
    except Exception as exc:
        logger.error(
            "WebSocket connection error",
            connection_id=connection_id,
            user_id=user_id,
            error=str(exc),
            correlation_id=correlation_id
        )
    finally:
        # Clean up connection
        await websocket_manager.disconnect(connection_id)
        
        websocket_logger.connection_closed(
            connection_id=connection_id,
            reason="Normal closure"
        )


@router.websocket("/admin")
async def admin_websocket_endpoint(
    websocket: WebSocket,
    admin_token: Optional[str] = Query(None, description="Admin authentication token"),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    resource_monitor: ResourceMonitor = Depends(get_resource_monitor)
):
    """
    Administrative WebSocket endpoint for system monitoring and management.
    
    Provides:
    - Real-time system metrics and performance data
    - Configuration change notifications
    - User connection monitoring
    - Service health status updates
    - Administrative alerts and warnings
    """
    connection_id = f"admin_{uuid.uuid4()}"
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    client_ip = websocket.client.host if websocket.client else "unknown"
    user_agent = websocket.headers.get("user-agent", "unknown")
    
    logger.info(
        "Admin WebSocket connection attempt",
        connection_id=connection_id,
        client_ip=client_ip,
        correlation_id=correlation_id
    )
    
    try:
        # Accept connection
        await websocket.accept()
        
        # Validate admin access
        if not _validate_admin_token(admin_token):
            await websocket.send_json({
                "type": "error",
                "data": {
                    "error": "Invalid admin token",
                    "code": "ADMIN_AUTH_FAILED"
                }
            })
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Register admin connection
        await websocket_manager.connect(
            websocket=websocket,
            connection_id=connection_id,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Authenticate as admin
        await websocket_manager.authenticate_connection(connection_id, "admin")
        
        # Send admin connection confirmation
        await _send_connection_message(
            websocket,
            connection_id,
            "admin_connection_established",
            {
                "connection_id": connection_id,
                "admin_features": [
                    "system_metrics",
                    "connection_monitoring",
                    "configuration_management",
                    "service_health",
                    "user_activity"
                ],
                "server_time": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Start admin monitoring loop
        await _handle_admin_websocket(
            websocket,
            websocket_manager,
            resource_monitor,
            connection_id
        )
        
    except WebSocketDisconnect:
        logger.info(
            "Admin WebSocket disconnected",
            connection_id=connection_id,
            correlation_id=correlation_id
        )
    except Exception as exc:
        logger.error(
            "Admin WebSocket connection error",
            connection_id=connection_id,
            error=str(exc),
            correlation_id=correlation_id
        )
    finally:
        await websocket_manager.disconnect(connection_id)


# WebSocket Management API Endpoints

@router.get("/connections/status")
async def get_connection_status(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> JSONResponse:
    """Get current WebSocket connection status and metrics."""
    try:
        metrics = websocket_manager.get_metrics()
        
        return JSONResponse({
            "success": True,
            "data": {
                "total_connections": metrics.total_connections,
                "authenticated_connections": metrics.authenticated_connections,
                "active_users": metrics.active_users,
                "messages_sent_total": metrics.messages_sent_total,
                "messages_received_total": metrics.messages_received_total,
                "bytes_sent_total": metrics.bytes_sent_total,
                "bytes_received_total": metrics.bytes_received_total,
                "average_connection_duration_seconds": metrics.average_connection_duration_seconds,
                "peak_concurrent_connections": metrics.peak_concurrent_connections,
                "connection_errors_total": metrics.connection_errors_total,
                "message_queue_size": metrics.message_queue_size,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as exc:
        logger.error(f"Failed to get connection status: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to get connection status"}
        )


@router.post("/connections/{connection_id}/disconnect")
async def disconnect_connection(
    connection_id: str,
    reason: Optional[str] = Query(None, description="Reason for disconnection"),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> JSONResponse:
    """Administratively disconnect a specific WebSocket connection."""
    try:
        success = await websocket_manager.disconnect(connection_id, reason)
        
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Connection {connection_id} disconnected",
                "data": {
                    "connection_id": connection_id,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "Connection not found"}
            )
            
    except Exception as exc:
        logger.error(f"Failed to disconnect connection {connection_id}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to disconnect connection"}
        )


@router.post("/broadcast")
async def broadcast_message(
    message_type: str,
    data: Dict[str, Any],
    target_users: Optional[List[str]] = None,
    priority: MessagePriority = MessagePriority.NORMAL,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
) -> JSONResponse:
    """Broadcast a message to WebSocket connections."""
    try:
        if target_users:
            # Send to specific users
            results = []
            for user_id in target_users:
                success = await websocket_manager.broadcast_to_user(
                    user_id,
                    {
                        "type": message_type,
                        "data": data,
                        "priority": priority.value,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                results.append({"user_id": user_id, "success": success})
            
            return JSONResponse({
                "success": True,
                "message": f"Message broadcast to {len(target_users)} users",
                "data": {
                    "message_type": message_type,
                    "target_users": len(target_users),
                    "results": results,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            # Broadcast to all users
            await websocket_manager.broadcast_to_all({
                "type": message_type,
                "data": data,
                "priority": priority.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return JSONResponse({
                "success": True,
                "message": "Message broadcast to all users",
                "data": {
                    "message_type": message_type,
                    "broadcast_type": "all_users",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
            
    except Exception as exc:
        logger.error(f"Failed to broadcast message: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to broadcast message"}
        )


# Helper Functions

async def _handle_websocket_messages(
    websocket: WebSocket,
    websocket_manager: WebSocketManager,
    connection_id: str,
    user_id: Optional[str],
    case_id: Optional[str]
) -> None:
    """Handle incoming WebSocket messages."""
    try:
        async for data in websocket.iter_text():
            try:
                # Parse message
                message = json.loads(data)
                message_type = message.get("type")
                message_data = message.get("data", {})
                
                websocket_logger.message_received(
                    connection_id=connection_id,
                    message_type=message_type,
                    message_size=len(data)
                )
                
                # Handle different message types
                if message_type == "ping":
                    await _handle_ping(websocket, connection_id)
                elif message_type == "subscribe":
                    await _handle_subscription(
                        websocket_manager,
                        connection_id,
                        message_data.get("channels", [])
                    )
                elif message_type == "unsubscribe":
                    await _handle_unsubscription(
                        websocket_manager,
                        connection_id,
                        message_data.get("channels", [])
                    )
                elif message_type == "heartbeat":
                    await _handle_heartbeat(websocket, connection_id)
                else:
                    # Handle via WebSocket manager
                    await websocket_manager.handle_message(connection_id, data)
                    
            except json.JSONDecodeError:
                await _send_error_message(
                    websocket,
                    connection_id,
                    "invalid_json",
                    "Invalid JSON message format"
                )
            except Exception as exc:
                logger.error(
                    "Error processing WebSocket message",
                    connection_id=connection_id,
                    error=str(exc)
                )
                await _send_error_message(
                    websocket,
                    connection_id,
                    "message_processing_error",
                    str(exc)
                )
                
    except WebSocketDisconnect:
        # Normal disconnection
        pass
    except Exception as exc:
        logger.error(
            "WebSocket message handling error",
            connection_id=connection_id,
            error=str(exc)
        )


async def _handle_admin_websocket(
    websocket: WebSocket,
    websocket_manager: WebSocketManager,
    resource_monitor: ResourceMonitor,
    connection_id: str
) -> None:
    """Handle admin WebSocket with system monitoring."""
    try:
        # Start periodic system metrics updates
        metrics_task = asyncio.create_task(
            _send_periodic_metrics(websocket, resource_monitor, connection_id)
        )
        
        # Handle admin messages
        message_task = asyncio.create_task(
            _handle_admin_messages(websocket, websocket_manager, connection_id)
        )
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [metrics_task, message_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            
    except Exception as exc:
        logger.error(
            "Admin WebSocket handling error",
            connection_id=connection_id,
            error=str(exc)
        )


async def _send_periodic_metrics(
    websocket: WebSocket,
    resource_monitor: ResourceMonitor,
    connection_id: str,
    interval: int = 5
) -> None:
    """Send periodic system metrics to admin connection."""
    while True:
        try:
            # Get current metrics
            metrics = await resource_monitor.get_current_metrics()
            
            # Send metrics update
            await websocket.send_json({
                "type": "system_metrics",
                "data": {
                    "metrics": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
            
            await asyncio.sleep(interval)
            
        except WebSocketDisconnect:
            break
        except Exception as exc:
            logger.error(
                "Error sending periodic metrics",
                connection_id=connection_id,
                error=str(exc)
            )
            await asyncio.sleep(interval)


async def _handle_admin_messages(
    websocket: WebSocket,
    websocket_manager: WebSocketManager,
    connection_id: str
) -> None:
    """Handle incoming admin messages."""
    async for data in websocket.iter_text():
        try:
            message = json.loads(data)
            message_type = message.get("type")
            message_data = message.get("data", {})
            
            if message_type == "get_connections":
                # Get connection status
                metrics = websocket_manager.get_metrics()
                await websocket.send_json({
                    "type": "connection_status",
                    "data": metrics.dict()
                })
            elif message_type == "disconnect_user":
                # Disconnect specific user
                user_id = message_data.get("user_id")
                if user_id:
                    await websocket_manager.disconnect_user(user_id)
            elif message_type == "broadcast_admin":
                # Admin broadcast
                await websocket_manager.broadcast_to_all({
                    "type": "admin_message",
                    "data": message_data
                })
                
        except json.JSONDecodeError:
            await _send_error_message(
                websocket,
                connection_id,
                "invalid_json",
                "Invalid JSON message format"
            )


async def _auto_subscribe_user_events(
    websocket_manager: WebSocketManager,
    connection_id: str,
    user_id: str,
    case_id: Optional[str]
) -> None:
    """Auto-subscribe to relevant event channels for authenticated users."""
    channels = [
        f"user:{user_id}",
        "system:alerts",
        "system:maintenance"
    ]
    
    if case_id:
        channels.extend([
            f"case:{case_id}",
            f"case:{case_id}:documents",
            f"case:{case_id}:search"
        ])
    
    for channel in channels:
        await websocket_manager.subscribe_to_channel(connection_id, channel)


async def _handle_ping(websocket: WebSocket, connection_id: str) -> None:
    """Handle ping message with pong response."""
    await websocket.send_json({
        "type": "pong",
        "data": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connection_id": connection_id
        }
    })


async def _handle_heartbeat(websocket: WebSocket, connection_id: str) -> None:
    """Handle heartbeat message."""
    await websocket.send_json({
        "type": "heartbeat_ack",
        "data": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connection_id": connection_id
        }
    })


async def _handle_subscription(
    websocket_manager: WebSocketManager,
    connection_id: str,
    channels: List[str]
) -> None:
    """Handle channel subscription requests."""
    for channel in channels:
        await websocket_manager.subscribe_to_channel(connection_id, channel)


async def _handle_unsubscription(
    websocket_manager: WebSocketManager,
    connection_id: str,
    channels: List[str]
) -> None:
    """Handle channel unsubscription requests."""
    for channel in channels:
        await websocket_manager.unsubscribe_from_channel(connection_id, channel)


async def _send_connection_message(
    websocket: WebSocket,
    connection_id: str,
    message_type: str,
    data: Dict[str, Any]
) -> None:
    """Send a connection-related message."""
    try:
        await websocket.send_json({
            "type": message_type,
            "data": data,
            "connection_id": connection_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        websocket_logger.message_sent(
            connection_id=connection_id,
            message_type=message_type
        )
        
    except Exception as exc:
        logger.error(
            "Failed to send connection message",
            connection_id=connection_id,
            message_type=message_type,
            error=str(exc)
        )


async def _send_error_message(
    websocket: WebSocket,
    connection_id: str,
    error_code: str,
    error_message: str
) -> None:
    """Send an error message to WebSocket client."""
    try:
        await websocket.send_json({
            "type": "error",
            "data": {
                "error_code": error_code,
                "error_message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "connection_id": connection_id
        })
        
        websocket_logger.error_occurred(
            connection_id=connection_id,
            error=error_message,
            error_code=error_code
        )
        
    except Exception as exc:
        logger.error(
            "Failed to send error message",
            connection_id=connection_id,
            error_code=error_code,
            error=str(exc)
        )


def _validate_admin_token(token: Optional[str]) -> bool:
    """Validate admin authentication token."""
    # Basic token validation - in production this would check against
    # a secure token store or authentication service
    settings = get_settings()
    admin_token = getattr(settings, 'admin_token', None)
    
    if not admin_token or not token:
        return False
    
    return token == admin_token


# Export router for main application
__all__ = ["router"]