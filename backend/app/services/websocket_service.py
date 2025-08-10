"""
WebSocket Service - Real-time updates for test execution and monitoring.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()
    session_id: Optional[str] = None


class ConnectionManager:
    """WebSocket connection manager for real-time updates."""
    
    def __init__(self):
        # Active connections organized by session
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global connections (for system-wide updates)
        self.global_connections: Set[WebSocket] = set()
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
        
        if session_id:
            # Session-specific connection
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
            logger.info(f"New WebSocket connection for session {session_id}")
        else:
            # Global connection
            self.global_connections.add(websocket)
            logger.info("New global WebSocket connection")
        
        # Send welcome message
        await self._send_to_websocket(websocket, {
            "type": "connection_established",
            "data": {
                "session_id": session_id,
                "connected_at": datetime.utcnow().isoformat(),
                "message": "Connected to Breakpoint LLM Testing Platform"
            }
        })
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        metadata = self.connection_metadata.get(websocket, {})
        session_id = metadata.get("session_id")
        
        if session_id and session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected from session {session_id}")
        
        self.global_connections.discard(websocket)
        self.connection_metadata.pop(websocket, None)
        logger.info("WebSocket disconnected")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to all connections in a specific session."""
        if session_id not in self.active_connections:
            logger.warning(f"No active connections for session {session_id}")
            return
        
        disconnected = set()
        for websocket in self.active_connections[session_id]:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_global(self, message: Dict[str, Any]):
        """Broadcast message to all global connections."""
        disconnected = set()
        for websocket in self.global_connections.copy():
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections for a specific user."""
        user_websockets = [
            ws for ws, metadata in self.connection_metadata.items()
            if metadata.get("user_id") == user_id
        ]
        
        disconnected = set()
        for websocket in user_websockets:
            try:
                await self._send_to_websocket(websocket, message)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def _send_to_websocket(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific WebSocket."""
        formatted_message = {
            "type": message.get("type", "update"),
            "data": message.get("data", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_text(json.dumps(formatted_message))
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get the number of active connections."""
        if session_id:
            return len(self.active_connections.get(session_id, set()))
        return len(self.global_connections) + sum(len(conns) for conns in self.active_connections.values())
    
    def get_session_connections(self) -> Dict[str, int]:
        """Get connection counts by session."""
        return {session_id: len(connections) for session_id, connections in self.active_connections.items()}


class WebSocketService:
    """Service for managing WebSocket communications and real-time updates."""
    
    def __init__(self):
        self.manager = ConnectionManager()
        self._ping_task = None
        self._start_ping_task()
    
    def _start_ping_task(self):
        """Start background task for ping/pong to keep connections alive."""
        if self._ping_task is None:
            self._ping_task = asyncio.create_task(self._ping_connections())
    
    async def _ping_connections(self):
        """Send periodic ping to maintain connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await self.manager.broadcast_global({
                    "type": "ping",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                })
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    async def connect_session(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """Connect to a specific test session."""
        await self.manager.connect(websocket, session_id, user_id)
    
    async def connect_global(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Connect for global updates."""
        await self.manager.connect(websocket, None, user_id)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket."""
        self.manager.disconnect(websocket)
    
    # Test execution updates
    async def notify_test_started(self, session_id: str, test_data: Dict[str, Any]):
        """Notify that a test has started."""
        await self.manager.send_to_session(session_id, {
            "type": "test_started",
            "data": {
                "test_id": test_data.get("test_id"),
                "test_type": test_data.get("test_type"),
                "started_at": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_test_completed(self, session_id: str, result_data: Dict[str, Any]):
        """Notify that a test has completed."""
        await self.manager.send_to_session(session_id, {
            "type": "test_completed",
            "data": {
                "test_id": result_data.get("test_id"),
                "passed": result_data.get("passed"),
                "execution_time": result_data.get("execution_time"),
                "tokens_used": result_data.get("tokens_used"),
                "vulnerability_detected": result_data.get("vulnerability_detected"),
                "completed_at": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_test_error(self, session_id: str, error_data: Dict[str, Any]):
        """Notify about a test error."""
        await self.manager.send_to_session(session_id, {
            "type": "test_error",
            "data": {
                "test_id": error_data.get("test_id"),
                "error_message": error_data.get("error_message"),
                "error_type": error_data.get("error_type"),
                "occurred_at": datetime.utcnow().isoformat()
            }
        })
    
    # Session updates
    async def notify_session_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Notify about session progress."""
        await self.manager.send_to_session(session_id, {
            "type": "session_progress",
            "data": {
                "session_id": session_id,
                "completed_tests": progress_data.get("completed_tests", 0),
                "total_tests": progress_data.get("total_tests", 0),
                "passed_tests": progress_data.get("passed_tests", 0),
                "failed_tests": progress_data.get("failed_tests", 0),
                "progress_percentage": progress_data.get("progress_percentage", 0),
                "estimated_completion": progress_data.get("estimated_completion"),
                "updated_at": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_session_completed(self, session_id: str, summary_data: Dict[str, Any]):
        """Notify that a session has completed."""
        await self.manager.send_to_session(session_id, {
            "type": "session_completed",
            "data": {
                "session_id": session_id,
                "total_tests": summary_data.get("total_tests"),
                "passed_tests": summary_data.get("passed_tests"),
                "failed_tests": summary_data.get("failed_tests"),
                "duration": summary_data.get("duration"),
                "pass_rate": summary_data.get("pass_rate"),
                "completed_at": datetime.utcnow().isoformat()
            }
        })
    
    # Failure analysis updates
    async def notify_vulnerability_detected(self, session_id: str, vulnerability_data: Dict[str, Any]):
        """Notify about a detected vulnerability."""
        await self.manager.send_to_session(session_id, {
            "type": "vulnerability_detected",
            "data": {
                "test_id": vulnerability_data.get("test_id"),
                "vulnerability_type": vulnerability_data.get("vulnerability_type"),
                "severity": vulnerability_data.get("severity"),
                "confidence": vulnerability_data.get("confidence"),
                "description": vulnerability_data.get("description"),
                "detected_at": datetime.utcnow().isoformat()
            }
        })
    
    # System updates
    async def notify_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status updates."""
        await self.manager.broadcast_global({
            "type": "system_status",
            "data": {
                "status": status_data.get("status"),
                "active_sessions": status_data.get("active_sessions", 0),
                "total_connections": self.manager.get_connection_count(),
                "system_load": status_data.get("system_load"),
                "updated_at": datetime.utcnow().isoformat()
            }
        })
    
    async def notify_maintenance_mode(self, maintenance_data: Dict[str, Any]):
        """Notify about maintenance mode."""
        await self.manager.broadcast_global({
            "type": "maintenance_mode",
            "data": {
                "enabled": maintenance_data.get("enabled"),
                "message": maintenance_data.get("message"),
                "estimated_duration": maintenance_data.get("estimated_duration"),
                "scheduled_at": maintenance_data.get("scheduled_at")
            }
        })
    
    # Performance metrics
    async def notify_performance_metrics(self, session_id: str, metrics_data: Dict[str, Any]):
        """Send performance metrics updates."""
        await self.manager.send_to_session(session_id, {
            "type": "performance_metrics",
            "data": {
                "session_id": session_id,
                "avg_response_time": metrics_data.get("avg_response_time"),
                "requests_per_minute": metrics_data.get("requests_per_minute"),
                "error_rate": metrics_data.get("error_rate"),
                "token_usage_rate": metrics_data.get("token_usage_rate"),
                "updated_at": datetime.utcnow().isoformat()
            }
        })
    
    # Utility methods
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": self.manager.get_connection_count(),
            "global_connections": len(self.manager.global_connections),
            "session_connections": self.manager.get_session_connections(),
            "active_sessions": len(self.manager.active_connections)
        }
    
    async def send_custom_message(self, session_id: str, message_type: str, data: Dict[str, Any]):
        """Send a custom message to a session."""
        await self.manager.send_to_session(session_id, {
            "type": message_type,
            "data": data
        })


# Global WebSocket service instance
websocket_service = WebSocketService()
