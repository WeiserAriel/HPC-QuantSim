"""
WebSocket connection manager for real-time dashboard updates.
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Set
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasting for the dashboard."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message to websocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any], channel: str = None):
        """Broadcast a message to all connected clients or specific channel subscribers."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected_connections = []
        
        for websocket in self.active_connections:
            try:
                # If channel is specified, only send to subscribers
                if channel:
                    metadata = self.connection_metadata.get(websocket, {})
                    subscriptions = metadata.get("subscriptions", set())
                    if channel not in subscriptions:
                        continue
                
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to broadcast to websocket: {e}")
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)
    
    async def broadcast_metrics_update(self, metrics: Dict[str, Any]):
        """Broadcast real-time metrics update."""
        message = {
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics
        }
        await self.broadcast(message, channel="metrics")
    
    async def broadcast_simulation_progress(self, progress: Dict[str, Any]):
        """Broadcast simulation progress update."""
        message = {
            "type": "simulation_progress",
            "timestamp": datetime.utcnow().isoformat(),
            "data": progress
        }
        await self.broadcast(message, channel="simulation")
    
    async def broadcast_new_result(self, result_data: Dict[str, Any]):
        """Broadcast new simulation result."""
        message = {
            "type": "new_result",
            "timestamp": datetime.utcnow().isoformat(),
            "data": result_data
        }
        await self.broadcast(message, channel="results")
    
    def subscribe_connection(self, websocket: WebSocket, channels: List[str]):
        """Subscribe a connection to specific data channels."""
        if websocket in self.connection_metadata:
            subscriptions = self.connection_metadata[websocket].get("subscriptions", set())
            subscriptions.update(channels)
            self.connection_metadata[websocket]["subscriptions"] = subscriptions
    
    def unsubscribe_connection(self, websocket: WebSocket, channels: List[str]):
        """Unsubscribe a connection from specific data channels."""
        if websocket in self.connection_metadata:
            subscriptions = self.connection_metadata[websocket].get("subscriptions", set())
            subscriptions.difference_update(channels)
            self.connection_metadata[websocket]["subscriptions"] = subscriptions
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_subscription_count(self, channel: str) -> int:
        """Get the number of connections subscribed to a specific channel."""
        count = 0
        for metadata in self.connection_metadata.values():
            subscriptions = metadata.get("subscriptions", set())
            if channel in subscriptions:
                count += 1
        return count

