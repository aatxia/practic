"""
ConnectionManager — tracks active WebSocket connections.

Kept intentionally simple in Phase 5 (no broadcast/pub-sub needed yet): each
client gets its own independent inference session. This exists mainly so we
can log connection counts (section 24: log WebSocket connections) and have a
single place to extend from later (e.g. per-connection session state once
the real-time inference pipeline lands in Phase 10).
"""
from __future__ import annotations

import itertools

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)

_id_counter = itertools.count(1)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> int:
        await websocket.accept()
        conn_id = next(_id_counter)
        self._connections[conn_id] = websocket
        logger.info("WebSocket connected id=%s total_active=%s", conn_id, len(self._connections))
        return conn_id

    def disconnect(self, conn_id: int) -> None:
        self._connections.pop(conn_id, None)
        logger.info(
            "WebSocket disconnected id=%s total_active=%s", conn_id, len(self._connections)
        )

    @property
    def active_count(self) -> int:
        return len(self._connections)


# Module-level singleton: one manager shared by every WebSocket route.
connection_manager = ConnectionManager()
