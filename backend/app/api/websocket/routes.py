from __future__ import annotations

from fastapi import APIRouter, WebSocket

from websocket.handler import websocket_endpoint

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def ws_route(websocket: WebSocket) -> None:
    await websocket_endpoint(websocket)
