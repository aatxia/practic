"""Shared WebSocket test helpers."""


def receive_skip_boundary(ws):
    """Reads the next message, transparently discarding any interleaved
    'boundary' messages first (websocket/protocol.py) -- these fire from a
    sustained no-hand streak (ws_token_boundary_no_hand_frames /
    ws_utterance_boundary_no_hand_frames, app/core/config.py) and are
    irrelevant noise to a test using an always-no-hand stub extractor that
    isn't specifically exercising continuous-dictation boundary events."""
    message = ws.receive_json()
    while message.get("type") == "boundary":
        message = ws.receive_json()
    return message
