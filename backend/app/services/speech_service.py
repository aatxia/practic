"""
Speech services: TextToSpeech (Phase 20) and SpeechRecognizer / STT (Phase 13).

For MVP, both directions default to the BROWSER's native Web APIs
(SpeechSynthesis for TTS, SpeechRecognition/getUserMedia+STT for STT),
configured via TTS_PROVIDER / STT_PROVIDER in .env. In that mode the
backend does not need to do audio processing at all -- the frontend
calls the browser API directly and only sends/receives text over the
existing WebSocket/API. These backend interfaces exist so a server-side
provider (e.g. a hosted TTS/STT engine) can be swapped in later without
changing the frontend contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class NotConfiguredError(RuntimeError):
    pass


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return synthesized audio bytes for `text`. Implemented in Phase 20
        only if/when a server-side TTS provider (not the browser API) is
        configured via TTS_PROVIDER."""
        raise NotImplementedError


class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """Return Ukrainian transcript for `audio`. Implemented in Phase 13
        only if/when a server-side STT provider (not the browser API) is
        configured via STT_PROVIDER."""
        raise NotImplementedError


class BrowserDelegatedTTS(TextToSpeech):
    """Used when TTS_PROVIDER=browser: synthesis happens client-side, so the
    backend never needs to produce audio. Calling this is a configuration
    error, not a missing feature."""

    def synthesize(self, text: str) -> bytes:
        raise NotConfiguredError(
            "TTS_PROVIDER=browser: audio synthesis happens in the browser "
            "(window.speechSynthesis), not on the backend."
        )


class BrowserDelegatedSTT(SpeechRecognizer):
    def transcribe(self, audio: bytes) -> str:
        raise NotConfiguredError(
            "STT_PROVIDER=browser: transcription happens in the browser, "
            "not on the backend."
        )
