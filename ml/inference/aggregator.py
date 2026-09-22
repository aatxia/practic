"""
aggregator — turns the raw per-frame prediction stream from a sliding-window
recognizer (Phase 10: one prediction per incoming frame, same held sign
repeated many times) into a stable gloss sequence.

This is NOT true sign-boundary linguistic segmentation (movement/hold-phase
detection, coarticulation handling, etc.) -- that's a substantially harder
problem this project doesn't attempt yet. It's a deterministic debounce
heuristic common in real-time gesture recognition: a gloss must be predicted
`stability_frames` times in a row (above `confidence_threshold`) before it's
"confirmed" and appended to the sequence, and it won't be re-confirmed while
the signer keeps holding the same sign. Honestly named and documented as a
heuristic, not dressed up as real segmentation.
"""
from __future__ import annotations


class GlossSequenceAggregator:
    def __init__(self, stability_frames: int = 5, confidence_threshold: float = 0.5) -> None:
        if stability_frames < 1:
            raise ValueError(f"stability_frames must be >= 1, got {stability_frames}")
        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.sequence: list[str] = []
        self._current_gloss: str | None = None
        self._stable_count = 0
        self._last_confirmed_gloss: str | None = None

    def update(self, gloss: str, confidence: float) -> bool:
        """Feed one raw prediction. Returns True iff this call just confirmed
        a NEW gloss (appended to `self.sequence`) -- the caller should treat
        that frame's prediction as final rather than interim."""
        if confidence < self.confidence_threshold:
            # Too uncertain to count toward stability -- neither confirms
            # nor silently keeps a stale streak alive.
            self._current_gloss = None
            self._stable_count = 0
            return False

        if gloss == self._current_gloss:
            self._stable_count += 1
        else:
            self._current_gloss = gloss
            self._stable_count = 1

        if self._stable_count >= self.stability_frames and gloss != self._last_confirmed_gloss:
            self._last_confirmed_gloss = gloss
            self.sequence.append(gloss)
            return True
        return False

    def reset(self) -> None:
        self.sequence = []
        self._current_gloss = None
        self._stable_count = 0
        self._last_confirmed_gloss = None
