"""
Centralized application configuration.

All tunable parameters are read from environment variables (see /.env.example
at the repo root). Nothing here is hardcoded — this is the single source of
truth the rest of the backend imports from.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is three levels up from this file: backend/app/core/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

# Make the sibling `ml/` package importable from anywhere in the backend
# (e.g. `from ml.preprocessing.landmarks import ...` in websocket/handler.py),
# regardless of the working directory uvicorn/pytest was launched from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Backend ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # --- WebSocket ---
    ws_max_message_size_bytes: int = 2_097_152
    ws_max_fps: int = 15
    ws_heartbeat_interval_sec: int = 30

    # --- Gloss-sequence aggregation (Phase 11: ml/inference/aggregator.py) ---
    ws_gloss_stability_frames: int = 5
    ws_gloss_confidence_threshold: float = 0.5

    # --- Word-level motion gate (ml/inference/motion_gate.py). The word
    # classifier has no trained "no gesture" class -- confirmed live via a
    # real diagnostic that a genuinely still/idle hand gets confidently
    # (~0.9+) misclassified as one of the 7 known words otherwise. Below
    # this mean per-frame hand-landmark motion (near 0 for a held-still
    # hand; a real recorded sign measures >=1.0, see the diagnostic this
    # was derived from), the classifier is simply not run that frame. ---
    ws_min_hand_motion: float = 0.4

    # --- Live-vs-replay diagnostics (temporary, opt-in, off by default --
    # see websocket/handler.py and scripts/diagnose_live_capture.py). Never
    # log/save anything unless explicitly turned on locally: this is for a
    # developer/signer investigating why live webcam recognition differs
    # from a recorded-clip replay test, not a production feature. ---
    # Logs one rate-limited "LIVE WORD DEBUG" block per ~5th frame once the
    # word buffer is full: buffer fill, motion gate PASS/REJECT, whether the
    # LSTM was actually called, its top-3 classes, and the aggregator's
    # streak/confirmation state -- so a live failure can be attributed to a
    # specific stage instead of one generic "not recognized".
    ws_debug_log_word_pipeline: bool = False
    # When set to a directory path, saves each full 32-frame word buffer as
    # a timestamped .npz (a real captured live window, not a recreation) --
    # replay with scripts/diagnose_live_capture.py to run it directly
    # through the loaded checkpoint, bypassing the motion gate and
    # aggregator entirely, to tell apart "the model is wrong on this live
    # input" from "the gate/aggregator threw away a correct prediction".
    ws_debug_save_feature_windows_dir: str = ""
    # Same idea as ws_debug_log_word_pipeline, for the separate single-frame
    # fingerspelling classifier: one rate-limited "[FINGER DEBUG]" block per
    # ~5th frame with exactly one hand visible -- handedness, hand motion,
    # top-3 letter classes/probabilities, the current prediction, debounce
    # streak, and whether it's interim or confirmed.
    ws_debug_log_fingerspelling_pipeline: bool = False
    # Same idea again, for wall-clock timing per pipeline stage (frame
    # decode, MediaPipe hands/pose/face, normalization, feature-vector
    # build, word-LSTM inference, fingerspelling inference, total) -- one
    # rate-limited "[TIMING]" block per ~30th frame. Exists so a real
    # performance investigation always starts from real numbers measured
    # on the machine that's actually slow, not a guess (this project's own
    # sandbox has no camera to reproduce a signer's live lag with) --
    # see PROJECT_STATUS.md's continuous-dictation performance phase.
    ws_debug_log_timing: bool = False

    # --- Utterance boundary detection (continuous-dictation UI: the camera
    # runs continuously, confirmed signs/letters accumulate into a live
    # utterance instead of replacing one another, see websocket/handler.py
    # and frontend/hooks/useUtterance.ts). A sustained hand absence is the
    # boundary signal, reusing the same no_hand_streak counter
    # recognition_mode's own word/letter-mode reset already tracks -- not a
    # second independent timer measuring the same thing. Two thresholds so
    # a short natural pause between signs doesn't end the whole utterance:
    # crossing the token threshold finalizes whatever's currently mid-flight
    # (mainly an in-progress fingerspelled word) into one token; crossing
    # the longer utterance threshold additionally means "run language
    # post-processing now", the same way voice-typing silence detection
    # distinguishes a word pause from an end-of-sentence pause. ---
    ws_token_boundary_no_hand_frames: int = 8
    ws_utterance_boundary_no_hand_frames: int = 30

    # --- Facial grammar markers (Phase 17: ml/features/facial_grammar.py) ---
    ws_facial_calibration_frames: int = 30
    ws_facial_raised_ratio: float = 0.25
    ws_facial_furrowed_ratio: float = 0.25

    # --- MediaPipe (Phase 6-7 CV pipeline) ---
    mediapipe_models_dir: Path = REPO_ROOT / "models" / "mediapipe"

    # --- ML model ---
    model_type: str = "lstm"
    model_checkpoint_path: str = "models/checkpoints/real_v1/latest.pt"
    model_sequence_length: int = 32
    model_device: str = "cpu"

    # --- Fingerspelling classifier (ml/fingerspelling/) -- a separate,
    # single-frame model from the word-level one above, optional: the
    # live per-frame letter path is skipped (not an error) when this
    # checkpoint isn't present, since it's an additive capability. ---
    fingerspelling_checkpoint_path: str = "ml/fingerspelling/checkpoints/latest.pt"
    fingerspelling_device: str = "cpu"
    fingerspelling_stability_frames: int = 8
    fingerspelling_confidence_threshold: float = 0.6
    # Fingerspelling classifies one still handshape per frame with no notion
    # of "is this a held letter or a hand mid-motion" -- confirmed live
    # (streaming real recorded word clips, e.g. data/real/raw/aatxia/
    # WANT_00.mp4, through the real /ws path) that a one-handed DYNAMIC word
    # sign can hold a single letter's prediction stably enough, for long
    # enough, to satisfy fingerspelling_stability_frames purely because the
    # hand shape mid-gesture resembles a known static letter -- while
    # genuinely still in motion (mean hand-landmark motion 0.6-2.9 at the
    # moment of confirmation, measured the same way as ws_min_hand_motion
    # below). A letter is only allowed to CONFIRM (not interim-display, see
    # websocket/handler.py) when the same short window's motion is below
    # this threshold -- i.e. the hand is actually being held still, the way
    # real fingerspelling is performed for a camera.
    fingerspelling_max_hand_motion: float = 0.4

    # --- Feature toggles ---
    features_hands: bool = True
    features_pose: bool = True
    features_face: bool = True

    # --- NLP ---
    nlp_backend: str = "rule_based"
    nlp_hf_model_name: str = ""

    # --- AI sentence-composition fallback (optional, OFF by default) ---
    # Only ever consulted when ml/nlp/gloss_to_text.py's rule-based composer
    # can't match a pattern for the recognized words (composed_text would
    # otherwise be null) -- see app/services/ai_sentence_composer.py's
    # module docstring for why this must never be the primary path. An
    # empty key disables the feature entirely: compose_with_ai() returns
    # None immediately, no network call ever attempted.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- TTS / STT ---
    tts_provider: str = "browser"
    stt_provider: str = "browser"

    # --- Privacy ---
    store_raw_video: bool = False
    send_video_to_third_party: bool = False

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_checkpoint_path_resolved(self) -> Path:
        """model_checkpoint_path is documented (.env.example) as relative to
        the repo root, not the process's cwd -- uvicorn is commonly launched
        from backend/, where a bare relative path would silently resolve to
        the wrong (nonexistent) location. Mirrors mediapipe_models_dir."""
        path = Path(self.model_checkpoint_path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def fingerspelling_checkpoint_path_resolved(self) -> Path:
        """Same repo-root-relative resolution as model_checkpoint_path_resolved."""
        path = Path(self.fingerspelling_checkpoint_path)
        return path if path.is_absolute() else REPO_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Settings are cached; call get_settings() rather than instantiating Settings() directly."""
    return Settings()
