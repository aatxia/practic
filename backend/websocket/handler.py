"""
WebSocket handler — the per-connection message loop.

Security (section 36):
- every message's type is validated against the protocol schema;
- payload size is capped at WS_MAX_MESSAGE_SIZE_BYTES;
- frame rate is capped at WS_MAX_FPS (excess frames are rejected with a
  clear error, not silently dropped -- easier to debug from the frontend);
- the connection is closed cleanly on protocol violations that indicate a
  misbehaving/hostile client (oversized payloads).

CV pipeline (Phase 6-7): each valid frame is decoded and run through the
real MediaPipe landmark extractor + normalization + feature vector builder.

Inference (Phase 9-10): feature vectors are buffered into a sliding window
(per connection); once the window is full, it's run through a trained
LSTM checkpoint if one exists. If no checkpoint has been trained yet, the
response stays an honest error reporting which modalities were actually
detected -- never a fabricated sign prediction (section 40: "NO FAKE AI").

Gloss aggregation (Phase 11): raw per-frame predictions are debounced by
GlossSequenceAggregator (ml/inference/aggregator.py) into a stable gloss
sequence -- most frames are still interim ("prediction", is_final=False);
a "final_prediction" (is_final=True) fires only when the same gloss has
been predicted `WS_GLOSS_STABILITY_FRAMES` times in a row above
`WS_GLOSS_CONFIDENCE_THRESHOLD`.

Facial grammar (Phase 17): every frame with a detected face is fed to a
per-connection BaselineCalibrator (ml/features/facial_grammar.py) --
independent of whether sign inference has a trained checkpoint, since it
serves a separate purpose. Once calibrated, its eyebrow-position marker
rides along on every PredictionMessage ("facial_grammar"), and flips a
just-confirmed gloss's composed text to a question ("?" instead of ".").
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.fingerspelling_provider import get_fingerspelling_service
from app.services.inference_provider import get_inference_service
from app.services.inference_service import MLNotReadyError
from app.services.translation_service import RuleBasedTranslationService
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ml.features.facial_grammar import BaselineCalibrator, FacialGrammarMarker
from ml.features.feature_vector import (
    FeatureConfig,
    build_feature_vector,
    feature_vector_size,
)
from ml.features.hands import HAND_FEATURE_SIZE
from ml.fingerspelling.dataset import hand_to_feature_vector
from ml.inference.aggregator import GlossSequenceAggregator
from ml.inference.motion_gate import mean_hand_motion
from ml.preprocessing.landmarks import (
    FeatureToggles,
    LandmarkExtractor,
    ModelNotFoundError,
)
from ml.preprocessing.normalization import normalize_frame
from ml.preprocessing.video_reader import FrameDecodeError, decode_base64_frame
from websocket.manager import connection_manager
from websocket.protocol import (
    BoundaryMessage,
    ConnectionMessage,
    ErrorMessage,
    LandmarksStatusMessage,
    LetterCandidate,
    LetterPredictionMessage,
    PredictionMessage,
    ProtocolError,
    WordCandidate,
    parse_client_message,
)

logger = get_logger(__name__)

# Window size for the fingerspelling motion gate (see letter_motion_buffer
# below) -- fixed, not tied to fingerspelling_stability_frames, since the
# ws_min_hand_motion/fingerspelling_max_hand_motion thresholds were derived
# from real recorded clips measured over exactly this many frames.
_LETTER_MOTION_WINDOW = 8

# Kept as the recognition_mode reset's own reasoning reference (see
# settings.ws_token_boundary_no_hand_frames, app/core/config.py, which is
# the actual configurable threshold both recognition_mode's reset AND the
# "token" boundary event below fire on -- one sustained-pause signal, not
# two independently tuned ones). Deliberately longer than a single-frame
# flicker: fingerspelling already relies on a brief hand release between
# two letters of the same word (see letter_aggregator's reset above), and
# that must not itself kick the signer back to word-priority mid-word, or
# finalize an in-progress fingerspelled word early.

# Stateless (a small hand-authored lexicon, see ml/nlp/gloss_to_text.py) --
# unlike the landmark extractor / inference service, there's nothing here
# worth lazily loading or caching failure state for.
translation_service = RuleBasedTranslationService()

def _xy_points(landmarks: object) -> list[tuple[float, float]] | None:
    """Raw (x, y) pairs for one modality's landmarks (an (N, 3) array), or
    None if that modality wasn't detected this frame -- see
    LandmarksStatusMessage's docstring for why this reads the pre-
    normalization coordinates, not ml/preprocessing/normalization.py's."""
    if landmarks is None:
        return None
    return [(float(x), float(y)) for x, y, _z in landmarks]  # type: ignore[misc]


def _save_debug_feature_window(
    out_dir: str, conn_id: object, frame_index: int, window: list[list[float]]
) -> None:
    """Temporary diagnostic only (ws_debug_save_feature_windows_dir) -- a
    REAL captured 32-frame word buffer, saved unmodified as-is, so it can be
    replayed offline through the exact loaded checkpoint later (see
    scripts/diagnose_live_capture.py), bypassing the motion gate and
    aggregator entirely. Never called unless explicitly opted into."""
    import numpy as np

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"window_conn{conn_id}_frame{frame_index:06d}_{time.time_ns()}.npz"
    np.savez_compressed(path, features=np.asarray(window, dtype=np.float32))


def _log_word_debug(
    conn_id: object,
    frame_index: int,
    normalized: object,
    feature_buffer: object,
    sequence_length: int,
    hand_motion: float | None,
    motion_threshold: float,
    motion_gate_passed: bool,
    lstm_called: bool,
    topk: list | None,
    gloss_aggregator: GlossSequenceAggregator,
    confirmed: bool,
    recognition_mode: str,
) -> None:
    """Temporary diagnostic only (ws_debug_log_word_pipeline) -- one
    rate-limited block per frame naming exactly which stage the word
    pipeline is in, so a live failure can be attributed to a specific state
    (motion gate reject / low confidence / wrong class / aggregator reject)
    instead of one generic "not recognized"."""
    top3_str = (
        "[" + ", ".join(f"{r.gloss}:{r.confidence:.3f}" for r in topk) + "]" if topk else "n/a"
    )
    logger.info(
        "LIVE WORD DEBUG\n"
        "  conn=%s frame=%s mode=%s\n"
        "  hands left/right=%s/%s\n"
        "  buffer=%s/%s\n"
        "  hand_motion=%s motion_threshold=%s motion_gate=%s\n"
        "  lstm_called=%s\n"
        "  top3=%s\n"
        "  aggregator_class=%s aggregator_streak=%s\n"
        "  confirmed=%s",
        conn_id,
        frame_index,
        recognition_mode,
        normalized.present["left_hand"],  # type: ignore[attr-defined]
        normalized.present["right_hand"],  # type: ignore[attr-defined]
        len(feature_buffer),  # type: ignore[arg-type]
        sequence_length,
        f"{hand_motion:.4f}" if hand_motion is not None else "n/a",
        motion_threshold,
        "PASS" if motion_gate_passed else "REJECT",
        "yes" if lstm_called else "no",
        top3_str,
        gloss_aggregator._current_gloss,  # noqa: SLF001 -- diagnostic-only read of internal debounce state
        gloss_aggregator._stable_count,  # noqa: SLF001
        "yes" if confirmed else "no",
    )


def _log_timing_debug(conn_id: object, frame_index: int, stage: str, elapsed_ms: float) -> None:
    """Temporary diagnostic only (ws_debug_log_timing) -- real wall-clock
    time for one pipeline stage, on whichever machine is actually running
    the backend (a performance investigation has to start from real
    numbers measured on the slow machine, not a guess elsewhere). One
    rate-limited line per stage per ~30th frame, not every frame -- the
    same rate-limiting convention as the other opt-in debug logs above."""
    logger.info("[TIMING] conn=%s frame=%s stage=%s ms=%.2f", conn_id, frame_index, stage, elapsed_ms)


def _log_fingerspelling_debug(
    handedness: str,
    hand_motion: float,
    topk: list | None,
    prediction_letter: str,
    letter_aggregator: GlossSequenceAggregator,
    stability_frames: int,
    confirmed: bool,
    recognition_mode: str,
) -> None:
    """Temporary diagnostic only (ws_debug_log_fingerspelling_pipeline) --
    one rate-limited block per frame with exactly one hand visible, naming
    handedness, motion, the full top-3 ranked distribution (not just
    predict()'s argmax), and the debounce streak -- so a wrong live letter
    can be attributed to "the model itself ranked it wrong" vs "it was
    right but never reached the confirmation streak"."""
    top3_str = "[" + ", ".join(f"{r.letter}:{r.confidence:.2f}" for r in topk) + "]" if topk else "n/a"
    logger.info(
        "[FINGER DEBUG]\n"
        "  hand=%s mode=%s\n"
        "  motion=%.2f\n"
        "  top3=%s\n"
        "  prediction=%s\n"
        "  streak=%s/%s\n"
        "  confirmed=%s",
        handedness,
        recognition_mode,
        hand_motion,
        top3_str,
        prediction_letter,
        letter_aggregator._stable_count,  # noqa: SLF001 -- diagnostic-only read of internal debounce state
        stability_frames,
        "true" if confirmed else "false",
    )


_landmark_extractor: LandmarkExtractor | None = None
_landmark_extractor_error: str | None = None


def _get_landmark_extractor() -> LandmarkExtractor | None:
    """Lazily create the (process-wide, reused across connections) landmark
    extractor. If the model files aren't downloaded yet, cache the failure
    so we don't re-check the filesystem on every single frame -- but still
    surface a clear, actionable error to the client."""
    global _landmark_extractor, _landmark_extractor_error
    if _landmark_extractor is not None or _landmark_extractor_error is not None:
        return _landmark_extractor

    settings = get_settings()
    try:
        _landmark_extractor = LandmarkExtractor(
            model_dir=settings.mediapipe_models_dir,
            features=FeatureToggles(
                hands=settings.features_hands,
                pose=settings.features_pose,
                face=settings.features_face,
            ),
        )
    except ModelNotFoundError as exc:
        _landmark_extractor_error = str(exc)
        logger.warning("Landmark extractor unavailable: %s", _landmark_extractor_error)
    return _landmark_extractor



async def websocket_endpoint(websocket: WebSocket) -> None:
    settings = get_settings()
    conn_id = await connection_manager.connect(websocket)
    await websocket.send_json(ConnectionMessage(status="ok").model_dump())

    min_interval_sec = 1.0 / settings.ws_max_fps
    last_frame_at = 0.0
    frames_received = 0
    already_closed = False
    # Per-connection: an isolated sliding window per signer, sized to the
    # loaded checkpoint's sequence_length once (if) inference is ready.
    feature_buffer: deque[list[float]] | None = None
    # Per-connection: debounces the raw per-frame prediction stream into a
    # stable gloss sequence (Phase 11 -- see ml/inference/aggregator.py).
    gloss_aggregator = GlossSequenceAggregator(
        stability_frames=settings.ws_gloss_stability_frames,
        confidence_threshold=settings.ws_gloss_confidence_threshold,
    )
    # Per-connection: same debounce heuristic, for the separate real
    # fingerspelling classifier's per-frame letter stream (ml/fingerspelling/).
    # Reset whenever the frame doesn't have exactly one hand (see below) so
    # the signer can spell the same letter twice in a row by briefly
    # releasing the handshape between them, the natural way fingerspelling
    # transitions between letters.
    letter_aggregator = GlossSequenceAggregator(
        stability_frames=settings.fingerspelling_stability_frames,
        confidence_threshold=settings.fingerspelling_confidence_threshold,
    )
    # Per-connection: short window of recent single-hand feature vectors,
    # just for measuring whether that hand is actually being held still
    # (real fingerspelling) or moving (a one-handed dynamic word sign in
    # progress) -- see fingerspelling_max_hand_motion's docstring in
    # app/core/config.py. Fixed size (not tied to fingerspelling_stability_
    # frames, which a deployment/test may tune much smaller) -- the
    # threshold above was derived from real recorded clips measured over
    # exactly this many frames (see PROJECT_STATUS.md); a shorter window
    # wouldn't reliably show real motion yet.
    letter_motion_buffer: deque[list[float]] = deque(maxlen=_LETTER_MOTION_WINDOW)
    # Per-connection: word gets first priority -- while recognition_mode is
    # "word", only the word LSTM is allowed to CONFIRM (fingerspelling still
    # shows interim per-frame guesses, never letter_confirmed). Only once
    # the word buffer has been full for ws_gloss_stability_frames frames
    # without confirming (the same amount of time a real matching sign
    # would need anyway) does it fall back to "letter" mode, where dactyl
    # composes a word letter by letter instead. A sustained hand absence
    # (ws_token_boundary_no_hand_frames) is the sign-boundary that resets
    # back to "word" for the next sign -- the same "reset = boundary"
    # convention already used for both aggregators individually.
    recognition_mode = "word"
    no_hand_streak = 0
    word_buffer_full_streak = 0
    # A clean match normally confirms within ws_gloss_stability_frames
    # frames (see the diagnostics that derived that setting), so 3x that is
    # a generous allowance for ordinary live jitter (a wrong gloss briefly
    # appearing mid-transition, momentarily resetting the aggregator's own
    # streak) before concluding this genuinely isn't one of the trained
    # words and falling back to letter-by-letter composition instead.
    _WORD_FALLBACK_GRACE_FRAMES = settings.ws_gloss_stability_frames * 3

    def _advance_word_fallback_streak(confirmed: bool) -> None:
        """Ticks word_buffer_full_streak / falls back to "letter" mode.
        Called both when the word buffer fills without confirming AND when
        the motion gate rejects a window outright -- a held-still hand
        (zero motion) is just as valid a signal that no word is being
        attempted as "buffer full but not confirming" is, and real
        fingerspelling is by definition motionless. Skipping this call on
        the motion-gate-reject path left static letters unable to ever
        fall back to "letter" mode (confirmed via live testing: a held
        letter photo streamed for 60 frames never confirmed)."""
        nonlocal word_buffer_full_streak, recognition_mode
        if recognition_mode != "word":
            return
        if confirmed:
            word_buffer_full_streak = 0
        else:
            word_buffer_full_streak += 1
            if word_buffer_full_streak >= _WORD_FALLBACK_GRACE_FRAMES:
                recognition_mode = "letter"

    # Per-connection: calibrates against this signer's own neutral face,
    # then classifies eyebrow position into a non-manual grammar marker
    # (Phase 17 -- see ml/features/facial_grammar.py).
    facial_calibrator = BaselineCalibrator(
        calibration_frames=settings.ws_facial_calibration_frames,
        raised_ratio=settings.ws_facial_raised_ratio,
        furrowed_ratio=settings.ws_facial_furrowed_ratio,
    )

    try:
        while True:
            raw_text = await websocket.receive_text()

            if len(raw_text.encode("utf-8")) > settings.ws_max_message_size_bytes:
                logger.warning("WebSocket id=%s sent oversized message, closing", conn_id)
                await websocket.send_json(
                    ErrorMessage(
                        message=(
                            f"Message exceeds maximum allowed size of "
                            f"{settings.ws_max_message_size_bytes} bytes"
                        )
                    ).model_dump()
                )
                await websocket.close(code=1009)  # 1009 = message too big
                already_closed = True
                break

            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    ErrorMessage(message="Message is not valid JSON").model_dump()
                )
                continue

            try:
                frame_message = parse_client_message(raw)
            except ProtocolError as exc:
                await websocket.send_json(ErrorMessage(message=str(exc)).model_dump())
                continue

            now = time.monotonic()
            if now - last_frame_at < min_interval_sec:
                await websocket.send_json(
                    ErrorMessage(
                        message=(
                            f"Frame rate exceeds configured max of {settings.ws_max_fps} FPS; "
                            "frame dropped"
                        )
                    ).model_dump()
                )
                continue
            last_frame_at = now
            frames_received += 1
            _log_timing = settings.ws_debug_log_timing and frames_received % 30 == 0
            _frame_total_start = time.perf_counter()

            _stage_start = time.perf_counter()
            try:
                decoded_frame = decode_base64_frame(frame_message.data)
            except FrameDecodeError as exc:
                await websocket.send_json(
                    ErrorMessage(message=f"Could not decode frame: {exc}").model_dump()
                )
                continue
            if _log_timing:
                _log_timing_debug(conn_id, frames_received, "decode", (time.perf_counter() - _stage_start) * 1000)

            extractor = _get_landmark_extractor()
            if extractor is None:
                await websocket.send_json(
                    ErrorMessage(
                        message=_landmark_extractor_error
                        or "Landmark extractor is not available."
                    ).model_dump()
                )
                continue

            # detect() is a blocking call; run it off the event loop so one
            # slow frame doesn't stall every other connection.
            _stage_start = time.perf_counter()
            raw_landmarks = await asyncio.to_thread(extractor.extract, decoded_frame)
            if _log_timing:
                # The single combined call actually made in production --
                # not hands/pose/face broken out separately, which would
                # mean running MediaPipe detection 3x more per frame than
                # production does just to observe it (see scripts/
                # profile_backend.py for a one-off breakdown of the three
                # instead, run offline against a recorded clip).
                _log_timing_debug(conn_id, frames_received, "mediapipe_combined", (time.perf_counter() - _stage_start) * 1000)
            _stage_start = time.perf_counter()
            normalized = normalize_frame(raw_landmarks)
            if _log_timing:
                _log_timing_debug(conn_id, frames_received, "normalize", (time.perf_counter() - _stage_start) * 1000)

            # Sent independent of everything below (ML readiness, buffering,
            # facial calibration) -- real per-frame detection, so the "is my
            # hand visible" indicator works from frame 1.
            await websocket.send_json(
                LandmarksStatusMessage(
                    left_hand=normalized.present["left_hand"],
                    right_hand=normalized.present["right_hand"],
                    pose=normalized.present["pose"],
                    face=normalized.present["face"],
                    left_hand_points=_xy_points(raw_landmarks.left_hand),
                    right_hand_points=_xy_points(raw_landmarks.right_hand),
                    pose_points=_xy_points(raw_landmarks.pose),
                ).model_dump()
            )

            # Sign-boundary signal for recognition_mode AND the continuous-
            # dictation "boundary" events below: genuinely no hand visible
            # at all, not "not exactly one hand" (hand_vector below also
            # fires on two hands present, a real sign in progress) -- and
            # sustained, not a single-frame flicker.
            no_hand_at_all = raw_landmarks.left_hand is None and raw_landmarks.right_hand is None
            no_hand_streak = no_hand_streak + 1 if no_hand_at_all else 0
            if no_hand_streak >= settings.ws_token_boundary_no_hand_frames and recognition_mode == "letter":
                recognition_mode = "word"
                word_buffer_full_streak = 0
                gloss_aggregator.reset()

            # Exactly-once-per-crossing, not repeated every frame the hand
            # stays away: no_hand_streak increments by exactly 1 per
            # no-hand frame and resets to 0 the instant a hand reappears,
            # so it passes through every integer on the way up -- no extra
            # "already fired" flag needed. "token" always fires first (or
            # alongside, if the two thresholds are configured equal) since
            # ws_utterance_boundary_no_hand_frames is expected to be >=
            # ws_token_boundary_no_hand_frames (see protocol.py).
            if no_hand_streak == settings.ws_token_boundary_no_hand_frames:
                await websocket.send_json(BoundaryMessage(kind="token").model_dump())
            if no_hand_streak == settings.ws_utterance_boundary_no_hand_frames:
                await websocket.send_json(BoundaryMessage(kind="utterance").model_dump())

            # Word priority (recognition_mode) only means anything when
            # there's an actual word model to prioritize -- with no real
            # checkpoint loaded (not configured, or still demo-mode-only),
            # recognition_mode would otherwise sit at its initial "word"
            # forever (the word block below that would ever advance it past
            # that never runs) and silently block fingerspelling from ever
            # confirming at all. Computed once here, reused by both blocks
            # below, so it always reflects the SAME word-readiness state
            # they'll each independently re-check.
            inference_service = get_inference_service()
            word_recognition_available = inference_service.is_ready() and not getattr(
                inference_service, "is_demo_mode", False
            )

            # Real, separately-trained classifier (ml/fingerspelling/, distinct
            # from the word-level checkpoint below whether that one is demo or
            # real) for single dactyl letters -- independent of word-level ML
            # readiness, same
            # as landmarks_status/facial grammar above. Silently absent (no
            # message at all) when the frame doesn't have exactly one hand,
            # or when no fingerspelling checkpoint is loaded -- this is an
            # additive capability, not something that needs its own error.
            hand_vector, _skip_reason = hand_to_feature_vector(raw_landmarks.left_hand, raw_landmarks.right_hand)
            if hand_vector is None:
                letter_aggregator.reset()
                letter_motion_buffer.clear()
            else:
                fingerspelling_service = get_fingerspelling_service()
                if fingerspelling_service.is_ready():
                    _stage_start = time.perf_counter()
                    # predict_with_alternatives runs exactly the same ONE
                    # forward pass predict() would -- the ranked
                    # alternatives are nearly free on top of it (a topk
                    # over the already-computed softmax output), so this
                    # never costs more per frame than the old predict()
                    # call did (see its own docstring). letter_alternatives
                    # is reused below for both the debug log and, only
                    # once confirmed, the real top_k sent to the client.
                    letter_prediction, letter_alternatives = await asyncio.to_thread(
                        fingerspelling_service.predict_with_alternatives, hand_vector.tolist(), 3
                    )
                    if _log_timing:
                        _log_timing_debug(
                            conn_id, frames_received, "fingerspelling_inference", (time.perf_counter() - _stage_start) * 1000
                        )

                    letter_motion_buffer.append(hand_vector.tolist())
                    hand_motion = mean_hand_motion(list(letter_motion_buffer), hand_vector.shape[0])
                    if recognition_mode != "letter" and word_recognition_available:
                        # Word gets first priority (see recognition_mode's
                        # docstring above) -- a letter still shows as an
                        # interim guess below (real diagnostic visibility,
                        # never withheld), but is never allowed to CONFIRM
                        # until the word model has had its full, fair chance
                        # and failed. Without this, dactyl (needing only
                        # fingerspelling_stability_frames to confirm) would
                        # keep winning the race against the word LSTM
                        # (needing a full sequence_length buffer plus its
                        # own stability streak) on every single sign.
                        letter_aggregator.reset()
                        letter_confirmed = False
                    # A one-handed DYNAMIC word sign can hold a single
                    # letter's prediction stably enough to reach
                    # stability_frames purely because the hand shape
                    # mid-gesture resembles a known static letter -- while
                    # genuinely still in motion (confirmed live, see
                    # fingerspelling_max_hand_motion's docstring). Only let
                    # the debounce run -- and thus only let a letter ever
                    # reach letter_confirmed -- while the hand is actually
                    # held still; a moving hand never even starts building a
                    # streak, the same "reset = boundary" role hand-absence
                    # already plays above.
                    elif hand_motion >= settings.fingerspelling_max_hand_motion:
                        letter_aggregator.reset()
                        letter_confirmed = False
                    else:
                        letter_confirmed = letter_aggregator.update(
                            letter_prediction.letter, letter_prediction.confidence
                        )

                    if settings.ws_debug_log_fingerspelling_pipeline and frames_received % 5 == 0:
                        _log_fingerspelling_debug(
                            handedness="Left" if raw_landmarks.left_hand is not None else "Right",
                            hand_motion=hand_motion,
                            topk=letter_alternatives,
                            prediction_letter=letter_prediction.letter,
                            letter_aggregator=letter_aggregator,
                            stability_frames=settings.fingerspelling_stability_frames,
                            confirmed=letter_confirmed,
                            recognition_mode=recognition_mode,
                        )

                    await websocket.send_json(
                        LetterPredictionMessage(
                            type="letter_confirmed" if letter_confirmed else "letter_prediction",
                            letter=letter_prediction.letter,
                            confidence=letter_prediction.confidence,
                            is_final=letter_confirmed,
                            # Only on the confirmed message -- see
                            # LetterCandidate's own docstring for why.
                            top_k=(
                                [LetterCandidate(letter=r.letter, confidence=r.confidence) for r in letter_alternatives]
                                if letter_confirmed
                                else None
                            ),
                        ).model_dump()
                    )

            # Calibrates/classifies regardless of ML readiness below -- an
            # honest FacialGrammarMarker.NONE when no face was detected at
            # all, never a guess from zero-filled landmarks.
            facial_marker = (
                facial_calibrator.update(normalized.face)
                if normalized.present["face"]
                else FacialGrammarMarker.NONE
            )

            feature_config = FeatureConfig(
                hands=settings.features_hands,
                pose=settings.features_pose,
                face=settings.features_face,
            )
            _stage_start = time.perf_counter()
            feature_vector = build_feature_vector(normalized, feature_config)
            if _log_timing:
                _log_timing_debug(conn_id, frames_received, "feature_vector_build", (time.perf_counter() - _stage_start) * 1000)

            # A demo/synthetic-only checkpoint (ml/datasets/synthetic.py) is
            # treated the same as "no checkpoint at all": its output carries
            # no real information about the signer's actual gesture (confirms
            # glosses close to at random on real camera input), so it must
            # never reach the live translation stream as if it were a real
            # recognition -- same honest not-ready message either way, with
            # no "demo"/internal-path wording for the UI to surface.
            # /health's ml_pipeline_status still reports "demo_mode" distinctly
            # for anyone debugging this server-side. word_recognition_available
            # (computed above) is the exact same check.
            if not word_recognition_available:
                # Real landmarks ARE extracted (Phase 6-7) -- but no real
                # trained checkpoint is loaded, so this stays an honest error
                # with real detection info, never a fabricated sign.
                await websocket.send_json(
                    ErrorMessage(
                        message=(
                            "Sign-recognition is not available yet. Landmarks were "
                            f"extracted: left_hand={normalized.present['left_hand']}, "
                            f"right_hand={normalized.present['right_hand']}, "
                            f"pose={normalized.present['pose']}, "
                            f"face={normalized.present['face']} "
                            f"(feature vector size: {feature_vector.shape[0]}/"
                            f"{feature_vector_size(feature_config)})."
                        )
                    ).model_dump()
                )
                continue

            sequence_length = inference_service.sequence_length
            if feature_buffer is None:
                feature_buffer = deque(maxlen=sequence_length)
            feature_buffer.append(feature_vector.tolist())

            if len(feature_buffer) < sequence_length:
                await websocket.send_json(
                    ErrorMessage(
                        message=(
                            f"Buffering: {len(feature_buffer)}/{sequence_length} frames "
                            "collected before the first prediction."
                        )
                    ).model_dump()
                )
                continue

            # Live-vs-replay diagnostics (temporary, opt-in -- see
            # ws_debug_save_feature_windows_dir's docstring in app/core/
            # config.py): a REAL captured live window, saved unmodified, so
            # it can be replayed offline through the exact same checkpoint
            # later, bypassing the motion gate and aggregator entirely.
            if settings.ws_debug_save_feature_windows_dir:
                _save_debug_feature_window(
                    settings.ws_debug_save_feature_windows_dir, conn_id, frames_received, list(feature_buffer)
                )

            # The word classifier has no trained "no gesture" class (see
            # ml/inference/motion_gate.py) -- confirmed live that a held-
            # still hand otherwise gets confidently misclassified as one of
            # the 7 known words. Skip inference entirely on a window with no
            # real hand movement, and reset the debounce streak: this also
            # doubles as the sign-boundary signal for gloss_aggregator (the
            # same role letter_aggregator.reset() plays above for a dropped
            # hand), so the same word can be confirmed again after a pause.
            hand_motion = (
                mean_hand_motion(list(feature_buffer), HAND_FEATURE_SIZE) if feature_config.hands else None
            )
            motion_gate_passed = hand_motion is None or hand_motion >= settings.ws_min_hand_motion

            if not motion_gate_passed:
                gloss_aggregator.reset()
                _advance_word_fallback_streak(confirmed=False)
                if settings.ws_debug_log_word_pipeline and frames_received % 5 == 0:
                    _log_word_debug(
                        conn_id, frames_received, normalized, feature_buffer, sequence_length,
                        hand_motion, settings.ws_min_hand_motion, motion_gate_passed,
                        lstm_called=False, topk=None, gloss_aggregator=gloss_aggregator, confirmed=False,
                        recognition_mode=recognition_mode,
                    )
                await websocket.send_json(
                    ErrorMessage(
                        message=(
                            "No hand motion detected -- waiting for a sign "
                            f"(motion={hand_motion:.3f}, need >= {settings.ws_min_hand_motion})."
                        )
                    ).model_dump()
                )
                continue

            try:
                # predict_with_alternatives runs exactly the same ONE
                # forward pass predict() would (see its own docstring) --
                # keep it off the event loop, same reasoning as
                # extractor.extract above. word_alternatives is reused
                # below for both the debug log and, only once confirmed,
                # the real top_k sent to the client.
                _stage_start = time.perf_counter()
                prediction, word_alternatives = await asyncio.to_thread(
                    inference_service.predict_with_alternatives, list(feature_buffer), 3
                )
                if _log_timing:
                    _log_timing_debug(conn_id, frames_received, "word_lstm_inference", (time.perf_counter() - _stage_start) * 1000)
            except MLNotReadyError as exc:
                await websocket.send_json(ErrorMessage(message=str(exc)).model_dump())
                continue

            # Word gets first priority (see recognition_mode's docstring
            # above): while in "letter" mode, the word LSTM keeps running
            # (real diagnostic visibility, an interim "prediction" message
            # still goes out below) but is never allowed to reach
            # final_prediction -- only one of {word, letter} ever confirms
            # at a time. While still in "word" mode, track how long the
            # buffer has been full without confirming (see
            # _advance_word_fallback_streak's docstring).
            if recognition_mode == "word":
                confirmed = gloss_aggregator.update(prediction.sign, prediction.confidence)
                _advance_word_fallback_streak(confirmed)
            else:
                confirmed = False

            if settings.ws_debug_log_word_pipeline and frames_received % 5 == 0:
                _log_word_debug(
                    conn_id, frames_received, normalized, feature_buffer, sequence_length,
                    hand_motion, settings.ws_min_hand_motion, motion_gate_passed,
                    lstm_called=True, topk=word_alternatives, gloss_aggregator=gloss_aggregator, confirmed=confirmed,
                    recognition_mode=recognition_mode,
                )

            display_text = prediction.text
            if confirmed:
                # Phase 12: translate the just-confirmed gloss into a real
                # Ukrainian sentence when the (intentionally small) rule-based
                # lexicon covers it; otherwise keep the raw gloss text rather
                # than guessing a composition -- see ml/nlp/gloss_to_text.py.
                # Phase 17: a concurrent eyebrow marker makes it a question --
                # written Ukrainian uses "?" for both yes/no and wh-questions,
                # so either marker flips the terminator the same way.
                # inference_service.is_demo_mode is never True here -- the
                # demo_mode check above already routes demo checkpoints to an
                # honest not-ready message before any prediction is made.
                try:
                    display_text = translation_service.gloss_to_text(
                        [prediction.sign],
                        is_question=facial_marker != FacialGrammarMarker.NONE,
                    )
                except ValueError:
                    pass
            if _log_timing:
                # The happy-path total: decode through word-LSTM inference
                # (and fingerspelling inference, when a hand was present).
                # Frames that exit earlier (buffering, motion-gated, no
                # checkpoint) have their own individual stage logs above but
                # no "total" line -- there is no complete pipeline run to
                # total for those.
                _log_timing_debug(conn_id, frames_received, "total_pipeline", (time.perf_counter() - _frame_total_start) * 1000)
            await websocket.send_json(
                PredictionMessage(
                    type="final_prediction" if confirmed else "prediction",
                    text=display_text,
                    gloss=prediction.sign,
                    confidence=prediction.confidence,
                    is_final=confirmed,
                    facial_grammar=facial_marker.value,
                    # Only on the confirmed message -- see WordCandidate's
                    # own docstring for why.
                    top_k=(
                        [WordCandidate(gloss=r.gloss, confidence=r.confidence) for r in word_alternatives]
                        if confirmed
                        else None
                    ),
                ).model_dump()
            )
            if confirmed:
                logger.info(
                    "WebSocket id=%s confirmed gloss #%s (sequence so far: %s)",
                    conn_id,
                    len(gloss_aggregator.sequence),
                    gloss_aggregator.sequence,
                )

            if frames_received % 30 == 0:
                logger.info(
                    "WebSocket id=%s received %s frames so far (message payloads not logged)",
                    conn_id,
                    frames_received,
                )

    except WebSocketDisconnect:
        logger.info("WebSocket id=%s disconnected by client", conn_id)
    finally:
        connection_manager.disconnect(conn_id)
        if not already_closed and websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                # Already closed by the ASGI server between our check and this call.
                pass
