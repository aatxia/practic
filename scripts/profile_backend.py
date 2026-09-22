#!/usr/bin/env python3
"""
Real, measured per-stage backend timing for the live recognition pipeline --
not guessed. Runs the exact same pipeline websocket/handler.py uses (real
MediaPipe detectors, real normalization/feature-vector code, real loaded
checkpoints) against real recorded frames, and reports median/p95 per stage.

Complements the opt-in WS_DEBUG_LOG_TIMING live diagnostic (websocket/
handler.py, app/core/config.py), which logs the same stages from an actual
running WebSocket session but only reports the single combined MediaPipe
call production actually makes. This script additionally profiles
hands/pose/face separately (running each detector alone) to show which of
the three actually dominates -- something the live diagnostic deliberately
doesn't do, since running MediaPipe detection 3x more per frame just to
observe it would itself make the app slower.

Usage (from repo root):
    python scripts/profile_backend.py [path/to/clip.mp4]

Defaults to a real recorded clip already in the repo if no path is given.
"""
from __future__ import annotations

import argparse
import base64
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from ml.features.feature_vector import FeatureConfig, build_feature_vector
from ml.fingerspelling.dataset import hand_to_feature_vector
from ml.fingerspelling.recognizer import FingerspellingRecognizer
from ml.inference.recognizer import SignRecognizer
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor
from ml.preprocessing.normalization import normalize_frame
from ml.preprocessing.video_reader import decode_base64_frame

DEFAULT_CLIP = REPO_ROOT / "data" / "real" / "raw" / "aatxia" / "WANT_00.mp4"


def load_clip_frames(path: str, max_frames: int = 90) -> list[bytes]:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok2:
            frames.append(buf.tobytes())
    cap.release()
    return frames


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return (time.perf_counter() - start) * 1000, result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def report(name: str, values_ms: list[float]) -> None:
    if not values_ms:
        print(f"{name:28s} (no samples -- this stage never ran)")
        return
    print(
        f"{name:28s} median={statistics.median(values_ms):7.2f}ms  "
        f"p95={percentile(values_ms, 95):7.2f}ms  n={len(values_ms)}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", nargs="?", default=str(DEFAULT_CLIP), help="Real recorded .mp4 clip to replay")
    parser.add_argument("--checkpoint", default="models/checkpoints/real_v1/latest.pt")
    parser.add_argument("--fingerspelling-checkpoint", default="ml/fingerspelling/checkpoints/latest.pt")
    parser.add_argument("--mediapipe-models-dir", default="models/mediapipe")
    args = parser.parse_args(argv)

    raw_frames = load_clip_frames(args.clip)
    print(f"Loaded {len(raw_frames)} real frames from {args.clip}\n")

    # base64-encode once, matching the actual wire format the WebSocket
    # handler decodes every frame -- measures the REAL decode path, not
    # raw cv2.imread.
    b64_frames = [f"data:image/jpeg;base64,{base64.b64encode(f).decode('ascii')}" for f in raw_frames]

    extractor = LandmarkExtractor(model_dir=args.mediapipe_models_dir, features=FeatureToggles(True, True, True))
    # Separate single-modality extractors, ONLY to break MediaPipe hands/
    # pose/face timing apart for this one-off report -- never done in the
    # live handler, which makes exactly one combined extract() call.
    hands_only = LandmarkExtractor(model_dir=args.mediapipe_models_dir, features=FeatureToggles(True, False, False))
    pose_only = LandmarkExtractor(model_dir=args.mediapipe_models_dir, features=FeatureToggles(False, True, False))
    face_only = LandmarkExtractor(model_dir=args.mediapipe_models_dir, features=FeatureToggles(False, False, True))

    word_recognizer = SignRecognizer(args.checkpoint, device="cpu")
    fs_recognizer = FingerspellingRecognizer(args.fingerspelling_checkpoint, device="cpu")
    feature_config = FeatureConfig(hands=True, pose=True, face=True)

    decode_ms, hands_ms, pose_ms, face_ms, combined_extract_ms = [], [], [], [], []
    normalize_ms, feature_vec_ms, word_lstm_ms, fs_ms, realistic_total_ms = [], [], [], [], []

    feature_buffer: list[list[float]] = []
    sequence_length = word_recognizer.sequence_length

    for b64, raw in zip(b64_frames, raw_frames):
        t, decoded = timed(decode_base64_frame, b64)
        decode_ms.append(t)

        # Isolated single-modality timings, for the breakdown only.
        t, _ = timed(hands_only.extract, decoded)
        hands_ms.append(t)
        t, _ = timed(pose_only.extract, decoded)
        pose_ms.append(t)
        t, _ = timed(face_only.extract, decoded)
        face_ms.append(t)

        # The ONE combined call production actually makes -- this is what
        # realistic_total_ms below is built from, not the three isolated
        # calls above.
        t, raw_landmarks = timed(extractor.extract, decoded)
        combined_extract_ms.append(t)
        realistic_total = combined_extract_ms[-1]

        t, normalized = timed(normalize_frame, raw_landmarks)
        normalize_ms.append(t)
        realistic_total += t

        t, feature_vector = timed(build_feature_vector, normalized, feature_config)
        feature_vec_ms.append(t)
        realistic_total += t
        feature_buffer.append(feature_vector.tolist())
        if len(feature_buffer) > sequence_length:
            feature_buffer.pop(0)

        if len(feature_buffer) == sequence_length:
            t, _ = timed(word_recognizer.predict, feature_buffer)
            word_lstm_ms.append(t)
            realistic_total += t

        hand_vector, _ = hand_to_feature_vector(raw_landmarks.left_hand, raw_landmarks.right_hand)
        if hand_vector is not None:
            t, _ = timed(fs_recognizer.predict, hand_vector.tolist())
            fs_ms.append(t)
            realistic_total += t

        realistic_total_ms.append(realistic_total + decode_ms[-1])

    print("=== Per-stage timing (real frames, real models, CPU) ===")
    report("frame decode", decode_ms)
    report("MediaPipe hands (isolated)", hands_ms)
    report("MediaPipe pose (isolated)", pose_ms)
    report("MediaPipe face (isolated)", face_ms)
    report("MediaPipe combined (real, 1 call)", combined_extract_ms)
    report("normalization", normalize_ms)
    report("feature-vector build", feature_vec_ms)
    report("word-LSTM inference", word_lstm_ms)
    report("fingerspelling inference", fs_ms)
    report("REALISTIC total per frame", realistic_total_ms)
    print()
    print(f"Implied max FPS from the realistic total: {1000 / statistics.median(realistic_total_ms):.1f}")

    extractor.close()
    hands_only.close()
    pose_only.close()
    face_only.close()


if __name__ == "__main__":
    main()
