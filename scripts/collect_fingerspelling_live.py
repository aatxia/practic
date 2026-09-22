#!/usr/bin/env python3
"""
collect_fingerspelling_live -- captures REAL live-webcam fingerspelling
samples for one known letter, through the EXACT same pipeline production
uses (ml/preprocessing/landmarks.py's LandmarkExtractor with the same
FeatureToggles(hands=True, pose=False, face=False) ml/fingerspelling/
build_dataset.py trains on, then ml/fingerspelling/dataset.py's
hand_to_feature_vector()) -- then immediately evaluates the CURRENT
checkpoint on exactly what was captured. No retraining, no threshold/
label/checkpoint changes -- this only tells you whether the checkpoint
generalizes to your live camera on a letter you say you're showing it.

This is an interactive, camera-driving script -- run it locally, on a
machine with a real webcam and display, not in a sandboxed/headless
environment (same constraint as scripts/record_dataset.py).

Usage:
    python scripts/collect_fingerspelling_live.py --letter С
    python scripts/collect_fingerspelling_live.py --letter Ж --count 30
    python scripts/collect_fingerspelling_live.py --letter Ш --camera-index 0

Controls:
    q   quit early (whatever was captured so far is still saved/evaluated,
        unless zero samples were captured)

The preview is deliberately NOT mirrored -- matches the live production
camera pipeline (frontend/hooks/useCamera.ts, scripts/record_dataset.py),
neither of which mirror either. It will feel visually backwards (like a
video call before mirroring was added); that's expected, not a bug.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from ml.fingerspelling.dataset import hand_to_feature_vector
from ml.fingerspelling.recognizer import FingerspellingRecognizer
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor

DEFAULT_CHECKPOINT = "ml/fingerspelling/checkpoints/latest.pt"
DEFAULT_MODELS_DIR = "models/mediapipe"
DEFAULT_OUTPUT_DIR = "data/fingerspelling_live_captures"


def capture(letter: str, count: int, camera_index: int, models_dir: str) -> tuple[list[list[float]], list[str]]:
    # pose=False, face=False: fingerspelling never consumes those (see
    # hand_to_feature_vector) -- same toggles ml/fingerspelling/
    # build_dataset.py trains with, just skipping unrelated detectors for
    # a faster live loop, not a different hand-extraction pipeline.
    extractor = LandmarkExtractor(models_dir, FeatureToggles(hands=True, pose=False, face=False))

    capture_device = cv2.VideoCapture(camera_index)
    if not capture_device.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")

    features: list[list[float]] = []
    handedness: list[str] = []
    window = f"collect_fingerspelling_live -- {letter}"

    try:
        while len(features) < count:
            ok, frame = capture_device.read()
            if not ok:
                raise SystemExit("Camera read failed")

            result = extractor.extract(frame)
            vector, skip_reason = hand_to_feature_vector(result.left_hand, result.right_hand)

            display = frame.copy()
            if vector is not None:
                features.append(vector.tolist())
                handedness.append("Left" if result.left_hand is not None else "Right")
                status = f"captured {len(features)}/{count}"
                color = (0, 200, 0)
            else:
                status = f"no valid hand ({skip_reason})"
                color = (0, 0, 255)

            cv2.putText(
                display, f"target: {letter}  {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
            cv2.putText(
                display, "hold the letter still -- q to stop early", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture_device.release()
        cv2.destroyAllWindows()

    return features, handedness


def evaluate(
    checkpoint_path: str, letter: str, features: list[list[float]], handedness: list[str]
) -> None:
    recognizer = FingerspellingRecognizer(checkpoint_path, device="cpu")
    classes = sorted(recognizer._index_to_label.values())  # noqa: SLF001 -- diagnostic script
    if letter not in classes:
        print(f"** WARNING ** '{letter}' is not one of this checkpoint's {len(classes)} trained classes: {classes}")

    top1_hits = 0
    top3_hits = 0
    predicted_counts: Counter[str] = Counter()
    prob_sum = np.zeros(len(classes), dtype=np.float64)

    for vector in features:
        top = recognizer.predict_topk(vector, k=len(classes))  # full ranked distribution
        predicted_counts[top[0].letter] += 1
        if top[0].letter == letter:
            top1_hits += 1
        if letter in {r.letter for r in top[:3]}:
            top3_hits += 1
        for r in top:
            prob_sum[classes.index(r.letter)] += r.confidence

    n = len(features)
    mean_probs = prob_sum / max(n, 1)

    print(f"\n=== Evaluation: target letter '{letter}', {n} real live-captured samples ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Handedness seen: {dict(Counter(handedness))}")
    print(f"Top-1 accuracy: {top1_hits}/{n} = {top1_hits / max(n, 1):.1%}")
    print(f"Top-3 accuracy: {top3_hits}/{n} = {top3_hits / max(n, 1):.1%}")
    print("\nConfusion counts (what it actually predicted as top-1, across all samples):")
    for pred_letter, cnt in predicted_counts.most_common():
        marker = " <-- correct" if pred_letter == letter else ""
        print(f"  {pred_letter:4s} {cnt:3d}/{n}{marker}")
    print("\nMean probability per class (descending):")
    for idx in np.argsort(-mean_probs):
        marker = " <-- target" if classes[idx] == letter else ""
        print(f"  {classes[idx]:4s} {mean_probs[idx]:.3f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--letter", required=True, help="The Ukrainian letter you are about to show the camera")
    parser.add_argument("--count", type=int, default=30, help="Number of valid single-hand samples to capture")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    features, handedness = capture(args.letter, args.count, args.camera_index, args.models_dir)
    if not features:
        raise SystemExit("No valid samples captured -- nothing saved, nothing evaluated.")

    session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.letter}_{session_id}.npz"
    np.savez_compressed(
        out_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.array([args.letter] * len(features), dtype="<U8"),
        handedness=np.array(handedness, dtype="<U8"),
        session_id=session_id,
    )
    print(f"Saved {len(features)} real captured samples to {out_path}")

    evaluate(args.checkpoint, args.letter, features, handedness)


if __name__ == "__main__":
    main()
