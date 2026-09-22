#!/usr/bin/env python3
"""
diagnose_live_capture -- replays REAL captured live-webcam feature windows
(saved by the backend when WS_DEBUG_SAVE_FEATURE_WINDOWS_DIR is set, see
backend/app/core/config.py and backend/websocket/handler.py) directly
through the loaded word-level LSTM checkpoint, bypassing the motion gate
and the gloss aggregator entirely.

This answers exactly one question: does the model itself recognize a real
live-captured window correctly, when nothing else in the pipeline (gate,
debounce) gets a chance to interfere? If yes, a live recognition failure is
in the gate/routing, not the model. If the model is ALSO wrong on the same
live-captured window, the problem is upstream (live capture, live
normalization/domain, or the model's generalization to this signer's live
performance) -- not the gate.

Usage:
    # 1. Locally, before running the real backend, set:
    #      WS_DEBUG_SAVE_FEATURE_WINDOWS_DIR=/tmp/uksl_live_windows
    #    then start the backend and perform a known word in front of the
    #    live webcam for a couple of seconds. Every full 32-frame window
    #    gets saved as a real .npz -- no synthetic/recreated data.
    #
    # 2. python scripts/diagnose_live_capture.py \\
    #        --windows-dir /tmp/uksl_live_windows \\
    #        --checkpoint models/checkpoints/real_v1/latest.pt \\
    #        --expected WANT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference.recognizer import SignRecognizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows-dir", required=True, help="Directory of .npz windows saved by the live backend")
    parser.add_argument("--checkpoint", default="models/checkpoints/real_v1/latest.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected", default=None, help="The word you actually performed, for a quick eyeball diff")
    parser.add_argument("-k", type=int, default=3, help="Top-k classes to print per window")
    args = parser.parse_args()

    windows_dir = Path(args.windows_dir)
    files = sorted(windows_dir.glob("window_*.npz"))
    if not files:
        raise SystemExit(
            f"No .npz windows found in {windows_dir} -- did you set "
            "WS_DEBUG_SAVE_FEATURE_WINDOWS_DIR and actually perform a sign in front of the live webcam?"
        )

    recognizer = SignRecognizer(args.checkpoint, device=args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  sequence_length={recognizer.sequence_length}  feature_config={recognizer.feature_config}")
    print(f"  classes={sorted(recognizer._index_to_label.values())}")  # noqa: SLF001 -- diagnostic script
    print(f"  demo_mode={recognizer.is_demo_mode}  source_tags={recognizer.source_tags}")
    print(f"Replaying {len(files)} real captured live windows from {windows_dir}\n")

    correct = 0
    for path in files:
        data = np.load(path)
        features = data["features"]
        print(f"{path.name}")
        print(f"  tensor shape (as fed to the model): (1, {features.shape[0]}, {features.shape[1]})")
        if features.shape[1] != feature_dim(recognizer):
            print(
                f"  ** MISMATCH ** captured feature dim {features.shape[1]} != checkpoint's expected "
                f"{feature_dim(recognizer)} -- this alone would explain a live failure, stop and fix this first."
            )
            continue

        top = recognizer.predict_topk(features.tolist(), k=args.k)
        for rank, r in enumerate(top, start=1):
            marker = " <-- expected" if args.expected and r.gloss == args.expected else ""
            print(f"    #{rank} {r.gloss:8s} {r.confidence:.3f}{marker}")
        if args.expected and top and top[0].gloss == args.expected:
            correct += 1
        print()

    if args.expected:
        print(f"Top-1 matched '{args.expected}' on {correct}/{len(files)} captured live windows.")


def feature_dim(recognizer: SignRecognizer) -> int:
    from ml.features.feature_vector import feature_vector_size

    return feature_vector_size(recognizer.feature_config)


if __name__ == "__main__":
    main()
