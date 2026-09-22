#!/usr/bin/env python3
"""
DEMO MODE ONLY -- generates a small synthetic dataset so the Phase 8/9
dataset pipeline can be exercised end-to-end before a real Ukrainian Sign
Language video dataset is available. See ml/datasets/synthetic.py and
PROJECT_STATUS.md (Phase 8) for why: no public isolated-sign УЖМ dataset
with signer_id labeling could be found.

Usage:
    python scripts/generate_demo_dataset.py [--output-dir data] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.datasets.synthetic import generate_demo_dataset
from ml.features.feature_vector import FeatureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data", help="Dataset root (default: data/)")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--samples-per-signer-gloss", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-hands", action="store_true", help="Disable the hands modality")
    parser.add_argument("--no-pose", action="store_true", help="Disable the pose modality")
    parser.add_argument("--no-face", action="store_true", help="Disable the face modality")
    args = parser.parse_args()

    config = FeatureConfig(hands=not args.no_hands, pose=not args.no_pose, face=not args.no_face)
    samples = generate_demo_dataset(
        output_dir=args.output_dir,
        config=config,
        sequence_length=args.sequence_length,
        samples_per_signer_gloss=args.samples_per_signer_gloss,
        seed=args.seed,
    )

    print(f"[DEMO MODE] Generated {len(samples)} synthetic samples under {args.output_dir}/")
    print(f"  annotations: {args.output_dir}/annotations/demo_annotations.jsonl")
    print(f"  features:    {args.output_dir}/processed/demo/*.npy")
    print("This is NOT real Ukrainian Sign Language data -- it only exercises the pipeline.")


if __name__ == "__main__":
    main()
