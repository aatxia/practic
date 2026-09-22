#!/usr/bin/env python3
"""
Signer-independent train/val/test split (section 11): assigns every sample
from a given signer_id to exactly one split, so no signer appears in both
train and test (see ml/datasets/split.py for why this is a hard rule, not
a preference). Reads a JSONL annotation file (ml/datasets/annotation.py)
and writes `<output-dir>/{train,val,test}.txt` (one sample_id per line) plus
`<output-dir>/split_manifest.json` (ratios, seed, per-signer assignment).

Usage:
    python scripts/create_dataset_split.py \\
        --annotations data/annotations/demo_annotations.jsonl \\
        --output-dir data/splits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.datasets.annotation import load_annotations
from ml.datasets.split import SplitRatios, signer_independent_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="Path to a JSONL annotation file")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_annotations(args.annotations)
    ratios = SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    split = signer_independent_split(samples, ratios=ratios, seed=args.seed)

    by_id = {sample.sample_id: sample for sample in samples}
    signer_assignment: dict[str, str] = {}
    for split_name, sample_ids in split.items():
        for sample_id in sample_ids:
            signer_assignment[by_id[sample_id].signer_id] = split_name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, sample_ids in split.items():
        text = "\n".join(sample_ids) + ("\n" if sample_ids else "")
        (output_dir / f"{split_name}.txt").write_text(text, encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {name: len(ids) for name, ids in split.items()},
        "signer_assignment": signer_assignment,
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "split_manifest.json").write_text(manifest_text, encoding="utf-8")

    for name, ids in split.items():
        signers = sorted({by_id[i].signer_id for i in ids})
        signer_list = ", ".join(signers) if signers else "-"
        print(f"{name}: {len(ids)} samples, {len(signers)} signers ({signer_list})")


if __name__ == "__main__":
    main()
