"""
annotation — the on-disk format for one labeled sign-language clip
(section 10), and honest load/write helpers for it.

Format: JSON Lines (one `SampleAnnotation` per line) so files can be
inspected with plain text tools, diffed in git, and streamed without
loading an entire dataset into memory.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleAnnotation:
    """One labeled clip.

    `clip_path` means different things depending on `source`:
    - for a real dataset, it points at a raw video file (`[start_frame,
      end_frame)` selects the signing segment within it) -- decoding it
      into a feature-vector sequence is Phase 9's job, via the existing
      ml/preprocessing + ml/features pipeline.
    - for `source="demo_synthetic"` (ml/datasets/synthetic.py), it points
      directly at a precomputed `.npy` feature-vector sequence, since there
      is no real video behind it at all.

    `source` must never be guessed: it says exactly where the label and the
    data came from (e.g. "demo_synthetic", or a real dataset's name), so
    downstream code can refuse to silently mix synthetic and real samples.
    """

    sample_id: str
    clip_path: str
    signer_id: str
    gloss: str
    start_frame: int
    end_frame: int
    fps: float
    source: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.signer_id:
            raise ValueError("signer_id must not be empty")
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be greater than start_frame "
                f"({self.start_frame}) for sample {self.sample_id!r}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps} for sample {self.sample_id!r}")


def load_annotations(path: Path | str) -> list[SampleAnnotation]:
    """Load a JSONL annotation file. Raises a `ValueError` naming the exact
    file and line on malformed input or a duplicate sample_id -- a bad row
    is never silently skipped, since that would quietly shrink the dataset."""
    path = Path(path)
    samples: list[SampleAnnotation] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sample = SampleAnnotation(**data)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid annotation row: {exc}") from exc
            if sample.sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample.sample_id!r}")
            seen_ids.add(sample.sample_id)
            samples.append(sample)
    return samples


def write_annotations(path: Path | str, samples: list[SampleAnnotation]) -> None:
    """Write samples as JSONL, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
