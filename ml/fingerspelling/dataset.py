"""
fingerspelling.dataset -- turns the real USL_alphabet_train photos into
normalized hand-landmark feature vectors for a single-frame classifier.

Unlike ml/datasets/synthetic.py's sequences of frames (fed to the LSTM
isolated-word recognizer), one dactyl letter is one still photo of a
handshape -- there is no temporal dimension here, so this reuses
ml/preprocessing/normalization.py's normalize_hand() directly on the single
detected hand, not the full multimodal per-frame pipeline built for video.

USL_alphabet_train/ has two kinds of subdirectories that must never be
treated as real classes:
  - `_excluded_non_ukrainian/` -- letters (Ъ, Ы, Э) that aren't part of the
    Ukrainian alphabet; confirmed with the person who uploaded this data
    that they don't belong here.
  - `_excluded_unclear/` -- an ambiguously-named folder ("TR", mixed
    Latin/Cyrillic) and a single-file folder from a corrupted directory
    name; neither has a confirmed real label.
Both are skipped by construction (any directory name starting with "_"),
never silently folded into a real letter's samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ml.preprocessing.landmarks import LandmarkExtractor
from ml.preprocessing.normalization import normalize_hand

_EXCLUDED_PREFIX = "_"


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    label: str


def scan_image_dataset(root: Path | str) -> list[ImageSample]:
    """One sample per image file directly inside a per-letter subdirectory
    of `root`. Subdirectories starting with "_" are skipped -- see module
    docstring."""
    root = Path(root)
    samples: list[ImageSample] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith(_EXCLUDED_PREFIX):
            continue
        label = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file():
                samples.append(ImageSample(image_path=image_path, label=label))
    return samples


@dataclass
class ExtractionResult:
    features: np.ndarray  # (N, 63) float32
    labels: list[str]
    # (path, reason) for every photo that couldn't honestly be turned into a
    # sample -- never silently dropped without a record.
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    # How many kept samples came from a detected left vs right hand, before
    # left-hand mirroring -- see extract_features' docstring.
    left_hand_count: int = 0
    right_hand_count: int = 0


def hand_to_feature_vector(
    left_hand: np.ndarray | None, right_hand: np.ndarray | None
) -> tuple[np.ndarray | None, str | None]:
    """Turns raw (un-normalized) single-frame hand landmarks -- the same
    `FrameLandmarks.left_hand`/`.right_hand` the live WebSocket handler
    already extracts every frame for the overlay (backend/websocket/
    handler.py) -- into the exact 63-dim feature vector this classifier
    was trained on. Shared by build_dataset.py's offline photo extraction
    and the live per-frame path (backend/app/services/fingerspelling_
    service.py) so training and serving can never silently drift apart.

    Returns (vector, None) on success, or (None, reason) when the frame
    doesn't have exactly one hand -- the caller decides what to do with
    "no hand" vs "both hands" (log it for a static photo, just skip this
    frame silently for a live video stream), this function only reports
    it honestly rather than guessing.

    A left-hand detection has its x-coordinate mirrored before
    normalization, so a left-hand and a right-hand frame of the same
    handshape land in the same feature space (handshape identity doesn't
    depend on which physical hand performs it, just its mirror image) --
    a deliberate, documented design choice, not something MediaPipe does
    on its own.
    """
    has_left = left_hand is not None
    has_right = right_hand is not None

    if has_left and has_right:
        return None, "both hands detected, ambiguous for a single-handshape photo"
    if not has_left and not has_right:
        return None, "no hand detected"

    if has_left:
        hand = left_hand.copy()
        hand[:, 0] *= -1.0  # mirror x so left- and right-hand frames share one feature space
    else:
        hand = right_hand

    normalized = normalize_hand(hand)
    return normalized.flatten().astype(np.float32), None


def _read_image_bgr(path: Path) -> np.ndarray | None:
    """cv2.imread(str(path)) silently returns None for non-ASCII paths on
    Windows (it goes through an ANSI-codepage fopen internally) -- a real
    problem here, since USL_alphabet_train's class directories are named
    with Cyrillic letters (see module docstring). Reading the bytes via
    pathlib (which handles Windows Unicode paths correctly) and decoding
    through cv2.imdecode works cross-platform -- the same technique
    ml/preprocessing/video_reader.py's decode_base64_frame already uses for
    the live WebSocket's frames, so this isn't a new pattern in this repo."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def extract_features(
    samples: list[ImageSample], extractor: LandmarkExtractor, progress_every: int = 0
) -> ExtractionResult:
    """Runs the real MediaPipe hand detector on every sample image and
    turns the single detected hand into a feature vector via
    hand_to_feature_vector() -- see its docstring for the normalization/
    mirroring convention.

    A photo with no detected hand, or with BOTH hands detected (these are
    meant to be single-handshape photos; two hands means something in the
    frame is ambiguous), is skipped and recorded -- never silently
    zero-filled or guessed, unlike the video pipeline where a missing hand
    is a legitimate "not signing right now" frame.

    `progress_every` (0 = silent, the default -- existing callers,
    including tests, see no output change): with tens of thousands of
    real photos, this loop is the slow part (one real detector call per
    image, no other work) and prints nothing on its own, which reads as
    hung rather than working. build_dataset.py passes a real value so a
    long real run stays visibly alive.
    """
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[tuple[Path, str]] = []
    left_count = 0
    right_count = 0

    for index, sample in enumerate(samples, start=1):
        image_bgr = _read_image_bgr(sample.image_path)
        if image_bgr is None:
            skipped.append((sample.image_path, "unreadable image file"))
            continue

        result = extractor.extract(image_bgr)
        vector, skip_reason = hand_to_feature_vector(result.left_hand, result.right_hand)
        if vector is None:
            skipped.append((sample.image_path, skip_reason))
            continue

        if result.left_hand is not None:
            left_count += 1
        else:
            right_count += 1

        features.append(vector)
        labels.append(sample.label)

        if progress_every and index % progress_every == 0:
            print(f"  ...{index}/{len(samples)} images processed")

    return ExtractionResult(
        features=np.stack(features) if features else np.zeros((0, 63), dtype=np.float32),
        labels=labels,
        skipped=skipped,
        left_hand_count=left_count,
        right_hand_count=right_count,
    )
