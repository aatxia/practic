#!/usr/bin/env python3
"""
Records real labeled video clips of you signing words on your own laptop's
webcam, one gloss at a time, and writes them in the same SampleAnnotation
format ml/training/train.py already reads (ml/datasets/annotation.py) --
so recorded clips plug straight into the existing real training pipeline
(ml/preprocessing/landmarks.py + ml/training/train.py), no new training
code needed.

This is an interactive, camera-driving script -- it must be run locally,
on a machine with a real webcam and display, not in a sandboxed/headless
environment. Needs OpenCV (`pip install opencv-python`; already a
dependency of ml/preprocessing/).

Usage:
    python scripts/record_dataset.py --signer-id oksana --output-dir data/real

Controls during a session:
    SPACE   at the "ready" prompt: start recording the current take (after
            a 3s countdown). After a take is recorded: keep it and move on.
    r       at the "keep this take?" prompt: discard it and record again
            (does not count toward --repeats).
    n       at the "ready" prompt: skip the rest of this word's remaining
            takes and move to the next word.
    q       quit -- everything already saved (and its annotation row) stays
            on disk; re-running with the same --signer-id/--output-dir
            resumes (next_take_index never overwrites an existing clip).

Default word list (20 words) is the highest-leverage subset of the
existing rule-based grammar's vocabulary (ml/nlp/lexicon.py): pronouns,
common verbs/nouns, and the NOT/PAST grammar markers -- those two aren't
"words" on their own but unlock negated and past-tense sentences for
*every* verb already in the lexicon, not just their own class. Pass
--words to record a different/custom list instead (any gloss code from
ml/nlp/lexicon.py, or a new one -- display_label() falls back to printing
the raw code if it isn't in the lexicon yet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml.datasets.annotation import SampleAnnotation, load_annotations, write_annotations
from ml.nlp.lexicon import (
    ADVERBS,
    NEGATION_GLOSS,
    NOUNS,
    PRONOUNS,
    STANDALONE,
    TENSE_PAST_GLOSS,
    VERBS,
)

DEFAULT_WORDS = [
    "I", "YOU", "WE",
    "WANT", "LIKE", "HAVE", "GO", "EAT", "DRINK", "WORK",
    "WATER", "BREAD", "BOOK", "HOUSE", "FRIEND", "CAR",
    NEGATION_GLOSS, TENSE_PAST_GLOSS,
    "TODAY", "YESTERDAY",
]

COUNTDOWN_SECONDS = 3


def display_label(gloss: str) -> str:
    """A short Ukrainian citation form to show on screen while recording --
    not a claim this IS the sign for the word (that's for the person
    signing to know), just a readable prompt of what they're recording."""
    if gloss in PRONOUNS:
        return PRONOUNS[gloss].lemma
    if gloss in VERBS:
        return VERBS[gloss].infinitive
    if gloss in NOUNS:
        return NOUNS[gloss].cases["nominative"]
    if gloss in ADVERBS:
        return ADVERBS[gloss].text
    if gloss in STANDALONE:
        return STANDALONE[gloss].text
    if gloss == NEGATION_GLOSS:
        return "не (заперечення)"
    if gloss == TENSE_PAST_GLOSS:
        return "минулий час (граматичний маркер)"
    return gloss


def next_take_index(output_dir: Path, signer_id: str, gloss: str) -> int:
    """The next unused take number for this signer+gloss -- lets a session
    be stopped (q) and resumed later without overwriting earlier
    recordings, and lets a discarded retake's slot be reused immediately."""
    clip_dir = output_dir / "raw" / signer_id
    existing = list(clip_dir.glob(f"{gloss}_*.mp4")) if clip_dir.exists() else []
    used = set()
    for path in existing:
        suffix = path.stem.removeprefix(f"{gloss}_")
        if suffix.isdigit():
            used.add(int(suffix))
    take = 0
    while take in used:
        take += 1
    return take


def build_annotation(
    clip_path: Path, signer_id: str, gloss: str, take: int, frame_count: int, fps: float
) -> SampleAnnotation:
    return SampleAnnotation(
        sample_id=f"{signer_id}_{gloss}_{take:02d}",
        clip_path=clip_path.as_posix(),  # forward slashes even on Windows -- portable back-reading on any OS
        signer_id=signer_id,
        gloss=gloss,
        start_frame=0,
        end_frame=frame_count,
        fps=fps,
        source="real_webcam",
    )


def _blank_frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _record_clip(capture, writer, label: str, seconds: float, fps: float, window: str) -> int:
    """Records for `seconds`, showing a live preview with a countdown
    burned into the frame so the clip is self-documenting even if the
    annotation file is later lost. Returns the number of frames written."""
    import cv2

    total_frames = max(1, int(seconds * fps))
    frame_count = 0
    while frame_count < total_frames:
        ok, frame = capture.read()
        if not ok:
            break
        remaining = seconds - frame_count / fps
        cv2.putText(frame, f"{label}  {remaining:0.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        writer.write(frame)
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        frame_count += 1
    return frame_count


def _countdown(capture, label: str, seconds: int, window: str, width: int, height: int) -> None:
    import cv2

    for remaining in range(seconds, 0, -1):
        ok, frame = capture.read()
        if not ok:
            frame = _blank_frame(width, height)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"{remaining}...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(window, frame)
        cv2.waitKey(1000)


def record_session(
    signer_id: str,
    output_dir: Path,
    words: list[str],
    repeats: int,
    clip_seconds: float,
    camera_index: int,
) -> None:
    import cv2

    window = "record_dataset (q=quit, n=skip word, SPACE=record/keep, r=retake)"

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. This script needs a real "
            "webcam and display -- it can't run in a headless/sandboxed environment."
        )
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    clip_dir = output_dir / "raw" / signer_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "annotations" / f"{signer_id}_annotations.jsonl"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(annotations_path) if annotations_path.exists() else []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"Signer: {signer_id} | {len(words)} words x {repeats} repeats | camera {width}x{height}@{fps:.0f}fps")
    print(f"Output: {output_dir}\n")

    try:
        for gloss in words:
            label = display_label(gloss)
            rep = 0
            while rep < repeats:
                take = next_take_index(output_dir, signer_id, gloss)
                ok, frame = capture.read()
                if not ok:
                    frame = _blank_frame(width, height)
                cv2.putText(
                    frame,
                    f"{label} -- take {rep + 1}/{repeats} -- SPACE=record n=skip q=quit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window, frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord("n"):
                    break
                if key != ord(" "):
                    continue

                _countdown(capture, label, COUNTDOWN_SECONDS, window, width, height)

                clip_path = clip_dir / f"{gloss}_{take:02d}.mp4"
                writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                frame_count = _record_clip(capture, writer, label, clip_seconds, fps, window)
                writer.release()

                if frame_count == 0:
                    clip_path.unlink(missing_ok=True)
                    print(f"[{gloss}] take {take}: aborted, nothing saved")
                    continue

                annotation = build_annotation(clip_path, signer_id, gloss, take, frame_count, fps)
                annotations.append(annotation)
                write_annotations(annotations_path, annotations)
                print(f"[{gloss}] take {take}: saved {clip_path.name} ({frame_count} frames)")

                ok, frame = capture.read()
                if not ok:
                    frame = _blank_frame(width, height)
                cv2.putText(frame, "Keep? SPACE=keep  r=retake", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(window, frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("r"):
                    annotations.pop()
                    write_annotations(annotations_path, annotations)
                    clip_path.unlink(missing_ok=True)
                    print(f"[{gloss}] take {take}: discarded, retaking")
                    continue

                rep += 1
    except KeyboardInterrupt:
        print("\nStopped early -- everything saved so far is kept.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Annotations: {annotations_path}")
    print(
        "Next (once you have enough real clips across a few signers):\n"
        f"  python -m ml.training.train --annotations {annotations_path} "
        f"--dataset-root {output_dir} --config configs/model.yaml --experiment-name real_v1"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signer-id", required=True, help="Your name/id -- used for the signer-independent split later")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real"), help="Dataset root (default: data/real)")
    parser.add_argument(
        "--words", nargs="+", default=None, help="Gloss codes to record (default: 20-word starter set, see module docstring)"
    )
    parser.add_argument("--repeats", type=int, default=15, help="Recordings per word (default: 15)")
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args(argv)

    record_session(
        signer_id=args.signer_id,
        output_dir=args.output_dir,
        words=args.words or DEFAULT_WORDS,
        repeats=args.repeats,
        clip_seconds=args.clip_seconds,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()
