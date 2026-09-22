#!/usr/bin/env python3
"""
collect_dynamic_signs — guided interactive recorder for the dynamic-word
vocabulary defined in ml/nlp/vocabulary.py (the 7 already-trained real_v1
classes, plus the 8 next classes proposed for real_v2 -- see
PROJECT_STATUS.md's continuous-dictation vocabulary-expansion phase).

Extends scripts/record_dataset.py's recording mechanics (countdown, take
numbering, resumability, annotation writing -- reused directly here, not
reimplemented) with two things that script doesn't have:

  - an interactive per-word menu, showing how many real recordings each
    word already has vs. the ~20/class target, so you can freely pick
    which word to record next instead of walking a fixed list in order;
  - real landmark-presence verification after each take: the just-
    recorded clip is immediately re-decoded through the SAME MediaPipe
    hand-detection pipeline production uses, and a take with too little
    detected hand is flagged as likely unusable, defaulting to a retake
    instead of silently saving a broken recording.

This is an interactive, camera-driving script -- run it locally, on a
machine with a real webcam and display, not in a sandboxed/headless
environment. Needs OpenCV (`pip install opencv-python`) and the MediaPipe
model files (`scripts/download_mediapipe_models.sh`).

Usage (from repo root):
    python scripts/collect_dynamic_signs.py --signer-id <your name>

Controls:
    At the word menu -- type a number, or press Enter to quit.
    SPACE   at the "ready" prompt: start recording the current take (after
            a 3s countdown).
    After a take is recorded:
        - if hand detection looks fine: SPACE=keep, r=retake.
        - if hand detection was too low: SPACE=keep anyway (only if you
          believe the check is wrong), r=retake (recommended).
    q       at the "ready" prompt: stop recording this word and go back
            to the word menu.

Never overwrites an existing take -- resuming later (same --signer-id/
--output-dir) picks up exactly where you left off, same guarantee as
scripts/record_dataset.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.annotation import load_annotations, write_annotations
from ml.nlp.vocabulary import TARGET_DYNAMIC_VOCABULARY
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor
from scripts.record_dataset import (
    COUNTDOWN_SECONDS,
    _blank_frame,
    _countdown,
    _record_clip,
    build_annotation,
    display_label,
    next_take_index,
)

TARGET_RECORDINGS_PER_WORD = 20
# Calibrated against real data, not guessed: measured this same check
# against 8 of the already-successful real_v1 training clips (data/real/
# raw/aatxia/*.mp4) -- hand-detection rate over the FULL recorded clip
# (including the natural pre-roll before the sign starts and pause after
# it ends, both intentionally padded into every take) ranged 11%-42%,
# never near 100% even for clips that trained a 92%+-accuracy checkpoint.
# Set well below that real range's floor so a genuinely broken take
# (camera pointed away, hand never in frame, capture fired by accident --
# closer to 0%) gets caught without ever flagging a normal, good
# recording just because it has the padding every take is expected to
# have.
MIN_HAND_DETECTION_RATE = 0.08


def count_existing_takes(output_dir: Path, signer_id: str, gloss: str) -> int:
    clip_dir = output_dir / "raw" / signer_id
    if not clip_dir.exists():
        return 0
    return len(list(clip_dir.glob(f"{gloss}_*.mp4")))


def verify_landmarks(clip_path: Path, extractor: LandmarkExtractor) -> tuple[float, int]:
    """Re-decodes the just-recorded clip through the real MediaPipe hand
    detector (the same one training/live inference use) and returns
    (hand_detection_rate, frame_count) -- a real measurement of the actual
    saved file, not an assumption from recording parameters alone."""
    import cv2

    capture = cv2.VideoCapture(str(clip_path))
    total = 0
    with_hand = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        total += 1
        result = extractor.extract(frame)
        if result.left_hand is not None or result.right_hand is not None:
            with_hand += 1
    capture.release()
    if total == 0:
        return 0.0, 0
    return with_hand / total, total


def choose_word(output_dir: Path, signer_id: str, words: list[str], target: int) -> str | None:
    print("\nОбери слово для запису (Enter -- вийти):")
    for i, gloss in enumerate(words, start=1):
        count = count_existing_takes(output_dir, signer_id, gloss)
        # Always shows the real number, even once the target is reached --
        # "готово" alone with no number was harder to sanity-check/debug
        # against what's actually on disk.
        status = f"{count}/{target} (готово)" if count >= target else f"{count}/{target}"
        print(f"  {i:2d}. {display_label(gloss):20s} ({gloss:10s}) -- {status}")
    choice = input("Номер слова: ").strip()
    if not choice:
        return None
    try:
        index = int(choice)
    except ValueError:
        print("Не розпізнано число, спробуй ще раз.")
        return choose_word(output_dir, signer_id, words, target)
    if not (1 <= index <= len(words)):
        print("Немає такого номера в списку.")
        return choose_word(output_dir, signer_id, words, target)
    return words[index - 1]


def record_word_session(
    signer_id: str,
    output_dir: Path,
    gloss: str,
    target: int,
    clip_seconds: float,
    camera_index: int,
    mediapipe_models_dir: str,
) -> None:
    import cv2

    label = display_label(gloss)
    window = f"collect_dynamic_signs -- {label} (q=stop word, SPACE=record/keep, r=retake)"

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. This script needs a real "
            "webcam and display -- it can't run in a headless/sandboxed environment."
        )
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    extractor = LandmarkExtractor(
        model_dir=mediapipe_models_dir, features=FeatureToggles(hands=True, pose=False, face=False)
    )

    clip_dir = output_dir / "raw" / signer_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "annotations" / f"{signer_id}_annotations.jsonl"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(annotations_path) if annotations_path.exists() else []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        existing = count_existing_takes(output_dir, signer_id, gloss)
        while existing < target:
            take = next_take_index(output_dir, signer_id, gloss)
            ok, frame = capture.read()
            if not ok:
                frame = _blank_frame(width, height)
            cv2.putText(
                frame,
                f"{label} -- {existing}/{target} -- SPACE=record q=stop word",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window, frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
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

            hand_rate, verified_frames = verify_landmarks(clip_path, extractor)
            print(
                f"[{gloss}] take {take}: {frame_count} frames recorded, hand detected in "
                f"{hand_rate:.0%} of {verified_frames} decoded frames"
            )

            if hand_rate < MIN_HAND_DETECTION_RATE:
                prompt_frame = _blank_frame(width, height)
                cv2.putText(
                    prompt_frame,
                    f"Low hand detection ({hand_rate:.0%}) -- SPACE=keep anyway  r=retake",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(window, prompt_frame)
                key = cv2.waitKey(0) & 0xFF
                if key != ord(" "):
                    clip_path.unlink(missing_ok=True)
                    print(f"[{gloss}] take {take}: discarded (low hand-detection rate), retaking")
                    continue
            else:
                ok, frame = capture.read()
                if not ok:
                    frame = _blank_frame(width, height)
                cv2.putText(frame, "Keep? SPACE=keep  r=retake", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(window, frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("r"):
                    clip_path.unlink(missing_ok=True)
                    print(f"[{gloss}] take {take}: discarded, retaking")
                    continue

            annotation = build_annotation(clip_path, signer_id, gloss, take, frame_count, fps)
            annotations.append(annotation)
            write_annotations(annotations_path, annotations)
            existing += 1
            print(f"[{gloss}] saved -- {existing}/{target} recordings for this word")
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signer-id", required=True, help="Your name/id -- used for the signer-independent split later")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real"), help="Dataset root (default: data/real)")
    parser.add_argument(
        "--words", nargs="+", default=None,
        help="Gloss codes to offer in the menu (default: the full 15-word target vocabulary from ml/nlp/vocabulary.py)",
    )
    parser.add_argument("--target", type=int, default=TARGET_RECORDINGS_PER_WORD, help="Recordings per word (default: 20)")
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mediapipe-models-dir", default="models/mediapipe")
    args = parser.parse_args(argv)

    words = args.words or list(TARGET_DYNAMIC_VOCABULARY)

    # Resolved, absolute path -- printed so it's obvious which directory is
    # actually being scanned/written to, not just the (possibly relative,
    # possibly surprising depending on the current working directory)
    # argument as typed.
    print(f"Signer: {args.signer_id} | Output: {args.output_dir.resolve()} | Target: {args.target}/word")

    try:
        while True:
            gloss = choose_word(args.output_dir, args.signer_id, words, args.target)
            if gloss is None:
                break
            record_word_session(
                signer_id=args.signer_id,
                output_dir=args.output_dir,
                gloss=gloss,
                target=args.target,
                clip_seconds=args.clip_seconds,
                camera_index=args.camera_index,
                mediapipe_models_dir=args.mediapipe_models_dir,
            )
    except KeyboardInterrupt:
        print("\nStopped early -- everything already saved (and its annotation row) stays on disk.")
        return

    print("\nDone. Next (once you have enough real clips):")
    print(
        f"  python -m ml.training.train --annotations {args.output_dir}/annotations/{args.signer_id}_annotations.jsonl "
        f"--dataset-root {args.output_dir} --experiment-name real_v2"
    )


if __name__ == "__main__":
    main()
