#!/usr/bin/env python3
"""
collect_dactyl_photos -- guided interactive photo capture for the Ukrainian
dactyl (fingerspelling) alphabet letters still missing a real directory
under data/USL_alphabet_train/ (see ml/nlp/fingerspelling.py's
UKRAINIAN_ALPHABET for the full 33-letter set this dataset covers).

This is a targeted follow-up to the existing 37k-photo dataset (already
covering 26 of 33 letters), not a replacement for it -- scraping images off
the internet was considered and rejected (wrong alphabet variants, posed
illustrations instead of real hands, copyright/licensing risk, and the same
fabricated-data trap this project's "NO FAKE AI" rule exists to avoid).
Every photo this script saves is a real picture of your own hand, verified
on the spot through the SAME MediaPipe hand detector + hand_to_feature_
vector() normalization the offline dataset builder (ml/fingerspelling/
build_dataset.py) and the live serving path (backend/app/services/
fingerspelling_service.py) both already use -- so a bad capture (no hand,
both hands, blurry) never silently becomes a training example.

This is an interactive, camera-driving script -- run it locally, on a
machine with a real webcam and display, not in a sandboxed/headless
environment. Needs OpenCV (`pip install opencv-python`) and the MediaPipe
model files (`scripts/download_mediapipe_models.sh`).

Usage (from repo root):
    python scripts/collect_dactyl_photos.py

Controls:
    At the letter menu -- type a number, or press Enter to quit.
    SPACE   at the "ready" prompt: capture a photo (after a 3s countdown).
    After a photo is captured:
        - if hand detection looks fine: SPACE=keep, r=retake.
        - if hand detection failed (no hand / both hands): SPACE=save
          anyway (only if you believe the check is wrong), r=retake
          (recommended).
    q       at the "ready" prompt: stop this letter and go back to the menu.

Never overwrites an existing photo -- resuming later (same --output-dir)
picks up exactly where you left off, same guarantee as
scripts/collect_dynamic_signs.py gives for video takes.

Does NOT rebuild features or retrain on its own -- prints the exact
follow-up commands at the end and stops, matching this project's
convention that a training step only ever runs on data a person has
reviewed, never automatically.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.fingerspelling.dataset import hand_to_feature_vector
from ml.nlp.fingerspelling import UKRAINIAN_ALPHABET
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor
from scripts.record_dataset import _blank_frame, _countdown

TARGET_PHOTOS_PER_LETTER = 20
COUNTDOWN_SECONDS = 3

# The same literal ml.nlp.fingerspelling.UKRAINIAN_ALPHABET is built from --
# duplicated here only so the letter menu has a stable, linguistically
# correct alphabetical order (a frozenset's iteration order isn't
# guaranteed to match it). test_collect_dactyl_photos_script.py asserts
# this stays in sync with the real UKRAINIAN_ALPHABET set.
_ALPHABET_ORDER = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"


def existing_letters(data_root: Path) -> set[str]:
    """Real letter directories already present under data_root -- any
    directory not starting with "_" (see ml/fingerspelling/dataset.py's
    _excluded_non_ukrainian/_excluded_unclear exclusion convention)."""
    if not data_root.exists():
        return set()
    return {entry.name for entry in data_root.iterdir() if entry.is_dir() and not entry.name.startswith("_")}


def missing_letters(data_root: Path) -> list[str]:
    """Ukrainian alphabet letters (uppercase, matching the existing
    directory naming convention) with no directory yet under data_root --
    in real alphabet order, not directory-listing order."""
    present = existing_letters(data_root)
    return [char.upper() for char in _ALPHABET_ORDER if char.upper() not in present]


def count_existing_photos(data_root: Path, letter: str) -> int:
    letter_dir = data_root / letter
    if not letter_dir.exists():
        return 0
    return len([path for path in letter_dir.iterdir() if path.is_file()])


def choose_letter(data_root: Path, letters: list[str], target: int) -> str | None:
    print("\nОбери літеру для фотографування (Enter -- вийти):")
    for i, letter in enumerate(letters, start=1):
        count = count_existing_photos(data_root, letter)
        status = f"{count}/{target} (готово)" if count >= target else f"{count}/{target}"
        print(f"  {i:2d}. {letter} -- {status}")
    choice = input("Номер літери: ").strip()
    if not choice:
        return None
    try:
        index = int(choice)
    except ValueError:
        print("Не розпізнано число, спробуй ще раз.")
        return choose_letter(data_root, letters, target)
    if not (1 <= index <= len(letters)):
        print("Немає такого номера в списку.")
        return choose_letter(data_root, letters, target)
    return letters[index - 1]


def next_photo_path(data_root: Path, letter: str) -> Path:
    """A timestamp-based filename matching the existing dataset's
    IMG_%Y%m%d_%H%M%S.jpg convention (see data/USL_alphabet_train/<letter>/
    for the existing photos this mirrors) -- never overwrites an existing
    file, falling back to a numeric suffix on a same-second collision."""
    letter_dir = data_root / letter
    letter_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = letter_dir / f"IMG_{stamp}.jpg"
    suffix = 1
    while candidate.exists():
        candidate = letter_dir / f"IMG_{stamp}_{suffix}.jpg"
        suffix += 1
    return candidate


def _save_photo(path: Path, frame) -> None:
    """cv2.imwrite silently fails for non-ASCII (Cyrillic) paths on Windows
    -- the write-side mirror of ml/fingerspelling/dataset.py's
    _read_image_bgr fix. Encoding in memory and writing the bytes via
    pathlib (which handles Windows Unicode paths correctly) works
    cross-platform."""
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Could not encode photo for {path}")
    path.write_bytes(buffer.tobytes())


def _should_keep_capturing(existing: int, target: int, started_above_target: bool) -> bool:
    """Whether the capture loop should take one more photo.

    Normally stops once `existing` reaches `target`. But when a session
    STARTS already at/above target -- e.g. re-shooting a letter whose
    existing photos turned out unusable (Ь had 1500 tiny 100x56 grayscale
    frames, all rejected by hand_to_feature_vector -- see
    ml/fingerspelling/features.skipped.txt) -- the on-disk count no longer
    means "done". Without this, choosing that letter would exit the
    capture loop immediately, before the camera window even opened, with
    zero chance to take a single new photo. Once a session starts above
    target, only `q` ends it, never the count."""
    return started_above_target or existing < target


def capture_letter_session(
    data_root: Path,
    letter: str,
    target: int,
    camera_index: int,
    mediapipe_models_dir: str,
) -> None:
    import cv2

    window = f"collect_dactyl_photos -- {letter} (q=stop letter, SPACE=capture/keep, r=retake)"

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. This script needs a real "
            "webcam and display -- it can't run in a headless/sandboxed environment."
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    extractor = LandmarkExtractor(
        model_dir=mediapipe_models_dir, features=FeatureToggles(hands=True, pose=False, face=False)
    )

    try:
        existing = count_existing_photos(data_root, letter)
        started_above_target = existing >= target
        while _should_keep_capturing(existing, target, started_above_target):
            ok, frame = capture.read()
            if not ok:
                frame = _blank_frame(width, height)
            cv2.putText(
                frame,
                f"{letter} -- {existing}/{target} -- SPACE=capture q=stop letter",
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

            _countdown(capture, letter, COUNTDOWN_SECONDS, window, width, height)

            ok, photo = capture.read()
            if not ok:
                print(f"[{letter}]: camera read failed, try again")
                continue

            result = extractor.extract(photo)
            vector, skip_reason = hand_to_feature_vector(result.left_hand, result.right_hand)

            if vector is None:
                prompt_frame = photo.copy()
                cv2.putText(
                    prompt_frame,
                    f"No good: {skip_reason} -- SPACE=save anyway  r=retake",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(window, prompt_frame)
                key = cv2.waitKey(0) & 0xFF
                if key != ord(" "):
                    print(f"[{letter}]: discarded ({skip_reason}), retaking")
                    continue
            else:
                preview = photo.copy()
                cv2.putText(preview, "Keep? SPACE=keep  r=retake", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("r"):
                    print(f"[{letter}]: discarded, retaking")
                    continue

            photo_path = next_photo_path(data_root, letter)
            _save_photo(photo_path, photo)
            existing += 1
            print(f"[{letter}] saved {photo_path.name} -- {existing}/{target} photos for this letter")
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/USL_alphabet_train"),
        help="Dataset root (default: data/USL_alphabet_train)",
    )
    parser.add_argument(
        "--letters", nargs="+", default=None,
        help="Letters to offer in the menu (default: whichever of the 33 Ukrainian letters have no directory yet)",
    )
    parser.add_argument("--target", type=int, default=TARGET_PHOTOS_PER_LETTER, help="Photos per letter (default: 20)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mediapipe-models-dir", default="models/mediapipe")
    args = parser.parse_args(argv)

    letters = args.letters or missing_letters(args.output_dir)
    if not letters:
        print("Немає літер без фото -- усі 33 літери вже мають директорію.")
        return

    print(f"Output: {args.output_dir.resolve()} | Target: {args.target}/letter | Letters: {', '.join(letters)}")

    try:
        while True:
            letter = choose_letter(args.output_dir, letters, args.target)
            if letter is None:
                break
            capture_letter_session(
                data_root=args.output_dir,
                letter=letter,
                target=args.target,
                camera_index=args.camera_index,
                mediapipe_models_dir=args.mediapipe_models_dir,
            )
    except KeyboardInterrupt:
        print("\nStopped early -- every already-saved photo stays on disk.")
        return

    print("\nDone. Next (once you have enough real photos):")
    print(
        f"  python -m ml.fingerspelling.build_dataset --data-root {args.output_dir} "
        "--models-dir models/mediapipe --output ml/fingerspelling/features.npz"
    )
    print("  python -m ml.fingerspelling.train")


if __name__ == "__main__":
    main()
