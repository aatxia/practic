#!/usr/bin/env bash
# Downloads pretrained MediaPipe Tasks model bundles (hand/pose/face
# landmarkers) into models/mediapipe/. These are Google-provided, Apache 2.0
# licensed model bundles -- an external ML dependency this project does not
# train itself (see docs/architecture.md).
#
# Official source: https://storage.googleapis.com/mediapipe-models/...
# If that host is blocked/unreachable in your network, hand/face fall back
# to a community GitHub mirror automatically. There is currently no known
# mirror for pose_landmarker.task -- if the official download fails, get it
# manually from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# and place it at models/mediapipe/pose_landmarker.task.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/mediapipe"
mkdir -p "$OUT_DIR"

download() {
  local name="$1" official_url="$2" mirror_url="${3:-}"
  local out="$OUT_DIR/$name"

  if [ -f "$out" ]; then
    echo "✓ $name already present, skipping"
    return 0
  fi

  echo "Downloading $name from official source..."
  if curl -fsSL -o "$out" "$official_url"; then
    echo "✓ $name downloaded (official source)"
    return 0
  fi

  if [ -n "$mirror_url" ]; then
    echo "  official source unreachable, trying GitHub mirror..."
    if curl -fsSL -o "$out" "$mirror_url"; then
      echo "✓ $name downloaded (GitHub mirror)"
      return 0
    fi
  fi

  echo "✗ Failed to download $name automatically." >&2
  rm -f "$out"
  return 1
}

download hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
  "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task"

download face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
  "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/face_landmarker.task"

download pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
  || echo "  -> see the manual instructions in this script's header comment"

echo ""
echo "MediaPipe models ready in: $OUT_DIR"
ls -la "$OUT_DIR"
