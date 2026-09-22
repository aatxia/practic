"""
normalization — makes raw MediaPipe landmark coordinates invariant to the
signer's position, distance from the camera, and body size (section 7).

Each `normalize_*` function centers landmarks on an anatomically meaningful
reference point and rescales by an anatomically meaningful distance, so the
same sign produces (approximately) the same feature vector whether the
signer is close to the camera or far, left of frame or centered, tall or
short.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.preprocessing.landmarks import (
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    FrameLandmarks,
)

# MediaPipe Pose landmark indices (33-point topology)
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# MediaPipe Hand landmark indices (21-point topology)
HAND_WRIST = 0
HAND_MIDDLE_FINGER_MCP = 9

_EPS = 1e-6


def _safe_scale(value: float) -> float:
    """Avoid division by ~0 when landmarks are degenerate (e.g. a single point)."""
    return value if value > _EPS else 1.0


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Center on the wrist, scale by wrist-to-middle-finger-MCP distance."""
    wrist = landmarks[HAND_WRIST]
    centered = landmarks - wrist
    scale = _safe_scale(float(np.linalg.norm(centered[HAND_MIDDLE_FINGER_MCP])))
    return centered / scale


def normalize_pose(landmarks: np.ndarray) -> np.ndarray:
    """Center on the shoulder midpoint, scale by shoulder width -- makes the
    representation invariant to torso position and distance from camera."""
    left_shoulder = landmarks[POSE_LEFT_SHOULDER]
    right_shoulder = landmarks[POSE_RIGHT_SHOULDER]
    center = (left_shoulder + right_shoulder) / 2.0
    centered = landmarks - center
    scale = _safe_scale(float(np.linalg.norm(left_shoulder - right_shoulder)))
    return centered / scale


def normalize_face(landmarks: np.ndarray) -> np.ndarray:
    """Center on the centroid of all face landmarks, scale by the max
    distance from centroid to any landmark (a proxy for face size in frame)."""
    center = landmarks.mean(axis=0)
    centered = landmarks - center
    scale = _safe_scale(float(np.linalg.norm(centered, axis=1).max()))
    return centered / scale


@dataclass
class NormalizedFrame:
    """Fixed-shape, zero-filled-when-absent normalized landmarks for one frame,
    ready to feed into a feature vector builder (ml/features/, Phase 7).

    `present` records which modalities were actually detected, so the model
    (or a debug overlay, section 41) can distinguish "hand at the origin"
    from "no hand detected" even though both are represented as zeros.
    """

    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray
    present: dict[str, bool]


def normalize_frame(frame: FrameLandmarks) -> NormalizedFrame:
    left_hand = (
        normalize_hand(frame.left_hand)
        if frame.left_hand is not None
        else np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    )
    right_hand = (
        normalize_hand(frame.right_hand)
        if frame.right_hand is not None
        else np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    )
    pose = (
        normalize_pose(frame.pose)
        if frame.pose is not None
        else np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    )
    face = (
        normalize_face(frame.face)
        if frame.face is not None
        else np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32)
    )

    return NormalizedFrame(
        left_hand=left_hand,
        right_hand=right_hand,
        pose=pose,
        face=face,
        present={
            "left_hand": frame.left_hand is not None,
            "right_hand": frame.right_hand is not None,
            "pose": frame.pose is not None,
            "face": frame.face is not None,
        },
    )
