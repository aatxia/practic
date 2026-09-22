/**
 * Phase 15: pose definitions for the placeholder 3D avatar.
 *
 * IMPORTANT (same "no fake AI" rule as the rest of this project): these are
 * NOT real Ukrainian Sign Language handshapes/movements. No motion-capture
 * or reference data for real УЖМ signs exists in this project (Phase 8: no
 * public annotated dataset was found). Each pose here is a simple,
 * hand-authored generic gesture (a wave, a nod, a head shake, hands brought
 * together) loosely evocative of its gloss's meaning for demo purposes --
 * never presented as linguistically accurate signing. See Avatar.tsx, which
 * renders a visible "DEMO" notice whenever it's playing.
 */

export interface JointRotations {
  head: { x: number; y: number };
  leftShoulder: { x: number; z: number };
  leftElbow: number;
  rightShoulder: { x: number; z: number };
  rightElbow: number;
}

export const NEUTRAL_POSE: JointRotations = {
  head: { x: 0, y: 0 },
  leftShoulder: { x: 0, z: 0 },
  leftElbow: 0,
  rightShoulder: { x: 0, z: 0 },
  rightElbow: 0,
};

export interface Pose {
  /** Target joint rotations (radians) held during this pose. */
  target: JointRotations;
  /** Optional joint that oscillates around the target while held, so the
   * pose reads as a motion (e.g. a wave) rather than a frozen stance. */
  wobble?: {
    joint: "rightElbow" | "leftElbow";
    amplitude: number;
    hz: number;
  };
}

const WAVE: Pose = {
  target: {
    head: { x: 0, y: 0 },
    leftShoulder: { x: 0, z: 0 },
    leftElbow: 0,
    rightShoulder: { x: -1.4, z: 0.3 },
    rightElbow: -1.2,
  },
  wobble: { joint: "rightElbow", amplitude: 0.35, hz: 1.5 },
};

const NOD: Pose = {
  target: { ...NEUTRAL_POSE, head: { x: 0.35, y: 0 } },
  wobble: undefined,
};

const SHAKE: Pose = {
  target: { ...NEUTRAL_POSE, head: { x: 0, y: 0.4 } },
  wobble: undefined,
};

const HANDS_TOGETHER: Pose = {
  target: {
    head: { x: 0.1, y: 0 },
    leftShoulder: { x: -0.9, z: -0.5 },
    leftElbow: -1.3,
    rightShoulder: { x: -0.9, z: 0.5 },
    rightElbow: -1.3,
  },
};

/** gloss -> named demo pose. Only the Phase 8 demo-dataset standalone
 * glosses have one; anything else has no defined animation (Avatar.tsx
 * holds NEUTRAL_POSE and says so, rather than guessing a gesture). */
export const GLOSS_POSES: Record<string, Pose> = {
  PRIVIT: WAVE,
  TAK: NOD,
  NI: SHAKE,
  DYAKUYU: HANDS_TOGETHER,
  BUD_LASKA: HANDS_TOGETHER,
};

export function poseForGloss(gloss: string): Pose | null {
  return GLOSS_POSES[gloss] ?? null;
}
