/**
 * Gesture targets for the rigged Cesium Man model (riggedPuppet.ts),
 * expressed in the SAME `JointRotations` shape poses.ts uses for the
 * procedural puppet -- reused here purely as a convenient container, not
 * because the numbers mean the same thing on this rig. GlossPlayer lerps
 * between whichever pose table it's constructed with; the field values
 * below are fed to riggedPuppet.ts's `applyWorldSwing()`, not to the
 * procedural puppet.ts.
 *
 * These numbers were NOT guessed: this rig's bones (Cesium Man, CC BY 4.0,
 * github.com/KhronosGroup/glTF-Sample-Assets) have their own local axis
 * conventions, unrelated to the procedural puppet's. Each value below was
 * determined empirically -- driving one bone at a time through a headless
 * Chromium render and reading the resulting WORLD-SPACE wrist coordinates
 * (not eyeballing screenshots, which turned out ambiguous for this rig) to
 * confirm which axis/sign actually raises, lowers, or swings each arm --
 * see PROJECT_STATUS.md for the full process. Same "no fake AI" principle
 * as poses.ts: still hand-authored demo gestures, not real УЖМ.
 */
import type { Pose } from "./poses";

// Bind pose (T-pose) has both arms held straight out to the sides. A
// world-Z swing of about 1 radian (sign mirrored per side) brings the arm
// down to a relaxed hang -- this is the rigged puppet's own "neutral",
// analogous to poses.ts's all-zero NEUTRAL_POSE but for a rig whose bind
// pose isn't already relaxed.
const RIG_NEUTRAL: Pose["target"] = {
  head: { x: 0, y: 0 },
  leftShoulder: { x: 0, z: -1.0 },
  leftElbow: 0,
  rightShoulder: { x: 0, z: 1.0 },
  rightElbow: 0,
};

export const RIG_NEUTRAL_POSE: Pose["target"] = RIG_NEUTRAL;

const RIG_WAVE: Pose = {
  target: {
    ...RIG_NEUTRAL,
    rightShoulder: { x: 2.0, z: 2.0 },
    rightElbow: 1.6,
  },
  wobble: { joint: "rightElbow", amplitude: 0.35, hz: 1.5 },
};

const RIG_NOD: Pose = { target: { ...RIG_NEUTRAL, head: { x: 0.3, y: 0 } } };

const RIG_SHAKE: Pose = { target: { ...RIG_NEUTRAL, head: { x: 0, y: 0.3 } } };

const RIG_HANDS_TOGETHER: Pose = {
  target: {
    ...RIG_NEUTRAL,
    head: { x: 0.1, y: 0 },
    leftShoulder: { x: 0.3, z: -2.7 },
    leftElbow: 1.8,
    rightShoulder: { x: 0.3, z: 2.7 },
    rightElbow: 1.8,
  },
};

/** Same gloss keys as poses.ts's GLOSS_POSES -- every gloss animatable on
 * the procedural puppet is animatable here too, so switching backends
 * never silently loses a gesture. */
export const RIGGED_GLOSS_POSES: Record<string, Pose> = {
  PRIVIT: RIG_WAVE,
  TAK: RIG_NOD,
  NI: RIG_SHAKE,
  DYAKUYU: RIG_HANDS_TOGETHER,
  BUD_LASKA: RIG_HANDS_TOGETHER,
};

export function riggedPoseForGloss(gloss: string): Pose | null {
  return RIGGED_GLOSS_POSES[gloss] ?? null;
}
