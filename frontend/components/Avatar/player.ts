/**
 * GlossPlayer — pure (no Three.js/DOM) sequencing logic for Avatar.tsx:
 * steps through a gloss sequence, holding each gloss's pose (poses.ts) for
 * a fixed duration with a short lerped transition in between, and reports
 * which glosses in the sequence have no defined pose at all (rather than
 * silently freezing or guessing one).
 */
import { type JointRotations, type Pose, NEUTRAL_POSE, poseForGloss } from "./poses";

export const TRANSITION_SECONDS = 0.35;
export const HOLD_SECONDS = 1.1;

export interface PlayerFrame {
  rotations: JointRotations;
  currentGloss: string | null;
  unanimatedGlosses: string[];
  finished: boolean;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpRotations(a: JointRotations, b: JointRotations, t: number): JointRotations {
  return {
    head: { x: lerp(a.head.x, b.head.x, t), y: lerp(a.head.y, b.head.y, t) },
    leftShoulder: {
      x: lerp(a.leftShoulder.x, b.leftShoulder.x, t),
      z: lerp(a.leftShoulder.z, b.leftShoulder.z, t),
    },
    leftElbow: lerp(a.leftElbow, b.leftElbow, t),
    rightShoulder: {
      x: lerp(a.rightShoulder.x, b.rightShoulder.x, t),
      z: lerp(a.rightShoulder.z, b.rightShoulder.z, t),
    },
    rightElbow: lerp(a.rightElbow, b.rightElbow, t),
  };
}

export class GlossPlayer {
  private sequence: string[] = [];
  private index = 0;
  private elapsed = 0;
  private fromPose: JointRotations;

  /**
   * Both defaults reproduce the original (procedural-puppet-only) behavior
   * exactly. A second puppet backend (e.g. a rigged model whose own "arms
   * relaxed" pose isn't all-zero, and whose gesture poses live in a
   * separate table) can supply its own neutral pose and lookup function
   * instead, reusing this class's timing/transition/wobble logic as-is.
   */
  constructor(
    private neutralPose: JointRotations = NEUTRAL_POSE,
    private poseLookup: (gloss: string) => Pose | null = poseForGloss,
  ) {
    this.fromPose = neutralPose;
  }

  play(sequence: string[]): void {
    this.sequence = sequence;
    this.index = 0;
    this.elapsed = 0;
    this.fromPose = this.neutralPose;
  }

  get unanimatedGlosses(): string[] {
    return this.sequence.filter((gloss) => this.poseLookup(gloss) === null);
  }

  update(deltaSeconds: number): PlayerFrame {
    if (this.index >= this.sequence.length) {
      return {
        rotations: this.neutralPose,
        currentGloss: null,
        unanimatedGlosses: this.unanimatedGlosses,
        finished: true,
      };
    }

    this.elapsed += deltaSeconds;
    const gloss = this.sequence[this.index]!;
    const pose = this.poseLookup(gloss);
    const target = pose?.target ?? this.neutralPose;

    let rotations: JointRotations;
    if (this.elapsed < TRANSITION_SECONDS) {
      rotations = lerpRotations(this.fromPose, target, this.elapsed / TRANSITION_SECONDS);
    } else if (pose?.wobble) {
      const holdElapsed = this.elapsed - TRANSITION_SECONDS;
      const offset = Math.sin(holdElapsed * pose.wobble.hz * 2 * Math.PI) * pose.wobble.amplitude;
      rotations = { ...target, [pose.wobble.joint]: target[pose.wobble.joint] + offset };
    } else {
      rotations = target;
    }

    if (this.elapsed >= TRANSITION_SECONDS + HOLD_SECONDS) {
      this.fromPose = target;
      this.index += 1;
      this.elapsed = 0;
    }

    return { rotations, currentGloss: gloss, unanimatedGlosses: this.unanimatedGlosses, finished: false };
  }
}
