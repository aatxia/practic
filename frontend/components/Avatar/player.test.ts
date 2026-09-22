import { describe, expect, it } from "vitest";
import { GlossPlayer, HOLD_SECONDS, TRANSITION_SECONDS } from "./player";
import { NEUTRAL_POSE } from "./poses";

describe("GlossPlayer", () => {
  it("reports finished with the neutral pose when nothing is playing", () => {
    const player = new GlossPlayer();
    const frame = player.update(0.016);

    expect(frame.finished).toBe(true);
    expect(frame.currentGloss).toBeNull();
    expect(frame.rotations).toEqual(NEUTRAL_POSE);
  });

  it("plays the first gloss immediately and reports it as current", () => {
    const player = new GlossPlayer();
    player.play(["PRIVIT", "TAK"]);

    const frame = player.update(0.01);
    expect(frame.finished).toBe(false);
    expect(frame.currentGloss).toBe("PRIVIT");
  });

  it("lerps toward the target pose during the transition window", () => {
    const player = new GlossPlayer();
    player.play(["TAK"]);

    const early = player.update(TRANSITION_SECONDS / 2);
    // NOD's target head.x is 0.35; halfway through the transition it should
    // be partway there, not already at the target.
    expect(early.rotations.head.x).toBeGreaterThan(0);
    expect(early.rotations.head.x).toBeLessThan(0.35);
  });

  it("holds at the target pose once the transition finishes", () => {
    const player = new GlossPlayer();
    player.play(["TAK"]);

    player.update(TRANSITION_SECONDS + 0.05);
    const held = player.update(0.05);
    expect(held.rotations.head.x).toBeCloseTo(0.35, 5);
  });

  it("advances to the next gloss after transition + hold elapses", () => {
    const player = new GlossPlayer();
    player.play(["TAK", "NI"]);

    player.update(TRANSITION_SECONDS + HOLD_SECONDS + 0.01);
    const frame = player.update(0.01);
    expect(frame.currentGloss).toBe("NI");
  });

  it("finishes after the last gloss's hold elapses", () => {
    const player = new GlossPlayer();
    player.play(["TAK"]);

    player.update(TRANSITION_SECONDS + HOLD_SECONDS + 0.01);
    const frame = player.update(0.01);
    expect(frame.finished).toBe(true);
    expect(frame.currentGloss).toBeNull();
  });

  it("lists glosses with no defined pose as unanimated, without dropping them", () => {
    const player = new GlossPlayer();
    player.play(["TAK", "UNKNOWN_GLOSS"]);

    expect(player.unanimatedGlosses).toEqual(["UNKNOWN_GLOSS"]);
    const frame = player.update(0.01);
    expect(frame.unanimatedGlosses).toEqual(["UNKNOWN_GLOSS"]);
  });

  it("holds neutral pose (not a guess) while playing a gloss with no pose", () => {
    const player = new GlossPlayer();
    player.play(["UNKNOWN_GLOSS"]);

    const frame = player.update(TRANSITION_SECONDS + 0.5);
    expect(frame.rotations).toEqual(NEUTRAL_POSE);
    expect(frame.currentGloss).toBe("UNKNOWN_GLOSS");
  });

  it("wobbles a waving pose's elbow around its target during the hold", () => {
    const player = new GlossPlayer();
    player.play(["PRIVIT"]);

    player.update(TRANSITION_SECONDS + 0.01);
    const a = player.update(0.1);
    const b = player.update(0.1);
    // WAVE's wobble oscillates rightElbow continuously, so consecutive
    // held frames should differ (it's not frozen at the static target).
    expect(a.rotations.rightElbow).not.toBeCloseTo(b.rotations.rightElbow, 5);
  });

  it("resets to neutral and restarts index/elapsed on a new play() call", () => {
    const player = new GlossPlayer();
    player.play(["TAK"]);
    player.update(TRANSITION_SECONDS + HOLD_SECONDS + 0.01);
    player.update(0.01);

    player.play(["NI"]);
    const frame = player.update(0.001);
    expect(frame.currentGloss).toBe("NI");
    expect(frame.rotations.head.x).toBeCloseTo(0, 5);
  });
});
