import { describe, expect, it } from "vitest";
import { NEUTRAL_POSE } from "./poses";
import { applyRotations, buildPuppet } from "./puppet";

describe("buildPuppet", () => {
  it("builds a scene graph with a mesh under the root for each body part", () => {
    const puppet = buildPuppet();

    expect(puppet.root.children.length).toBeGreaterThan(0);
    // head, leftShoulder, rightShoulder must all be descendants of root.
    expect(puppet.root.children).toContain(puppet.head);
    expect(puppet.root.children).toContain(puppet.leftShoulder);
    expect(puppet.root.children).toContain(puppet.rightShoulder);
  });

  it("nests the elbow pivot under its shoulder pivot (joint hierarchy)", () => {
    const puppet = buildPuppet();

    expect(puppet.leftShoulder.children).toContain(puppet.leftElbow);
    expect(puppet.rightShoulder.children).toContain(puppet.rightElbow);
  });
});

describe("applyRotations", () => {
  it("applies neutral rotations as all-zero", () => {
    const puppet = buildPuppet();
    applyRotations(puppet, NEUTRAL_POSE);

    expect(puppet.head.rotation.x).toBe(0);
    expect(puppet.head.rotation.y).toBe(0);
    expect(puppet.leftElbow.rotation.x).toBe(0);
    expect(puppet.rightElbow.rotation.x).toBe(0);
  });

  it("maps each joint's rotation fields onto the corresponding pivot", () => {
    const puppet = buildPuppet();
    applyRotations(puppet, {
      head: { x: 0.1, y: 0.2 },
      leftShoulder: { x: 0.3, z: 0.4 },
      leftElbow: 0.5,
      rightShoulder: { x: 0.6, z: 0.7 },
      rightElbow: 0.8,
    });

    expect(puppet.head.rotation.x).toBeCloseTo(0.1);
    expect(puppet.head.rotation.y).toBeCloseTo(0.2);
    expect(puppet.leftShoulder.rotation.x).toBeCloseTo(0.3);
    expect(puppet.leftShoulder.rotation.z).toBeCloseTo(0.4);
    expect(puppet.leftElbow.rotation.x).toBeCloseTo(0.5);
    expect(puppet.rightShoulder.rotation.x).toBeCloseTo(0.6);
    expect(puppet.rightShoulder.rotation.z).toBeCloseTo(0.7);
    expect(puppet.rightElbow.rotation.x).toBeCloseTo(0.8);
  });
});
