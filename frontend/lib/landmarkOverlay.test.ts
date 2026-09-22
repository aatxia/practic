import { describe, expect, it, vi } from "vitest";
import { computeObjectCoverTransform, drawLandmarksOverlay } from "./landmarkOverlay";
import type { LandmarksStatusMessage } from "@/types/api";

describe("computeObjectCoverTransform", () => {
  it("scales to fill the box width and crops top/bottom for a 4:3 video in a 16:9 box", () => {
    // 4:3 video (640x480) inside a 16:9 box (e.g. 320x180, matching aspect-video) --
    // width is the binding dimension here (320/640=0.5 > 180/480=0.375), so the
    // rendered video exactly fills the box width and overflows (crops) vertically.
    const transform = computeObjectCoverTransform(320, 180, 640, 480);

    expect(transform).not.toBeNull();
    expect(transform!.scale).toBeCloseTo(0.5);
    expect(transform!.offsetX).toBeCloseTo(0);
    const renderedHeight = 480 * transform!.scale;
    expect(transform!.offsetY).toBeCloseTo((180 - renderedHeight) / 2);
    expect(transform!.offsetY).toBeLessThan(0);
  });

  it("returns null for a zero-sized box or video (not yet measured)", () => {
    expect(computeObjectCoverTransform(0, 0, 640, 480)).toBeNull();
    expect(computeObjectCoverTransform(320, 180, 0, 0)).toBeNull();
  });

  it("maps a centered point back to the center of the box", () => {
    const transform = computeObjectCoverTransform(320, 180, 640, 480)!;
    const centerX = transform.offsetX + 0.5 * 640 * transform.scale;
    const centerY = transform.offsetY + 0.5 * 480 * transform.scale;
    expect(centerX).toBeCloseTo(160);
    expect(centerY).toBeCloseTo(90);
  });
});

function createMockCtx(): CanvasRenderingContext2D {
  return {
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    strokeStyle: "",
    fillStyle: "",
    lineWidth: 0,
  } as unknown as CanvasRenderingContext2D;
}

describe("drawLandmarksOverlay", () => {
  it("clears the canvas and draws nothing else when there is no status", () => {
    const ctx = createMockCtx();
    drawLandmarksOverlay(ctx, 320, 180, 640, 480, null);

    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 320, 180);
    expect(ctx.stroke).not.toHaveBeenCalled();
  });

  it("draws a skeleton only for the modalities that were actually detected", () => {
    const ctx = createMockCtx();
    const status: LandmarksStatusMessage = {
      type: "landmarks_status",
      left_hand: false,
      right_hand: true,
      pose: false,
      face: false,
      left_hand_points: null,
      right_hand_points: Array.from({ length: 21 }, (_, i) => [i / 21, i / 21] as [number, number]),
      pose_points: null,
    };

    drawLandmarksOverlay(ctx, 320, 180, 640, 480, status);

    // 21 points, 21 connections for the one detected hand -> stroke + fill each called once per draw pass.
    expect(ctx.stroke).toHaveBeenCalledTimes(1);
    expect(ctx.fill).toHaveBeenCalledTimes(21);
  });

  it("does nothing (beyond clearing) when the video hasn't reported a size yet", () => {
    const ctx = createMockCtx();
    const status: LandmarksStatusMessage = {
      type: "landmarks_status",
      left_hand: true,
      right_hand: false,
      pose: false,
      face: false,
      left_hand_points: [[0.5, 0.5]],
      right_hand_points: null,
      pose_points: null,
    };

    drawLandmarksOverlay(ctx, 320, 180, 0, 0, status);

    expect(ctx.clearRect).toHaveBeenCalled();
    expect(ctx.fill).not.toHaveBeenCalled();
  });
});
