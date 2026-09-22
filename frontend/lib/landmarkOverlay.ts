import type { LandmarkPoint, LandmarksStatusMessage } from "@/types/api";

/** MediaPipe's own 21-point hand topology (thumb/index/middle/ring/pinky +
 * palm base), same connection set their own demos draw. */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

/** Upper-body subset of MediaPipe's 33-point pose topology -- shoulders,
 * elbows, wrists, hips -- the part relevant to signing, not the legs. */
export const POSE_CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
];

export interface ObjectCoverTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/** How a video with intrinsic size (videoWidth, videoHeight) is actually
 * scaled and cropped when displayed with CSS `object-fit: cover` inside a
 * box of size (boxWidth, boxHeight) -- the same math the browser itself
 * applies, needed here to map [0,1]-normalized landmark points back onto
 * the visible, possibly-cropped video. */
export function computeObjectCoverTransform(
  boxWidth: number,
  boxHeight: number,
  videoWidth: number,
  videoHeight: number,
): ObjectCoverTransform | null {
  if (boxWidth <= 0 || boxHeight <= 0 || videoWidth <= 0 || videoHeight <= 0) return null;

  const scale = Math.max(boxWidth / videoWidth, boxHeight / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;

  return {
    scale,
    offsetX: (boxWidth - renderedWidth) / 2,
    offsetY: (boxHeight - renderedHeight) / 2,
  };
}

function toScreenPoint(
  [nx, ny]: LandmarkPoint,
  transform: ObjectCoverTransform,
  videoWidth: number,
  videoHeight: number,
): [number, number] {
  return [
    transform.offsetX + nx * videoWidth * transform.scale,
    transform.offsetY + ny * videoHeight * transform.scale,
  ];
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  points: LandmarkPoint[],
  connections: [number, number][],
  transform: ObjectCoverTransform,
  videoWidth: number,
  videoHeight: number,
  color: string,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of connections) {
    const pointA = points[a];
    const pointB = points[b];
    if (!pointA || !pointB) continue;
    const [ax, ay] = toScreenPoint(pointA, transform, videoWidth, videoHeight);
    const [bx, by] = toScreenPoint(pointB, transform, videoWidth, videoHeight);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();

  ctx.fillStyle = color;
  for (const point of points) {
    const [x, y] = toScreenPoint(point, transform, videoWidth, videoHeight);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/** Draws real detected hand/pose points directly on the video frame, the
 * way MediaPipe's own demos do -- canvas is expected to already be sized
 * (in CSS pixels, via context scale) to match the displayed video box. */
export function drawLandmarksOverlay(
  ctx: CanvasRenderingContext2D,
  boxWidth: number,
  boxHeight: number,
  videoWidth: number,
  videoHeight: number,
  status: LandmarksStatusMessage | null,
): void {
  ctx.clearRect(0, 0, boxWidth, boxHeight);
  if (!status) return;

  const transform = computeObjectCoverTransform(boxWidth, boxHeight, videoWidth, videoHeight);
  if (!transform) return;

  if (status.left_hand_points) {
    drawSkeleton(
      ctx,
      status.left_hand_points,
      HAND_CONNECTIONS,
      transform,
      videoWidth,
      videoHeight,
      "#f97316",
    );
  }
  if (status.right_hand_points) {
    drawSkeleton(
      ctx,
      status.right_hand_points,
      HAND_CONNECTIONS,
      transform,
      videoWidth,
      videoHeight,
      "#22c55e",
    );
  }
  if (status.pose_points) {
    drawSkeleton(
      ctx,
      status.pose_points,
      POSE_CONNECTIONS,
      transform,
      videoWidth,
      videoHeight,
      "#38bdf8",
    );
  }
}
