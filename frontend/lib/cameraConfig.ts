import type { CameraConfig } from "@/types/camera";

function parseIntEnv(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  fps: parseIntEnv(process.env.NEXT_PUBLIC_CAMERA_FPS, 12),
  width: parseIntEnv(process.env.NEXT_PUBLIC_CAMERA_WIDTH, 640),
  height: parseIntEnv(process.env.NEXT_PUBLIC_CAMERA_HEIGHT, 480),
};

/** Per master-prompt section 3: keep FPS within a sane 10-15 range regardless of config. */
export function clampFps(fps: number): number {
  return Math.min(15, Math.max(10, fps));
}
