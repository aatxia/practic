export interface CameraConfig {
  fps: number;
  width: number;
  height: number;
}

export type CameraStatus =
  | "idle"
  | "requesting_permission"
  | "streaming"
  | "stopped"
  | "error";

export type CameraErrorReason =
  | "permission_denied"
  | "no_camera_found"
  | "camera_in_use"
  | "unsupported"
  | "unknown";

export interface CameraError {
  reason: CameraErrorReason;
  message: string;
}

/**
 * A single captured frame, produced at the configured FPS while streaming.
 * `dataUrl` is a base64 JPEG data URL -- ready to be sent as the WebSocket
 * `frame` message payload in Phase 5 (see types/api.ts FrameMessage).
 */
export interface CapturedFrame {
  dataUrl: string;
  timestamp: number;
  width: number;
  height: number;
}
