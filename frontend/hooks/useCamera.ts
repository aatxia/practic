"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { clampFps, DEFAULT_CAMERA_CONFIG } from "@/lib/cameraConfig";
import type { CameraConfig, CameraError, CameraStatus, CapturedFrame } from "@/types/camera";

interface UseCameraOptions {
  config?: Partial<CameraConfig>;
  /**
   * Called at the configured FPS while streaming, with a captured frame.
   * Left undefined until Phase 5 wires this into the WebSocket client --
   * intentionally NOT sending anything anywhere on its own (see master-prompt
   * section 3: "не відправляй кожен frame безконтрольно").
   */
  onFrame?: (frame: CapturedFrame) => void;
}

interface UseCameraResult {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  error: CameraError | null;
  config: CameraConfig;
  start: () => Promise<void>;
  stop: () => void;
}

function mapGetUserMediaError(err: unknown): CameraError {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return { reason: "unsupported", message: "Цей браузер не підтримує доступ до камери." };
  }

  const name = err instanceof Error ? err.name : "";

  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return {
        reason: "permission_denied",
        message: "Доступ до камери відхилено. Дозволь доступ у налаштуваннях браузера.",
      };
    case "NotFoundError":
    case "DevicesNotFoundError":
      return { reason: "no_camera_found", message: "Камеру не знайдено на цьому пристрої." };
    case "NotReadableError":
    case "TrackStartError":
      return {
        reason: "camera_in_use",
        message: "Камера вже використовується іншим застосунком.",
      };
    default:
      return {
        reason: "unknown",
        message: err instanceof Error ? err.message : "Невідома помилка доступу до камери.",
      };
  }
}

export function useCamera(options: UseCameraOptions = {}): UseCameraResult {
  const { onFrame } = options;
  const config: CameraConfig = {
    ...DEFAULT_CAMERA_CONFIG,
    ...options.config,
    fps: clampFps(options.config?.fps ?? DEFAULT_CAMERA_CONFIG.fps),
  };

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<CameraError | null>(null);

  const stopCaptureLoop = useCallback(() => {
    if (captureIntervalRef.current !== null) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, []);

  const startCaptureLoop = useCallback(() => {
    if (!onFrame) return;
    stopCaptureLoop();

    const intervalMs = 1000 / config.fps;
    captureIntervalRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < video.HAVE_CURRENT_DATA) return;

      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }
      const canvas = canvasRef.current;
      // config.width/height is only an "ideal" hint to getUserMedia (see
      // start() below) -- the camera can (and often does, e.g. any 16:9
      // sensor asked for a 4:3 640x480) return a stream at a different
      // native resolution. Forcing drawImage to config.width/height would
      // stretch/squash that frame to fit, distorting hand/finger
      // proportions in a way neither the training photos nor recorded
      // clips (never force-resized) ever were -- a real, previously-
      // shipped bug that skewed live recognition off the training
      // distribution. Always capturing at the video's own actual
      // videoWidth/videoHeight guarantees no distortion, regardless of
      // what resolution the camera actually granted, and matches what
      // lib/landmarkOverlay.ts already assumes for drawing the overlay.
      const width = video.videoWidth || config.width;
      const height = video.videoHeight || config.height;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, width, height);
      onFrame({
        dataUrl: canvas.toDataURL("image/jpeg", 0.8),
        timestamp: Date.now(),
        width,
        height,
      });
    }, intervalMs);
  }, [config.fps, config.width, config.height, onFrame, stopCaptureLoop]);

  const stop = useCallback(() => {
    stopCaptureLoop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus((prev) => (prev === "error" ? prev : "stopped"));
  }, [stopCaptureLoop]);

  const start = useCallback(async () => {
    setError(null);
    setStatus("requesting_permission");

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      const mappedError = mapGetUserMediaError(new Error("unsupported"));
      setError(mappedError);
      setStatus("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: config.width },
          height: { ideal: config.height },
          frameRate: { ideal: config.fps, max: 30 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        const playResult = videoRef.current.play();
        if (playResult && typeof playResult.then === "function") {
          // autoplay can reject before a user gesture on some browsers; preview still attaches
          await playResult.catch(() => {});
        }
      }
      setStatus("streaming");
      startCaptureLoop();
    } catch (err) {
      setError(mapGetUserMediaError(err));
      setStatus("error");
    }
  }, [config.width, config.height, config.fps, startCaptureLoop]);

  useEffect(() => {
    return () => {
      stopCaptureLoop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [stopCaptureLoop]);

  return { videoRef, status, error, config, start, stop };
}
