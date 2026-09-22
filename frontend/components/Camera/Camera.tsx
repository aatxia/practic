"use client";

import { useEffect, useRef, useState } from "react";
import { useCamera } from "@/hooks/useCamera";
import { drawLandmarksOverlay } from "@/lib/landmarkOverlay";
import type { CapturedFrame } from "@/types/camera";
import type { LandmarksStatusMessage } from "@/types/api";

const STATUS_LABEL: Record<string, string> = {
  idle: "Не запущено",
  requesting_permission: "Очікую дозвіл...",
  streaming: "Активна",
  stopped: "Зупинено",
  error: "Помилка",
};

interface CameraProps {
  /** Called at the configured FPS while streaming (see useCamera). Optional --
   * omit for a plain preview-only camera (e.g. standalone demo usage). */
  onFrame?: (frame: CapturedFrame) => void;
  /** Latest real per-frame detection (backend/websocket/protocol.py's
   * landmarks_status) -- when given, real detected hand/pose points are
   * drawn directly on the video, the way MediaPipe's own demos do.
   * Optional -- omit for a plain preview-only camera. */
  landmarksStatus?: LandmarksStatusMessage | null;
}

export function Camera({ onFrame, landmarksStatus = null }: CameraProps = {}): React.ReactElement {
  const { videoRef, status, error, config, start, stop } = useCamera({ onFrame });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const [boxSize, setBoxSize] = useState({ width: 0, height: 0 });
  // On by default -- it's the main way to see detection actually working --
  // but drawing a live skeleton over your own face/hands isn't for everyone,
  // so it's a real, one-click off switch, not just the separate side dots.
  const [overlayEnabled, setOverlayEnabled] = useState(true);

  const isStreaming = status === "streaming";
  const isBusy = status === "requesting_permission";

  // config.width/height is only the "ideal" hint passed to getUserMedia
  // (useCamera.ts) -- the camera can grant a different actual resolution
  // (e.g. a 16:9 sensor asked for 640x480 4:3). Reading the video
  // element's own actual dimensions once they're known keeps this label
  // honest about what's really being captured and sent, not just requested.
  const [actualResolution, setActualResolution] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const updateResolution = () => {
      if (video.videoWidth && video.videoHeight) {
        setActualResolution({ width: video.videoWidth, height: video.videoHeight });
      }
    };
    video.addEventListener("loadedmetadata", updateResolution);
    updateResolution();
    return () => video.removeEventListener("loadedmetadata", updateResolution);
  }, [videoRef, isStreaming]);

  useEffect(() => {
    const box = boxRef.current;
    if (!box || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setBoxSize({ width, height });
    });
    observer.observe(box);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || boxSize.width === 0 || boxSize.height === 0) return;

    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
    canvas.width = boxSize.width * dpr;
    canvas.height = boxSize.height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    drawLandmarksOverlay(
      ctx,
      boxSize.width,
      boxSize.height,
      video.videoWidth,
      video.videoHeight,
      isStreaming && overlayEnabled ? landmarksStatus : null,
    );
  }, [landmarksStatus, boxSize, isStreaming, overlayEnabled, videoRef]);

  return (
    <div className="flex flex-col gap-3">
      <div
        ref={boxRef}
        className="relative flex aspect-video items-center justify-center overflow-hidden rounded-xl bg-slate-900"
      >
        {/* Live camera preview -- no captions applicable to a real-time video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`h-full w-full object-cover ${isStreaming ? "block" : "hidden"}`}
        />
        {/* Real detected hand/pose points, drawn on top of the video --
            see frontend/lib/landmarkOverlay.ts */}
        <canvas
          ref={canvasRef}
          className={`pointer-events-none absolute inset-0 h-full w-full ${isStreaming && overlayEnabled ? "block" : "hidden"}`}
        />
        {isStreaming && (
          <button
            type="button"
            onClick={() => setOverlayEnabled((current) => !current)}
            aria-pressed={overlayEnabled}
            className="absolute right-2 top-2 rounded-md bg-slate-900/70 px-2 py-1 text-xs font-medium text-white hover:bg-slate-900/90"
          >
            {overlayEnabled ? "Сховати індикатори на відео" : "Показати індикатори на відео"}
          </button>
        )}
        {!isStreaming && (
          <p className="px-4 text-center text-sm text-slate-400">
            {status === "error" ? error?.message : "Камера не активна"}
          </p>
        )}
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>
          {isStreaming && actualResolution
            ? `${actualResolution.width}×${actualResolution.height}`
            : `${config.width}×${config.height}`}{" "}
          · {config.fps} FPS
        </span>
        <span
          className={
            status === "streaming"
              ? "font-medium text-emerald-600"
              : status === "error"
                ? "font-medium text-red-600"
                : "text-slate-400"
          }
        >
          {STATUS_LABEL[status]}
        </span>
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void start()}
          disabled={isStreaming || isBusy}
          className="flex-1 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Увімкнути камеру
        </button>
        <button
          type="button"
          onClick={stop}
          disabled={!isStreaming}
          className="flex-1 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
        >
          Вимкнути
        </button>
      </div>
    </div>
  );
}
