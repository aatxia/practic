"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  BoundaryMessage,
  LandmarksStatusMessage,
  LetterPredictionMessage,
  PredictionMessage,
  FrameMessage,
  ServerMessage,
} from "@/types/api";
import type { CapturedFrame } from "@/types/camera";

export type WebSocketStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface UseWebSocketResult {
  status: WebSocketStatus;
  lastMessage: ServerMessage | null;
  /** Most recent landmarks_status, held independently of lastMessage --
   * every processed frame immediately follows it with a prediction/error
   * message, so a consumer reading lastMessage alone would see it for a
   * single render and then lose it. */
  landmarksStatus: LandmarksStatusMessage | null;
  /** Most recent letter_prediction/letter_confirmed, held the same way as
   * landmarksStatus and for the same reason -- the real fingerspelling
   * classifier (ml/fingerspelling/), separate from lastMessage's word-level
   * (still synthetic-only) prediction stream. */
  letterMessage: LetterPredictionMessage | null;
  /** Most recent prediction/final_prediction, held the same way as
   * landmarksStatus/letterMessage and for the same reason -- every
   * processed frame that DOES attempt a word-level guess is immediately
   * followed by the next frame's landmarks_status (and possibly a
   * letter_prediction/letter_confirmed), which would otherwise overwrite
   * lastMessage and make a consumer relying on it alone see a real
   * interim/confirmed guess for one render, then lose it. Stays null
   * while no word-level guess is in flight (buffering, motion-gated,
   * not-ready) -- an honest "nothing attempted", not stale leftover data. */
  wordMessage: PredictionMessage | null;
  /** Most recent "boundary" event (see types/api.ts's BoundaryMessage) --
   * unlike landmarksStatus/letterMessage/wordMessage, a consumer must react
   * to EACH occurrence, not just read "the current one": since every
   * incoming message is a freshly parsed object, comparing this value's
   * object identity against the last one a consumer has seen (the same
   * "already handled?" pattern used for wordMessage/letterMessage
   * elsewhere) reliably distinguishes two consecutive boundary events even
   * when they share the same `kind`. */
  boundaryMessage: BoundaryMessage | null;
  connect: () => void;
  disconnect: () => void;
  sendFrame: (frame: CapturedFrame) => void;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";

// The terminal message for whichever frame is currently in flight -- see
// awaitingFrameResponseRef below. landmarks_status and letter_prediction/
// letter_confirmed are interim (each frame gets 0-1 of the latter, always
// followed by exactly one of these), connection is sent once at connect
// and isn't tied to a frame at all.
const TERMINAL_MESSAGE_TYPES = new Set(["prediction", "final_prediction", "error"]);

function isServerMessage(value: unknown): value is ServerMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    typeof (value as { type: unknown }).type === "string"
  );
}

export function useWebSocket(): UseWebSocketResult {
  const socketRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<WebSocketStatus>("idle");
  const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null);
  const [landmarksStatus, setLandmarksStatus] = useState<LandmarksStatusMessage | null>(null);
  const [letterMessage, setLetterMessage] = useState<LetterPredictionMessage | null>(null);
  const [wordMessage, setWordMessage] = useState<PredictionMessage | null>(null);
  const [boundaryMessage, setBoundaryMessage] = useState<BoundaryMessage | null>(null);
  // Backpressure: the backend processes one frame's real MediaPipe
  // extraction (+ inference) fully before reading the next message off this
  // connection -- on a slow machine that easily takes longer than the
  // capture loop's interval. Without this, sendFrame would keep firing on
  // its own timer regardless, queuing up frames the server hasn't gotten to
  // yet; every later landmarks_status/prediction would then describe an
  // increasingly stale frame, the display drifting further behind the (locally
  // rendered, always-live) camera preview forever, not just running a bit slow.
  // Capping this connection to one in-flight frame means a slow backend
  // instead just shows a lower-but-live indicator rate -- never growing lag.
  const awaitingFrameResponseRef = useRef(false);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) {
      return; // already connecting/connected
    }
    if (typeof WebSocket === "undefined") {
      setStatus("error");
      return;
    }

    awaitingFrameResponseRef.current = false;
    setStatus("connecting");
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    // React StrictMode (dev only) double-invokes effects: mount -> cleanup -> mount.
    // That closes this exact socket almost immediately while a second one opens.
    // Without this guard, the first socket's belated onclose/onerror would
    // overwrite the status set by the second (current) socket's onopen.
    const isStale = () => socketRef.current !== socket;

    socket.onopen = () => {
      if (isStale()) return;
      setStatus("open");
    };
    socket.onclose = () => {
      if (isStale()) return;
      setStatus("closed");
    };
    socket.onerror = () => {
      if (isStale()) return;
      setStatus("error");
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      if (isStale()) return;
      try {
        const parsed: unknown = JSON.parse(event.data);
        if (isServerMessage(parsed)) {
          setLastMessage(parsed);
          if (parsed.type === "landmarks_status") {
            setLandmarksStatus(parsed);
          } else if (parsed.type === "letter_prediction" || parsed.type === "letter_confirmed") {
            setLetterMessage(parsed);
          } else if (parsed.type === "boundary") {
            // Not a terminal message (see TERMINAL_MESSAGE_TYPES below) --
            // the backend may send it as an EXTRA message alongside a
            // frame's normal landmarks_status/prediction/error, not in
            // place of one, so it must never gate sendFrame's backpressure.
            setBoundaryMessage(parsed);
          } else if (TERMINAL_MESSAGE_TYPES.has(parsed.type)) {
            if (parsed.type === "prediction" || parsed.type === "final_prediction") {
              setWordMessage(parsed);
            } else {
              // A terminal "error" (buffering/motion-gated/not-ready) means
              // no word-level guess was attempted this frame -- clear any
              // previous one rather than leave it looking still current.
              setWordMessage(null);
            }
            // Only now is it safe to let sendFrame release the next frame.
            awaitingFrameResponseRef.current = false;
          }
        }
      } catch {
        // Malformed message from the server -- ignore rather than crash the UI.
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setStatus("closed");
  }, []);

  const sendFrame = useCallback((frame: CapturedFrame) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    // Drop (don't queue) this frame if the previous one's response hasn't
    // arrived yet -- see awaitingFrameResponseRef above.
    if (awaitingFrameResponseRef.current) return;

    const message: FrameMessage = {
      type: "frame",
      timestamp: frame.timestamp,
      data: frame.dataUrl,
    };
    awaitingFrameResponseRef.current = true;
    socket.send(JSON.stringify(message));
  }, []);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  return {
    status,
    lastMessage,
    landmarksStatus,
    letterMessage,
    wordMessage,
    boundaryMessage,
    connect,
    disconnect,
    sendFrame,
  };
}
