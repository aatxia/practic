"use client";

/**
 * A minimal, always-visible "is the app basically working" indicator
 * (app/layout.tsx's header, every page) -- deliberately NOT the same thing
 * as the translator's own live WebSocket connection state (hooks/
 * useWebSocket.ts), which already reflects real-time readiness once a
 * session is open. This is a slower, coarser signal (is the backend
 * reachable at all), so it doesn't need frequent polling or technical
 * detail in the default view -- see PROJECT_STATUS.md's continuous-
 * dictation performance investigation, which found this polling every 5s
 * on every page (with raw "model:"/"ML pipeline:" wording always visible)
 * both unnecessarily frequent and exactly the kind of technical wording
 * that shouldn't be in the normal interface. Detail is still available,
 * just behind a click, the same "diagnostics behind a toggle" convention
 * used elsewhere (components/Translator/TranslatorView.tsx).
 */
import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import type { HealthResponse } from "@/types/api";

type Status = "checking" | "connected" | "disconnected";

// A basic reachability check doesn't need near-real-time granularity --
// was 5000ms, six times the request volume for no real benefit to the
// signer (see this component's module docstring).
const HEALTH_CHECK_INTERVAL_MS = 30000;

const STATUS_LABEL: Record<Status, string> = {
  connected: "Готово",
  checking: "Перевірка...",
  disconnected: "Немає з'єднання",
};

export function ConnectionStatus(): React.ReactElement {
  const [status, setStatus] = useState<Status>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function check(): Promise<void> {
      try {
        const result = await fetchHealth();
        if (!cancelled) {
          setHealth(result);
          setStatus("connected");
        }
      } catch {
        if (!cancelled) {
          setStatus("disconnected");
        }
      }
    }

    void check();
    const interval = setInterval(() => void check(), HEALTH_CHECK_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const dotColor =
    status === "connected"
      ? "bg-emerald-500"
      : status === "checking"
        ? "bg-amber-400"
        : "bg-red-500";

  return (
    <div className="flex flex-col gap-1 text-sm">
      <button
        type="button"
        onClick={() => setShowDetails((current) => !current)}
        className="flex items-center gap-2"
      >
        <span className={`h-2.5 w-2.5 rounded-full ${dotColor}`} aria-hidden />
        <span className="font-medium text-slate-700">{STATUS_LABEL[status]}</span>
      </button>
      {showDetails && health && (
        <span className="text-xs text-slate-400" data-testid="connection-status-details">
          model: {health.model_type} · ML pipeline: {health.ml_pipeline_status}
        </span>
      )}
    </div>
  );
}
