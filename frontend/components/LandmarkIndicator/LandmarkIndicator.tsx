"use client";

import { useState } from "react";
import type { LandmarksStatusMessage } from "@/types/api";

interface LandmarkIndicatorProps {
  /** Latest real per-frame detection (backend/websocket/protocol.py's
   * landmarks_status), or null before the first frame is processed. */
  status: LandmarksStatusMessage | null;
}

const ITEMS: { key: "left_hand" | "right_hand" | "pose"; label: string }[] = [
  { key: "left_hand", label: "Ліва рука" },
  { key: "right_hand", label: "Права рука" },
  { key: "pose", label: "Поза" },
];

/** Opt-in visibility indicator (off by default): lets the signer check
 * whether the camera actually sees their hands/pose before or while
 * signing, using the same real per-frame detection the recognition
 * pipeline itself sees -- never a guess. */
export function LandmarkIndicator({ status }: LandmarkIndicatorProps): React.ReactElement {
  const [visible, setVisible] = useState(false);

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => setVisible((current) => !current)}
        aria-pressed={visible}
        className="self-start text-xs font-medium text-slate-500 hover:text-brand-600"
      >
        {visible ? "Сховати індикатори" : "Показати індикатори рук"}
      </button>
      {visible && (
        <div className="flex gap-4 text-xs">
          {ITEMS.map(({ key, label }) => {
            const detected = status?.[key] ?? false;
            return (
              <span key={key} className="inline-flex items-center gap-1.5">
                <span
                  className={`h-2 w-2 rounded-full ${detected ? "bg-emerald-500" : "bg-slate-300"}`}
                  aria-hidden
                />
                <span className={detected ? "text-slate-700" : "text-slate-400"}>{label}</span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
