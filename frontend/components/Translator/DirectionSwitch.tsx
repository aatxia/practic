"use client";

import { ArrowLeftRight } from "lucide-react";
import type { TranslationDirection } from "@/types/direction";

interface DirectionSwitchProps {
  direction: TranslationDirection;
  onChange: (direction: TranslationDirection) => void;
}

const ACTIVE = "bg-brand-600 text-white";
const INACTIVE = "bg-white text-slate-600 hover:bg-slate-100";

/** Switches which panel is active -- never reloads the page or navigates,
 * just flips a piece of state the parent uses to pick which panel to
 * render (see TranslatorView.tsx). */
export function DirectionSwitch({ direction, onChange }: DirectionSwitchProps): React.ReactElement {
  return (
    <div
      role="group"
      aria-label="Напрямок перекладу"
      className="mx-auto flex w-fit items-center gap-1 rounded-full border border-slate-200 bg-slate-50 p-1"
    >
      <button
        type="button"
        onClick={() => onChange("gestures-to-text")}
        aria-pressed={direction === "gestures-to-text"}
        className={`rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
          direction === "gestures-to-text" ? ACTIVE : INACTIVE
        }`}
      >
        Жести → Українська
      </button>
      <button
        type="button"
        onClick={() => onChange(direction === "gestures-to-text" ? "text-to-gestures" : "gestures-to-text")}
        title="Поміняти напрямок"
        aria-label="Поміняти напрямок"
        className="rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
      >
        <ArrowLeftRight className="h-4 w-4" aria-hidden />
      </button>
      <button
        type="button"
        onClick={() => onChange("text-to-gestures")}
        aria-pressed={direction === "text-to-gestures"}
        className={`rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
          direction === "text-to-gestures" ? ACTIVE : INACTIVE
        }`}
      >
        Українська → Жести
      </button>
    </div>
  );
}
