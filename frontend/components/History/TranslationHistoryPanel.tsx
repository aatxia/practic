"use client";

import { Copy, RotateCcw, Sparkles, Trash2, Volume2 } from "lucide-react";
import type { TranslationHistoryEntry } from "@/hooks/useTranslationHistory";

/** The real Ukrainian text a history entry represents -- prefers the
 * backend-composed sentence, falls back to the per-word labels when no
 * full sentence was composed (same fallback TranslatorView already uses
 * for lastFinalTranslation). Never glossSequence's own internal tokens. */
export function historyEntryText(entry: TranslationHistoryEntry): string {
  return entry.composedText ?? entry.glossLabels.map((label) => label.text).join(" ");
}

const DIRECTION_LABEL: Record<TranslationHistoryEntry["direction"], string> = {
  "gestures-to-text": "Жести → Українська",
  "text-to-gestures": "Українська → Жести",
};

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" });
}

interface TranslationHistoryPanelProps {
  history: TranslationHistoryEntry[];
  onCopy: (entry: TranslationHistoryEntry) => void;
  onSpeak: (entry: TranslationHistoryEntry) => void;
  onReuse: (entry: TranslationHistoryEntry) => void;
  onDelete: (id: string) => void;
  onClearAll: () => void;
  /** Hides the Speak action entirely when the browser has no speech
   * synthesis support, rather than rendering a button that silently does
   * nothing when clicked. */
  ttsSupported: boolean;
}

/** Recent-translations list shown on the main screen -- every entry
 * carries its own real actions (never decorative): Copy/Speak/Reuse/Delete,
 * plus a global "Очистити історію". Persisted via useTranslationHistory
 * (localStorage), not a database. */
export function TranslationHistoryPanel({
  history,
  onCopy,
  onSpeak,
  onReuse,
  onDelete,
  onClearAll,
  ttsSupported,
}: TranslationHistoryPanelProps): React.ReactElement | null {
  if (history.length === 0) return null;

  return (
    <div data-testid="translation-history" className="rounded-2xl border border-slate-200 bg-white p-6">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Історія</h3>
        <button
          type="button"
          onClick={onClearAll}
          className="text-xs font-medium text-slate-400 hover:text-slate-600"
        >
          Очистити історію
        </button>
      </div>
      <ol className="flex flex-col gap-2 text-sm text-slate-600">
        {history.map((entry) => (
          <li key={entry.id} className="flex items-center justify-between gap-2 rounded-lg border border-slate-100 px-3 py-2">
            <div className="min-w-0 flex-1">
              <p className="truncate text-slate-800">{historyEntryText(entry)}</p>
              <p className="flex items-center gap-1 text-xs text-slate-400">
                {DIRECTION_LABEL[entry.direction]} · {formatTimestamp(entry.timestamp)}
                {entry.composedTextSource === "ai_fallback" && (
                  // Same honesty rule as the live "Готове речення" badge --
                  // an AI-composed sentence is never shown without this
                  // label, even after it's landed in history.
                  <span
                    className="inline-flex items-center gap-0.5 rounded-full bg-brand-50 px-1.5 py-px font-semibold uppercase tracking-wide text-brand-600"
                    title="Це речення склав ШІ -- не гарантовано граматично точне"
                  >
                    <Sparkles className="h-2 w-2" aria-hidden />
                    ШІ
                  </span>
                )}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-0.5">
              <button
                type="button"
                onClick={() => onCopy(entry)}
                title="Копіювати"
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              >
                <Copy className="h-3.5 w-3.5" aria-hidden />
              </button>
              {ttsSupported && (
                <button
                  type="button"
                  onClick={() => onSpeak(entry)}
                  title="Озвучити"
                  className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                >
                  <Volume2 className="h-3.5 w-3.5" aria-hidden />
                </button>
              )}
              <button
                type="button"
                onClick={() => onReuse(entry)}
                title="Використати знову"
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              </button>
              <button
                type="button"
                onClick={() => onDelete(entry.id)}
                title="Видалити"
                className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-red-600"
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden />
              </button>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
