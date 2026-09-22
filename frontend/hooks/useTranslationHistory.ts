"use client";

/**
 * Owns the CAMERA -> Ukrainian translation result state, kept strictly
 * separate from the live/active utterance (hooks/useUtterance.ts) and from
 * the unrelated text/voice -> gesture flow (types/translation.ts's
 * TranslationState, shown by components/Transcript). Three distinct pieces
 * of state, matching the product requirement that a finalized translation
 * must never be conflated with or cleared by new signing activity:
 *
 *   - lastFinalTranslation: the most recently completed sentence. Persists
 *     on screen until EITHER a new translation replaces it OR the user
 *     explicitly clears it (clearCurrent) -- never auto-cleared by new
 *     signing starting, a reconnect, or any diagnostic/interim update.
 *   - history: every completed translation this session (and, via
 *     localStorage, previous sessions), newest first.
 *
 * addEntry() is the only way either piece of state changes (besides the
 * explicit clear actions) -- called exactly once per genuinely completed
 * utterance (see useUtterance.ts's boundary-message identity guard, which
 * is what actually prevents a duplicate call from ever reaching here).
 */
import { useCallback, useEffect, useState } from "react";
import type { TranslationDirection } from "@/types/direction";
import type { GlossLabel } from "@/types/translation";

export interface TranslationHistoryEntry {
  id: string;
  composedText: string | null;
  glossSequence: string[];
  glossLabels: GlossLabel[];
  timestamp: number;
  direction: TranslationDirection;
  /** The exact Ukrainian text the user typed/spoke, for a text-to-gestures
   * entry only -- kept separate from composedText (which is the backend's
   * re-COMPOSED sentence from the gloss sequence, and can legitimately
   * differ, e.g. fingerspelling capitalization, see
   * ml/nlp/text_to_gloss.py's own documented round-trip caveat). null for
   * a gestures-to-text entry, which has no "typed input" of its own. */
  sourceText: string | null;
  /** Which engine produced composedText -- "rule_based" is grammar-
   * guaranteed (ml/nlp/gloss_to_text.py); "ai_fallback" is a real LLM's
   * best attempt (backend/app/services/ai_sentence_composer.py), only
   * ever used when rule_based couldn't compose a pattern, and must be
   * shown as AI-generated, not verified (see types/api.ts's
   * GlossToTextResponse.composed_text_source). null for a
   * text-to-gestures entry (no gloss-to-text call involved) or an entry
   * saved before this field existed -- never assumed to be "rule_based"
   * just because that was the only option back then. */
  composedTextSource: "rule_based" | "ai_fallback" | null;
}

const STORAGE_KEY = "uksl.translationHistory.v1";
// Bounds localStorage growth -- a translation session is not expected to
// need more than this many past results kept around.
const MAX_HISTORY_ENTRIES = 50;

function isValidEntry(value: unknown): value is Omit<TranslationHistoryEntry, "direction" | "sourceText"> {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.id === "string" &&
    (entry.composedText === null || typeof entry.composedText === "string") &&
    Array.isArray(entry.glossSequence) &&
    Array.isArray(entry.glossLabels) &&
    typeof entry.timestamp === "number"
  );
}

// A value pre-dating the `direction`/`sourceText` fields (saved by an
// older build of this hook) is still a valid entry -- it just predates
// those fields ever being tracked, so it's normalized to a safe default
// (the original camera->text flow this hook was built for, and no known
// source text) rather than dropped.
function normalizeDirection(value: unknown): TranslationDirection {
  return value === "text-to-gestures" ? "text-to-gestures" : "gestures-to-text";
}

function normalizeSourceText(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function normalizeComposedTextSource(value: unknown): "rule_based" | "ai_fallback" | null {
  return value === "rule_based" || value === "ai_fallback" ? value : null;
}

function loadHistory(): TranslationHistoryEntry[] {
  // SSR (no window), a private-browsing/storage-blocked browser, or a
  // corrupted value must never crash the app -- an empty history is always
  // a safe fallback, never a reason to throw.
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidEntry).map((entry) => ({
      ...entry,
      direction: normalizeDirection((entry as Record<string, unknown>).direction),
      sourceText: normalizeSourceText((entry as Record<string, unknown>).sourceText),
      composedTextSource: normalizeComposedTextSource((entry as Record<string, unknown>).composedTextSource),
    }));
  } catch {
    return [];
  }
}

function saveHistory(entries: TranslationHistoryEntry[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage full or blocked -- history still works for this session in
    // memory, it just won't survive a refresh. Not worth surfacing as an
    // error to the signer.
  }
}

export interface UseTranslationHistoryResult {
  lastFinalTranslation: TranslationHistoryEntry | null;
  history: TranslationHistoryEntry[];
  /** Appends a completed translation and makes it the new
   * lastFinalTranslation. A no-op for an empty gloss sequence -- an empty
   * utterance is not a translation and must not create a history entry
   * or overwrite whatever was last shown. `direction` defaults to
   * "gestures-to-text" (this hook's original, and still primary, use). */
  addEntry: (
    composedText: string | null,
    glossSequence: string[],
    glossLabels: GlossLabel[],
    direction?: TranslationDirection,
    sourceText?: string | null,
    composedTextSource?: "rule_based" | "ai_fallback" | null
  ) => void;
  /** Hides the currently shown final translation -- history is untouched. */
  clearCurrent: () => void;
  /** Clears all history -- does not touch whatever is currently shown as
   * lastFinalTranslation. */
  clearHistory: () => void;
  /** Removes one entry by id -- from history, and from lastFinalTranslation
   * too if that's the entry being removed (never leaves a "current"
   * translation pointing at something no longer in history). */
  removeEntry: (id: string) => void;
}

export function useTranslationHistory(): UseTranslationHistoryResult {
  // Starts empty on EVERY render, server or client, and loads the real
  // localStorage value in an effect instead of a useState initializer --
  // a lazy initializer runs during the client's first render too (before
  // hydration reconciles), and since the server render always sees no
  // `window` and produces `[]`, a non-empty saved history would make that
  // very first client render already show the "Готове речення" toolbar
  // the server didn't render, a real hydration mismatch reported live.
  // An effect only ever runs client-side, strictly after hydration
  // completes, so the two renders that must match now do.
  const [history, setHistory] = useState<TranslationHistoryEntry[]>([]);
  const [lastFinalTranslation, setLastFinalTranslation] = useState<TranslationHistoryEntry | null>(null);

  useEffect(() => {
    const loaded = loadHistory();
    setHistory(loaded);
    setLastFinalTranslation(loaded[0] ?? null);
  }, []);

  const addEntry = useCallback(
    (
      composedText: string | null,
      glossSequence: string[],
      glossLabels: GlossLabel[],
      direction: TranslationDirection = "gestures-to-text",
      sourceText: string | null = null,
      composedTextSource: "rule_based" | "ai_fallback" | null = null
    ) => {
      if (glossSequence.length === 0) return;
      const entry: TranslationHistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        composedText,
        glossSequence,
        glossLabels,
        timestamp: Date.now(),
        direction,
        sourceText,
        composedTextSource,
      };
      setLastFinalTranslation(entry);
      setHistory((current) => {
        const next = [entry, ...current].slice(0, MAX_HISTORY_ENTRIES);
        saveHistory(next);
        return next;
      });
    },
    []
  );

  const clearCurrent = useCallback(() => setLastFinalTranslation(null), []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
  }, []);

  const removeEntry = useCallback((id: string) => {
    setHistory((current) => {
      const next = current.filter((entry) => entry.id !== id);
      saveHistory(next);
      return next;
    });
    setLastFinalTranslation((current) => (current?.id === id ? null : current));
  }, []);

  return { lastFinalTranslation, history, addEntry, clearCurrent, clearHistory, removeEntry };
}
