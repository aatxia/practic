"use client";

import { Check, Sparkles } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

/** One correctable option -- normalized from either WordCandidate or
 * LetterCandidate (types/api.ts) so this component doesn't need to know
 * which classifier produced it. `key` is what gets passed to onSelect;
 * `label` is the human-readable text to show. */
export interface CorrectionOption {
  key: string;
  label: string;
  confidence: number;
}

interface CorrectionPopoverProps {
  /** What's shown on the trigger while closed -- the current value. */
  triggerLabel: string;
  options: CorrectionOption[];
  currentKey: string;
  onSelect: (key: string) => void;
  /** Accessible name for both the trigger and the option list -- callers
   * pass the same "Виправити слово «X»" / "Виправити літеру «X»"
   * convention used everywhere else in this panel. */
  ariaLabel: string;
  /** Fires whenever this popover's own open state changes (toggled open,
   * or closed via selection/outside click/Escape) -- lets a caller pause
   * live recognition while any correction is being made. Without this, a
   * signer moving their hand to the mouse to pick a candidate could get
   * read as "hand left the frame," firing the same pause-boundary that
   * ends a word/sentence and splicing a spurious token into what's being
   * corrected (see GestureToTextPanel.tsx). */
  onOpenChange?: (open: boolean) => void;
}

/** Replaces a bare native `<select>` with a small floating card: each
 * candidate shows its real confidence as a bar, not just a number, and the
 * model's own top pick is called out -- makes the "the model already knew
 * the right answer was #2" case visible at a glance instead of requiring a
 * click into a plain option list to find out. Closes on an outside click,
 * Escape, or picking an option. */
export function CorrectionPopover({
  triggerLabel,
  options,
  currentKey,
  onSelect,
  ariaLabel,
  onOpenChange,
}: CorrectionPopoverProps): React.ReactElement {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);
  const listboxId = useId();

  useEffect(() => {
    onOpenChange?.(open);
    // onOpenChange intentionally excluded -- callers pass an inline
    // closure that's a new function identity every render; depending on
    // it would fire this effect (and the notification) on every parent
    // re-render, not just on a real open/close transition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (event: MouseEvent): void => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const handleSelect = (key: string): void => {
    onSelect(key);
    setOpen(false);
  };

  return (
    <span ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        className="inline-flex items-center gap-0.5 rounded-md px-0.5 text-inherit decoration-dotted decoration-brand-300 underline-offset-4 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-300"
      >
        {triggerLabel}
        <Sparkles className="h-2.5 w-2.5 shrink-0 text-brand-400" aria-hidden />
      </button>
      {open && (
        <div
          id={listboxId}
          role="listbox"
          aria-label={ariaLabel}
          className="animate-correction-pop-in absolute left-1/2 top-full z-20 mt-1.5 w-44 -translate-x-1/2 rounded-xl border border-slate-200 bg-white p-1.5 text-left shadow-lg"
        >
          {options.map((option, index) => {
            const isCurrent = option.key === currentKey;
            const percent = Math.round(option.confidence * 100);
            return (
              <button
                key={option.key}
                type="button"
                role="option"
                aria-selected={isCurrent}
                aria-label={`${option.label} ${percent}%`}
                onClick={() => handleSelect(option.key)}
                className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs transition-colors ${
                  isCurrent ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-50"
                }`}
              >
                <span className="flex min-w-0 flex-1 items-center gap-1 font-medium">
                  {index === 0 && (
                    <span className="rounded bg-brand-100 px-1 py-px text-[9px] font-semibold uppercase tracking-wide text-brand-600">
                      топ
                    </span>
                  )}
                  <span className="truncate">{option.label}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1">
                  <span className="h-1 w-8 overflow-hidden rounded-full bg-slate-100">
                    <span className="block h-full rounded-full bg-brand-400" style={{ width: `${percent}%` }} />
                  </span>
                  <span className="w-7 text-right tabular-nums text-slate-400">{percent}%</span>
                  {isCurrent && <Check className="h-3 w-3 shrink-0 text-brand-600" aria-hidden />}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </span>
  );
}
