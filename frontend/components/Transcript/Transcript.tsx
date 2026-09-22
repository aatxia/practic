"use client";

import { CaseSensitive } from "lucide-react";
import type { TranslationState } from "@/types/translation";

interface TranscriptProps {
  state: TranslationState;
}

export function Transcript({ state }: TranscriptProps): React.ReactElement {
  if (state.status === "idle") {
    return (
      <p className="text-sm text-slate-400">
        Введи текст або скажи щось голосом і натисни «Перекласти».
      </p>
    );
  }

  if (state.status === "loading") {
    return <p className="text-sm text-slate-400">Перекладаю...</p>;
  }

  if (state.status === "error") {
    return <p className="text-sm text-red-600">{state.message}</p>;
  }

  // Never show glossSequence's own tokens (e.g. "WANT", "CAR") -- those are
  // internal English identifiers, not Ukrainian. glossLabels/composedText
  // (backend/app/api/routes/translate.py) are the real Ukrainian words.
  return (
    <div className="flex flex-col gap-2">
      {state.composedText && (
        <p className="text-base font-semibold text-slate-900">{state.composedText}</p>
      )}
      <div className="flex flex-wrap gap-2" data-testid="transcript-gloss-labels">
        {state.glossLabels.map((label, index) => (
          <span
            key={`${label.text}-${index}`}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-sm ${
              label.isFingerspell
                ? "border border-slate-200 bg-slate-50 text-slate-700"
                : "bg-brand-50 text-brand-700"
            }`}
          >
            {label.isFingerspell && <CaseSensitive className="h-3.5 w-3.5" aria-hidden />}
            {label.text}
          </span>
        ))}
      </div>
      <p className="text-xs text-slate-400">
        {state.composedText
          ? "Відео-переклад нижче, у порядку показу."
          : "Складено зі слів нижче — повного речення для цієї комбінації ще немає."}{" "}
        Позначені літерами чипи — дактилологія (слово поза лексиконом, розкладене по літерах).
      </p>
    </div>
  );
}
