"use client";

import { ClipboardPaste, Eraser, RefreshCw } from "lucide-react";

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  /** Clears the input text (and whatever result it produced -- the
   * caller decides that part). */
  onClear: () => void;
  /** Reads the system clipboard and appends/replaces the input with it.
   * Optional -- omit to hide the button (e.g. an environment where
   * clipboard read access isn't meaningful). */
  onPaste?: () => void;
  /** Re-runs the translation for the CURRENT value without the user
   * retyping it. Optional -- omit to hide the button. */
  onRepeat?: () => void;
  disabled?: boolean;
}

export function TextInput({
  value,
  onChange,
  onSubmit,
  onClear,
  onPaste,
  onRepeat,
  disabled = false,
}: TextInputProps): React.ReactElement {
  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        const trimmed = value.trim();
        if (trimmed) onSubmit(trimmed);
      }}
    >
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Введіть текст українською..."
        rows={3}
        disabled={disabled}
        className="w-full resize-none rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-brand-500 focus:outline-none disabled:bg-slate-50"
      />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
        >
          Перекласти
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={!value}
          className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
        >
          <Eraser className="h-3.5 w-3.5" aria-hidden />
          Очистити
        </button>
        {onPaste && (
          <button
            type="button"
            onClick={onPaste}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            <ClipboardPaste className="h-3.5 w-3.5" aria-hidden />
            Вставити
          </button>
        )}
        {onRepeat && (
          <button
            type="button"
            onClick={onRepeat}
            disabled={disabled || !value.trim()}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden />
            Повторити переклад
          </button>
        )}
      </div>
    </form>
  );
}
