"use client";

import { GestureVideo } from "@/components/GestureVideo";
import { TextInput } from "@/components/TextInput";
import { Transcript } from "@/components/Transcript";
import { VoiceInput } from "@/components/VoiceInput";
import type { TranslationState } from "@/types/translation";

interface TextToGesturePanelProps {
  inputText: string;
  onInputTextChange: (text: string) => void;
  onTranslate: (text: string) => void;
  onClear: () => void;
  onPaste: () => void;
  onRepeatTranslate: () => void;
  translationState: TranslationState;
  translationGlossSequence: string[];
}

/** Ukrainian text (typed or spoken) -> gesture-video playback. Unknown
 * words automatically fall back to real per-letter dactyl clips instead of
 * being skipped (see components/GestureVideo/ -- already implemented). */
export function TextToGesturePanel({
  inputText,
  onInputTextChange,
  onTranslate,
  onClear,
  onPaste,
  onRepeatTranslate,
  translationState,
  translationGlossSequence,
}: TextToGesturePanelProps): React.ReactElement {
  return (
    <div className="flex flex-col gap-6">
      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Текст / Голос → Жести
        </h2>
        <div className="flex flex-col gap-3">
          <VoiceInput onTranscript={onInputTextChange} />
          <TextInput
            value={inputText}
            onChange={onInputTextChange}
            onSubmit={onTranslate}
            onClear={onClear}
            onPaste={onPaste}
            onRepeat={onRepeatTranslate}
            disabled={translationState.status === "loading"}
          />
          <Transcript state={translationState} />
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Відео-переклад
        </h2>
        <GestureVideo glossSequence={translationGlossSequence} />
      </section>
    </div>
  );
}
