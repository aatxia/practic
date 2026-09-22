"use client";

import { Mic, Square } from "lucide-react";
import { useEffect, useRef } from "react";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";

const STATUS_LABEL: Record<string, string> = {
  idle: "Не активовано",
  listening: "Слухаю...",
  stopped: "Готово",
  error: "Помилка",
};

interface VoiceInputProps {
  /** Called with the accumulated final transcript whenever it changes. */
  onTranscript: (text: string) => void;
}

export function VoiceInput({ onTranscript }: VoiceInputProps): React.ReactElement {
  const { status, transcript, error, start, stop } = useSpeechRecognition();
  const lastReportedRef = useRef("");

  useEffect(() => {
    if (transcript && transcript !== lastReportedRef.current) {
      lastReportedRef.current = transcript;
      onTranscript(transcript);
    }
  }, [transcript, onTranscript]);

  const isListening = status === "listening";

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => (isListening ? stop() : start())}
        aria-pressed={isListening}
        className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold text-white ${
          isListening ? "bg-red-600 hover:bg-red-700" : "bg-brand-600 hover:bg-brand-700"
        }`}
      >
        {isListening ? (
          <>
            <Square className="h-4 w-4" aria-hidden />
            Зупинити
          </>
        ) : (
          <>
            <Mic className="h-4 w-4" aria-hidden />
            Голос
          </>
        )}
      </button>
      <span className={`text-xs ${status === "error" ? "text-red-600" : "text-slate-400"}`}>
        {status === "error" ? error?.message : STATUS_LABEL[status]}
      </span>
    </div>
  );
}
