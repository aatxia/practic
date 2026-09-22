"use client";

import { useCallback, useRef, useState } from "react";
import type { SpeechError, SpeechStatus } from "@/types/speech";

/**
 * Phase 13: minimal shape of the browser's Web Speech API (SpeechRecognition)
 * this hook actually uses -- not the full spec, and not shipped in every
 * browser's TS lib, so declared locally rather than assumed global.
 */
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: { transcript: string };
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionErrorEventLike {
  error: string;
}

interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getSpeechRecognitionConstructor(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

function mapSpeechError(errorCode: string): SpeechError {
  switch (errorCode) {
    case "not-allowed":
    case "permission-denied":
      return { reason: "permission_denied", message: "Доступ до мікрофона відхилено." };
    case "no-speech":
      return { reason: "no_speech", message: "Мовлення не розпізнано. Спробуй ще раз." };
    case "network":
      return { reason: "network", message: "Помилка мережі під час розпізнавання мовлення." };
    default:
      return { reason: "unknown", message: `Невідома помилка розпізнавання мовлення: ${errorCode}` };
  }
}

interface UseSpeechRecognitionResult {
  status: SpeechStatus;
  transcript: string;
  error: SpeechError | null;
  start: () => void;
  stop: () => void;
}

export function useSpeechRecognition(): UseSpeechRecognitionResult {
  const [status, setStatus] = useState<SpeechStatus>("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<SpeechError | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  const start = useCallback(() => {
    const Constructor = getSpeechRecognitionConstructor();
    if (!Constructor) {
      setError({ reason: "unsupported", message: "Цей браузер не підтримує розпізнавання мовлення." });
      setStatus("error");
      return;
    }

    setError(null);
    setTranscript("");

    const recognition = new Constructor();
    recognition.lang = "uk-UA";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result?.isFinal) {
          finalTranscript += result[0].transcript;
        }
      }
      if (finalTranscript) {
        setTranscript((prev) => (prev ? `${prev} ${finalTranscript}` : finalTranscript).trim());
      }
    };

    recognition.onerror = (event) => {
      setError(mapSpeechError(event.error));
      setStatus("error");
    };

    recognition.onend = () => {
      setStatus((prev) => (prev === "error" ? prev : "stopped"));
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      // start() can throw synchronously (e.g. Chrome's InvalidStateError
      // when called again before the previous session fully ended) --
      // uncaught, this silently killed the click with no visible error at
      // all, which read as "voice input just doesn't work."
      setError({
        reason: "unknown",
        message: `Не вдалося запустити розпізнавання мовлення: ${err instanceof Error ? err.message : String(err)}`,
      });
      setStatus("error");
      return;
    }
    setStatus("listening");
  }, []);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
  }, []);

  return { status, transcript, error, start, stop };
}
