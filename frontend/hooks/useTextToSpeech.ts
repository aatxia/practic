"use client";

/**
 * React state wrapper over lib/tts.ts's browser speechSynthesis call --
 * tracks isSpeaking (for a Speak/Stop button) and the last spoken text (for
 * a separate "Повторити" (repeat) control that doesn't need the caller to
 * pass the text again).
 */
import { useCallback, useRef, useState } from "react";
import { cancelSpeech, isSpeechSynthesisSupported, speak as speakText } from "@/lib/tts";

export interface UseTextToSpeechResult {
  isSupported: boolean;
  isSpeaking: boolean;
  /** Speaks `text` aloud, replacing any speech already in progress. */
  speak: (text: string) => void;
  /** Re-speaks the last text passed to speak() -- a no-op if nothing has
   * been spoken yet this session. */
  repeat: () => void;
  stop: () => void;
}

export function useTextToSpeech(): UseTextToSpeechResult {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const lastTextRef = useRef<string | null>(null);

  const speak = useCallback((text: string) => {
    if (!text) return;
    lastTextRef.current = text;
    setIsSpeaking(true);
    speakText(text, {
      onEnd: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  }, []);

  const repeat = useCallback(() => {
    if (lastTextRef.current) speak(lastTextRef.current);
  }, [speak]);

  const stop = useCallback(() => {
    cancelSpeech();
    setIsSpeaking(false);
  }, []);

  return { isSupported: isSpeechSynthesisSupported(), isSpeaking, speak, repeat, stop };
}
