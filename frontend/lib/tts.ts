/**
 * Thin wrapper over the browser's Web Speech API SpeechSynthesis (text ->
 * spoken audio) for reading a finalized Ukrainian translation aloud. This
 * is the OPPOSITE direction of hooks/useSpeechRecognition.ts (voice -> text
 * input) -- unrelated Web Speech API, unrelated flow.
 *
 * Real per this project's "no fake AI" principle: if the browser has no
 * speechSynthesis support, or no Ukrainian voice installed, this fails
 * gracefully (isSupported()/speak() just no-op or report failure) rather
 * than faking speech with the wrong language or throwing.
 */

export function isSpeechSynthesisSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** Best available voice for Ukrainian speech, or null if the browser has
 * none installed -- browsers vary widely in bundled/downloadable voices, so
 * this is never assumed to exist. Falls back to the default voice with
 * lang="uk-UA" set on the utterance either way (some browsers still pick a
 * reasonable voice from the lang alone even with no exact voice match). */
function findUkrainianVoice(): SpeechSynthesisVoice | null {
  if (!isSpeechSynthesisSupported()) return null;
  const voices = window.speechSynthesis.getVoices();
  return voices.find((voice) => voice.lang.toLowerCase().startsWith("uk")) ?? null;
}

export interface SpeakOptions {
  /** Called when this utterance finishes speaking normally. */
  onEnd?: () => void;
  /** Called when this utterance is cancelled or fails to speak. */
  onError?: () => void;
}

/** Speaks `text` aloud in Ukrainian. Cancels any speech already in
 * progress first (never queues/overlaps two utterances). A no-op, not a
 * throw, when speechSynthesis isn't supported at all -- the caller can
 * check isSpeechSynthesisSupported() to decide whether to show the
 * control in the first place. */
export function speak(text: string, options: SpeakOptions = {}): void {
  if (!isSpeechSynthesisSupported() || !text) {
    options.onError?.();
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "uk-UA";
  const voice = findUkrainianVoice();
  if (voice) utterance.voice = voice;
  utterance.onend = () => options.onEnd?.();
  utterance.onerror = () => options.onError?.();
  window.speechSynthesis.speak(utterance);
}

export function cancelSpeech(): void {
  if (!isSpeechSynthesisSupported()) return;
  window.speechSynthesis.cancel();
}
