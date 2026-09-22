/**
 * Centralized playback timing/speed constants for GestureVideo.tsx --
 * kept here, not hardcoded inline in the component, so a transition or
 * speed option only needs changing in one place.
 */

/** Pause inserted automatically between two consecutive clips, in ms. */
export const PLAYBACK_TRANSITIONS_MS = {
  /** Between two letters of the same fingerspelled word -- short, since
   * it's still one word being spelled out. */
  letterToLetter: 150,
  /** Between two different words/gestures (including the last letter of a
   * fingerspelled word and the next word) -- longer, a real word/sentence
   * boundary the viewer should be able to notice. */
  wordToWord: 500,
} as const;

export const PLAYBACK_SPEEDS: readonly number[] = [0.5, 0.75, 1, 1.5, 2];
export const DEFAULT_PLAYBACK_SPEED = 1;
