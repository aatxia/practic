/**
 * Structured continuous-dictation utterance tokens (see hooks/useUtterance.ts).
 *
 * Distinct from the raw gloss sequence lib/glossDisplay.ts groups for
 * display: these preserve confidence and timing per confirmed unit, not
 * just a display label, and a fingerspelled word is one token with its
 * own per-letter detail, not a run of separate letter entries a caller
 * has to re-group.
 */
import type { LetterCandidate, WordCandidate } from "./api";

/** One confirmed dynamic sign (backend's "final_prediction"). */
export interface WordToken {
  kind: "word";
  /** Raw gloss code, e.g. "WANT" -- an internal identifier, not Ukrainian
   * text (see lib/glossLabels.ts for the display form). */
  value: string;
  confidence: number;
  /** Client-side Date.now() at the moment this token was confirmed --
   * the backend doesn't currently timestamp individual predictions, so
   * this measures confirmation order/timing, not a video-frame-accurate
   * sign start/end. */
  timestamp: number;
  /** The model's real ranked alternatives at the moment this word was
   * confirmed (types/api.ts's PredictionMessage.top_k) -- lets a person
   * correct a wrong CONFIRMED word from what the model itself actually
   * ranked, never a guessed list. null when the message that confirmed
   * this token carried none. */
  topK: WordCandidate[] | null;
}

/** One confirmed dactyl letter (backend's "letter_confirmed") -- not a
 * standalone utterance token on its own; it only ever appears inside a
 * FingerspelledToken.letters, accumulated by useUtterance until a token
 * boundary finalizes the word it's spelling. */
export interface LetterToken {
  value: string;
  confidence: number;
  timestamp: number;
  /** Mirrors WordToken.topK for the fingerspelling classifier. */
  topK: LetterCandidate[] | null;
}

/** One finalized fingerspelled word -- assembled from a run of confirmed
 * letters (useUtterance's currentFingerspelledLetters) once a "token"
 * boundary event says the pause means the word is done, not just a
 * between-letters release. `value` is exactly the concatenated letters,
 * lowercased first-letter-capitalized for display (see
 * lib/glossDisplay.ts's existing convention) -- never dictionary-corrected
 * (Phase 5's lexical decoder is a later, separate concern), so an unknown
 * name or term always comes through as typed. */
export interface FingerspelledToken {
  kind: "fingerspelled";
  value: string;
  letters: LetterToken[];
  /** Timestamp of the token-boundary finalization, not the first letter --
   * matches WordToken's "when this token became part of the utterance". */
  timestamp: number;
}

export type UtteranceToken = WordToken | FingerspelledToken;
