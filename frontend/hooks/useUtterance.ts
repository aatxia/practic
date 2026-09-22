"use client";

/**
 * Continuous-dictation utterance state machine (Phase 2/3 of the
 * continuous-dictation redesign -- see PROJECT_STATUS.md). Replaces the
 * old flat liveGlossSequence: word confirmations become WordTokens
 * immediately; confirmed letters accumulate into a temporary buffer and
 * only become one FingerspelledToken once a backend "boundary" event
 * (websocket/protocol.py) says the pause means the fingerspelled word is
 * done, not just a between-letters release. A longer "utterance" boundary
 * additionally finalizes the whole accumulated sequence and hands it to
 * the caller for language post-processing (typically POST
 * /translate/gloss-to-text) -- this hook never calls that itself, keeping
 * it independent of any particular composition backend (see Part 6 of the
 * continuous-dictation plan: a future LM post-editor is a drop-in
 * replacement for whatever onUtteranceComplete does with the sequence).
 *
 * Continuous by design: nothing here requires a manual "start" -- it
 * reacts to whichever word/letter/boundary messages useWebSocket already
 * exposes, for as long as the connection is open. A caller wanting a
 * manual fallback control (see TranslatorView.tsx) just calls reset().
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { FINGERSPELL_WORD_BOUNDARY, isFingerspellGloss } from "@/lib/glossDisplay";
import type { BoundaryMessage, LetterCandidate, LetterPredictionMessage, PredictionMessage, WordCandidate } from "@/types/api";
import type { FingerspelledToken, LetterToken, UtteranceToken, WordToken } from "@/types/tokens";

interface UseUtteranceOptions {
  wordMessage: PredictionMessage | null;
  letterMessage: LetterPredictionMessage | null;
  boundaryMessage: BoundaryMessage | null;
  /** Called once per "utterance" boundary, with the just-finalized raw
   * gloss sequence -- only when it's non-empty (nothing was signed isn't
   * an utterance). Expands each token back into the wire gloss format
   * ml/nlp/gloss_to_text.py already understands, including a
   * FingerspelledToken as a run of "FS_<letter>" tokens -- no backend
   * change needed to consume this. */
  onUtteranceComplete: (glossSequence: string[]) => void;
}

export interface UseUtteranceResult {
  /** Finalized tokens in the current utterance so far, in confirmation
   * order -- words and finalized fingerspelled words mixed together.
   * Cleared the moment onUtteranceComplete fires for them. */
  confirmedTokens: UtteranceToken[];
  /** True right after the LAST entry of confirmedTokens (a word, or a
   * just-finalized fingerspelled word) was confirmed, until confirmPending()
   * is called or that token is corrected/undone. A caller pauses sending
   * new camera frames while this is true, so continuous recognition
   * doesn't run ahead onto the next sign before this one is acknowledged --
   * otherwise a run of the same word can get confirmed several times with
   * nothing between them ("Я мати мати мати мати мати"). Never true for an
   * in-progress fingerspelling letter -- only a completed WORD gates
   * recognition, not each individual letter. */
  awaitingConfirmation: boolean;
  /** Letters accumulated for the fingerspelled word currently being
   * spelled, not yet finalized -- empty when no fingerspelling is in
   * progress. Grows one letter at a time as letter_confirmed messages
   * arrive; a caller renders this as the live growing word. */
  currentFingerspelledLetters: LetterToken[];
  /** Current unconfirmed word guess (backend's interim "prediction"), or
   * null when none is in flight. */
  interimWordGloss: string | null;
  /** Current unconfirmed letter guess, or null. */
  interimLetter: string | null;
  /** confirmedTokens expanded back into the raw wire gloss format --
   * what a manual "translate now" fallback control would send. */
  rawGlossSequence: string[];
  /** Clears all accumulated state (confirmedTokens, the in-progress
   * fingerspelling buffer) -- for a manual "reset session" fallback
   * control, not needed for normal continuous use. */
  reset: () => void;
  /** Removes the last confirmed token (a WordToken or a finalized
   * FingerspelledToken) from the current utterance -- a manual correction
   * for a misrecognized sign, without discarding everything confirmed
   * before it. No-op when there is nothing confirmed yet. Never touches
   * the in-progress fingerspelling buffer -- see undoLastLetter for that. */
  undoLastToken: () => void;
  /** Removes the last confirmed letter from the in-progress fingerspelling
   * buffer -- a manual correction for a misrecognized letter, mid-word.
   * No-op when the buffer is empty. Never touches confirmedTokens. */
  undoLastLetter: () => void;
  /** Manually finalizes whatever is in the fingerspelling buffer into a
   * FingerspelledToken right now, without waiting for a backend "token"
   * boundary -- for when the signer is done spelling but the automatic
   * pause detection hasn't fired yet. No-op when the buffer is empty. */
  finishFingerspelledWord: () => void;
  /** Clears only the in-progress fingerspelling buffer, discarding those
   * letters without turning them into a token -- for when the signer
   * wants to restart spelling a word from scratch. Never touches
   * confirmedTokens. */
  clearFingerspellingBuffer: () => void;
  /** Replaces a CONFIRMED word token's gloss with one of its own real
   * ranked alternatives (WordToken.topK) -- for when recognition confirms
   * the wrong word and it's already saved. `tokenIndex` indexes
   * confirmedTokens; a no-op for an out-of-range index or a token that
   * isn't kind "word" (correcting a fingerspelled word's overall value
   * isn't well-defined the same way -- see correctLetter for its letters). */
  correctWordToken: (tokenIndex: number, candidate: WordCandidate) => void;
  /** Replaces one letter in the in-progress fingerspelling buffer (not yet
   * finalized into a token) with one of its own real ranked alternatives
   * (LetterToken.topK) -- for correcting a misrecognized letter mid-word
   * without retyping everything after it. `letterIndex` indexes
   * currentFingerspelledLetters; a no-op for an out-of-range index. */
  correctLetter: (letterIndex: number, candidate: LetterCandidate) => void;
  /** Mirrors correctLetter for a letter inside an ALREADY-FINALIZED
   * fingerspelled word (confirmedTokens[tokenIndex].letters) -- both the
   * letter and the token's own concatenated `value` are updated together,
   * so the finalized word stays internally consistent. `tokenIndex` indexes
   * confirmedTokens, `letterIndex` indexes that token's own letters; a
   * no-op for an out-of-range index or a token that isn't kind
   * "fingerspelled". */
  correctFinalizedLetter: (tokenIndex: number, letterIndex: number, candidate: LetterCandidate) => void;
  /** Clears awaitingConfirmation without changing the pending word itself
   * -- "yes, that's right, continue" for the plain case where nothing
   * needs correcting. A no-op when nothing is pending. */
  confirmPending: () => void;
}

function tokenToGlossTokens(token: UtteranceToken): string[] {
  return token.kind === "word" ? [token.value] : token.value.split("").map((letter) => `FS_${letter}`);
}

// Two separately-fingerspelled words signed back to back (no dynamic word
// in between) would otherwise both flatten into one continuous run of
// FS_<letter> tokens with nothing marking where one word ends and the next
// starts -- the same ambiguity ml/nlp/text_to_gloss.py guards against for
// typed text (see FINGERSPELL_WORD_BOUNDARY's own docstring). Inserted only
// when it's actually needed, i.e. right before a FingerspelledToken whose
// letters would otherwise sit directly after another fingerspelled run.
function tokensToGlossSequence(tokens: UtteranceToken[]): string[] {
  const sequence: string[] = [];
  for (const token of tokens) {
    if (token.kind === "fingerspelled" && isFingerspellGloss(sequence[sequence.length - 1] ?? "")) {
      sequence.push(FINGERSPELL_WORD_BOUNDARY);
    }
    sequence.push(...tokenToGlossTokens(token));
  }
  return sequence;
}

export function useUtterance({
  wordMessage,
  letterMessage,
  boundaryMessage,
  onUtteranceComplete,
}: UseUtteranceOptions): UseUtteranceResult {
  const [confirmedTokens, setConfirmedTokens] = useState<UtteranceToken[]>([]);
  const [currentFingerspelledLetters, setCurrentFingerspelledLetters] = useState<LetterToken[]>([]);
  // True right after a WORD (a dynamic sign, or a just-finalized
  // fingerspelled word) is confirmed and appended to confirmedTokens --
  // always the LAST entry, since both pushToken and
  // finalizeFingerspelledWord only ever append. A caller (TranslatorView)
  // pauses sending new camera frames while this is true, so recognition
  // waits for an explicit confirmPending() (or a correction, which counts
  // as one) before moving on to the next word. Without this gate,
  // continuous back-to-back recognition can produce runs of the same word
  // confirmed several times in a row with nothing between them to say
  // "yes, that one, keep going" ("Я мати мати мати мати мати").
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);

  // Ref mirrors of the two state values above: read synchronously inside
  // effects (finalizing a fingerspelled word on a boundary, composing the
  // sequence handed to onUtteranceComplete) without a stale closure or
  // needing the state itself in an effect's dependency array -- which
  // would re-run that effect on every unrelated update to that state.
  const tokensRef = useRef<UtteranceToken[]>([]);
  const lettersRef = useRef<LetterToken[]>([]);

  // "Already processed this exact message object?" guards -- React
  // StrictMode (dev only) double-invokes effects; without these a single
  // confirmation would append twice (same pattern already established in
  // this codebase's useWebSocket.ts connect()).
  const lastWordRef = useRef<PredictionMessage | null>(null);
  const lastLetterRef = useRef<LetterPredictionMessage | null>(null);
  const lastBoundaryRef = useRef<BoundaryMessage | null>(null);

  const pushToken = useCallback((token: UtteranceToken) => {
    tokensRef.current = [...tokensRef.current, token];
    setConfirmedTokens(tokensRef.current);
    setAwaitingConfirmation(true);
  }, []);

  // Shared by the automatic "token" boundary path below and the manual
  // finishFingerspelledWord() control -- both turn whatever's in the
  // buffer into one FingerspelledToken the same way. A no-op for an empty
  // buffer (nothing to finalize).
  const finalizeFingerspelledWord = useCallback(() => {
    if (lettersRef.current.length === 0) return;
    const letters = lettersRef.current;
    lettersRef.current = [];
    setCurrentFingerspelledLetters([]);
    const token: FingerspelledToken = {
      kind: "fingerspelled",
      value: letters.map((letter) => letter.value).join(""),
      letters,
      timestamp: Date.now(),
    };
    tokensRef.current = [...tokensRef.current, token];
    setConfirmedTokens(tokensRef.current);
    setAwaitingConfirmation(true);
  }, []);

  useEffect(() => {
    if (!wordMessage || wordMessage.type !== "final_prediction") return;
    if (lastWordRef.current === wordMessage) return;
    lastWordRef.current = wordMessage;
    const token: WordToken = {
      kind: "word",
      value: wordMessage.gloss,
      confidence: wordMessage.confidence,
      timestamp: Date.now(),
      topK: wordMessage.top_k,
    };
    pushToken(token);
  }, [wordMessage, pushToken]);

  useEffect(() => {
    if (!letterMessage?.is_final) return;
    if (lastLetterRef.current === letterMessage) return;
    lastLetterRef.current = letterMessage;
    lettersRef.current = [
      ...lettersRef.current,
      {
        value: letterMessage.letter,
        confidence: letterMessage.confidence,
        timestamp: Date.now(),
        topK: letterMessage.top_k,
      },
    ];
    setCurrentFingerspelledLetters(lettersRef.current);
  }, [letterMessage]);

  useEffect(() => {
    if (!boundaryMessage) return;
    if (lastBoundaryRef.current === boundaryMessage) return;
    lastBoundaryRef.current = boundaryMessage;

    // Both "token" and "utterance" boundaries finalize a pending
    // fingerspelled word -- an utterance ending is itself a token
    // ending too, never left dangling as an unfinalized buffer.
    finalizeFingerspelledWord();

    if (boundaryMessage.kind === "utterance" && tokensRef.current.length > 0) {
      const sequence = tokensToGlossSequence(tokensRef.current);
      tokensRef.current = [];
      setConfirmedTokens([]);
      setAwaitingConfirmation(false);
      onUtteranceComplete(sequence);
    }
  }, [boundaryMessage, onUtteranceComplete, finalizeFingerspelledWord]);

  const reset = useCallback(() => {
    tokensRef.current = [];
    lettersRef.current = [];
    setConfirmedTokens([]);
    setCurrentFingerspelledLetters([]);
    setAwaitingConfirmation(false);
  }, []);

  const undoLastToken = useCallback(() => {
    if (tokensRef.current.length === 0) return;
    tokensRef.current = tokensRef.current.slice(0, -1);
    setConfirmedTokens(tokensRef.current);
    setAwaitingConfirmation(false);
  }, []);

  const undoLastLetter = useCallback(() => {
    if (lettersRef.current.length === 0) return;
    lettersRef.current = lettersRef.current.slice(0, -1);
    setCurrentFingerspelledLetters(lettersRef.current);
  }, []);

  const clearFingerspellingBuffer = useCallback(() => {
    lettersRef.current = [];
    setCurrentFingerspelledLetters([]);
  }, []);

  const confirmPending = useCallback(() => setAwaitingConfirmation(false), []);

  const correctWordToken = useCallback((tokenIndex: number, candidate: WordCandidate) => {
    const current = tokensRef.current[tokenIndex];
    if (!current || current.kind !== "word") return;
    const corrected: WordToken = { ...current, value: candidate.gloss, confidence: candidate.confidence };
    tokensRef.current = tokensRef.current.map((token, index) => (index === tokenIndex ? corrected : token));
    setConfirmedTokens(tokensRef.current);
    // Picking a correction is itself an act of confirming this word --
    // no separate "now click confirm too" step needed.
    setAwaitingConfirmation(false);
  }, []);

  const correctLetter = useCallback((letterIndex: number, candidate: LetterCandidate) => {
    if (letterIndex < 0 || letterIndex >= lettersRef.current.length) return;
    const current = lettersRef.current[letterIndex]!;
    const corrected: LetterToken = { ...current, value: candidate.letter, confidence: candidate.confidence };
    lettersRef.current = lettersRef.current.map((letter, index) => (index === letterIndex ? corrected : letter));
    setCurrentFingerspelledLetters(lettersRef.current);
  }, []);

  const correctFinalizedLetter = useCallback(
    (tokenIndex: number, letterIndex: number, candidate: LetterCandidate) => {
      const current = tokensRef.current[tokenIndex];
      if (!current || current.kind !== "fingerspelled") return;
      if (letterIndex < 0 || letterIndex >= current.letters.length) return;
      const correctedLetter: LetterToken = {
        ...current.letters[letterIndex]!,
        value: candidate.letter,
        confidence: candidate.confidence,
      };
      const correctedLetters = current.letters.map((letter, index) =>
        index === letterIndex ? correctedLetter : letter
      );
      const corrected: FingerspelledToken = {
        ...current,
        letters: correctedLetters,
        value: correctedLetters.map((letter) => letter.value).join(""),
      };
      tokensRef.current = tokensRef.current.map((token, index) => (index === tokenIndex ? corrected : token));
      setConfirmedTokens(tokensRef.current);
      setAwaitingConfirmation(false);
    },
    []
  );

  return {
    confirmedTokens,
    awaitingConfirmation,
    currentFingerspelledLetters,
    interimWordGloss: wordMessage?.type === "prediction" ? wordMessage.gloss : null,
    interimLetter: letterMessage && !letterMessage.is_final ? letterMessage.letter : null,
    rawGlossSequence: tokensToGlossSequence(confirmedTokens),
    reset,
    undoLastToken,
    undoLastLetter,
    finishFingerspelledWord: finalizeFingerspelledWord,
    clearFingerspellingBuffer,
    correctWordToken,
    correctLetter,
    correctFinalizedLetter,
    confirmPending,
  };
}
