import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { BoundaryMessage, LetterCandidate, LetterPredictionMessage, PredictionMessage, WordCandidate } from "@/types/api";
import { useUtterance } from "./useUtterance";

function word(gloss: string, isFinal: boolean, confidence = 0.9, topK: WordCandidate[] | null = null): PredictionMessage {
  return {
    type: isFinal ? "final_prediction" : "prediction",
    text: gloss,
    gloss,
    confidence,
    is_final: isFinal,
    facial_grammar: "NONE",
    top_k: topK,
  };
}

function letter(
  value: string,
  isFinal: boolean,
  confidence = 0.9,
  topK: LetterCandidate[] | null = null
): LetterPredictionMessage {
  return {
    type: isFinal ? "letter_confirmed" : "letter_prediction",
    letter: value,
    confidence,
    is_final: isFinal,
    top_k: topK,
  };
}

function boundary(kind: "token" | "utterance"): BoundaryMessage {
  return { type: "boundary", kind };
}

describe("useUtterance", () => {
  it("appends a confirmed word directly as a WordToken", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "word", value: "WANT", confidence: 0.9 }),
    ]);
    expect(result.current.rawGlossSequence).toEqual(["WANT"]);
  });

  it("does NOT append an interim (unconfirmed) word prediction", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("WANT", false), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    expect(result.current.confirmedTokens).toEqual([]);
    expect(result.current.interimWordGloss).toBe("WANT");
  });

  it("keeps several word tokens in the SAME utterance across 'token' boundaries -- a token boundary never ends the whole sentence", () => {
    // The exact scenario from the spec: I -> [token] -> WANT -> [token] ->
    // WATER -> [utterance] must produce ONE utterance [I, WANT, WATER],
    // not three separate ones.
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    expect(onUtteranceComplete).not.toHaveBeenCalled();
    expect(result.current.confirmedTokens).toHaveLength(1);

    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token") /* new object */, onUtteranceComplete });
    expect(onUtteranceComplete).not.toHaveBeenCalled();
    expect(result.current.confirmedTokens).toHaveLength(2);

    rerender({ wordMessage: word("WATER", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("utterance"), onUtteranceComplete });

    expect(onUtteranceComplete).toHaveBeenCalledTimes(1);
    expect(onUtteranceComplete).toHaveBeenCalledWith(["I", "WANT", "WATER"]);
  });

  it("a 'token' boundary with nothing being fingerspelled does not insert an empty/phantom token", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    // No letters at all yet.
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    expect(result.current.confirmedTokens).toEqual([]);

    // A confirmed word, then a token boundary with no pending letters --
    // must not add a spurious second token alongside the real one.
    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token") /* new object */, onUtteranceComplete });
    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "I" })]);
  });

  it("an interim (unconfirmed) letter_prediction never joins currentFingerspelledLetters", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: null, letterMessage: letter("Н", false), boundaryMessage: null, onUtteranceComplete });
    expect(result.current.currentFingerspelledLetters).toEqual([]);
    expect(result.current.interimLetter).toBe("Н");

    // A confirmed letter follows the interim guess -- only the confirmed
    // one is buffered, never a duplicate from the earlier interim.
    rerender({ wordMessage: null, letterMessage: letter("Н", true), boundaryMessage: null, onUtteranceComplete });
    expect(result.current.currentFingerspelledLetters).toHaveLength(1);
  });

  it("raw recognition (the sequence handed to onUtteranceComplete) is preserved verbatim, independent of composition", () => {
    // useUtterance's only job is to hand off the raw sequence -- it must
    // never be lost, reordered, or mutated by anything composition-related
    // (which is entirely the caller's concern, see onUtteranceComplete).
    let handedOff: string[] | null = null;
    const onUtteranceComplete = vi.fn((sequence: string[]) => {
      handedOff = sequence;
    });
    const { rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    for (const value of ["Н", "А"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("utterance") /* new object */, onUtteranceComplete });

    expect(handedOff).toEqual(["I", "FS_Н", "FS_А", "WANT"]);
  });

  it("accumulates confirmed letters into currentFingerspelledLetters, not directly into confirmedTokens", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["Н", "А", "С", "Т", "Я"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }

    expect(result.current.confirmedTokens).toEqual([]);
    expect(result.current.currentFingerspelledLetters.map((l) => l.value)).toEqual(["Н", "А", "С", "Т", "Я"]);
  });

  it("Test B: finalizes the letter buffer into ONE FingerspelledToken on a 'token' boundary, then lets the utterance continue", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    for (const value of ["Н", "А", "С", "Т", "Я"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });

    expect(result.current.currentFingerspelledLetters).toEqual([]);
    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "word", value: "I" }),
      expect.objectContaining({ kind: "fingerspelled", value: "НАСТЯ" }),
    ]);

    // The utterance continues -- a word after the fingerspelled word.
    rerender({ wordMessage: word("KNOW", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.rawGlossSequence).toEqual([
      "I",
      "FS_Н",
      "FS_А",
      "FS_С",
      "FS_Т",
      "FS_Я",
      "KNOW",
    ]);
    expect(onUtteranceComplete).not.toHaveBeenCalled();
  });

  it("Test C: two separate fingerspelled words in the same utterance become two separate FingerspelledTokens, not one merged run", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["А", "Б", "О"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });

    for (const value of ["В", "І"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token") /* new object */, onUtteranceComplete });

    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "fingerspelled", value: "АБО" }),
      expect.objectContaining({ kind: "fingerspelled", value: "ВІ" }),
    ]);
    // Never merged into one run -- exactly two distinct tokens, each with
    // its own letters, and the wire-format sequence keeps them as two
    // separate FS_ runs, separated by FINGERSPELL_WORD_BOUNDARY: without
    // it, "тобі" + "допомогти" -- both fingerspelled -- would flatten into
    // one indistinguishable run and despell back as the wrong single word
    // "тобідопомогти".
    expect(result.current.rawGlossSequence).toEqual(["FS_А", "FS_Б", "FS_О", "FS_BOUNDARY", "FS_В", "FS_І"]);
  });

  it("Test A: an 'utterance' boundary finalizes and hands off the whole raw sequence, then clears state for the next one", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const gloss of ["I", "WANT", "WATER"]) {
      rerender({ wordMessage: word(gloss, true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("utterance"), onUtteranceComplete });

    expect(onUtteranceComplete).toHaveBeenCalledTimes(1);
    expect(onUtteranceComplete).toHaveBeenCalledWith(["I", "WANT", "WATER"]);
    expect(result.current.confirmedTokens).toEqual([]);
    expect(result.current.rawGlossSequence).toEqual([]);
  });

  it("an 'utterance' boundary with nothing signed does not call onUtteranceComplete", () => {
    const onUtteranceComplete = vi.fn();
    const { rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("utterance"), onUtteranceComplete });

    expect(onUtteranceComplete).not.toHaveBeenCalled();
  });

  it("an 'utterance' boundary also finalizes a pending fingerspelled word first, including it in the handed-off sequence", () => {
    const onUtteranceComplete = vi.fn();
    const { rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["Х", "І"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    // Straight to "utterance" -- no separate "token" boundary first.
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("utterance"), onUtteranceComplete });

    expect(onUtteranceComplete).toHaveBeenCalledWith(["FS_Х", "FS_І"]);
  });

  it("reset() clears everything without calling onUtteranceComplete", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.confirmedTokens).toHaveLength(1);

    act(() => result.current.reset());

    expect(result.current.confirmedTokens).toEqual([]);
    expect(result.current.currentFingerspelledLetters).toEqual([]);
    expect(onUtteranceComplete).not.toHaveBeenCalled();
  });

  it("awaitingConfirmation becomes true the moment a word is confirmed", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });
    expect(result.current.awaitingConfirmation).toBe(false);

    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    expect(result.current.awaitingConfirmation).toBe(true);
  });

  it("confirmPending() clears awaitingConfirmation without touching the confirmed token itself", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.awaitingConfirmation).toBe(true);

    act(() => result.current.confirmPending());

    expect(result.current.awaitingConfirmation).toBe(false);
    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "WANT" })]);
  });

  it("finalizing a fingerspelled word (via a boundary) also sets awaitingConfirmation", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: null, letterMessage: letter("А", true), boundaryMessage: null, onUtteranceComplete });
    // Individual letters, still mid-word, never gate recognition on their own.
    expect(result.current.awaitingConfirmation).toBe(false);

    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });

    expect(result.current.awaitingConfirmation).toBe(true);
  });

  it("correctWordToken counts as confirming -- awaitingConfirmation clears when a correction is picked", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.awaitingConfirmation).toBe(true);

    act(() => result.current.correctWordToken(0, { gloss: "HAVE", confidence: 0.4 }));

    expect(result.current.awaitingConfirmation).toBe(false);
  });

  it("undoLastToken also clears awaitingConfirmation for the token it removes", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.awaitingConfirmation).toBe(true);

    act(() => result.current.undoLastToken());

    expect(result.current.awaitingConfirmation).toBe(false);
  });

  it("an 'utterance' boundary clears awaitingConfirmation along with everything else it finalizes", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.awaitingConfirmation).toBe(true);

    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: boundary("utterance"), onUtteranceComplete });

    expect(result.current.awaitingConfirmation).toBe(false);
  });

  it("undoLastToken removes only the most recently confirmed token", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    expect(result.current.confirmedTokens).toHaveLength(2);

    act(() => result.current.undoLastToken());

    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "I" })]);
    expect(result.current.rawGlossSequence).toEqual(["I"]);
  });

  it("undoLastToken is a no-op when nothing is confirmed yet", () => {
    const onUtteranceComplete = vi.fn();
    const { result } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    act(() => result.current.undoLastToken());

    expect(result.current.confirmedTokens).toEqual([]);
  });

  it("undoLastLetter removes only the last letter of the in-progress fingerspelling buffer, never a confirmed token", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    for (const value of ["Н", "А", "С"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }

    act(() => result.current.undoLastLetter());

    expect(result.current.currentFingerspelledLetters.map((l) => l.value)).toEqual(["Н", "А"]);
    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "I" })]);
  });

  it("undoLastLetter is a no-op when the fingerspelling buffer is empty", () => {
    const onUtteranceComplete = vi.fn();
    const { result } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    act(() => result.current.undoLastLetter());

    expect(result.current.currentFingerspelledLetters).toEqual([]);
  });

  it("finishFingerspelledWord manually finalizes the buffer into a token, without waiting for a boundary event", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["Н", "А", "С", "Т", "Я"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }

    act(() => result.current.finishFingerspelledWord());

    expect(result.current.currentFingerspelledLetters).toEqual([]);
    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "fingerspelled", value: "НАСТЯ" }),
    ]);
    expect(onUtteranceComplete).not.toHaveBeenCalled();
  });

  it("finishFingerspelledWord is a no-op when the buffer is empty -- never inserts a phantom token", () => {
    const onUtteranceComplete = vi.fn();
    const { result } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    act(() => result.current.finishFingerspelledWord());

    expect(result.current.confirmedTokens).toEqual([]);
  });

  it("clearFingerspellingBuffer discards the in-progress letters without creating a token, and never touches confirmedTokens", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    for (const value of ["Н", "А"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }

    act(() => result.current.clearFingerspellingBuffer());

    expect(result.current.currentFingerspelledLetters).toEqual([]);
    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "I" })]);

    // The buffer being cleared doesn't resurrect on a later boundary --
    // confirms the ref backing it was actually cleared, not just the
    // state mirror.
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "I" })]);
  });

  it("a confirmed word token carries the real top_k alternatives from the message that confirmed it", () => {
    const onUtteranceComplete = vi.fn();
    const topK = [{ gloss: "WANT", confidence: 0.6 }, { gloss: "HAVE", confidence: 0.3 }];
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("WANT", true, 0.9, topK), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    expect(result.current.confirmedTokens).toEqual([expect.objectContaining({ kind: "word", value: "WANT", topK })]);
  });

  it("correctWordToken replaces a confirmed word's gloss and confidence with the chosen candidate", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("WANT", true, 0.6), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    act(() => result.current.correctWordToken(0, { gloss: "HAVE", confidence: 0.3 }));

    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "word", value: "HAVE", confidence: 0.3 }),
    ]);
    expect(result.current.rawGlossSequence).toEqual(["HAVE"]);
  });

  it("correctWordToken only touches the token at the given index, leaving others alone", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    rerender({ wordMessage: word("WANT", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    act(() => result.current.correctWordToken(1, { gloss: "HAVE", confidence: 0.3 }));

    expect(result.current.rawGlossSequence).toEqual(["I", "HAVE"]);
  });

  it("correctWordToken is a no-op for an out-of-range index or a non-word token", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    act(() => result.current.correctWordToken(5, { gloss: "HAVE", confidence: 0.3 }));
    expect(result.current.rawGlossSequence).toEqual(["I"]);

    for (const value of ["Н", "А"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    // index 1 is now the finalized fingerspelled word, not a WordToken.
    act(() => result.current.correctWordToken(1, { gloss: "HAVE", confidence: 0.3 }));
    expect(result.current.rawGlossSequence).toEqual(["I", "FS_Н", "FS_А"]);
  });

  it("a confirmed letter in the fingerspelling buffer carries its real top_k alternatives", () => {
    const onUtteranceComplete = vi.fn();
    const topK = [{ letter: "С", confidence: 0.5 }, { letter: "Ш", confidence: 0.3 }];
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: null, letterMessage: letter("С", true, 0.5, topK), boundaryMessage: null, onUtteranceComplete });

    expect(result.current.currentFingerspelledLetters).toEqual([expect.objectContaining({ value: "С", topK })]);
  });

  it("correctLetter replaces one letter in the in-progress buffer without touching the others", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["Н", "А", "Ш"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }

    act(() => result.current.correctLetter(2, { letter: "С", confidence: 0.5 }));

    expect(result.current.currentFingerspelledLetters.map((l) => l.value)).toEqual(["Н", "А", "С"]);
  });

  it("correctLetter is a no-op for an out-of-range index", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: null, letterMessage: letter("Н", true), boundaryMessage: null, onUtteranceComplete });

    act(() => result.current.correctLetter(5, { letter: "С", confidence: 0.5 }));

    expect(result.current.currentFingerspelledLetters.map((l) => l.value)).toEqual(["Н"]);
  });

  it("correctFinalizedLetter fixes one letter of an already-finalized fingerspelled word and keeps its value in sync", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["Н", "А", "С", "Т", "Я"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "fingerspelled", value: "НАСТЯ" }),
    ]);

    // "С" was actually "Ш" -- fix letter index 2 in the finalized word.
    act(() => result.current.correctFinalizedLetter(0, 2, { letter: "Ш", confidence: 0.6 }));

    const token = result.current.confirmedTokens[0];
    expect(token).toEqual(
      expect.objectContaining({
        kind: "fingerspelled",
        value: "НАШТЯ",
      })
    );
    if (token?.kind === "fingerspelled") {
      expect(token.letters.map((l) => l.value)).toEqual(["Н", "А", "Ш", "Т", "Я"]);
      expect(token.letters[2]?.confidence).toBe(0.6);
    }
    expect(result.current.rawGlossSequence).toEqual(["FS_Н", "FS_А", "FS_Ш", "FS_Т", "FS_Я"]);
  });

  it("correctFinalizedLetter is a no-op for an out-of-range token index, an out-of-range letter index, or a non-fingerspelled token", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: word("I", true), letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    for (const value of ["Н", "А"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    // confirmedTokens is now [WordToken("I"), FingerspelledToken("НА")].

    act(() => result.current.correctFinalizedLetter(0, 0, { letter: "Ш", confidence: 0.6 }));
    expect(result.current.rawGlossSequence).toEqual(["I", "FS_Н", "FS_А"]);

    act(() => result.current.correctFinalizedLetter(1, 9, { letter: "Ш", confidence: 0.6 }));
    expect(result.current.rawGlossSequence).toEqual(["I", "FS_Н", "FS_А"]);

    act(() => result.current.correctFinalizedLetter(9, 0, { letter: "Ш", confidence: 0.6 }));
    expect(result.current.rawGlossSequence).toEqual(["I", "FS_Н", "FS_А"]);
  });

  it("correctFinalizedLetter only touches the targeted token, leaving other finalized words alone", () => {
    const onUtteranceComplete = vi.fn();
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    for (const value of ["А", "Б", "О"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token"), onUtteranceComplete });
    for (const value of ["В", "І"]) {
      rerender({ wordMessage: null, letterMessage: letter(value, true), boundaryMessage: null, onUtteranceComplete });
    }
    rerender({ wordMessage: null, letterMessage: null, boundaryMessage: boundary("token") /* new object */, onUtteranceComplete });

    act(() => result.current.correctFinalizedLetter(1, 0, { letter: "Т", confidence: 0.6 }));

    expect(result.current.confirmedTokens).toEqual([
      expect.objectContaining({ kind: "fingerspelled", value: "АБО" }),
      expect.objectContaining({ kind: "fingerspelled", value: "ТІ" }),
    ]);
  });

  it("does not double-append the same message object across a re-render (StrictMode double-invoke safety)", () => {
    const onUtteranceComplete = vi.fn();
    const w = word("I", true);
    const { result, rerender } = renderHook((props) => useUtterance(props), {
      initialProps: { wordMessage: null as PredictionMessage | null, letterMessage: null as LetterPredictionMessage | null, boundaryMessage: null as BoundaryMessage | null, onUtteranceComplete },
    });

    rerender({ wordMessage: w, letterMessage: null, boundaryMessage: null, onUtteranceComplete });
    // Same object reference, unrelated re-render (e.g. a parent state
    // change) -- must not append a second time.
    rerender({ wordMessage: w, letterMessage: null, boundaryMessage: null, onUtteranceComplete });

    expect(result.current.confirmedTokens).toHaveLength(1);
  });
});
