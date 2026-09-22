"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TranslationHistoryPanel, historyEntryText } from "@/components/History/TranslationHistoryPanel";
import { SignToSpeech } from "@/components/Illustration";
import { useTextToSpeech } from "@/hooks/useTextToSpeech";
import type { TranslationHistoryEntry } from "@/hooks/useTranslationHistory";
import { useTranslationHistory } from "@/hooks/useTranslationHistory";
import { useUtterance } from "@/hooks/useUtterance";
import { useWebSocket } from "@/hooks/useWebSocket";
import { ApiError, glossToText, textToGloss } from "@/lib/api";
import type { TranslationDirection } from "@/types/direction";
import type { TranslationState } from "@/types/translation";
import type { CapturedFrame } from "@/types/camera";
import { DirectionSwitch } from "./DirectionSwitch";
import { GestureToTextPanel } from "./GestureToTextPanel";
import { TextToGesturePanel } from "./TextToGesturePanel";

/** Whether a captured camera frame should actually be sent to the
 * backend right now. A plain, exported predicate (rather than inlined in
 * handleFrame) so the gating logic -- paused, a correction popover is
 * open, or the last confirmed word is still awaiting an explicit
 * confirm -- is unit-testable without needing to drive an actual camera
 * capture loop through jsdom. */
export function shouldSendFrame(
  isTranslating: boolean,
  correctionActive: boolean,
  awaitingConfirmation: boolean
): boolean {
  return isTranslating && !correctionActive && !awaitingConfirmation;
}

export function TranslatorView(): React.ReactElement {
  const { status, landmarksStatus, letterMessage, wordMessage, boundaryMessage, connect, disconnect, sendFrame } =
    useWebSocket();
  const [direction, setDirection] = useState<TranslationDirection>("gestures-to-text");
  const [inputText, setInputText] = useState("");
  const [translationState, setTranslationState] = useState<TranslationState>({ status: "idle" });
  // GestureVideo (real reference clips, see components/GestureVideo/) shows
  // exactly the sentence just translated via text/voice above, or the
  // sentence just finalized from the camera below.
  const [translationGlossSequence, setTranslationGlossSequence] = useState<string[]>([]);

  // Continuous dictation: the camera runs all the time (no "Почати
  // речення" gate needed for normal use -- see useUtterance.ts and
  // websocket/handler.py's "boundary" events). cameraStatus is purely
  // about what the LIVE line shows while a translation is being composed;
  // the finalized result itself lives in translationHistory, a completely
  // separate piece of state that new signing activity never touches (see
  // its own module docstring -- this is the fix for "the translated
  // sentence appears and then immediately disappears").
  const [cameraStatus, setCameraStatus] = useState<"listening" | "processing">("listening");
  const [cameraError, setCameraError] = useState<string | null>(null);
  // Start/Pause/Resume (Part 3 of the product overhaul): gates whether
  // captured frames are actually sent to the backend, without touching the
  // camera preview or the WebSocket connection itself -- pausing just
  // means recognition stops consuming new frames for a while.
  const [isTranslating, setIsTranslating] = useState(true);
  // Continuous dictation starts active by default (see isTranslating's own
  // comment) -- hasStartedOnce mirrors that so a later Pause shows
  // "Продовжити", never "Почати переклад" a second time.
  const [hasStartedOnce, setHasStartedOnce] = useState(true);
  const translationHistory = useTranslationHistory();
  const tts = useTextToSpeech();

  const handleUtteranceComplete = useCallback(async (sequence: string[]) => {
    setCameraStatus("processing");
    setCameraError(null);
    try {
      const result = await glossToText(sequence);
      translationHistory.addEntry(
        result.composed_text,
        sequence,
        result.gloss_labels.map((label) => ({ text: label.text, isFingerspell: label.is_fingerspell })),
        "gestures-to-text",
        null,
        result.composed_text_source
      );
      setTranslationGlossSequence(sequence);
    } catch (err) {
      setCameraError(err instanceof ApiError ? err.message : "Не вдалося скласти речення.");
    } finally {
      setCameraStatus("listening");
    }
  }, [translationHistory]);

  const utterance = useUtterance({
    wordMessage,
    letterMessage,
    boundaryMessage,
    onUtteranceComplete: handleUtteranceComplete,
  });

  // isTranslating is read from a ref, not the closure, so this callback's
  // identity stays stable across pause/resume -- useCamera.ts's capture
  // loop captures whichever onFrame it was given at the moment start() was
  // called and never re-reads it, so a changing identity here would NOT
  // actually stop frames once already streaming (see useCamera.ts).
  const isTranslatingRef = useRef(isTranslating);
  useEffect(() => {
    isTranslatingRef.current = isTranslating;
  }, [isTranslating]);
  // Same ref pattern as isTranslatingRef, for the same reason: read
  // synchronously inside handleFrame's stable closure, not the state
  // itself (which would force handleFrame's identity to change on every
  // correction popover open/close -- useCamera's capture loop wouldn't
  // pick that up once already streaming). Set from GestureToTextPanel's
  // onCorrectionActiveChange -- see CorrectionPopover.tsx's onOpenChange
  // docstring: without this, moving the hand to the mouse to pick a
  // correction can be read as a sign-boundary pause, finalizing whatever
  // was mid-correction into the wrong sentence.
  const correctionActiveRef = useRef(false);
  const handleCorrectionActiveChange = useCallback((active: boolean) => {
    correctionActiveRef.current = active;
  }, []);
  // Same ref pattern again, mirroring utterance.awaitingConfirmation --
  // pauses frames the moment a word is confirmed, until confirmPending()
  // (or a correction) says it's fine to move on to the next sign.
  const awaitingConfirmationRef = useRef(false);
  useEffect(() => {
    awaitingConfirmationRef.current = utterance.awaitingConfirmation;
  }, [utterance.awaitingConfirmation]);
  const handleFrame = useCallback(
    (frame: CapturedFrame) => {
      if (shouldSendFrame(isTranslatingRef.current, correctionActiveRef.current, awaitingConfirmationRef.current)) {
        sendFrame(frame);
      }
    },
    [sendFrame]
  );

  const handleStartOrResume = useCallback(() => {
    setIsTranslating(true);
    setHasStartedOnce(true);
  }, []);
  const handlePause = useCallback(() => setIsTranslating(false), []);

  // Manual fallback controls (see Part 1 of the continuous-dictation plan:
  // "manual controls may remain as fallback/debug controls, but continuous
  // signing should work without them") -- for when automatic pause
  // detection doesn't fire the way the signer expects (e.g. the camera
  // keeps seeing a hand that isn't really signing anymore).
  const handleFinishNow = useCallback(() => {
    if (utterance.rawGlossSequence.length === 0) return;
    void handleUtteranceComplete(utterance.rawGlossSequence);
    utterance.reset();
  }, [utterance, handleUtteranceComplete]);

  const handleResetPhrase = useCallback(() => {
    utterance.reset();
    setCameraError(null);
  }, [utterance]);

  // "Очистити все": current utterance + current fingerspelling + final
  // result -- but NEVER history (that has its own separate, explicitly
  // confirmed "Очистити історію" action). GestureToTextPanel asks for
  // confirmation before calling this.
  const handleClearAll = useCallback(() => {
    utterance.reset();
    setCameraError(null);
    translationHistory.clearCurrent();
  }, [utterance, translationHistory]);

  useEffect(() => {
    connect();
    return () => disconnect();
    // Connect once on mount; connect/disconnect identities are stable (useCallback).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleTranslate = useCallback(async (text: string) => {
    setTranslationState({ status: "loading" });
    try {
      const result = await textToGloss(text);
      const glossLabels = result.gloss_labels.map((label) => ({
        text: label.text,
        isFingerspell: label.is_fingerspell,
      }));
      setTranslationState({
        status: "success",
        glossSequence: result.gloss_sequence,
        glossLabels,
        composedText: result.composed_text,
      });
      setTranslationGlossSequence(result.gloss_sequence);
      translationHistory.addEntry(result.composed_text, result.gloss_sequence, glossLabels, "text-to-gestures", text);
    } catch (err) {
      setTranslationState({
        status: "error",
        message: err instanceof ApiError ? err.message : "Не вдалося перекласти текст.",
      });
    }
  }, [translationHistory]);

  const handleClearText = useCallback(() => {
    setInputText("");
    setTranslationState({ status: "idle" });
    setTranslationGlossSequence([]);
  }, []);

  const handlePasteText = useCallback(() => {
    void navigator.clipboard
      ?.readText?.()
      .then((text) => {
        if (text) setInputText(text);
      })
      .catch(() => {
        // Clipboard read denied/unavailable -- a no-op, never a crash or a
        // raw browser error surfaced to the signer.
      });
  }, []);

  const handleRepeatTranslate = useCallback(() => {
    if (inputText.trim()) void handleTranslate(inputText);
  }, [inputText, handleTranslate]);

  const handleReuseHistoryEntry = useCallback((entry: TranslationHistoryEntry) => {
    const text = entry.sourceText ?? historyEntryText(entry);
    setDirection("text-to-gestures");
    setInputText(text);
    setTranslationGlossSequence(entry.glossSequence);
    setTranslationState({
      status: "success",
      glossSequence: entry.glossSequence,
      glossLabels: entry.glossLabels,
      composedText: entry.composedText,
    });
  }, []);

  const handleCopyHistoryEntry = useCallback((entry: TranslationHistoryEntry) => {
    void navigator.clipboard?.writeText?.(historyEntryText(entry)).catch(() => {});
  }, []);

  const handleSpeakHistoryEntry = useCallback(
    (entry: TranslationHistoryEntry) => tts.speak(historyEntryText(entry)),
    [tts]
  );

  return (
    <div className="flex flex-col gap-6">
      <section className="grid items-center gap-6 py-4 sm:grid-cols-[1.1fr_0.9fr] sm:py-8">
        <div className="flex flex-col gap-3">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Жест стає словом миттєво
          </h1>
          <p className="max-w-md text-sm text-slate-500 sm:text-base">
            Камера розпізнає жест української жестової мови й одразу перекладає його на
            граматично коректне речення — а текст і голос перекладає назад у жести.
          </p>
        </div>
        <div className="mx-auto h-44 w-full max-w-sm sm:h-56">
          <SignToSpeech />
        </div>
      </section>

      <DirectionSwitch direction={direction} onChange={setDirection} />

      {direction === "gestures-to-text" ? (
        <GestureToTextPanel
          onFrame={handleFrame}
          landmarksStatus={landmarksStatus}
          wsStatus={status}
          wordMessage={wordMessage}
          letterMessage={letterMessage}
          utterance={utterance}
          translationHistory={translationHistory}
          tts={tts}
          cameraStatus={cameraStatus}
          cameraError={cameraError}
          isTranslating={isTranslating}
          hasStartedOnce={hasStartedOnce}
          onStartOrResume={handleStartOrResume}
          onPause={handlePause}
          onFinishNow={handleFinishNow}
          onResetPhrase={handleResetPhrase}
          onClearAll={handleClearAll}
          onCorrectionActiveChange={handleCorrectionActiveChange}
        />
      ) : (
        <TextToGesturePanel
          inputText={inputText}
          onInputTextChange={setInputText}
          onTranslate={(text) => void handleTranslate(text)}
          onClear={handleClearText}
          onPaste={handlePasteText}
          onRepeatTranslate={handleRepeatTranslate}
          translationState={translationState}
          translationGlossSequence={translationGlossSequence}
        />
      )}

      <TranslationHistoryPanel
        history={translationHistory.history}
        onCopy={handleCopyHistoryEntry}
        onSpeak={handleSpeakHistoryEntry}
        onReuse={handleReuseHistoryEntry}
        onDelete={translationHistory.removeEntry}
        onClearAll={translationHistory.clearHistory}
        ttsSupported={tts.isSupported}
      />
    </div>
  );
}
