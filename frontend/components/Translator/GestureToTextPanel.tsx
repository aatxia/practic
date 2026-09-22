"use client";

import { Check, ChevronDown, ChevronUp, Copy, HelpCircle, Pause, Repeat, Sparkles, Trash2, Volume2 } from "lucide-react";
import { Fragment, useCallback, useEffect, useState } from "react";
import { Camera } from "@/components/Camera";
import { LandmarkIndicator } from "@/components/LandmarkIndicator";
import { CorrectionPopover } from "@/components/Translator/CorrectionPopover";
import type { UseTextToSpeechResult } from "@/hooks/useTextToSpeech";
import type { UseTranslationHistoryResult } from "@/hooks/useTranslationHistory";
import type { UseUtteranceResult } from "@/hooks/useUtterance";
import type { WebSocketStatus } from "@/hooks/useWebSocket";
import { glossLabel } from "@/lib/glossLabels";
import type { CapturedFrame } from "@/types/camera";
import type {
  LandmarksStatusMessage,
  LetterCandidate,
  LetterPredictionMessage,
  PredictionMessage,
  WordCandidate,
} from "@/types/api";
import type { LetterToken, WordToken } from "@/types/tokens";

const WS_STATUS_LABEL: Record<WebSocketStatus, string> = {
  idle: "Не з'єднано",
  connecting: "З'єднання...",
  open: "З'єднано",
  closed: "З'єднання закрито",
  error: "Помилка з'єднання",
};

// Phase 17: non-manual grammar marker label, shown only when detected --
// "NONE" (no face / not yet calibrated / neutral) has no badge at all.
const FACIAL_GRAMMAR_LABEL: Partial<Record<string, string>> = {
  EYEBROWS_RAISED: "Брови підняті (питання «так/ні»)",
  EYEBROWS_FURROWED: "Брови насуплені (питання «хто/що/де»)",
};

/** A confirmed word, with a small dropdown to correct it in place when the
 * model's own real ranked alternatives (topK) offer more than one option --
 * fixes a wrong SAVED word (the gesture matches a word, but recognition
 * settles on the wrong one) by picking from what the model itself
 * actually ranked, never a guess. */
function ConfirmedWordChip({
  token,
  onCorrect,
  onOpenChange,
}: {
  token: WordToken;
  onCorrect: (candidate: WordCandidate) => void;
  onOpenChange?: (open: boolean) => void;
}): React.ReactElement {
  const label = glossLabel(token.value);
  const alternatives = token.topK;
  if (!alternatives || alternatives.length < 2) {
    return <span>{label}</span>;
  }
  return (
    <CorrectionPopover
      triggerLabel={label}
      currentKey={token.value}
      ariaLabel={`Виправити слово «${label}»`}
      options={alternatives.map((candidate) => ({
        key: candidate.gloss,
        label: glossLabel(candidate.gloss),
        confidence: candidate.confidence,
      }))}
      onSelect={(key) => {
        const candidate = alternatives.find((c) => c.gloss === key);
        if (candidate) onCorrect(candidate);
      }}
      onOpenChange={onOpenChange}
    />
  );
}

/** Mirrors ConfirmedWordChip for one letter of the still-growing
 * fingerspelling buffer -- lets a misrecognized letter be fixed mid-word
 * without retyping everything after it. */
function FingerspellLetterChip({
  token,
  displayChar,
  onCorrect,
  onOpenChange,
}: {
  token: LetterToken;
  /** The visible character -- first-letter-uppercase-rest-lowercase
   * position within the word (same convention as lib/glossDisplay.ts) is
   * purely cosmetic and kept separate from token.value, which stays the
   * real uppercase letter the backend sent and what <select>/its
   * <option>s match against. */
  displayChar: string;
  onCorrect: (candidate: LetterCandidate) => void;
  onOpenChange?: (open: boolean) => void;
}): React.ReactElement {
  const alternatives = token.topK;
  if (!alternatives || alternatives.length < 2) {
    return <span>{displayChar}</span>;
  }
  return (
    <CorrectionPopover
      triggerLabel={displayChar}
      currentKey={token.value}
      ariaLabel={`Виправити літеру «${token.value}»`}
      options={alternatives.map((candidate) => ({
        key: candidate.letter,
        label: candidate.letter,
        confidence: candidate.confidence,
      }))}
      onSelect={(key) => {
        const candidate = alternatives.find((c) => c.letter === key);
        if (candidate) onCorrect(candidate);
      }}
      onOpenChange={onOpenChange}
    />
  );
}

interface GestureToTextPanelProps {
  onFrame: (frame: CapturedFrame) => void;
  landmarksStatus: LandmarksStatusMessage | null;
  wsStatus: WebSocketStatus;
  wordMessage: PredictionMessage | null;
  letterMessage: LetterPredictionMessage | null;
  utterance: UseUtteranceResult;
  translationHistory: UseTranslationHistoryResult;
  tts: UseTextToSpeechResult;
  cameraStatus: "listening" | "processing";
  cameraError: string | null;
  isTranslating: boolean;
  hasStartedOnce: boolean;
  onStartOrResume: () => void;
  onPause: () => void;
  onFinishNow: () => void;
  onResetPhrase: () => void;
  onClearAll: () => void;
  /** Fires whenever whether ANY correction popover is open changes -- lets
   * the caller pause sending camera frames while a correction is in
   * progress (see CorrectionPopover.tsx's onOpenChange docstring for why:
   * moving the hand to the mouse mid-correction was getting read as a
   * sign-boundary pause and splicing a spurious token into the sentence
   * being corrected). Optional so this panel still renders standalone in
   * tests that don't care about frame gating. */
  onCorrectionActiveChange?: (active: boolean) => void;
}

export function GestureToTextPanel({
  onFrame,
  landmarksStatus,
  wsStatus,
  wordMessage,
  letterMessage,
  utterance,
  translationHistory,
  tts,
  cameraStatus,
  cameraError,
  isTranslating,
  hasStartedOnce,
  onStartOrResume,
  onPause,
  onFinishNow,
  onResetPhrase,
  onClearAll,
  onCorrectionActiveChange,
}: GestureToTextPanelProps): React.ReactElement {
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [confirmingClearAll, setConfirmingClearAll] = useState(false);
  // Which correction popovers (by a stable id per chip) are currently
  // open -- a Set rather than a bare counter so a stray extra open/close
  // notification can never push the count negative or stuck positive.
  const [openCorrectionIds, setOpenCorrectionIds] = useState<Set<string>>(new Set());
  const handleCorrectionOpenChange = useCallback((id: string, open: boolean) => {
    setOpenCorrectionIds((current) => {
      const next = new Set(current);
      if (open) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);
  const correctionActive = openCorrectionIds.size > 0;
  useEffect(() => {
    onCorrectionActiveChange?.(correctionActive);
  }, [correctionActive, onCorrectionActiveChange]);

  const handleCopyFinal = (): void => {
    const text = translationHistory.lastFinalTranslation?.composedText;
    if (!text) return;
    // navigator.clipboard may be unavailable (older browser, non-HTTPS,
    // test environment) -- copying is a convenience, never worth crashing
    // the app over.
    void navigator.clipboard?.writeText?.(text).catch(() => {});
  };

  const handleSpeakFinal = (): void => {
    const text = translationHistory.lastFinalTranslation?.composedText;
    if (text) tts.speak(text);
  };

  const handleClearAllClick = (): void => setConfirmingClearAll(true);
  const handleConfirmClearAll = (): void => {
    setConfirmingClearAll(false);
    onClearAll();
  };
  const handleCancelClearAll = (): void => setConfirmingClearAll(false);

  const interimWordLabel = utterance.interimWordGloss ? glossLabel(utterance.interimWordGloss) : null;

  const hasConfirmedTokens = utterance.confirmedTokens.length > 0;
  const hasFingerspellingBuffer = utterance.currentFingerspelledLetters.length > 0;
  const hasAnyActiveContent = hasConfirmedTokens || hasFingerspellingBuffer;

  // Never shows a raw backend status/error string (e.g. "Sign-recognition
  // is not available yet...", "Buffering: 5/32") -- that's developer-facing
  // plumbing, not a translation; it stays out of this line entirely rather
  // than flickering it on every frame. Shown only while nothing is
  // confirmed yet -- once something is, the chips below replace it.
  const statusFallback = !isTranslating
    ? "Пауза"
    : cameraStatus === "processing"
      ? "Обробка..."
      : wsStatus === "open"
        ? "Слухаю..."
        : "Підключення до камери...";

  const recognizingHint = [interimWordLabel, utterance.interimLetter].filter(Boolean).join(" · ");
  const showRecognizingHint = isTranslating && cameraStatus !== "processing" && recognizingHint.length > 0;

  const facialGrammarLabel = wordMessage ? FACIAL_GRAMMAR_LABEL[wordMessage.facial_grammar] : undefined;

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <section className="flex flex-col rounded-2xl border border-slate-200 bg-white p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">Камера</h2>
        <Camera onFrame={onFrame} landmarksStatus={landmarksStatus} />
        <div className="mt-3">
          <LandmarkIndicator status={landmarksStatus} />
        </div>
        {(correctionActive || utterance.awaitingConfirmation) && (
          // Visible, not just a silent frame-sending pause -- otherwise
          // recognition quietly stopping (during a correction, or while
          // waiting for the last word to be confirmed) reads as the app
          // hanging, not as deliberate behavior.
          <p
            className="mt-2 flex items-center justify-center gap-1.5 rounded-lg bg-brand-50 px-3 py-1.5 text-center text-xs font-medium text-brand-700"
            data-testid="correction-pause-banner"
          >
            <Pause className="h-3 w-3 shrink-0" aria-hidden />
            {correctionActive ? "Розпізнавання призупинено -- виправлення" : "Розпізнавання призупинено -- підтвердіть слово"}
          </p>
        )}
      </section>

      <section className="flex flex-col rounded-2xl border border-slate-200 bg-white p-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Переклад</h2>
        </div>
        <div
          className="flex flex-1 flex-col items-center justify-center gap-2 rounded-xl bg-slate-100 p-4 text-center text-sm text-slate-500"
          data-testid="live-sentence-status"
        >
          {hasAnyActiveContent ? (
            <div className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-1">
              {utterance.confirmedTokens.map((token, tokenIndex) => (
                // A literal trailing space (not just the flex gap) after
                // every word, so the text content itself has real word
                // boundaries -- matters for screen readers and for
                // copy/paste, not just the visual layout.
                <Fragment key={`${token.timestamp}-${tokenIndex}`}>
                  {token.kind === "word" ? (
                    <ConfirmedWordChip
                      token={token}
                      onCorrect={(candidate) => utterance.correctWordToken(tokenIndex, candidate)}
                      onOpenChange={(open) => handleCorrectionOpenChange(`word-${tokenIndex}`, open)}
                    />
                  ) : (
                    <span className="inline-flex items-center gap-0.5">
                      {token.letters.map((letterToken, letterIndex) => (
                        <FingerspellLetterChip
                          key={`${letterToken.timestamp}-${letterIndex}`}
                          token={letterToken}
                          displayChar={
                            letterIndex === 0 ? letterToken.value.toUpperCase() : letterToken.value.toLowerCase()
                          }
                          onCorrect={(candidate) => utterance.correctFinalizedLetter(tokenIndex, letterIndex, candidate)}
                          onOpenChange={(open) =>
                            handleCorrectionOpenChange(`finalized-letter-${tokenIndex}-${letterIndex}`, open)
                          }
                        />
                      ))}
                    </span>
                  )}{" "}
                </Fragment>
              ))}
              {hasFingerspellingBuffer && (
                <span className="inline-flex items-center gap-0.5">
                  {utterance.currentFingerspelledLetters.map((letterToken, letterIndex) => (
                    <FingerspellLetterChip
                      key={`${letterToken.timestamp}-${letterIndex}`}
                      token={letterToken}
                      displayChar={
                        letterIndex === 0 ? letterToken.value.toUpperCase() : letterToken.value.toLowerCase()
                      }
                      onCorrect={(candidate) => utterance.correctLetter(letterIndex, candidate)}
                      onOpenChange={(open) => handleCorrectionOpenChange(`buffer-letter-${letterIndex}`, open)}
                    />
                  ))}
                </span>
              )}
            </div>
          ) : (
            <span>{statusFallback}</span>
          )}
          {utterance.awaitingConfirmation && (
            // The step-by-step "ask about each word, only move on once
            // confirmed" flow: recognition is already paused (see the
            // banner under the camera), so this is the explicit "yes,
            // that one, continue" action for the plain case where the
            // last word needs no correction. Picking a candidate from
            // that word's own correction popover works too -- see
            // useUtterance.ts's correctWordToken/correctFinalizedLetter,
            // which clear awaitingConfirmation the same way.
            <div className="flex items-center gap-2" data-testid="confirm-pending-banner">
              <span className="text-xs font-medium text-slate-500">Підтвердіть останнє слово, щоб продовжити</span>
              <button
                type="button"
                onClick={utterance.confirmPending}
                className="inline-flex shrink-0 items-center gap-1 rounded-lg bg-brand-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-brand-700"
              >
                <Check className="h-3 w-3" aria-hidden />
                Підтвердити
              </button>
            </div>
          )}
          {showRecognizingHint && (
            <span className="text-xs text-slate-400" data-testid="recognizing-hint">
              Розпізнаю: {recognizingHint}...
            </span>
          )}
          {facialGrammarLabel && (
            <span className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700">
              <HelpCircle className="h-3 w-3" aria-hidden />
              {facialGrammarLabel}
            </span>
          )}
        </div>
        {cameraError && (
          <p className="mt-2 text-center text-xs text-red-600" data-testid="camera-translation-error">
            {cameraError}
          </p>
        )}

        {/* Primary controls -- every button here has real, working
            behavior wired to hooks/useUtterance.ts, never decorative. */}
        <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
          {isTranslating ? (
            <button
              type="button"
              onClick={onPause}
              className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              Пауза
            </button>
          ) : (
            <button
              type="button"
              onClick={onStartOrResume}
              className="shrink-0 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700"
            >
              {hasStartedOnce ? "Продовжити" : "Почати переклад"}
            </button>
          )}
          <button
            type="button"
            onClick={onFinishNow}
            disabled={!hasConfirmedTokens && !hasFingerspellingBuffer}
            className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            Завершити речення
          </button>
          <button
            type="button"
            onClick={utterance.undoLastToken}
            disabled={!hasConfirmedTokens}
            className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            Скасувати останнє
          </button>
          <button
            type="button"
            onClick={onResetPhrase}
            disabled={!hasAnyActiveContent}
            className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            Скинути фразу
          </button>
        </div>

        {/* Fingerspelling-specific controls -- only meaningful while a
            dactyl word is actively being spelled. */}
        {hasFingerspellingBuffer && (
          <div className="mt-2 flex flex-wrap items-center justify-end gap-2" data-testid="fingerspelling-controls">
            <button
              type="button"
              onClick={utterance.undoLastLetter}
              className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              Скасувати останню літеру
            </button>
            <button
              type="button"
              onClick={utterance.finishFingerspelledWord}
              className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              Завершити дактильне слово
            </button>
            <button
              type="button"
              onClick={utterance.clearFingerspellingBuffer}
              className="shrink-0 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              Очистити дактиль
            </button>
          </div>
        )}

        {/* Persistent final result -- see useTranslationHistory.ts's
            docstring: never auto-cleared by new signing, a reconnect, or
            any diagnostic update, only by a new completed translation or
            an explicit Clear click. */}
        <div className="mt-4 border-t border-slate-100 pt-4" data-testid="final-translation">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Готове речення</h3>
              {translationHistory.lastFinalTranslation?.composedTextSource === "ai_fallback" && (
                // The rule-based grammar engine couldn't compose this one
                // (see ml/nlp/gloss_to_text.py) -- an LLM's real best
                // attempt filled in instead, but unlike the rule-based
                // result it's NOT grammar-guaranteed, so it's always
                // labeled, never shown with the same confidence.
                <span
                  className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand-600"
                  title="Це речення склав ШІ, коли правило-орієнтований модуль не зміг -- не гарантовано граматично точне"
                  data-testid="ai-composed-badge"
                >
                  <Sparkles className="h-2.5 w-2.5" aria-hidden />
                  Складено ШІ
                </span>
              )}
            </div>
            {translationHistory.lastFinalTranslation && (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={handleCopyFinal}
                  title="Копіювати"
                  className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                >
                  <Copy className="h-3.5 w-3.5" aria-hidden />
                </button>
                {tts.isSupported && (
                  <>
                    <button
                      type="button"
                      onClick={handleSpeakFinal}
                      title="Озвучити"
                      className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                    >
                      <Volume2 className="h-3.5 w-3.5" aria-hidden />
                    </button>
                    <button
                      type="button"
                      onClick={tts.repeat}
                      title="Повторити озвучення"
                      disabled={tts.isSpeaking}
                      className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <Repeat className="h-3.5 w-3.5" aria-hidden />
                    </button>
                  </>
                )}
                <button
                  type="button"
                  onClick={translationHistory.clearCurrent}
                  title="Очистити переклад"
                  className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                >
                  <Trash2 className="h-3.5 w-3.5" aria-hidden />
                </button>
              </div>
            )}
          </div>
          {translationHistory.lastFinalTranslation ? (
            <p className="text-base font-semibold text-slate-900">
              {translationHistory.lastFinalTranslation.composedText ??
                translationHistory.lastFinalTranslation.glossLabels.map((label) => label.text).join(" ")}
            </p>
          ) : (
            <p className="text-sm text-slate-400">Тут з&apos;явиться завершене речення.</p>
          )}
        </div>

        <div className="mt-3 flex items-center justify-between">
          {confirmingClearAll ? (
            <div className="flex items-center gap-2 text-xs" data-testid="clear-all-confirm">
              <span className="text-slate-500">Точно очистити все?</span>
              <button type="button" onClick={handleConfirmClearAll} className="font-semibold text-red-600 hover:underline">
                Так
              </button>
              <button type="button" onClick={handleCancelClearAll} className="font-medium text-slate-400 hover:underline">
                Скасувати
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={handleClearAllClick}
              className="text-xs font-medium text-slate-400 hover:text-red-600"
            >
              Очистити все
            </button>
          )}

          <button
            type="button"
            onClick={() => setShowDiagnostics((current) => !current)}
            className="flex items-center gap-1 text-xs font-medium text-slate-400 hover:text-slate-600"
          >
            {showDiagnostics ? <ChevronUp className="h-3 w-3" aria-hidden /> : <ChevronDown className="h-3 w-3" aria-hidden />}
            Діагностика
          </button>
        </div>

        {showDiagnostics && (
          // Real per-frame readout from both classifiers plus connection
          // state -- developer-facing, not a translation.
          <div
            className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-500"
            data-testid="recognition-diagnostics"
          >
            <span>WS: {WS_STATUS_LABEL[wsStatus]}</span>
            <span>mode: {cameraStatus}</span>
            <span>
              Слово:{" "}
              {wordMessage
                ? `${glossLabel(wordMessage.gloss)} (${Math.round(wordMessage.confidence * 100)}%, ${
                    wordMessage.is_final ? "підтверджено" : "проміжно"
                  })`
                : "—"}
            </span>
            <span>
              Літера:{" "}
              {letterMessage
                ? `${letterMessage.letter} (${Math.round(letterMessage.confidence * 100)}%, ${
                    letterMessage.is_final ? "підтверджено" : "проміжно"
                  })`
                : "—"}
            </span>
            <span>
              Рука:{" "}
              {landmarksStatus ? (landmarksStatus.left_hand || landmarksStatus.right_hand ? "видно" : "не видно") : "—"}
            </span>
            <span>Поза: {landmarksStatus ? (landmarksStatus.pose ? "видно" : "не видно") : "—"}</span>
          </div>
        )}
      </section>
    </div>
  );
}
