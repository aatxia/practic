"use client";

import { Play, Repeat, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { groupGlossesForVideo } from "./groupGlossesForVideo";
import { DEFAULT_PLAYBACK_SPEED, PLAYBACK_SPEEDS, PLAYBACK_TRANSITIONS_MS } from "./playbackSettings";
import { hasPlayableVideo, type FingerspelledWordToken, type PlaybackToken } from "./playbackToken";

interface GestureVideoProps {
  /** Confirmed gloss sequence to play back, in order. Empty = idle. */
  glossSequence: string[];
}

/** One playable clip in the flattened playback queue: either a whole-word
 * gesture clip (letterIndex null) or one letter of a fingerspelled word. */
interface QueueEntry {
  tokenIndex: number;
  letterIndex: number | null;
  videoSrc: string;
}

function buildQueue(tokens: PlaybackToken[]): QueueEntry[] {
  const queue: QueueEntry[] = [];
  tokens.forEach((token, tokenIndex) => {
    if (token.kind === "gesture") {
      queue.push({ tokenIndex, letterIndex: null, videoSrc: token.videoSrc });
      return;
    }
    if (token.kind === "fingerspell") {
      token.letters.forEach((letter, letterIndex) => {
        if (letter.videoSrc !== null) {
          queue.push({ tokenIndex, letterIndex, videoSrc: letter.videoSrc });
        }
      });
    }
    // "missing" tokens contribute nothing playable.
  });
  return queue;
}

function fingerspellProgress(token: FingerspelledWordToken, currentLetterIndex: number | null): string {
  return token.letters
    .map((letter, index) => (index === currentLetterIndex ? `[${letter.letter}]` : letter.letter))
    .join(" ");
}

/** Real reference clips of an actual signer (see gestureVideos.ts /
 * dactylVideos.ts) -- replaces the hand-authored 3D avatar approximation
 * (components/Avatar/) as the translation display: a gloss sequence plays
 * back as its real videos, one after another. A word with no dedicated
 * gesture clip never just disappears -- it falls back to real per-letter
 * dactyl (fingerspelling) clips instead (see groupGlossesForVideo.ts /
 * playbackToken.ts), the same way a deaf signer spells an unfamiliar word
 * letter by letter rather than skipping it. */
export function GestureVideo({ glossSequence }: GestureVideoProps): React.ReactElement {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pendingAdvanceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [playIndex, setPlayIndex] = useState(0);
  // Forces the playback effect below to re-run even when playIndex is
  // being set to the SAME value it already had (e.g. "Спочатку"/"Повторити
  // слово" clicked while already on the first clip/word): the effect only
  // depends on playIndex and the resolved videoSrc, neither of which
  // changes in that case, so without this the video would just silently
  // stay wherever it already was instead of rewinding and replaying.
  const [replayNonce, setReplayNonce] = useState(0);
  // Set only by a per-word replay button -- overrides the default "play
  // the whole sequence" playlist with just that one word's clip(s) until
  // the next "Переглянути все" click or a new translation arrives.
  const [singleReplayTokenIndex, setSingleReplayTokenIndex] = useState<number | null>(null);
  const [playbackRate, setPlaybackRate] = useState(DEFAULT_PLAYBACK_SPEED);
  // Tracks which glossSequence the state above belongs to, so a new
  // translation (different array) resets playback to its own beginning --
  // set during render (React's documented way to reset state on a prop
  // change: https://react.dev/learn/you-might-not-need-an-effect), not in
  // an effect, so it doesn't cost an extra cascading render.
  const [ownSequence, setOwnSequence] = useState(glossSequence);

  function clearPendingAdvance(): void {
    if (pendingAdvanceRef.current !== null) {
      clearTimeout(pendingAdvanceRef.current);
      pendingAdvanceRef.current = null;
    }
  }

  if (ownSequence !== glossSequence) {
    setOwnSequence(glossSequence);
    setSingleReplayTokenIndex(null);
    setPlayIndex(0);
    setReplayNonce((n) => n + 1);
  }

  // Cancels any auto-advance timer scheduled for the PREVIOUS sequence.
  // Not done inline in the render-time reset above (refs must not be read
  // or written during render) -- an effect keyed on glossSequence itself
  // still runs before a stale timer could ever fire, since the playIndex
  // reset above is already synchronous by the time this commits.
  useEffect(() => {
    return clearPendingAdvance;
  }, [glossSequence]);

  const tokens = useMemo(() => groupGlossesForVideo(glossSequence), [glossSequence]);
  const queue = useMemo(() => buildQueue(tokens), [tokens]);
  const playableTokenIndexes = useMemo(() => new Set(queue.map((entry) => entry.tokenIndex)), [queue]);
  const fullyMissingTokens = useMemo(
    () => tokens.filter((token) => !hasPlayableVideo(token)),
    [tokens],
  );
  const partiallyMissingFingerspellTokens = useMemo(
    () =>
      tokens.filter(
        (token): token is FingerspelledWordToken =>
          token.kind === "fingerspell" &&
          token.letters.some((letter) => letter.videoSrc !== null) &&
          token.letters.some((letter) => letter.videoSrc === null),
      ),
    [tokens],
  );

  const playlist = singleReplayTokenIndex !== null
    ? queue.filter((entry) => entry.tokenIndex === singleReplayTokenIndex)
    : queue;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || playlist.length === 0) return;
    video.src = playlist[playIndex]!.videoSrc;
    video.currentTime = 0;
    video.playbackRate = playbackRate;
    const playResult = video.play();
    if (playResult && typeof playResult.then === "function") {
      // Autoplay can be blocked outside a user gesture -- the video stays
      // paused on its first frame rather than throwing; the person can
      // press play manually via the native controls. (jsdom's play() also
      // doesn't return a promise at all, same reasoning as useCamera.ts.)
      playResult.catch(() => {});
    }
    // playlist's identity changes every render (derived, not memoized as a
    // single value) -- keying off playIndex and the actual video source it
    // should be showing avoids reloading the same clip on unrelated
    // re-renders; replayNonce is what makes Restart/Repeat/etc. actually
    // rewind+replay even when playIndex numerically stays the same.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playIndex, replayNonce, playlist[playIndex]?.videoSrc]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = playbackRate;
  }, [playbackRate]);

  // The one place playIndex is ever set -- always also bumps replayNonce,
  // so the playback effect reruns (rewind + play) even when index happens
  // to equal what it already was.
  function goToIndex(index: number): void {
    clearPendingAdvance();
    setPlayIndex(index);
    setReplayNonce((n) => n + 1);
  }

  function handleEnded(): void {
    const nextIndex = playIndex + 1;
    if (nextIndex >= playlist.length) return;
    const current = playlist[playIndex]!;
    const next = playlist[nextIndex]!;
    const delay = next.tokenIndex === current.tokenIndex
      ? PLAYBACK_TRANSITIONS_MS.letterToLetter
      : PLAYBACK_TRANSITIONS_MS.wordToWord;
    clearPendingAdvance();
    pendingAdvanceRef.current = setTimeout(() => goToIndex(nextIndex), delay);
  }

  function playAll(): void {
    if (queue.length === 0) return;
    setSingleReplayTokenIndex(null);
    goToIndex(0);
  }

  function playWord(tokenIndex: number): void {
    setSingleReplayTokenIndex(tokenIndex);
    goToIndex(0);
  }

  function handleRestart(): void {
    goToIndex(0);
  }

  function handleRepeatWord(): void {
    const current = playlist[playIndex];
    if (!current) return;
    const startIndex = playlist.findIndex((entry) => entry.tokenIndex === current.tokenIndex);
    if (startIndex !== -1) goToIndex(startIndex);
  }

  function handleNextWord(): void {
    const current = playlist[playIndex];
    if (!current) return;
    const nextIndex = playlist.findIndex((entry, index) => index > playIndex && entry.tokenIndex !== current.tokenIndex);
    if (nextIndex === -1) return;
    goToIndex(nextIndex);
  }

  function handlePreviousWord(): void {
    const current = playlist[playIndex];
    if (!current) return;
    const currentWordStart = playlist.findIndex((entry) => entry.tokenIndex === current.tokenIndex);
    if (playIndex > currentWordStart) {
      goToIndex(currentWordStart);
      return;
    }
    for (let index = currentWordStart - 1; index >= 0; index -= 1) {
      if (playlist[index]!.tokenIndex !== current.tokenIndex) {
        const previousWordStart = playlist.findIndex((entry) => entry.tokenIndex === playlist[index]!.tokenIndex);
        goToIndex(previousWordStart);
        return;
      }
    }
  }

  if (glossSequence.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-xl bg-slate-100 px-4 text-center text-sm text-slate-400">
        Покажи слово чи речення, щоб побачити відео перекладу.
      </div>
    );
  }

  // A real translation happened, but not one of its words has any video
  // (as a gesture or fingerspelled letter) -- honestly says so instead of
  // falling back to the "nothing translated yet" placeholder, which would
  // hide that a translation exists and simply has no video for it.
  if (queue.length === 0) {
    return (
      <div className="flex h-72 flex-col items-center justify-center gap-1 rounded-xl bg-slate-100 px-4 text-center text-sm text-slate-400">
        <span>Немає відео для жодного слова з цього перекладу.</span>
        <span className="text-xs">
          {tokens.map((token) => (token.kind === "fingerspell" ? token.originalWord : token.label)).join(", ")}
        </span>
      </div>
    );
  }

  const currentEntry = playlist[playIndex];
  const currentToken = currentEntry ? tokens[currentEntry.tokenIndex] : null;
  const totalWordsInPlaylist = new Set(playlist.map((entry) => entry.tokenIndex)).size;
  const wordPosition = currentEntry
    ? new Set(playlist.slice(0, playlist.findIndex((entry) => entry.tokenIndex === currentEntry.tokenIndex) + 1).map((e) => e.tokenIndex)).size
    : 0;
  const canNavigateWords = totalWordsInPlaylist > 1;

  return (
    <div className="flex flex-col gap-2">
      {currentEntry && currentToken && (
        <div data-testid="playback-progress" className="text-xs text-slate-500">
          <span>
            {wordPosition}/{totalWordsInPlaylist}
          </span>
          {currentToken.kind === "gesture" && <span> · Жест: {currentToken.label}</span>}
          {currentToken.kind === "fingerspell" && (
            <span>
              {" "}
              · {currentToken.originalWord} ·{" "}
              <span data-testid="fingerspell-letters">
                {fingerspellProgress(currentToken, currentEntry.letterIndex)}
              </span>{" "}
              ({(currentEntry.letterIndex ?? 0) + 1}/{currentToken.letters.length})
            </span>
          )}
        </div>
      )}

      <video
        ref={videoRef}
        onEnded={handleEnded}
        controls
        playsInline
        data-testid="gesture-video"
        // Sized by the clip's own (portrait) aspect ratio, capped at 18rem
        // tall and centered -- forcing it to the full column width with
        // object-contain (the old className) left ugly black pillarboxing
        // on both sides, since these reference clips are taller than they
        // are wide.
        className="mx-auto max-h-72 w-auto max-w-full rounded-xl bg-slate-900 object-contain"
      />

      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={handleRestart}
          title="Спочатку"
          className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
        >
          <RotateCcw className="h-3 w-3" aria-hidden />
        </button>
        <button
          type="button"
          onClick={handlePreviousWord}
          disabled={!canNavigateWords}
          title="Попереднє слово"
          className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <SkipBack className="h-3 w-3" aria-hidden />
        </button>
        <button
          type="button"
          onClick={handleRepeatWord}
          title="Повторити слово"
          className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
        >
          <Repeat className="h-3 w-3" aria-hidden />
        </button>
        <button
          type="button"
          onClick={handleNextWord}
          disabled={!canNavigateWords}
          title="Наступне слово"
          className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <SkipForward className="h-3 w-3" aria-hidden />
        </button>
        <select
          data-testid="playback-speed"
          value={playbackRate}
          onChange={(event) => setPlaybackRate(Number(event.target.value))}
          className="rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600"
          aria-label="Швидкість відтворення"
        >
          {PLAYBACK_SPEEDS.map((speed) => (
            <option key={speed} value={speed}>
              {speed}×
            </option>
          ))}
        </select>
      </div>

      {playableTokenIndexes.size > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={playAll}
            title="Відтворити всі відео по черзі"
            className="inline-flex items-center gap-1 rounded-full bg-brand-600 px-2.5 py-0.5 text-xs font-medium text-white hover:bg-brand-700"
          >
            <Play className="h-3 w-3" aria-hidden />
            Переглянути все
          </button>
          {tokens.map((token, tokenIndex) => {
            if (!playableTokenIndexes.has(tokenIndex)) return null;
            const label = token.kind === "fingerspell" ? token.originalWord : token.label;
            return (
              <button
                key={token.key}
                type="button"
                onClick={() => playWord(tokenIndex)}
                title={`Повторити «${label}»`}
                className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-xs font-medium text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
              >
                <RotateCcw className="h-3 w-3" aria-hidden />
                {label}
              </button>
            );
          })}
        </div>
      )}

      {fullyMissingTokens.length > 0 && (
        <p className="text-xs text-slate-400">
          Немає відео для: {fullyMissingTokens.map((token) => (token.kind === "fingerspell" ? token.originalWord : token.label)).join(", ")}
        </p>
      )}

      {partiallyMissingFingerspellTokens.length > 0 && (
        <p className="text-xs text-amber-600" data-testid="partial-fingerspell-notice">
          Без відео для деяких літер:{" "}
          {partiallyMissingFingerspellTokens
            .map(
              (token) =>
                `${token.originalWord} (${token.letters
                  .filter((letter) => letter.videoSrc === null)
                  .map((letter) => letter.letter)
                  .join(", ")})`,
            )
            .join("; ")}
        </p>
      )}
    </div>
  );
}
