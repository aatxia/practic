/**
 * PlaybackToken — the unit GestureVideo.tsx plays back, one per semantic
 * word: either a real gesture-vocabulary clip, a word spelled out
 * letter-by-letter (a real per-letter dactyl clip each, see
 * dactylVideos.ts), or a known word with no uploaded gesture video yet.
 * Built once per gloss sequence by groupGlossesForVideo() -- everything
 * downstream works off this structured list instead of re-deriving "is
 * this a fingerspell run" from raw gloss strings itself (never ad-hoc
 * frontend string checks).
 *
 * A word never disappears merely for lacking a dedicated gesture clip: if
 * it isn't a "gesture" token, it's either "fingerspell" (spelled out) or,
 * only when it has no lexicon entry at all AND fingerspelling produced no
 * video-backed letters either, "missing" -- reported honestly, never
 * silently dropped or faked.
 */

export interface LetterToken {
  /** Single uppercase Ukrainian dactyl letter, e.g. "Н". */
  letter: string;
  /** null = no uploaded video for this letter yet -- a real, honest gap
   * (see dactylVideos.ts), never a substituted or guessed clip. Playback
   * skips it gracefully instead of crashing or faking a handshape. */
  videoSrc: string | null;
}

export interface GestureToken {
  kind: "gesture";
  key: string;
  gloss: string;
  label: string;
  videoSrc: string;
}

/** A known word/gesture with no uploaded video at all. */
export interface MissingGestureToken {
  kind: "missing";
  key: string;
  gloss: string;
  label: string;
}

export interface FingerspelledWordToken {
  kind: "fingerspell";
  key: string;
  /** Best-effort cased reconstruction of the spelled word (title case --
   * gloss tokens don't carry the original casing, only the letters), e.g.
   * "Настя". Kept separate from normalizedWord so a caller that wants just
   * the letters actually spelled, with no guessed casing, has that too. */
  originalWord: string;
  /** The letters actually spelled, lowercase, e.g. "настя". */
  normalizedWord: string;
  letters: LetterToken[];
}

export type PlaybackToken = GestureToken | MissingGestureToken | FingerspelledWordToken;

/** True if a token has at least one real clip to play (a gesture video, or
 * at least one of its letters has one) -- false only for a token nothing
 * downstream can show any video for at all. */
export function hasPlayableVideo(token: PlaybackToken): boolean {
  if (token.kind === "gesture") return true;
  if (token.kind === "missing") return false;
  return token.letters.some((letter) => letter.videoSrc !== null);
}
