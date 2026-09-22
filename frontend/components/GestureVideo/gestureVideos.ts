/**
 * Maps a gloss token to its real reference video (public/videos/, uploaded
 * by the signer herself -- an actual person performing the sign, not a
 * hand-authored 3D pose approximation like components/Avatar/). Filenames
 * don't always match the internal gloss code 1:1 (the negation gloss is
 * "NOT" internally, ml/nlp/lexicon.py, but the uploaded file is
 * "NO.mp4") -- this table is the single place that mapping is made
 * explicit, not something a caller should guess at.
 *
 * Only glosses with an actual uploaded video appear here. A gloss missing
 * from this table has no video -- callers must treat that as "no video for
 * this gesture" (see GestureVideo.tsx's "Немає відео для" list), never
 * substitute a different gloss's video, which would show the wrong sign as
 * if it were correct.
 */
export const GESTURE_VIDEOS: Partial<Record<string, string>> = {
  WE: "/videos/WE.mp4",
  WANT: "/videos/WANT.mp4",
  HAVE: "/videos/HAVE.mp4",
  WATER: "/videos/WATER.mp4",
  NOT: "/videos/NO.mp4",
  PAST: "/videos/PAST.mp4",
  I: "/videos/I.mp4",
  EAT: "/videos/EAT.mp4",
  FLOWER: "/videos/FLOWER.mp4",
  TOMORROW: "/videos/TOMORROW.mp4",
  YESTERDAY: "/videos/YESTERDAY.mp4",
};

export function videoForGloss(gloss: string): string | null {
  return GESTURE_VIDEOS[gloss] ?? null;
}
