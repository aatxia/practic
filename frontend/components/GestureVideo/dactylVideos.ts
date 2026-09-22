/**
 * Centralized Ukrainian dactyl (fingerspelling) letter -> reference video
 * map -- the SINGLE place this project maps a dactyl letter to its real
 * uploaded clip (public/videos/), mirroring ml/nlp/fingerspelling.py's
 * UKRAINIAN_ALPHABET so both layers agree on what "a letter" even means.
 * Never scatter this mapping across components -- a caller with a letter
 * calls videoForLetter(), same discipline as gestureVideos.ts's
 * videoForGloss() for whole-word gestures.
 *
 * A letter missing from this table has no video -- callers must show a
 * real "no video for this letter" state (see GestureVideo.tsx), never
 * substitute a different letter's clip or fabricate one; see this
 * project's "no fake AI/gestures" rule.
 */

// Mirrors ml/nlp/fingerspelling.py's UKRAINIAN_ALPHABET exactly -- kept as
// a literal copy, not fetched, since the completeness check below needs a
// synchronous, build-time-checkable key set.
export const UKRAINIAN_DACTYL_ALPHABET: readonly string[] = [
  "а", "б", "в", "г", "ґ", "д", "е", "є", "ж", "з", "и", "і", "ї", "й", "к",
  "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
  "ь", "ю", "я",
];

/**
 * Real per-letter reference clips uploaded to public/videos/, keyed by
 * uppercase letter (matches the FS_<LETTER> gloss convention directly --
 * ml/nlp/fingerspelling.py's FINGERSPELL_PREFIX).
 *
 * "А" (Cyrillic, U+0410) is deliberately NOT mapped here even though
 * public/videos/A.mp4 exists: that file's name is Latin "A" (U+0041), not
 * Cyrillic "А" -- the two glyphs render identically, but they are not the
 * same character the FS_А gloss token would look up. Whether that file was
 * actually meant to be the "А" demo is a real, open question this code
 * can't safely answer by guessing at a filename. Until it's renamed to the
 * Cyrillic "А.mp4", "А" stays honestly unmapped (see the "unsupported
 * letter" handling in GestureVideo.tsx) rather than risking the wrong clip
 * playing for a letter forever, unnoticed, because the glyphs look the
 * same on screen.
 */
export const DACTYL_VIDEOS: Partial<Record<string, string>> = {
  Б: "/videos/Б.mp4",
  В: "/videos/В.mp4",
  Г: "/videos/Г.mp4",
  Ґ: "/videos/Ґ.mp4",
  Д: "/videos/Д.mp4",
  Е: "/videos/Е.mp4",
  Є: "/videos/Є.mp4",
  Ж: "/videos/Ж.mp4",
  З: "/videos/З.mp4",
  И: "/videos/И.mp4",
  І: "/videos/І.mp4",
  Ї: "/videos/Ї.mp4",
  Й: "/videos/Й.mp4",
  К: "/videos/К.mp4",
  Л: "/videos/Л.mp4",
  М: "/videos/М.mp4",
  Н: "/videos/Н.mp4",
  О: "/videos/О.mp4",
  П: "/videos/П.mp4",
  Р: "/videos/Р.mp4",
  С: "/videos/С.mp4",
  Т: "/videos/Т.mp4",
  У: "/videos/У.mp4",
  Ф: "/videos/Ф.mp4",
  Х: "/videos/Х.mp4",
  Ц: "/videos/Ц.mp4",
  Ч: "/videos/Ч.mp4",
  Ш: "/videos/Ш.mp4",
  Щ: "/videos/Щ.mp4",
  Ь: "/videos/Ь.mp4",
  Ю: "/videos/Ю.mp4",
  Я: "/videos/Я.mp4",
};

/** letter: a single uppercase Ukrainian dactyl letter (e.g. "Н"), same
 * casing FS_<LETTER> gloss tokens use. Returns null for a letter with no
 * uploaded clip yet -- never a substitute, never a fabricated handshape. */
export function videoForLetter(letter: string): string | null {
  return DACTYL_VIDEOS[letter] ?? null;
}
