/**
 * Groups a gloss sequence for display: a run of consecutive Phase 16
 * fingerspelling tokens ("FS_<letter>") reads as noise one chip per letter,
 * so it's collapsed into one chip showing the spelled word. Regular glosses
 * are unaffected -- one chip each, same as before Phase 16. TENSE_PAST_GLOSS
 * ("PAST") is dropped entirely: it's a pure grammar marker with no
 * Ukrainian word and no gesture of its own (see ml/nlp/lexicon.py), so a
 * chip for it would be neither a real word nor a missing animation to
 * report -- just noise.
 */
import { glossLabel, TENSE_PAST_GLOSS } from "./glossLabels";

export interface GlossDisplayItem {
  key: string;
  label: string;
  isFingerspell: boolean;
  /** Raw gloss token(s) this item stands for, in order -- e.g. a single
   * regular gloss's own token, or a fingerspelled word's per-letter "FS_x"
   * run. Lets a caller (Avatar.tsx's per-gesture replay button) hand the
   * exact sub-sequence back to GlossPlayer.play() without re-deriving it. */
  tokens: string[];
}

const FINGERSPELL_PATTERN = /^FS_.$/u;

export function isFingerspellGloss(gloss: string): boolean {
  return FINGERSPELL_PATTERN.test(gloss);
}

/** Separates two DIFFERENT fingerspelled words sitting next to each other
 * in a flat gloss sequence (mirrors ml/nlp/fingerspelling.py's
 * FINGERSPELL_WORD_BOUNDARY) -- without it, two consecutive FS_ runs would
 * be indistinguishable from one long run and get glued into a single word
 * (e.g. "тобі" + "допомогти" -> the wrong "тобідопомогти"). Never itself a
 * fingerspell letter (isFingerspellGloss returns false for it), so callers
 * can tell it apart on sight; it contributes no chip/label of its own. */
export const FINGERSPELL_WORD_BOUNDARY = "FS_BOUNDARY";

// Gloss tokens are internal identifiers, not Ukrainian text (e.g. "WATER",
// "YOU_PL") -- glossLabel() looks up the real Ukrainian word (lib/glossLabels.ts).
export function formatGlossLabel(gloss: string): string {
  return glossLabel(gloss);
}

export function groupGlossesForDisplay(sequence: string[]): GlossDisplayItem[] {
  const items: GlossDisplayItem[] = [];
  let i = 0;
  while (i < sequence.length) {
    const gloss = sequence[i]!;
    if (gloss === TENSE_PAST_GLOSS || gloss === FINGERSPELL_WORD_BOUNDARY) {
      i += 1;
      continue;
    }
    if (!isFingerspellGloss(gloss)) {
      items.push({
        key: `${gloss}-${i}`,
        label: formatGlossLabel(gloss),
        isFingerspell: false,
        tokens: [gloss],
      });
      i += 1;
      continue;
    }

    const start = i;
    let word = "";
    const tokens: string[] = [];
    while (i < sequence.length && isFingerspellGloss(sequence[i]!)) {
      word += sequence[i]!.slice(3);
      tokens.push(sequence[i]!);
      i += 1;
    }
    items.push({
      key: `fs-${start}`,
      label: `${word.charAt(0)}${word.slice(1).toLowerCase()}`,
      isFingerspell: true,
      tokens,
    });
  }
  return items;
}
