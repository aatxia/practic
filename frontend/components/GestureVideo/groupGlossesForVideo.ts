/**
 * Groups a gloss sequence into PlaybackTokens for video playback: a run of
 * consecutive FS_ (fingerspelling) tokens becomes one FingerspelledWordToken
 * with a real per-letter video for each letter that has one (dactylVideos.ts)
 * -- unlike lib/glossDisplay.ts's groupGlossesForDisplay(), which only needs
 * a label for TEXT display, this also needs to resolve real playable video
 * sources, both for whole-word gestures and for each individual letter.
 *
 * Deliberately does NOT drop TENSE_PAST_GLOSS ("PAST"): that filtering
 * exists in glossDisplay.ts only because the 3D avatar has no
 * hand-authored pose for it, but a real reference video for PAST does
 * exist (gestureVideos.ts) and should be shown like any other gloss.
 */
import { FINGERSPELL_WORD_BOUNDARY, isFingerspellGloss } from "@/lib/glossDisplay";
import { glossLabel } from "@/lib/glossLabels";
import { videoForLetter } from "./dactylVideos";
import { videoForGloss } from "./gestureVideos";
import type { LetterToken, PlaybackToken } from "./playbackToken";

function titleCase(word: string): string {
  return `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`;
}

export function groupGlossesForVideo(sequence: string[]): PlaybackToken[] {
  const tokens: PlaybackToken[] = [];
  let i = 0;
  while (i < sequence.length) {
    const gloss = sequence[i]!;

    if (gloss === FINGERSPELL_WORD_BOUNDARY) {
      i += 1;
      continue;
    }

    if (!isFingerspellGloss(gloss)) {
      const label = glossLabel(gloss);
      const videoSrc = videoForGloss(gloss);
      tokens.push(
        videoSrc !== null
          ? { kind: "gesture", key: `${gloss}-${i}`, gloss, label, videoSrc }
          : { kind: "missing", key: `${gloss}-${i}`, gloss, label },
      );
      i += 1;
      continue;
    }

    const start = i;
    let word = "";
    const letters: LetterToken[] = [];
    while (i < sequence.length && isFingerspellGloss(sequence[i]!)) {
      const letter = sequence[i]!.slice(3);
      word += letter;
      letters.push({ letter, videoSrc: videoForLetter(letter) });
      i += 1;
    }
    tokens.push({
      kind: "fingerspell",
      key: `fs-${start}`,
      originalWord: titleCase(word),
      normalizedWord: word.toLowerCase(),
      letters,
    });
  }
  return tokens;
}
