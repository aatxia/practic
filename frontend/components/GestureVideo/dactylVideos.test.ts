import { describe, expect, it } from "vitest";
import { DACTYL_VIDEOS, UKRAINIAN_DACTYL_ALPHABET, videoForLetter } from "./dactylVideos";

describe("dactylVideos", () => {
  it("has 33 letters in the Ukrainian dactyl alphabet, matching ml/nlp/fingerspelling.py", () => {
    expect(UKRAINIAN_DACTYL_ALPHABET).toHaveLength(33);
    expect(new Set(UKRAINIAN_DACTYL_ALPHABET).size).toBe(33);
  });

  it("every mapped key is a single uppercase letter from the Ukrainian dactyl alphabet", () => {
    for (const key of Object.keys(DACTYL_VIDEOS)) {
      expect(key).toHaveLength(1);
      expect(UKRAINIAN_DACTYL_ALPHABET).toContain(key.toLowerCase());
      expect(key).toBe(key.toUpperCase());
    }
  });

  it("never assigns the same video path to two different letters", () => {
    const paths = Object.values(DACTYL_VIDEOS);
    expect(new Set(paths).size).toBe(paths.length);
  });

  it("every mapped path is a well-formed /videos/<file> reference", () => {
    for (const path of Object.values(DACTYL_VIDEOS)) {
      expect(path).toMatch(/^\/videos\/.+\.(mp4|webm)$/);
    }
  });

  it("covers at least 30 of the 33 letters -- a real, near-complete dataset, not a placeholder", () => {
    expect(Object.keys(DACTYL_VIDEOS).length).toBeGreaterThanOrEqual(30);
  });

  it("returns the exact mapped path for a known letter", () => {
    expect(videoForLetter("Н")).toBe("/videos/Н.mp4");
  });

  it("returns null, never a substituted clip, for a letter with no uploaded video", () => {
    // "А" (Cyrillic) is deliberately unmapped -- see dactylVideos.ts's
    // comment on the ambiguous Latin-named public/videos/A.mp4 file.
    expect(videoForLetter("А")).toBeNull();
  });

  it("returns null for a letter outside the Ukrainian dactyl alphabet entirely", () => {
    expect(videoForLetter("Q")).toBeNull();
  });
});
