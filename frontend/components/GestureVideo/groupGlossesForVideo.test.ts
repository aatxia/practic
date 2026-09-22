import { describe, expect, it } from "vitest";
import { groupGlossesForVideo } from "./groupGlossesForVideo";

describe("groupGlossesForVideo", () => {
  it("attaches the real video for each gloss that has one", () => {
    const tokens = groupGlossesForVideo(["WE", "WANT", "WATER"]);

    expect(tokens.map((token) => (token.kind === "gesture" ? token.videoSrc : null))).toEqual([
      "/videos/WE.mp4",
      "/videos/WANT.mp4",
      "/videos/WATER.mp4",
    ]);
    expect(tokens.every((token) => token.kind === "gesture")).toBe(true);
  });

  it("maps the NOT gloss to the uploaded NO.mp4 file, not a guessed filename", () => {
    const [token] = groupGlossesForVideo(["NOT"]);

    expect(token?.kind).toBe("gesture");
    expect(token?.kind === "gesture" && token.videoSrc).toBe("/videos/NO.mp4");
  });

  it("does NOT drop the PAST marker the way groupGlossesForDisplay does -- it has a real video", () => {
    const tokens = groupGlossesForVideo(["WE", "PAST", "WANT"]);

    expect(tokens.map((token) => (token.kind === "fingerspell" ? token.originalWord : token.label))).toEqual([
      "ми",
      "минулий час",
      "хотіти",
    ]);
    expect(tokens[1]?.kind === "gesture" && tokens[1].videoSrc).toBe("/videos/PAST.mp4");
  });

  it("reports a 'missing' token, never a substituted or guessed clip, for a gloss with no uploaded video", () => {
    const [token] = groupGlossesForVideo(["YOU"]);

    expect(token?.kind).toBe("missing");
  });

  it("collapses a fingerspelled run into one FingerspelledWordToken with a real per-letter video each", () => {
    const [token] = groupGlossesForVideo(["FS_О", "FS_К", "FS_О"]);

    expect(token?.kind).toBe("fingerspell");
    if (token?.kind !== "fingerspell") throw new Error("expected a fingerspell token");
    expect(token.originalWord).toBe("Око");
    expect(token.normalizedWord).toBe("око");
    expect(token.letters).toEqual([
      { letter: "О", videoSrc: "/videos/О.mp4" },
      { letter: "К", videoSrc: "/videos/К.mp4" },
      { letter: "О", videoSrc: "/videos/О.mp4" },
    ]);
  });

  it("spells a proper name letter by letter, e.g. Настя -> Н-А-С-Т-Я, with А gracefully unsupported", () => {
    const [token] = groupGlossesForVideo(["FS_Н", "FS_А", "FS_С", "FS_Т", "FS_Я"]);

    expect(token?.kind).toBe("fingerspell");
    if (token?.kind !== "fingerspell") throw new Error("expected a fingerspell token");
    expect(token.originalWord).toBe("Настя");
    // "А" has no uploaded video yet (see dactylVideos.ts) -- reported
    // honestly as null, not a crash and not a substituted clip.
    expect(token.letters.map((letter) => letter.videoSrc)).toEqual([
      "/videos/Н.mp4",
      null,
      "/videos/С.mp4",
      "/videos/Т.mp4",
      "/videos/Я.mp4",
    ]);
  });

  it("keeps a mixed sentence as separate words, not one item per letter", () => {
    // "Я хочу воду, Настя" -> 4 words: I, WANT, WATER, FingerspelledWordToken(Настя)
    const tokens = groupGlossesForVideo(["I", "WANT", "WATER", "FS_Н", "FS_А", "FS_С", "FS_Т", "FS_Я"]);

    expect(tokens).toHaveLength(4);
    expect(tokens.map((token) => token.kind)).toEqual(["gesture", "gesture", "gesture", "fingerspell"]);
  });

  it("handles an empty sequence", () => {
    expect(groupGlossesForVideo([])).toEqual([]);
  });

  it("keeps two consecutive fingerspelled words as two separate tokens, never glued into one", () => {
    // "тобі" + "допомогти", both outside the lexicon -- without the
    // boundary marker this collapsed into one wrong word/playback token.
    const tokens = groupGlossesForVideo([
      "FS_Т", "FS_О", "FS_Б", "FS_І",
      "FS_BOUNDARY",
      "FS_Д", "FS_О", "FS_П", "FS_О", "FS_М", "FS_О", "FS_Г", "FS_Т", "FS_И",
    ]);

    expect(tokens).toHaveLength(2);
    expect(tokens.every((token) => token.kind === "fingerspell")).toBe(true);
    expect(tokens.map((token) => token.kind === "fingerspell" && token.originalWord)).toEqual([
      "Тобі",
      "Допомогти",
    ]);
  });

  it("the boundary marker produces no playback token of its own", () => {
    const tokens = groupGlossesForVideo(["I", "FS_А", "FS_BOUNDARY", "FS_Б"]);

    expect(tokens).toHaveLength(3);
    expect(tokens.map((token) => token.kind)).toEqual(["gesture", "fingerspell", "fingerspell"]);
  });
});
