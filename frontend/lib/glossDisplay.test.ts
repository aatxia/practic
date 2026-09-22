import { describe, expect, it } from "vitest";
import { groupGlossesForDisplay } from "./glossDisplay";

describe("groupGlossesForDisplay", () => {
  it("renders each regular gloss as its Ukrainian word, not the internal gloss code", () => {
    const items = groupGlossesForDisplay(["I", "WANT", "WATER"]);

    expect(items.map((item) => item.label)).toEqual(["я", "хотіти", "вода"]);
    expect(items.every((item) => !item.isFingerspell)).toBe(true);
  });

  it("collapses a run of fingerspelling tokens into one spelled-word item", () => {
    const letters = ["FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_А"];
    const items = groupGlossesForDisplay(letters);

    expect(items).toHaveLength(1);
    expect(items[0]?.isFingerspell).toBe(true);
    expect(items[0]?.label).toBe("Оксана");
    expect(items[0]?.tokens).toEqual(letters);
  });

  it("gives each regular gloss item its own single-token tokens array", () => {
    const items = groupGlossesForDisplay(["I", "WANT"]);

    expect(items.map((item) => item.tokens)).toEqual([["I"], ["WANT"]]);
  });

  it("groups a fingerspelled run inside a larger sequence without touching the rest", () => {
    const items = groupGlossesForDisplay(["I", "LIKE", "FS_О", "FS_К", "FS_С", "FS_А", "FS_Н", "FS_У"]);

    expect(items.map((item) => item.label)).toEqual(["я", "любити", "Оксану"]);
    expect(items.map((item) => item.isFingerspell)).toEqual([false, false, true]);
  });

  it("handles an empty sequence", () => {
    expect(groupGlossesForDisplay([])).toEqual([]);
  });

  it("looks up the real Ukrainian word for a multi-word gloss token, not underscore-formatted English", () => {
    const items = groupGlossesForDisplay(["YOU_PL", "WORK_N", "DO_POBACHENNYA"]);

    expect(items.map((item) => item.label)).toEqual(["ви", "робота", "до побачення"]);
    expect(items.some((item) => item.label.includes("_"))).toBe(false);
  });

  it("drops the past-tense marker entirely -- no chip, not even an English one", () => {
    const items = groupGlossesForDisplay(["CAR", "PAST", "RIDE"]);

    expect(items.map((item) => item.label)).toEqual(["машина", "їхати"]);
  });

  it("falls back to underscore-as-space (never a raw untouched code) for a gloss outside the known table", () => {
    const items = groupGlossesForDisplay(["SOME_UNKNOWN_GLOSS"]);

    expect(items.map((item) => item.label)).toEqual(["SOME UNKNOWN GLOSS"]);
  });

  it("keeps two consecutive fingerspelled words separate, never glued into one", () => {
    // "тобі" and "допомогти" are both outside the lexicon; without the
    // FINGERSPELL_WORD_BOUNDARY marker between their FS_ runs, this
    // collapsed into the single wrong word "Тобідопомогти".
    const items = groupGlossesForDisplay([
      "FS_Т", "FS_О", "FS_Б", "FS_І",
      "FS_BOUNDARY",
      "FS_Д", "FS_О", "FS_П", "FS_О", "FS_М", "FS_О", "FS_Г", "FS_Т", "FS_И",
    ]);

    expect(items).toHaveLength(2);
    expect(items.map((item) => item.label)).toEqual(["Тобі", "Допомогти"]);
    expect(items.every((item) => item.isFingerspell)).toBe(true);
  });

  it("the boundary marker contributes no chip of its own", () => {
    const items = groupGlossesForDisplay(["I", "FS_А", "FS_BOUNDARY", "FS_Б"]);

    expect(items.map((item) => item.label)).toEqual(["я", "А", "Б"]);
  });
});
