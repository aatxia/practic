import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useTranslationHistory } from "./useTranslationHistory";

const STORAGE_KEY = "uksl.translationHistory.v1";

describe("useTranslationHistory", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("addEntry sets lastFinalTranslation and prepends to history", () => {
    const { result } = renderHook(() => useTranslationHistory());

    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], [{ text: "Я", isFingerspell: false }]));

    expect(result.current.lastFinalTranslation?.composedText).toBe("Я хочу води.");
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.composedText).toBe("Я хочу води.");
  });

  it("a second entry becomes the new lastFinalTranslation without removing the first from history", () => {
    const { result } = renderHook(() => useTranslationHistory());

    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    act(() => result.current.addEntry("Ми йдемо додому.", ["WE", "GO", "HOME"], []));

    expect(result.current.lastFinalTranslation?.composedText).toBe("Ми йдемо додому.");
    expect(result.current.history).toHaveLength(2);
    // Newest first.
    expect(result.current.history[0]?.composedText).toBe("Ми йдемо додому.");
    expect(result.current.history[1]?.composedText).toBe("Я хочу води.");
  });

  it("an empty gloss sequence is not a translation -- no entry, lastFinalTranslation unchanged", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));

    act(() => result.current.addEntry(null, [], []));

    expect(result.current.history).toHaveLength(1);
    expect(result.current.lastFinalTranslation?.composedText).toBe("Я хочу води.");
  });

  it("clearCurrent hides lastFinalTranslation without touching history", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));

    act(() => result.current.clearCurrent());

    expect(result.current.lastFinalTranslation).toBeNull();
    expect(result.current.history).toHaveLength(1);
  });

  it("clearHistory empties history without touching lastFinalTranslation", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));

    act(() => result.current.clearHistory());

    expect(result.current.history).toEqual([]);
    expect(result.current.lastFinalTranslation?.composedText).toBe("Я хочу води.");
  });

  it("persists to localStorage and a fresh hook instance restores it", () => {
    const first = renderHook(() => useTranslationHistory());
    act(() => first.result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    first.unmount();

    const second = renderHook(() => useTranslationHistory());
    expect(second.result.current.history).toHaveLength(1);
    expect(second.result.current.lastFinalTranslation?.composedText).toBe("Я хочу води.");
  });

  it("clearHistory persists -- a fresh hook instance does not resurrect cleared entries", () => {
    const first = renderHook(() => useTranslationHistory());
    act(() => first.result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    act(() => first.result.current.clearHistory());
    first.unmount();

    const second = renderHook(() => useTranslationHistory());
    expect(second.result.current.history).toEqual([]);
  });

  it("ignores a corrupted localStorage value and starts with empty history instead of throwing", () => {
    window.localStorage.setItem(STORAGE_KEY, "{not valid json");

    expect(() => renderHook(() => useTranslationHistory())).not.toThrow();
    const { result } = renderHook(() => useTranslationHistory());
    expect(result.current.history).toEqual([]);
    expect(result.current.lastFinalTranslation).toBeNull();
  });

  it("ignores a localStorage value that isn't an array of valid entries", () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ not: "an array" }));

    const { result } = renderHook(() => useTranslationHistory());
    expect(result.current.history).toEqual([]);
  });

  it("filters out individually malformed entries while keeping valid ones", () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        { id: "1", composedText: "Я хочу води.", glossSequence: ["I"], glossLabels: [], timestamp: 1 },
        { garbage: true },
        null,
        "a string",
      ])
    );

    const { result } = renderHook(() => useTranslationHistory());
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.composedText).toBe("Я хочу води.");
  });

  it("defaults direction to gestures-to-text when not specified", () => {
    const { result } = renderHook(() => useTranslationHistory());

    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));

    expect(result.current.history[0]?.direction).toBe("gestures-to-text");
  });

  it("records the text-to-gestures direction when specified", () => {
    const { result } = renderHook(() => useTranslationHistory());

    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], [], "text-to-gestures"));

    expect(result.current.history[0]?.direction).toBe("text-to-gestures");
  });

  it("treats a pre-existing localStorage entry with no direction field as gestures-to-text, not a malformed entry", () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([{ id: "1", composedText: "Я хочу води.", glossSequence: ["I"], glossLabels: [], timestamp: 1 }])
    );

    const { result } = renderHook(() => useTranslationHistory());

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.direction).toBe("gestures-to-text");
  });

  it("removeEntry deletes only the matching entry from history", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    act(() => result.current.addEntry("Ми йдемо додому.", ["WE", "GO", "HOME"], []));
    const idToRemove = result.current.history[1]!.id;

    act(() => result.current.removeEntry(idToRemove));

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.composedText).toBe("Ми йдемо додому.");
  });

  it("removeEntry also clears lastFinalTranslation when it's the entry being removed", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    const id = result.current.lastFinalTranslation!.id;

    act(() => result.current.removeEntry(id));

    expect(result.current.lastFinalTranslation).toBeNull();
  });

  it("removeEntry leaves lastFinalTranslation untouched when a different entry is removed", () => {
    const { result } = renderHook(() => useTranslationHistory());
    act(() => result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    act(() => result.current.addEntry("Ми йдемо додому.", ["WE", "GO", "HOME"], []));
    const olderId = result.current.history[1]!.id;

    act(() => result.current.removeEntry(olderId));

    expect(result.current.lastFinalTranslation?.composedText).toBe("Ми йдемо додому.");
  });

  it("removeEntry persists -- a fresh hook instance does not resurrect the removed entry", () => {
    const first = renderHook(() => useTranslationHistory());
    act(() => first.result.current.addEntry("Я хочу води.", ["I", "WANT", "WATER"], []));
    const id = first.result.current.history[0]!.id;
    act(() => first.result.current.removeEntry(id));
    first.unmount();

    const second = renderHook(() => useTranslationHistory());
    expect(second.result.current.history).toEqual([]);
  });

  it("caps history length so localStorage does not grow unbounded", () => {
    const { result } = renderHook(() => useTranslationHistory());

    act(() => {
      for (let i = 0; i < 60; i += 1) {
        result.current.addEntry(`Речення ${i}.`, [`GLOSS_${i}`], []);
      }
    });

    expect(result.current.history.length).toBeLessThanOrEqual(50);
    // Newest entries are the ones kept.
    expect(result.current.history[0]?.composedText).toBe("Речення 59.");
  });
});
