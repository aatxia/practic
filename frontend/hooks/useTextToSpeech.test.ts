import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useTextToSpeech } from "./useTextToSpeech";

class MockUtterance {
  lang = "";
  voice: unknown = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public text: string) {}
}

function stubSynthesis() {
  const synth = { getVoices: vi.fn(() => []), speak: vi.fn(), cancel: vi.fn() };
  vi.stubGlobal("speechSynthesis", synth);
  vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);
  return synth;
}

describe("useTextToSpeech", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports unsupported when the browser has no speechSynthesis", () => {
    const { result } = renderHook(() => useTextToSpeech());
    expect(result.current.isSupported).toBe(false);
  });

  it("speak() sets isSpeaking true, then false once the utterance ends", () => {
    const synth = stubSynthesis();
    const { result } = renderHook(() => useTextToSpeech());

    act(() => result.current.speak("Я хочу води."));
    expect(result.current.isSpeaking).toBe(true);

    const utterance = synth.speak.mock.calls[0]![0] as MockUtterance;
    act(() => utterance.onend?.());

    expect(result.current.isSpeaking).toBe(false);
  });

  it("repeat() re-speaks the last spoken text without the caller passing it again", () => {
    const synth = stubSynthesis();
    const { result } = renderHook(() => useTextToSpeech());

    act(() => result.current.speak("Я хочу води."));
    act(() => result.current.repeat());

    expect(synth.speak).toHaveBeenCalledTimes(2);
    expect((synth.speak.mock.calls[1]![0] as MockUtterance).text).toBe("Я хочу води.");
  });

  it("repeat() is a no-op before anything has been spoken", () => {
    const synth = stubSynthesis();
    const { result } = renderHook(() => useTextToSpeech());

    act(() => result.current.repeat());

    expect(synth.speak).not.toHaveBeenCalled();
  });

  it("stop() cancels speech and resets isSpeaking", () => {
    stubSynthesis();
    const { result } = renderHook(() => useTextToSpeech());

    act(() => result.current.speak("Я хочу води."));
    expect(result.current.isSpeaking).toBe(true);

    act(() => result.current.stop());

    expect(result.current.isSpeaking).toBe(false);
  });

  it("speak() with empty text never sets isSpeaking", () => {
    stubSynthesis();
    const { result } = renderHook(() => useTextToSpeech());

    act(() => result.current.speak(""));

    expect(result.current.isSpeaking).toBe(false);
  });
});
