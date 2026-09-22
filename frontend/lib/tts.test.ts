import { afterEach, describe, expect, it, vi } from "vitest";
import { cancelSpeech, isSpeechSynthesisSupported, speak } from "./tts";

class MockUtterance {
  lang = "";
  voice: unknown = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public text: string) {}
}

function makeMockSynthesis(voices: { lang: string; name: string }[] = []) {
  return {
    getVoices: vi.fn(() => voices),
    speak: vi.fn(),
    cancel: vi.fn(),
  };
}

describe("tts", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reports unsupported when the browser has no speechSynthesis at all", () => {
    expect(isSpeechSynthesisSupported()).toBe(false);
  });

  it("reports supported when speechSynthesis exists", () => {
    vi.stubGlobal("speechSynthesis", makeMockSynthesis());
    expect(isSpeechSynthesisSupported()).toBe(true);
  });

  it("speak() is a graceful no-op, never a throw, when unsupported", () => {
    expect(() => speak("Привіт")).not.toThrow();
  });

  it("speak() cancels any speech already in progress before speaking the new text", () => {
    const synth = makeMockSynthesis();
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);

    speak("Привіт");

    expect(synth.cancel).toHaveBeenCalled();
    expect(synth.speak).toHaveBeenCalledTimes(1);
  });

  it("speak() sets lang to uk-UA on the utterance", () => {
    const synth = makeMockSynthesis();
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);

    speak("Привіт");

    const utterance = synth.speak.mock.calls[0]![0] as MockUtterance;
    expect(utterance.lang).toBe("uk-UA");
    expect(utterance.text).toBe("Привіт");
  });

  it("speak() picks a Ukrainian voice when one is installed", () => {
    const ukVoice = { lang: "uk-UA", name: "Ukrainian" };
    const synth = makeMockSynthesis([{ lang: "en-US", name: "English" }, ukVoice]);
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);

    speak("Привіт");

    const utterance = synth.speak.mock.calls[0]![0] as MockUtterance;
    expect(utterance.voice).toBe(ukVoice);
  });

  it("speak() still works (via lang alone) when no Ukrainian voice is installed", () => {
    const synth = makeMockSynthesis([{ lang: "en-US", name: "English" }]);
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);

    speak("Привіт");

    const utterance = synth.speak.mock.calls[0]![0] as MockUtterance;
    expect(utterance.voice).toBeNull();
    expect(synth.speak).toHaveBeenCalledTimes(1);
  });

  it("speak() calls onEnd when the utterance finishes", () => {
    const synth = makeMockSynthesis();
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);
    const onEnd = vi.fn();

    speak("Привіт", { onEnd });
    const utterance = synth.speak.mock.calls[0]![0] as MockUtterance;
    utterance.onend?.();

    expect(onEnd).toHaveBeenCalled();
  });

  it("speak() calls onError instead of throwing when unsupported", () => {
    const onError = vi.fn();

    speak("Привіт", { onError });

    expect(onError).toHaveBeenCalled();
  });

  it("speak() is a no-op for empty text", () => {
    const synth = makeMockSynthesis();
    vi.stubGlobal("speechSynthesis", synth);
    vi.stubGlobal("SpeechSynthesisUtterance", MockUtterance);

    speak("");

    expect(synth.speak).not.toHaveBeenCalled();
  });

  it("cancelSpeech() is a graceful no-op when unsupported", () => {
    expect(() => cancelSpeech()).not.toThrow();
  });

  it("cancelSpeech() cancels in-progress speech when supported", () => {
    const synth = makeMockSynthesis();
    vi.stubGlobal("speechSynthesis", synth);

    cancelSpeech();

    expect(synth.cancel).toHaveBeenCalled();
  });
});
