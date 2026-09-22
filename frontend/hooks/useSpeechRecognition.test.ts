import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSpeechRecognition } from "./useSpeechRecognition";

interface MockResult {
  isFinal: boolean;
  0: { transcript: string };
}

class MockSpeechRecognition {
  lang = "";
  interimResults = false;
  continuous = false;
  onresult: ((event: { resultIndex: number; results: MockResult[] }) => void) | null = null;
  onerror: ((event: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;

  start = vi.fn();
  stop = vi.fn(() => {
    this.onend?.();
  });
}

let lastInstance: MockSpeechRecognition | null = null;

class TrackedMockSpeechRecognition extends MockSpeechRecognition {
  constructor() {
    super();
    lastInstance = this;
  }
}

describe("useSpeechRecognition", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    lastInstance = null;
  });

  it("reports unsupported when the browser has no SpeechRecognition API", () => {
    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error?.reason).toBe("unsupported");
  });

  it("starts listening and sets lang to uk-UA", () => {
    vi.stubGlobal("SpeechRecognition", TrackedMockSpeechRecognition);

    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });

    expect(result.current.status).toBe("listening");
    expect(lastInstance?.lang).toBe("uk-UA");
    expect(lastInstance?.start).toHaveBeenCalled();
  });

  it("accumulates final transcript results", () => {
    vi.stubGlobal("SpeechRecognition", TrackedMockSpeechRecognition);

    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });

    act(() => {
      lastInstance?.onresult?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "привіт" } }],
      });
    });

    expect(result.current.transcript).toBe("привіт");
  });

  it("maps not-allowed errors to permission_denied", () => {
    vi.stubGlobal("SpeechRecognition", TrackedMockSpeechRecognition);

    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });
    act(() => {
      lastInstance?.onerror?.({ error: "not-allowed" });
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error?.reason).toBe("permission_denied");
  });

  it("surfaces a clear error instead of crashing silently when start() throws synchronously", () => {
    // Chrome throws InvalidStateError synchronously from start() if it's
    // called again before the previous session fully ended -- uncaught,
    // this killed the click handler with no visible feedback at all.
    class ThrowingSpeechRecognition extends MockSpeechRecognition {
      override start = vi.fn(() => {
        throw new DOMException("already started", "InvalidStateError");
      });
    }
    vi.stubGlobal("SpeechRecognition", ThrowingSpeechRecognition);

    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toContain("already started");
  });

  it("sets status to stopped when stop() is called", async () => {
    vi.stubGlobal("SpeechRecognition", TrackedMockSpeechRecognition);

    const { result } = renderHook(() => useSpeechRecognition());

    act(() => {
      result.current.start();
    });
    act(() => {
      result.current.stop();
    });

    await waitFor(() => {
      expect(result.current.status).toBe("stopped");
    });
  });
});
