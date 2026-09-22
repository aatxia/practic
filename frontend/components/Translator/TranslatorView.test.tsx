import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TranslatorView, shouldSendFrame } from "./TranslatorView";

describe("shouldSendFrame", () => {
  it("sends while translating, no correction open, and nothing awaiting confirmation", () => {
    expect(shouldSendFrame(true, false, false)).toBe(true);
  });

  it("never sends while paused, whatever the other two flags are", () => {
    expect(shouldSendFrame(false, false, false)).toBe(false);
    expect(shouldSendFrame(false, true, false)).toBe(false);
    expect(shouldSendFrame(false, false, true)).toBe(false);
  });

  it("never sends while a correction popover is open, even mid-translation", () => {
    // Moving the hand to the mouse to pick a correction can otherwise be
    // read as a sign-boundary pause, splicing a spurious token into the
    // sentence being corrected -- this is the guard that stops it.
    expect(shouldSendFrame(true, true, false)).toBe(false);
  });

  it("never sends while the last confirmed word is awaiting an explicit confirm", () => {
    // Without this, continuous recognition can confirm the same word
    // several times in a row with nothing to say "yes, keep going"
    // between them -- this is the guard that pauses for that.
    expect(shouldSendFrame(true, false, true)).toBe(false);
  });
});

class MockWebSocket {
  static OPEN = 1;
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;

  constructor(public url: string) {
    lastInstance = this;
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    }, 0);
  }

  send(): void {}
  close(): void {
    this.onclose?.();
  }
}

let lastInstance: MockWebSocket | null = null;

function sendServerMessage(data: Record<string, unknown>): void {
  act(() => {
    lastInstance?.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  });
}

class MockSpeechSynthesisUtterance {
  lang = "";
  voice: unknown = null;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public text: string) {}
}

function stubNavigator(extra: Record<string, unknown> = {}): void {
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: vi.fn() },
    ...extra,
  });
}

describe("TranslatorView", () => {
  beforeEach(() => {
    lastInstance = null;
    vi.stubGlobal("WebSocket", MockWebSocket);
    stubNavigator();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("no backend in this test")));
    // useTranslationHistory persists to localStorage -- a real jsdom
    // localStorage survives across tests in this file unless cleared, so
    // one test's history would otherwise leak into the next test's
    // initial render.
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows a friendly connecting/listening state, never raw WebSocket wording", async () => {
    render(<TranslatorView />);

    expect(screen.getByTestId("live-sentence-status").textContent).toContain("Підключення до камери");
    expect(screen.queryByText(/WebSocket/)).not.toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю");
    });
  });

  it("never shows a raw backend status/error string in the live translation line", async () => {
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({
      type: "error",
      message:
        "Sign-recognition is not available yet. Landmarks were extracted: left_hand=False, right_hand=False, pose=True, face=True (feature vector size: 222/222).",
    });

    // Developer-facing plumbing text must never reach the UI.
    expect(screen.queryByText(/Sign-recognition/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Landmarks were extracted/)).not.toBeInTheDocument();
  });

  it("accumulates a camera-confirmed word into the live line immediately -- no button press required", async () => {
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));

    sendServerMessage({
      type: "final_prediction",
      text: "I",
      gloss: "I",
      confidence: 0.9,
      is_final: true,
      facial_grammar: "NONE",
    });

    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("я");
    });
    expect(screen.queryByRole("button", { name: "Почати речення" })).not.toBeInTheDocument();
  });

  it("shows real interim confidence for both classifiers in a diagnostics readout (hidden by default), never a raw gloss code", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());
    expect(screen.queryByTestId("recognition-diagnostics")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Діагностика/ }));
    await waitFor(() => expect(screen.getByTestId("recognition-diagnostics")).toBeInTheDocument());

    expect(screen.getByTestId("recognition-diagnostics").textContent).toContain("—");

    sendServerMessage({
      type: "prediction",
      text: "WANT",
      gloss: "WANT",
      confidence: 0.42,
      is_final: false,
      facial_grammar: "NONE",
    });
    sendServerMessage({ type: "letter_prediction", letter: "Б", confidence: 0.37, is_final: false });

    await waitFor(() => {
      const diagnostics = screen.getByTestId("recognition-diagnostics").textContent;
      expect(diagnostics).toContain("хотіти");
      expect(diagnostics).toContain("42%");
      expect(diagnostics).toContain("Б");
      expect(diagnostics).toContain("37%");
    });
    expect(screen.queryByText("WANT")).not.toBeInTheDocument();
  });

  it("distinguishes an interim guess from a confirmed one in the diagnostics readout", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());
    await user.click(screen.getByRole("button", { name: /Діагностика/ }));
    await waitFor(() => expect(screen.getByTestId("recognition-diagnostics")).toBeInTheDocument());

    sendServerMessage({ type: "letter_prediction", letter: "Ш", confidence: 0.52, is_final: false });
    await waitFor(() => {
      const diagnostics = screen.getByTestId("recognition-diagnostics").textContent;
      expect(diagnostics).toContain("Ш");
      expect(diagnostics).toContain("52%");
      expect(diagnostics).toContain("проміжно");
      expect(diagnostics).not.toContain("підтверджено");
    });

    sendServerMessage({ type: "letter_confirmed", letter: "Ш", confidence: 0.98, is_final: true });
    await waitFor(() => {
      expect(screen.getByTestId("recognition-diagnostics").textContent).toContain("підтверджено");
    });
  });

  it("shows a question-marker badge (Phase 17) when eyebrows are raised", async () => {
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({
      type: "final_prediction",
      text: "Так?",
      gloss: "TAK",
      confidence: 0.91,
      is_final: true,
      facial_grammar: "EYEBROWS_RAISED",
    });

    await waitFor(() => {
      expect(screen.getByText(/Брови підняті/)).toBeInTheDocument();
    });
  });

  it("translates typed text into a real Ukrainian sentence via the Phase 14 API, never the internal gloss codes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/text-to-gloss")) {
          return {
            ok: true,
            json: async () => ({
              gloss_sequence: ["I", "WANT", "WATER"],
              gloss_labels: [
                { text: "Я", is_fingerspell: false },
                { text: "хотіти", is_fingerspell: false },
                { text: "вода", is_fingerspell: false },
              ],
              composed_text: "Я хочу води.",
            }),
          };
        }
        throw new Error("no backend in this test");
      }),
    );
    const user = userEvent.setup();

    render(<TranslatorView />);
    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    await user.type(screen.getByPlaceholderText(/Введіть текст/), "Я хочу води.");
    await user.click(screen.getByRole("button", { name: "Перекласти" }));

    await waitFor(() => {
      expect(screen.getAllByText("Я хочу води.").length).toBeGreaterThan(0);
    });
    const transcriptLabels = within(screen.getByTestId("transcript-gloss-labels"));
    expect(transcriptLabels.getByText("хотіти")).toBeInTheDocument();
    expect(transcriptLabels.getByText("вода")).toBeInTheDocument();
    expect(screen.queryByText("WANT")).not.toBeInTheDocument();
    expect(screen.queryByText("WATER")).not.toBeInTheDocument();
  });

  it("appends a confirmed dactyl letter to the SAME live line as word gestures, not a separate panel", async () => {
    render(<TranslatorView />);

    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("я");
    });

    sendServerMessage({ type: "letter_confirmed", letter: "А", confidence: 0.9, is_final: true });
    await waitFor(() => {
      const text = screen.getByTestId("live-sentence-status").textContent;
      expect(text).toContain("я");
      expect(text).toContain("А");
    });
    expect(screen.queryByText("Дактиль")).not.toBeInTheDocument();
  });

  it("grows a fingerspelled word letter by letter, then finalizes it into ONE token on a 'token' boundary", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    const progression = ["Н", "На", "Нас", "Наст", "Настя"];
    for (const [index, letter] of ["Н", "А", "С", "Т", "Я"].entries()) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
      await waitFor(() => {
        expect(screen.getByTestId("live-sentence-status").textContent).toContain(progression[index]);
      });
    }
    expect(screen.getByTestId("live-sentence-status").textContent).toContain("Настя");

    sendServerMessage({ type: "boundary", kind: "token" });
    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("Настя");
    });
  });

  it("two fingerspelled words signed back to back stay two separate words in the live line, never glued into one", async () => {
    // The same flattening risk that affects typed text ("тобі допомогти" ->
    // "Тобідопомогти") exists here for two dactyl words spelled one after
    // another with only a 'token' boundary between them.
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    for (const letter of ["А", "Б", "О"]) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
    }
    sendServerMessage({ type: "boundary", kind: "token" });
    for (const letter of ["В", "І"]) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
    }
    await waitFor(() => {
      const text = screen.getByTestId("live-sentence-status").textContent;
      expect(text).toContain("Або");
      expect(text).toContain("Ві");
      expect(text).not.toContain("Абові");
    });
  });

  it("composes the accumulated utterance automatically on an 'utterance' boundary -- no button press required", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          const body = JSON.parse(init?.body as string) as { gloss_sequence: string[] };
          expect(body.gloss_sequence).toEqual(["I", "WANT", "WATER"]);
          return {
            ok: true,
            json: async () => ({
              gloss_labels: [
                { text: "Я", is_fingerspell: false },
                { text: "хотіти", is_fingerspell: false },
                { text: "вода", is_fingerspell: false },
              ],
              composed_text: "Я хочу води.",
            }),
          };
        }
        throw new Error("no backend in this test");
      }),
    );
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    const expectedLabels = ["я", "хотіти", "вода"];
    for (const [index, gloss] of ["I", "WANT", "WATER"].entries()) {
      sendServerMessage({
        type: "final_prediction",
        text: gloss,
        gloss,
        confidence: 0.9,
        is_final: true,
        facial_grammar: "NONE",
      });
      await waitFor(() => {
        expect(screen.getByTestId("live-sentence-status").textContent).toContain(expectedLabels[index]);
      });
    }

    sendServerMessage({ type: "boundary", kind: "utterance" });

    await waitFor(() => {
      expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");
    });
    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю");
    });

    sendServerMessage({ type: "final_prediction", text: "WE", gloss: "WE", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("ми");
    });
    expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");
  });

  it("'Завершити речення' manually finalizes the current utterance as a fallback control", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          return {
            ok: true,
            json: async () => ({
              gloss_labels: [{ text: "Я", is_fingerspell: false }],
              composed_text: "Я.",
            }),
          };
        }
        throw new Error("no backend in this test");
      }),
    );
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    expect(screen.getByRole("button", { name: "Завершити речення" })).toBeDisabled();

    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Завершити речення" })).toBeEnabled());

    await user.click(screen.getByRole("button", { name: "Завершити речення" }));

    await waitFor(() => {
      expect(screen.getByTestId("final-translation").textContent).toContain("Я.");
    });
  });

  it("'Скинути фразу' clears the current utterance without composing anything", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("must not be called"));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    await user.click(screen.getByRole("button", { name: "Скинути фразу" }));

    await waitFor(() => {
      expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю");
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  async function composeOnce(composedText: string, gloss = "I", liveLineLabel = "я") {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          return { ok: true, json: async () => ({ gloss_labels: [{ text: composedText, is_fingerspell: false }], composed_text: composedText }) };
        }
        throw new Error("no backend in this test");
      }),
    );
    sendServerMessage({ type: "final_prediction", text: gloss, gloss, confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain(liveLineLabel));
    sendServerMessage({ type: "boundary", kind: "utterance" });
    await waitFor(() => expect(screen.getByTestId("final-translation").textContent).toContain(composedText));
  }

  it("the final translation remains visible after resetting the active utterance ('Скинути фразу')", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    sendServerMessage({ type: "final_prediction", text: "WE", gloss: "WE", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("ми"));

    await user.click(screen.getByRole("button", { name: "Скинути фразу" }));

    expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));
  });

  it("translation history receives each completed result exactly once, in order, and an empty utterance never creates an entry", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "boundary", kind: "utterance" });
    expect(screen.queryByTestId("translation-history")).not.toBeInTheDocument();

    await composeOnce("Я хочу води.");
    await waitFor(() => {
      const history = within(screen.getByTestId("translation-history"));
      expect(history.getAllByRole("listitem")).toHaveLength(1);
      expect(history.getByText("Я хочу води.")).toBeInTheDocument();
    });

    sendServerMessage({ type: "final_prediction", text: "WE", gloss: "WE", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("ми"));
    await composeOnce("Ми йдемо додому.", "GO", "іти");

    await waitFor(() => {
      expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(2);
    });
  });

  it("'Очистити переклад' (current) and 'Очистити історію' behave independently", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");
    await waitFor(() => expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(1));

    await user.click(screen.getByTitle("Очистити переклад"));
    expect(screen.getByTestId("final-translation").textContent).toContain("Тут з'явиться завершене речення.");
    expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(1);

    await composeOnce("Ми йдемо додому.", "GO", "іти");
    await user.click(screen.getByRole("button", { name: "Очистити історію" }));
    expect(screen.queryByTestId("translation-history")).not.toBeInTheDocument();
    expect(screen.getByTestId("final-translation").textContent).toContain("Ми йдемо додому.");
  });

  it("copies the final translation to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubNavigator({ clipboard: { writeText } });
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");
    fireEvent.click(within(screen.getByTestId("final-translation")).getByTitle("Копіювати"));

    expect(writeText).toHaveBeenCalledWith("Я хочу води.");
  });

  it("duplicate finalization for the SAME boundary message does not duplicate the history entry", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          return { ok: true, json: async () => ({ gloss_labels: [{ text: "Я", is_fingerspell: false }], composed_text: "Я хочу води." }) };
        }
        throw new Error("no backend in this test");
      }),
    );
    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    const boundaryEvent = { data: JSON.stringify({ type: "boundary", kind: "utterance" }) } as MessageEvent<string>;
    act(() => {
      lastInstance?.onmessage?.(boundaryEvent);
      lastInstance?.onmessage?.(boundaryEvent);
    });

    await waitFor(() => expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води."));
    expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(1);
  });

  it("shows an AI-composed badge only when the backend says composed_text_source is ai_fallback, never for rule_based", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          return {
            ok: true,
            json: async () => ({
              gloss_labels: [{ text: "Я", is_fingerspell: false }],
              // The rule-based composer couldn't fit a pattern here -- an
              // LLM's real best attempt filled composed_text instead.
              composed_text: "Вода Я (ШІ-версія).",
              composed_text_source: "ai_fallback",
            }),
          };
        }
        throw new Error("no backend in this test");
      }),
    );
    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    fireEvent.click(screen.getByRole("button", { name: "Завершити речення" }));

    await waitFor(() => expect(screen.getByTestId("final-translation").textContent).toContain("Вода Я (ШІ-версія)."));
    expect(screen.getByTestId("ai-composed-badge")).toBeInTheDocument();
  });

  it("shows no AI badge when the backend's rule-based composer produced the sentence", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          return {
            ok: true,
            json: async () => ({
              gloss_labels: [{ text: "Я", is_fingerspell: false }],
              composed_text: "Я.",
              composed_text_source: "rule_based",
            }),
          };
        }
        throw new Error("no backend in this test");
      }),
    );
    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    fireEvent.click(screen.getByRole("button", { name: "Завершити речення" }));

    await waitFor(() => expect(screen.getByTestId("final-translation").textContent).toContain("Я."));
    expect(screen.queryByTestId("ai-composed-badge")).not.toBeInTheDocument();
  });

  it("restores translation history from localStorage on mount", async () => {
    const stored = [
      {
        id: "abc-1",
        composedText: "Я хочу води.",
        glossSequence: ["I", "WANT", "WATER"],
        glossLabels: [{ text: "Я", isFingerspell: false }],
        timestamp: 1,
      },
    ];
    window.localStorage.setItem("uksl.translationHistory.v1", JSON.stringify(stored));

    render(<TranslatorView />);

    expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");
    expect(within(screen.getByTestId("translation-history")).getByText("Я хочу води.")).toBeInTheDocument();
  });

  it("a corrupted localStorage value is ignored safely -- an empty history, never a crash", async () => {
    window.localStorage.setItem("uksl.translationHistory.v1", "{not valid json");

    expect(() => render(<TranslatorView />)).not.toThrow();
    expect(screen.queryByTestId("translation-history")).not.toBeInTheDocument();
  });

  it("diagnostics updates (interim predictions, toggling the panel) never modify the final translation or history", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    await user.click(screen.getByRole("button", { name: /Діагностика/ }));
    sendServerMessage({ type: "prediction", text: "WANT", gloss: "WANT", confidence: 0.5, is_final: false, facial_grammar: "NONE" });
    sendServerMessage({ type: "letter_prediction", letter: "Б", confidence: 0.4, is_final: false });
    await waitFor(() => expect(screen.getByTestId("recognition-diagnostics")).toBeInTheDocument());

    expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");
    expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(1);
  });

  // --- New product-overhaul controls ---

  it("shows an interim guess in a separately labeled 'Розпізнаю' line, never mixed into the main line with '?'/'…' clutter", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));

    sendServerMessage({ type: "prediction", text: "WANT", gloss: "WANT", confidence: 0.42, is_final: false, facial_grammar: "NONE" });

    await waitFor(() => expect(screen.getByTestId("recognizing-hint")).toBeInTheDocument());
    expect(screen.getByTestId("recognizing-hint").textContent).toContain("хотіти");
    // The old confusing format appended "?" directly to the main line --
    // must never appear anywhere now.
    expect(screen.getByTestId("live-sentence-status").textContent).not.toContain("?");
  });

  it("a confirmed word's line never shows a redundant interim guess for the same word right after it", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    sendServerMessage({ type: "prediction", text: "I", gloss: "I", confidence: 0.5, is_final: false, facial_grammar: "NONE" });

    await waitFor(() => expect(screen.getByTestId("recognizing-hint")).toBeInTheDocument());
    // Confirmed "я" and the interim "я" guess are now visually separated --
    // never concatenated into a confusing "я я?" on one line.
    const mainLine = screen.getByTestId("live-sentence-status");
    expect(mainLine.textContent).not.toContain("?");
  });

  it("Pause stops the live line, and Resume/Start returns it to listening", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));

    await user.click(screen.getByRole("button", { name: "Пауза" }));
    expect(screen.getByTestId("live-sentence-status").textContent).toContain("Пауза");

    await user.click(screen.getByRole("button", { name: "Продовжити" }));
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));
  });

  it("a confirmed word with real ranked alternatives shows a correction popover, and picking one replaces the wrong saved word", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    // The gesture confirms "WANT", but the model's own real top_k ranked
    // "HAVE" as a close second -- exactly the "recognized the wrong word
    // and it got saved" case this popover fixes.
    sendServerMessage({
      type: "final_prediction",
      text: "WANT",
      gloss: "WANT",
      confidence: 0.6,
      is_final: true,
      facial_grammar: "NONE",
      top_k: [
        { gloss: "WANT", confidence: 0.6 },
        { gloss: "HAVE", confidence: 0.35 },
      ],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("хотіти"));

    const trigger = screen.getByRole("button", { name: "Виправити слово «хотіти»" });
    await user.click(trigger);
    const option = screen.getByRole("option", { name: "мати 35%" });
    await user.click(option);

    // The picked option is closed and the trigger now reflects the new
    // value -- the real assertion is what's now shown, not a raw
    // textContent search that would also match the still-open option list.
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Виправити слово «мати»" })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: "Виправити слово «хотіти»" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("confirming a word pauses recognition and shows a visible indicator, until the correction popover is used to pick a candidate", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    // A confirmed word already pauses recognition on its own (see the
    // "each word waits for an explicit confirm" tests below) -- this
    // test is specifically about the SAME banner staying up while the
    // correction popover for that word is open, then clearing once a
    // candidate is picked (which counts as confirming it too).
    sendServerMessage({
      type: "final_prediction",
      text: "WANT",
      gloss: "WANT",
      confidence: 0.6,
      is_final: true,
      facial_grammar: "NONE",
      top_k: [
        { gloss: "WANT", confidence: 0.6 },
        { gloss: "HAVE", confidence: 0.35 },
      ],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("хотіти"));
    expect(screen.getByTestId("correction-pause-banner")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Виправити слово «хотіти»" }));
    expect(screen.getByTestId("correction-pause-banner")).toBeInTheDocument();

    await user.click(screen.getByRole("option", { name: "мати 35%" }));
    await waitFor(() => expect(screen.queryByTestId("correction-pause-banner")).not.toBeInTheDocument());
  });

  it("each confirmed word pauses recognition and waits for an explicit Підтвердити before the next one", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    // Even a word with only one real candidate (no correction popover to
    // pick from) still needs an explicit confirm -- otherwise continuous
    // recognition can produce runs of the same word confirmed several
    // times with nothing between them ("Я мати мати мати мати мати").
    sendServerMessage({
      type: "final_prediction",
      text: "I",
      gloss: "I",
      confidence: 0.95,
      is_final: true,
      facial_grammar: "NONE",
      top_k: [{ gloss: "I", confidence: 0.95 }],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    expect(screen.getByTestId("correction-pause-banner")).toBeInTheDocument();
    expect(screen.getByTestId("confirm-pending-banner")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Підтвердити" }));

    expect(screen.queryByTestId("correction-pause-banner")).not.toBeInTheDocument();
    expect(screen.queryByTestId("confirm-pending-banner")).not.toBeInTheDocument();
  });

  it("'Скасувати останнє' on a still-pending word clears the confirm banner along with the word itself", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({
      type: "final_prediction",
      text: "I",
      gloss: "I",
      confidence: 0.95,
      is_final: true,
      facial_grammar: "NONE",
      top_k: [{ gloss: "I", confidence: 0.95 }],
    });
    await waitFor(() => expect(screen.getByTestId("confirm-pending-banner")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Скасувати останнє" }));

    expect(screen.queryByTestId("confirm-pending-banner")).not.toBeInTheDocument();
    expect(screen.queryByTestId("correction-pause-banner")).not.toBeInTheDocument();
  });

  it("a confirmed word with only one real candidate shows no correction popover -- nothing to correct to", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({
      type: "final_prediction",
      text: "I",
      gloss: "I",
      confidence: 0.95,
      is_final: true,
      facial_grammar: "NONE",
      top_k: [{ gloss: "I", confidence: 0.95 }],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));

    expect(screen.queryByRole("button", { name: /Виправити слово/ })).not.toBeInTheDocument();
  });

  it("a confirmed letter with real ranked alternatives shows a correction popover, and picking one fixes the letter mid-word", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "letter_confirmed", letter: "Н", confidence: 0.9, is_final: true, top_k: [{ letter: "Н", confidence: 0.9 }] });
    // The gesture confirms "С", but the model's own real top_k ranked "Ш" close behind.
    sendServerMessage({
      type: "letter_confirmed",
      letter: "С",
      confidence: 0.55,
      is_final: true,
      top_k: [
        { letter: "С", confidence: 0.55 },
        { letter: "Ш", confidence: 0.4 },
      ],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Нс"));

    const trigger = screen.getByRole("button", { name: "Виправити літеру «С»" });
    await user.click(trigger);
    const option = screen.getByRole("option", { name: "Ш 40%" });
    await user.click(option);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Виправити літеру «Ш»" })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: "Виправити літеру «С»" })).not.toBeInTheDocument();
  });

  it("a letter inside an ALREADY-FINALIZED fingerspelled word can still be corrected from its real ranked alternatives", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "letter_confirmed", letter: "Н", confidence: 0.9, is_final: true, top_k: [{ letter: "Н", confidence: 0.9 }] });
    // The gesture confirms "С", model's real top_k ranked "Ш" close behind.
    sendServerMessage({
      type: "letter_confirmed",
      letter: "С",
      confidence: 0.55,
      is_final: true,
      top_k: [
        { letter: "С", confidence: 0.55 },
        { letter: "Ш", confidence: 0.4 },
      ],
    });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Нс"));

    // Finalize the word ("token" boundary) BEFORE correcting it -- this is
    // specifically the already-saved-word case, not the still-growing buffer.
    sendServerMessage({ type: "boundary", kind: "token" });
    await waitFor(() => expect(screen.queryByTestId("fingerspelling-controls")).not.toBeInTheDocument());

    const trigger = screen.getByRole("button", { name: "Виправити літеру «С»" });
    await user.click(trigger);
    const option = screen.getByRole("option", { name: "Ш 40%" });
    await user.click(option);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Виправити літеру «Ш»" })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: "Виправити літеру «С»" })).not.toBeInTheDocument();

    // The correction is reflected in what actually gets sent when the
    // sentence is finished, not just in the visible chip.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url.toString().endsWith("/translate/gloss-to-text")) {
          const body = JSON.parse(init?.body as string) as { gloss_sequence: string[] };
          expect(body.gloss_sequence).toEqual(["FS_Н", "FS_Ш"]);
          return { ok: true, json: async () => ({ gloss_labels: [], composed_text: "Нш." }) };
        }
        throw new Error("no backend in this test");
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Завершити речення" }));
    await waitFor(() => expect(screen.getByTestId("final-translation").textContent).toContain("Нш."));
  });

  it("'Скасувати останнє' removes only the most recently confirmed word", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    sendServerMessage({ type: "final_prediction", text: "I", gloss: "I", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("я"));
    sendServerMessage({ type: "final_prediction", text: "WANT", gloss: "WANT", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("хотіти"));

    await user.click(screen.getByRole("button", { name: "Скасувати останнє" }));

    await waitFor(() => {
      const text = screen.getByTestId("live-sentence-status").textContent;
      expect(text).toContain("я");
      expect(text).not.toContain("хотіти");
    });
  });

  it("'Скасувати останню літеру' removes only the last letter of the in-progress fingerspelled word", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    for (const letter of ["Н", "А"]) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
    }
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("На"));

    await user.click(screen.getByRole("button", { name: "Скасувати останню літеру" }));

    await waitFor(() => {
      const text = screen.getByTestId("live-sentence-status").textContent;
      expect(text).toContain("Н");
      expect(text).not.toContain("На");
    });
  });

  it("'Завершити дактильне слово' manually finalizes the buffer without waiting for a boundary event", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    for (const letter of ["О", "К", "О"]) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
    }
    await waitFor(() => expect(screen.getByTestId("fingerspelling-controls")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Завершити дактильне слово" }));

    await waitFor(() => expect(screen.queryByTestId("fingerspelling-controls")).not.toBeInTheDocument());
    expect(screen.getByTestId("live-sentence-status").textContent).toContain("Око");
  });

  it("'Очистити дактиль' discards the unfinished fingerspelling buffer entirely", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    for (const letter of ["О", "К"]) {
      sendServerMessage({ type: "letter_confirmed", letter, confidence: 0.9, is_final: true });
    }
    await waitFor(() => expect(screen.getByTestId("fingerspelling-controls")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Очистити дактиль" }));

    await waitFor(() => expect(screen.queryByTestId("fingerspelling-controls")).not.toBeInTheDocument());
    expect(screen.getByTestId("live-sentence-status").textContent).not.toContain("Ок");
  });

  it("'Очистити все' asks for confirmation, then clears current utterance/fingerspelling/final result but not history", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");
    sendServerMessage({ type: "final_prediction", text: "WE", gloss: "WE", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("ми"));

    await user.click(screen.getByRole("button", { name: "Очистити все" }));
    expect(screen.getByTestId("clear-all-confirm")).toBeInTheDocument();

    // Cancel -- nothing is cleared.
    await user.click(screen.getByRole("button", { name: "Скасувати" }));
    expect(screen.getByTestId("final-translation").textContent).toContain("Я хочу води.");

    await user.click(screen.getByRole("button", { name: "Очистити все" }));
    await user.click(screen.getByRole("button", { name: "Так" }));

    expect(screen.getByTestId("final-translation").textContent).toContain("Тут з'явиться завершене речення.");
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("Слухаю"));
    // History is untouched -- only "Очистити історію" clears that.
    expect(within(screen.getByTestId("translation-history")).getAllByRole("listitem")).toHaveLength(1);
  });

  it("Speak reads the final translation aloud, and Repeat speaks it again", async () => {
    const speak = vi.fn();
    vi.stubGlobal("speechSynthesis", { getVoices: vi.fn(() => []), speak, cancel: vi.fn() });
    vi.stubGlobal("SpeechSynthesisUtterance", MockSpeechSynthesisUtterance);
    const user = userEvent.setup();
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    await user.click(within(screen.getByTestId("final-translation")).getByTitle("Озвучити"));
    expect(speak).toHaveBeenCalledTimes(1);
    expect((speak.mock.calls[0]![0] as MockSpeechSynthesisUtterance).text).toBe("Я хочу води.");

    const utterance = speak.mock.calls[0]![0] as MockSpeechSynthesisUtterance;
    act(() => utterance.onend?.());

    await user.click(screen.getByTitle("Повторити озвучення"));
    expect(speak).toHaveBeenCalledTimes(2);
  });

  it("Speak/Repeat controls are hidden when the browser has no speech synthesis support", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    expect(screen.queryByTitle("Озвучити")).not.toBeInTheDocument();
    expect(screen.queryByTitle("Повторити озвучення")).not.toBeInTheDocument();
  });

  it("switching direction shows the text-to-gesture panel instead of the camera, without reloading", async () => {
    const user = userEvent.setup();
    render(<TranslatorView />);

    expect(screen.getByTestId("live-sentence-status")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/Введіть текст/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    expect(screen.queryByTestId("live-sentence-status")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Введіть текст/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Жести → Українська" }));
    expect(screen.getByTestId("live-sentence-status")).toBeInTheDocument();
  });

  it("text panel: Clear empties the input and resets the gesture video to idle", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ gloss_sequence: ["I"], gloss_labels: [{ text: "Я", is_fingerspell: false }], composed_text: "Я." }),
      })),
    );
    const user = userEvent.setup();
    render(<TranslatorView />);
    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    await user.type(screen.getByPlaceholderText(/Введіть текст/), "Я.");
    await user.click(screen.getByRole("button", { name: "Перекласти" }));
    await waitFor(() => expect(screen.getAllByText("Я.").length).toBeGreaterThan(0));

    await user.click(screen.getByRole("button", { name: "Очистити" }));

    expect(screen.getByPlaceholderText(/Введіть текст/)).toHaveValue("");
    expect(screen.getByText(/Покажи слово чи речення/)).toBeInTheDocument();
  });

  it("text panel: Paste fills the input from the clipboard", async () => {
    // userEvent.setup() installs its own clipboard polyfill on navigator --
    // it must run BEFORE stubNavigator() here, or it would silently
    // overwrite this test's custom readText stub.
    const user = userEvent.setup();
    const readText = vi.fn().mockResolvedValue("Привіт");
    stubNavigator({ clipboard: { readText } });
    render(<TranslatorView />);
    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    await user.click(screen.getByRole("button", { name: /Вставити/ }));

    await waitFor(() => expect(screen.getByPlaceholderText(/Введіть текст/)).toHaveValue("Привіт"));
  });

  it("text panel: Repeat translation re-runs translate for the current text", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ gloss_sequence: ["I"], gloss_labels: [{ text: "Я", is_fingerspell: false }], composed_text: "Я." }),
    }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<TranslatorView />);
    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    await user.type(screen.getByPlaceholderText(/Введіть текст/), "Я.");
    await user.click(screen.getByRole("button", { name: "Перекласти" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Повторити переклад/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("text-to-gesture translations are recorded in history with the text-to-gestures direction", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ gloss_sequence: ["I"], gloss_labels: [{ text: "Я", is_fingerspell: false }], composed_text: "Я." }),
      })),
    );
    const user = userEvent.setup();
    render(<TranslatorView />);
    await user.click(screen.getByRole("button", { name: "Українська → Жести" }));

    await user.type(screen.getByPlaceholderText(/Введіть текст/), "Я.");
    await user.click(screen.getByRole("button", { name: "Перекласти" }));

    await waitFor(() => {
      expect(within(screen.getByTestId("translation-history")).getByText("Я.")).toBeInTheDocument();
      expect(within(screen.getByTestId("translation-history")).getByText(/Українська → Жести/)).toBeInTheDocument();
    });
  });

  it("history: Copy copies that entry's text, independent of the currently shown final translation", async () => {
    // @testing-library/user-event's own setup() installs a clipboard
    // polyfill on navigator -- stubbing navigator.clipboard AFTER it
    // would be silently overwritten, so this test uses fireEvent (never
    // userEvent) throughout and stubs navigator without it.
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubNavigator({ clipboard: { writeText } });
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    fireEvent.click(within(screen.getByTestId("translation-history")).getByTitle("Копіювати"));

    expect(writeText).toHaveBeenCalledWith("Я хочу води.");
  });

  it("history: Speak reads that entry's text aloud", async () => {
    const speak = vi.fn();
    vi.stubGlobal("speechSynthesis", { getVoices: vi.fn(() => []), speak, cancel: vi.fn() });
    vi.stubGlobal("SpeechSynthesisUtterance", MockSpeechSynthesisUtterance);
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    fireEvent.click(within(screen.getByTestId("translation-history")).getByTitle("Озвучити"));

    expect(speak).toHaveBeenCalledTimes(1);
    expect((speak.mock.calls[0]![0] as MockSpeechSynthesisUtterance).text).toBe("Я хочу води.");
  });

  it("history: Reuse switches to the text-to-gestures panel with that entry's text and video", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");

    fireEvent.click(within(screen.getByTestId("translation-history")).getByTitle("Використати знову"));

    expect(screen.queryByTestId("live-sentence-status")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Введіть текст/)).toHaveValue("Я хочу води.");
  });

  it("history: Delete removes only that entry, leaving the rest and the current final translation intact", async () => {
    render(<TranslatorView />);
    await waitFor(() => expect(lastInstance).not.toBeNull());

    await composeOnce("Я хочу води.");
    sendServerMessage({ type: "final_prediction", text: "WE", gloss: "WE", confidence: 0.9, is_final: true, facial_grammar: "NONE" });
    await waitFor(() => expect(screen.getByTestId("live-sentence-status").textContent).toContain("ми"));
    await composeOnce("Ми йдемо додому.", "GO", "іти");

    const olderEntry = within(screen.getByTestId("translation-history")).getByText("Я хочу води.");
    const deleteButton = within(olderEntry.closest("li")!).getByTitle("Видалити");
    fireEvent.click(deleteButton);

    expect(within(screen.getByTestId("translation-history")).queryByText("Я хочу води.")).not.toBeInTheDocument();
    expect(within(screen.getByTestId("translation-history")).getByText("Ми йдемо додому.")).toBeInTheDocument();
    expect(screen.getByTestId("final-translation").textContent).toContain("Ми йдемо додому.");
  });
});
