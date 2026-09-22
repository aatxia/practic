import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { VoiceInput } from "./VoiceInput";

class MockSpeechRecognition {
  lang = "";
  interimResults = false;
  continuous = false;
  onresult: ((event: { resultIndex: number; results: { isFinal: boolean; 0: { transcript: string } }[] }) => void) | null =
    null;
  onerror: ((event: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;

  start = vi.fn();
  stop = vi.fn(() => {
    this.onend?.();
  });

  constructor() {
    lastInstance = this;
  }
}

let lastInstance: MockSpeechRecognition | null = null;

describe("VoiceInput", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    lastInstance = null;
  });

  it("shows an unsupported error when clicked without SpeechRecognition available", async () => {
    const user = userEvent.setup();

    render(<VoiceInput onTranscript={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: /Голос/ }));

    expect(screen.getByText(/не підтримує розпізнавання мовлення/)).toBeInTheDocument();
  });

  it("calls onTranscript once a final transcript is available", async () => {
    vi.stubGlobal("SpeechRecognition", MockSpeechRecognition);
    const onTranscript = vi.fn();
    const user = userEvent.setup();

    render(<VoiceInput onTranscript={onTranscript} />);
    await user.click(screen.getByRole("button", { name: /Голос/ }));

    act(() => {
      lastInstance?.onresult?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: "привіт" } }],
      });
    });

    expect(onTranscript).toHaveBeenCalledWith("привіт");
  });
});
