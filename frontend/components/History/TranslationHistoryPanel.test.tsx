import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TranslationHistoryEntry } from "@/hooks/useTranslationHistory";
import { TranslationHistoryPanel } from "./TranslationHistoryPanel";

function entry(overrides: Partial<TranslationHistoryEntry> = {}): TranslationHistoryEntry {
  return {
    id: "1",
    composedText: "Я хочу води.",
    glossSequence: ["I", "WANT", "WATER"],
    glossLabels: [],
    timestamp: Date.parse("2026-01-01T10:00:00Z"),
    direction: "gestures-to-text",
    sourceText: null,
    composedTextSource: "rule_based",
    ...overrides,
  };
}

describe("TranslationHistoryPanel", () => {
  it("renders nothing for an empty history", () => {
    const { container } = render(
      <TranslationHistoryPanel
        history={[]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    expect(container.firstChild).toBeNull();
  });

  it("shows each entry's composed text and direction label", () => {
    render(
      <TranslationHistoryPanel
        history={[entry(), entry({ id: "2", composedText: null, glossLabels: [{ text: "Ти", isFingerspell: false }], direction: "text-to-gestures" })]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    expect(screen.getByText("Я хочу води.")).toBeInTheDocument();
    expect(screen.getByText("Ти")).toBeInTheDocument();
    expect(screen.getByText(/Жести → Українська/)).toBeInTheDocument();
    expect(screen.getByText(/Українська → Жести/)).toBeInTheDocument();
  });

  it("shows an AI badge only for an entry whose sentence was AI-composed, never a rule-based one", () => {
    render(
      <TranslationHistoryPanel
        history={[
          entry({ id: "1", composedTextSource: "ai_fallback" }),
          entry({ id: "2", composedText: "Ти хотіти вода.", composedTextSource: "rule_based" }),
        ]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    expect(screen.getAllByText("ШІ")).toHaveLength(1);
  });

  it("Copy button calls onCopy with the entry", () => {
    const onCopy = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={onCopy}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    fireEvent.click(screen.getByTitle("Копіювати"));

    expect(onCopy).toHaveBeenCalledWith(entry());
  });

  it("Speak button calls onSpeak with the entry", () => {
    const onSpeak = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={vi.fn()}
        onSpeak={onSpeak}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    fireEvent.click(screen.getByTitle("Озвучити"));

    expect(onSpeak).toHaveBeenCalledWith(entry());
  });

  it("Speak button is hidden when ttsSupported is false", () => {
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported={false}
      />
    );

    expect(screen.queryByTitle("Озвучити")).not.toBeInTheDocument();
  });

  it("Reuse button calls onReuse with the entry", () => {
    const onReuse = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={onReuse}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    fireEvent.click(screen.getByTitle("Використати знову"));

    expect(onReuse).toHaveBeenCalledWith(entry());
  });

  it("Delete button calls onDelete with the entry's id", () => {
    const onDelete = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={onDelete}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    fireEvent.click(screen.getByTitle("Видалити"));

    expect(onDelete).toHaveBeenCalledWith("1");
  });

  it("Clear-all button calls onClearAll", () => {
    const onClearAll = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry()]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={onClearAll}
        ttsSupported
      />
    );

    fireEvent.click(screen.getByText("Очистити історію"));

    expect(onClearAll).toHaveBeenCalled();
  });

  it("each entry's actions operate on that entry only, not a neighboring one", () => {
    const onDelete = vi.fn();
    render(
      <TranslationHistoryPanel
        history={[entry({ id: "1" }), entry({ id: "2", composedText: "Ми йдемо додому." })]}
        onCopy={vi.fn()}
        onSpeak={vi.fn()}
        onReuse={vi.fn()}
        onDelete={onDelete}
        onClearAll={vi.fn()}
        ttsSupported
      />
    );

    fireEvent.click(screen.getAllByTitle("Видалити")[1]!);

    expect(onDelete).toHaveBeenCalledWith("2");
  });
});
