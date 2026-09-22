import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Transcript } from "./Transcript";

describe("Transcript", () => {
  it("shows an idle hint by default", () => {
    render(<Transcript state={{ status: "idle" }} />);
    expect(screen.getByText(/Введи текст або скажи щось голосом/)).toBeInTheDocument();
  });

  it("shows a loading message", () => {
    render(<Transcript state={{ status: "loading" }} />);
    expect(screen.getByText("Перекладаю...")).toBeInTheDocument();
  });

  it("shows the error message", () => {
    render(<Transcript state={{ status: "error", message: "Unrecognized word 'кавун'" }} />);
    expect(screen.getByText("Unrecognized word 'кавун'")).toBeInTheDocument();
  });

  it("renders the composed Ukrainian sentence and each Ukrainian word chip, never the internal gloss codes", () => {
    render(
      <Transcript
        state={{
          status: "success",
          glossSequence: ["I", "WANT", "WATER"],
          glossLabels: [
            { text: "Я", isFingerspell: false },
            { text: "хотіти", isFingerspell: false },
            { text: "вода", isFingerspell: false },
          ],
          composedText: "Я хочу води.",
        }}
      />,
    );

    expect(screen.getByText("Я хочу води.")).toBeInTheDocument();
    expect(screen.getByText("Я")).toBeInTheDocument();
    expect(screen.getByText("хотіти")).toBeInTheDocument();
    expect(screen.getByText("вода")).toBeInTheDocument();
    expect(screen.queryByText("WANT")).not.toBeInTheDocument();
    expect(screen.queryByText("WATER")).not.toBeInTheDocument();
  });

  it("groups a fingerspelled word (Phase 16) into one readable chip", () => {
    render(
      <Transcript
        state={{
          status: "success",
          glossSequence: ["I", "WANT", "FS_К", "FS_А", "FS_В", "FS_У", "FS_Н"],
          glossLabels: [
            { text: "Я", isFingerspell: false },
            { text: "хотіти", isFingerspell: false },
            { text: "Кавун", isFingerspell: true },
          ],
          composedText: "Я хочу Кавун.",
        }}
      />,
    );

    expect(screen.getByText("Я")).toBeInTheDocument();
    expect(screen.getByText("хотіти")).toBeInTheDocument();
    expect(screen.getByText("Кавун")).toBeInTheDocument();
    expect(screen.queryByText("FS_К")).not.toBeInTheDocument();
    expect(screen.queryByText("WANT")).not.toBeInTheDocument();
  });

  it("shows a fallback note (no bold sentence) when the sequence doesn't compose into a full sentence", () => {
    render(
      <Transcript
        state={{
          status: "success",
          glossSequence: ["WATER"],
          glossLabels: [{ text: "вода", isFingerspell: false }],
          composedText: null,
        }}
      />,
    );

    expect(screen.getByText("вода")).toBeInTheDocument();
    expect(screen.getByText(/повного речення для цієї комбінації ще немає/)).toBeInTheDocument();
  });
});
