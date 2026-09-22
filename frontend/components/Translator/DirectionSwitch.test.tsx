import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DirectionSwitch } from "./DirectionSwitch";

describe("DirectionSwitch", () => {
  it("marks the active direction as pressed", () => {
    render(<DirectionSwitch direction="gestures-to-text" onChange={vi.fn()} />);

    expect(screen.getByText("Жести → Українська")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Українська → Жести")).toHaveAttribute("aria-pressed", "false");
  });

  it("clicking the inactive side switches direction", () => {
    const onChange = vi.fn();
    render(<DirectionSwitch direction="gestures-to-text" onChange={onChange} />);

    fireEvent.click(screen.getByText("Українська → Жести"));

    expect(onChange).toHaveBeenCalledWith("text-to-gestures");
  });

  it("clicking the already-active side is a harmless no-op re-selection", () => {
    const onChange = vi.fn();
    render(<DirectionSwitch direction="gestures-to-text" onChange={onChange} />);

    fireEvent.click(screen.getByText("Жести → Українська"));

    expect(onChange).toHaveBeenCalledWith("gestures-to-text");
  });

  it("the swap button toggles to the opposite direction", () => {
    const onChange = vi.fn();
    render(<DirectionSwitch direction="gestures-to-text" onChange={onChange} />);

    fireEvent.click(screen.getByTitle("Поміняти напрямок"));

    expect(onChange).toHaveBeenCalledWith("text-to-gestures");
  });

  it("the swap button toggles back from text-to-gestures", () => {
    const onChange = vi.fn();
    render(<DirectionSwitch direction="text-to-gestures" onChange={onChange} />);

    fireEvent.click(screen.getByTitle("Поміняти напрямок"));

    expect(onChange).toHaveBeenCalledWith("gestures-to-text");
  });
});
