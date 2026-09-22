import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Avatar } from "./Avatar";

describe("Avatar", () => {
  it("shows the WebGL-unsupported fallback in jsdom (no real GL context available)", () => {
    render(<Avatar glossSequence={["PRIVIT"]} />);

    expect(screen.getByText(/WebGL не підтримується/)).toBeInTheDocument();
  });
});
