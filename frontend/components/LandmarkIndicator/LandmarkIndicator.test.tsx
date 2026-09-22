import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { LandmarkIndicator } from "./LandmarkIndicator";

describe("LandmarkIndicator", () => {
  it("is collapsed (opt-in) by default", () => {
    render(<LandmarkIndicator status={null} />);

    expect(screen.queryByText("Ліва рука")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Показати індикатори/ })).toBeInTheDocument();
  });

  it("shows detection state for each modality once toggled on", async () => {
    const user = userEvent.setup();
    render(
      <LandmarkIndicator
        status={{
          type: "landmarks_status",
          left_hand: true,
          right_hand: false,
          pose: true,
          face: false,
          left_hand_points: null,
          right_hand_points: null,
          pose_points: null,
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Показати індикатори/ }));

    expect(screen.getByText("Ліва рука")).toBeInTheDocument();
    expect(screen.getByText("Права рука")).toBeInTheDocument();
    expect(screen.getByText("Поза")).toBeInTheDocument();
  });

  it("treats a null status (no frame processed yet) as nothing detected", async () => {
    const user = userEvent.setup();
    render(<LandmarkIndicator status={null} />);

    await user.click(screen.getByRole("button", { name: /Показати індикатори/ }));

    // Renders the labels without crashing on a missing status -- detection
    // dots default to "not detected" rather than guessing.
    expect(screen.getByText("Ліва рука")).toBeInTheDocument();
  });

  it("toggles back to hidden on a second click", async () => {
    const user = userEvent.setup();
    render(<LandmarkIndicator status={null} />);

    const button = screen.getByRole("button", { name: /Показати індикатори/ });
    await user.click(button);
    expect(screen.getByText("Ліва рука")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Сховати індикатори/ }));
    expect(screen.queryByText("Ліва рука")).not.toBeInTheDocument();
  });
});
