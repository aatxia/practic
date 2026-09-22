import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Camera } from "./Camera";

describe("Camera", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("shows the inactive state and enabled 'start' button before streaming", () => {
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn() } });

    render(<Camera />);

    expect(screen.getByText("Камера не активна")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Увімкнути камеру" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Вимкнути" })).toBeDisabled();
  });

  it("switches to streaming state after granting camera permission", async () => {
    const track = { stop: vi.fn() };
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track] });
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const user = userEvent.setup();
    render(<Camera />);

    await user.click(screen.getByRole("button", { name: "Увімкнути камеру" }));

    await waitFor(() => {
      expect(screen.getByText("Активна")).toBeInTheDocument();
    });
  });

  it("shows a Ukrainian error message when permission is denied", async () => {
    const deniedError = Object.assign(new Error("denied"), { name: "NotAllowedError" });
    const getUserMedia = vi.fn().mockRejectedValue(deniedError);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const user = userEvent.setup();
    render(<Camera />);

    await user.click(screen.getByRole("button", { name: "Увімкнути камеру" }));

    await waitFor(() => {
      expect(screen.getByText(/Доступ до камери відхилено/)).toBeInTheDocument();
    });
  });

  it("accepts a landmarksStatus prop (overlay canvas) without crashing, streaming or not", async () => {
    const track = { stop: vi.fn() };
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track] });
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const status = {
      type: "landmarks_status" as const,
      left_hand: false,
      right_hand: true,
      pose: true,
      face: false,
      left_hand_points: null,
      right_hand_points: [[0.4, 0.5]] as [number, number][],
      pose_points: null,
    };

    const user = userEvent.setup();
    const { container } = render(<Camera landmarksStatus={status} />);
    expect(container.querySelector("canvas")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Увімкнути камеру" }));

    await waitFor(() => {
      expect(screen.getByText("Активна")).toBeInTheDocument();
    });
  });

  it("toggles the on-video overlay off and on via its own button, independent of streaming", async () => {
    const track = { stop: vi.fn() };
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track] });
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const user = userEvent.setup();
    const { container } = render(<Camera />);

    await user.click(screen.getByRole("button", { name: "Увімкнути камеру" }));
    await waitFor(() => expect(screen.getByText("Активна")).toBeInTheDocument());

    const toggle = screen.getByRole("button", { name: "Сховати індикатори на відео" });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(container.querySelector("canvas")).toHaveClass("block");

    await user.click(toggle);

    expect(screen.getByRole("button", { name: "Показати індикатори на відео" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(container.querySelector("canvas")).toHaveClass("hidden");
  });
});
