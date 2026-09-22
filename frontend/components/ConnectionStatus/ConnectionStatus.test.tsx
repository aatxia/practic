import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectionStatus } from "./ConnectionStatus";

function stubHealthyFetch() {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: () =>
      Promise.resolve({
        status: "ok",
        app_env: "development",
        model_type: "lstm",
        features: { hands: true, pose: true, face: true },
        ml_pipeline_status: "not_implemented",
      }),
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("ConnectionStatus", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("shows a friendly Ukrainian label when the backend /health check succeeds -- no raw ML/model wording by default", async () => {
    stubHealthyFetch();

    render(<ConnectionStatus />);

    await waitFor(() => {
      expect(screen.getByText("Готово")).toBeInTheDocument();
    });
    // Technical detail is hidden until explicitly requested (see the
    // toggle test below) -- never shown by default.
    expect(screen.queryByText(/ML pipeline/)).not.toBeInTheDocument();
    expect(screen.queryByText(/model:/)).not.toBeInTheDocument();
    expect(screen.queryByText("Connected")).not.toBeInTheDocument();
  });

  it("shows a friendly Ukrainian label when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network error")));

    render(<ConnectionStatus />);

    await waitFor(() => {
      expect(screen.getByText("Немає з'єднання")).toBeInTheDocument();
    });
    expect(screen.queryByText("Disconnected")).not.toBeInTheDocument();
  });

  it("reveals model/ML-pipeline detail only after an explicit click, and hides it again on a second click", async () => {
    stubHealthyFetch();
    const user = userEvent.setup();

    render(<ConnectionStatus />);
    await waitFor(() => expect(screen.getByText("Готово")).toBeInTheDocument());

    await user.click(screen.getByText("Готово"));
    expect(screen.getByTestId("connection-status-details").textContent).toContain("ML pipeline: not_implemented");

    await user.click(screen.getByText("Готово"));
    expect(screen.queryByTestId("connection-status-details")).not.toBeInTheDocument();
  });

  it("polls /health no more often than once every 30 seconds, not every 5", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchMock = stubHealthyFetch();

    render(<ConnectionStatus />);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await act(() => vi.advanceTimersByTimeAsync(5000));
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(() => vi.advanceTimersByTimeAsync(25000));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
