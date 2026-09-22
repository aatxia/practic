import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCamera } from "./useCamera";

function makeFakeStream(): MediaStream {
  const track = { stop: vi.fn(), kind: "video" } as unknown as MediaStreamTrack;
  return {
    getTracks: () => [track],
  } as unknown as MediaStream;
}

describe("useCamera", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("starts streaming when permission is granted", async () => {
    const fakeStream = makeFakeStream();
    const getUserMedia = vi.fn().mockResolvedValue(fakeStream);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const { result } = renderHook(() => useCamera());

    expect(result.current.status).toBe("idle");

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.status).toBe("streaming");
    });
    expect(getUserMedia).toHaveBeenCalledWith(
      expect.objectContaining({ audio: false, video: expect.any(Object) }),
    );
  });

  it("reports permission_denied when getUserMedia rejects with NotAllowedError", async () => {
    const deniedError = Object.assign(new Error("denied"), { name: "NotAllowedError" });
    const getUserMedia = vi.fn().mockRejectedValue(deniedError);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const { result } = renderHook(() => useCamera());

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });
    expect(result.current.error?.reason).toBe("permission_denied");
  });

  it("reports unsupported when the browser has no mediaDevices API", async () => {
    vi.stubGlobal("navigator", {});

    const { result } = renderHook(() => useCamera());

    await act(async () => {
      await result.current.start();
    });

    await waitFor(() => {
      expect(result.current.status).toBe("error");
    });
    expect(result.current.error?.reason).toBe("unsupported");
  });

  it("stops all tracks and resets status when stop() is called", async () => {
    const fakeStream = makeFakeStream();
    const getUserMedia = vi.fn().mockResolvedValue(fakeStream);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const { result } = renderHook(() => useCamera());

    await act(async () => {
      await result.current.start();
    });
    await waitFor(() => expect(result.current.status).toBe("streaming"));

    act(() => {
      result.current.stop();
    });

    expect(result.current.status).toBe("stopped");
    const [track] = fakeStream.getTracks();
    expect(track).toBeDefined();
    expect(track?.stop).toHaveBeenCalled();
  });

  it("clamps configured FPS into the 10-15 range", () => {
    const { result } = renderHook(() => useCamera({ config: { fps: 60 } }));
    expect(result.current.config.fps).toBe(15);
  });

  it("captures frames at the camera's actual native resolution, never stretched to the requested ideal", async () => {
    // jsdom has no real canvas 2D context ("without installing the canvas
    // npm package") -- stub just enough of it (drawImage) to let the real
    // capture code run past canvas.getContext("2d"), the same no-op-canvas
    // approach as this suite's other canvas-touching tests.
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage,
    } as unknown as CanvasRenderingContext2D);

    const fakeStream = makeFakeStream();
    const getUserMedia = vi.fn().mockResolvedValue(fakeStream);
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });

    const onFrame = vi.fn();
    const { result } = renderHook(() =>
      useCamera({ onFrame, config: { width: 640, height: 480, fps: 15 } }),
    );

    // A real 16:9 webcam asked for the 4:3 "ideal" 640x480 hint can grant a
    // different native resolution -- forcing the captured frame to 640x480
    // anyway would stretch it, distorting hand/finger proportions relative
    // to training data that was never force-resized this way.
    const video = document.createElement("video");
    Object.defineProperty(video, "videoWidth", { value: 1280, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 720, configurable: true });
    Object.defineProperty(video, "readyState", { value: 4, configurable: true });
    result.current.videoRef.current = video;

    await act(async () => {
      await result.current.start();
    });
    await waitFor(() => expect(result.current.status).toBe("streaming"));

    await waitFor(() => expect(onFrame).toHaveBeenCalled(), { timeout: 2000 });

    const frame = onFrame.mock.calls[0]?.[0];
    expect(frame.width).toBe(1280);
    expect(frame.height).toBe(720);
  });
});
