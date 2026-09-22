import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useWebSocket } from "./useWebSocket";

class MockWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  sentMessages: string[] = [];

  constructor(public url: string) {
    instances.push(this);
  }

  send(data: string): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }

  // Test helpers, not part of the real WebSocket API
  simulateOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: unknown): void {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  }
}

let instances: MockWebSocket[] = [];

describe("useWebSocket", () => {
  beforeEach(() => {
    instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("transitions to 'open' once the socket connects", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => {
      result.current.connect();
    });
    expect(result.current.status).toBe("connecting");

    act(() => {
      instances[0]?.simulateOpen();
    });

    await waitFor(() => {
      expect(result.current.status).toBe("open");
    });
  });

  it("stores the last valid server message", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    act(() => {
      instances[0]?.simulateMessage({
        type: "error",
        message: "Sign-recognition ML pipeline is not implemented yet",
      });
    });

    await waitFor(() => {
      expect(result.current.lastMessage?.type).toBe("error");
    });
  });

  it("keeps landmarksStatus after a later, different message arrives", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    act(() => {
      instances[0]?.simulateMessage({
        type: "landmarks_status",
        left_hand: true,
        right_hand: false,
        pose: true,
        face: false,
      });
    });
    await waitFor(() => {
      expect(result.current.landmarksStatus?.left_hand).toBe(true);
    });

    // The very next message (real protocol: a prediction/error always
    // follows landmarks_status for the same frame) overwrites lastMessage
    // but must NOT erase the sticky landmarksStatus.
    act(() => {
      instances[0]?.simulateMessage({
        type: "error",
        message: "Buffering: 1/32 frames collected before the first prediction.",
      });
    });
    await waitFor(() => {
      expect(result.current.lastMessage?.type).toBe("error");
    });
    expect(result.current.landmarksStatus?.left_hand).toBe(true);
    expect(result.current.landmarksStatus?.right_hand).toBe(false);
  });

  it("ignores malformed (non-JSON) server messages without crashing", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    act(() => {
      instances[0]?.onmessage?.({ data: "not json" } as MessageEvent<string>);
    });

    expect(result.current.lastMessage).toBeNull();
  });

  it("only sends a frame once the socket is open", async () => {
    const { result } = renderHook(() => useWebSocket());
    const frame = { dataUrl: "data:image/jpeg;base64,AAA", timestamp: 123, width: 640, height: 480 };

    act(() => result.current.connect());
    result.current.sendFrame(frame); // socket still CONNECTING -- must be a no-op
    expect(instances[0]?.sentMessages).toHaveLength(0);

    act(() => instances[0]?.simulateOpen());
    act(() => result.current.sendFrame(frame));

    expect(instances[0]?.sentMessages).toHaveLength(1);
    const sent = JSON.parse(instances[0]!.sentMessages[0]!);
    expect(sent).toEqual({ type: "frame", timestamp: 123, data: frame.dataUrl });
  });

  it("drops a new frame while the previous one's response hasn't arrived yet", async () => {
    const { result } = renderHook(() => useWebSocket());
    const frame = { dataUrl: "data:image/jpeg;base64,AAA", timestamp: 1, width: 640, height: 480 };

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    act(() => result.current.sendFrame(frame));
    expect(instances[0]?.sentMessages).toHaveLength(1);

    // Backend hasn't replied to the first frame yet -- a second capture-loop
    // tick must be dropped, not queued (that's what causes unbounded lag on
    // a backend slower than the capture rate).
    act(() => result.current.sendFrame({ ...frame, timestamp: 2 }));
    expect(instances[0]?.sentMessages).toHaveLength(1);

    // landmarks_status is the interim message for a frame -- still waiting.
    act(() => {
      instances[0]?.simulateMessage({
        type: "landmarks_status",
        left_hand: false,
        right_hand: false,
        pose: false,
        face: false,
        left_hand_points: null,
        right_hand_points: null,
        pose_points: null,
      });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 3 }));
    expect(instances[0]?.sentMessages).toHaveLength(1);

    // The terminal message for that frame arrives -- now the next frame may send.
    act(() => {
      instances[0]?.simulateMessage({ type: "error", message: "Buffering: 1/32" });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 4 }));
    expect(instances[0]?.sentMessages).toHaveLength(2);
  });

  it("keeps letterMessage sticky the same way as landmarksStatus", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    act(() => {
      instances[0]?.simulateMessage({
        type: "letter_prediction",
        letter: "А",
        confidence: 0.7,
        is_final: false,
      });
    });
    await waitFor(() => {
      expect(result.current.letterMessage?.letter).toBe("А");
    });
    expect(result.current.letterMessage?.is_final).toBe(false);

    // The real per-frame terminal message (word-level path) follows --
    // must not erase the sticky letterMessage.
    act(() => {
      instances[0]?.simulateMessage({ type: "error", message: "Buffering: 1/32" });
    });
    await waitFor(() => {
      expect(result.current.lastMessage?.type).toBe("error");
    });
    expect(result.current.letterMessage?.letter).toBe("А");

    act(() => {
      instances[0]?.simulateMessage({
        type: "letter_confirmed",
        letter: "Б",
        confidence: 0.9,
        is_final: true,
      });
    });
    await waitFor(() => {
      expect(result.current.letterMessage?.letter).toBe("Б");
    });
    expect(result.current.letterMessage?.is_final).toBe(true);
  });

  it("exposes boundary events and updates on each new one, even with the same kind", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());

    expect(result.current.boundaryMessage).toBeNull();

    act(() => {
      instances[0]?.simulateMessage({ type: "boundary", kind: "token" });
    });
    await waitFor(() => {
      expect(result.current.boundaryMessage?.kind).toBe("token");
    });
    const firstBoundary = result.current.boundaryMessage;

    act(() => {
      instances[0]?.simulateMessage({ type: "boundary", kind: "token" });
    });
    await waitFor(() => {
      // A NEW message object even though `kind` is unchanged -- a
      // consumer comparing by reference (see useUtterance.ts) must be
      // able to tell this apart from the first "token" boundary.
      expect(result.current.boundaryMessage).not.toBe(firstBoundary);
    });
  });

  it("does not release the next frame on a letter message -- only the true terminal message counts", async () => {
    const { result } = renderHook(() => useWebSocket());
    const frame = { dataUrl: "data:image/jpeg;base64,AAA", timestamp: 1, width: 640, height: 480 };

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());
    act(() => result.current.sendFrame(frame));
    expect(instances[0]?.sentMessages).toHaveLength(1);

    act(() => {
      instances[0]?.simulateMessage({
        type: "letter_prediction",
        letter: "А",
        confidence: 0.7,
        is_final: false,
      });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 2 }));
    expect(instances[0]?.sentMessages).toHaveLength(1); // still dropped -- word-level terminal message hasn't arrived

    act(() => {
      instances[0]?.simulateMessage({ type: "error", message: "Buffering: 1/32" });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 3 }));
    expect(instances[0]?.sentMessages).toHaveLength(2);
  });

  it("does not release the next frame on a boundary event either -- the backend sends it as an EXTRA message, not in place of the terminal one", async () => {
    const { result } = renderHook(() => useWebSocket());
    const frame = { dataUrl: "data:image/jpeg;base64,AAA", timestamp: 1, width: 640, height: 480 };

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());
    act(() => result.current.sendFrame(frame));
    expect(instances[0]?.sentMessages).toHaveLength(1);

    act(() => {
      instances[0]?.simulateMessage({ type: "boundary", kind: "token" });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 2 }));
    expect(instances[0]?.sentMessages).toHaveLength(1); // still dropped -- terminal message hasn't arrived yet

    act(() => {
      instances[0]?.simulateMessage({ type: "error", message: "Buffering: 1/32" });
    });
    act(() => result.current.sendFrame({ ...frame, timestamp: 3 }));
    expect(instances[0]?.sentMessages).toHaveLength(2);
  });

  it("transitions to 'closed' when disconnect() is called", async () => {
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    act(() => instances[0]?.simulateOpen());
    act(() => result.current.disconnect());

    expect(result.current.status).toBe("closed");
  });

  it("ignores a belated close event from a stale (already-replaced) socket", async () => {
    // Simulates React StrictMode's dev-only mount -> cleanup -> mount cycle:
    // the first socket is closed, a second one opens, and then the first
    // socket's close event finally fires. The stale event must not override
    // the current (open) status.
    const { result } = renderHook(() => useWebSocket());

    act(() => result.current.connect());
    const firstSocket = instances[0]!;
    act(() => result.current.disconnect());

    act(() => result.current.connect());
    const secondSocket = instances[1]!;
    act(() => secondSocket.simulateOpen());
    expect(result.current.status).toBe("open");

    // Belated close event from the first (now-replaced) socket arrives late.
    act(() => firstSocket.onclose?.());

    expect(result.current.status).toBe("open");
  });
});
