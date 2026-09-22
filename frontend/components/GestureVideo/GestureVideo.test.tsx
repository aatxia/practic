import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GestureVideo } from "./GestureVideo";
import { PLAYBACK_TRANSITIONS_MS } from "./playbackSettings";

// jsdom has no real media pipeline -- HTMLMediaElement.play() throws "Not
// implemented" there. GestureVideo already swallows that rejection (real
// browsers can also reject play() outside a user gesture), but jsdom
// doesn't even return a promise by default, so stub it to one that does.
function stubVideoPlayback(): void {
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
}

function endAndAdvance(video: HTMLVideoElement, ms: number): void {
  fireEvent.ended(video);
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

describe("GestureVideo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    stubVideoPlayback();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows an idle placeholder, no video element, for an empty sequence", () => {
    render(<GestureVideo glossSequence={[]} />);

    expect(screen.getByText(/Покажи слово чи речення/)).toBeInTheDocument();
    expect(document.querySelector("video")).toBeNull();
  });

  it("plays the real video for a gloss that has one, labeled as a gesture", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);

    const video = document.querySelector("video");
    expect(video).not.toBeNull();
    expect(video?.getAttribute("src")).toBe("/videos/WE.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("1/2");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("Жест: ми");
  });

  it("advances to the next clip, after the word-to-word transition, when the current one ends", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);

    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);

    expect(video.getAttribute("src")).toBe("/videos/WATER.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("2/2");
  });

  it("does not advance before the transition delay elapses", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);

    const video = document.querySelector("video")!;
    fireEvent.ended(video);
    act(() => {
      vi.advanceTimersByTime(PLAYBACK_TRANSITIONS_MS.wordToWord - 50);
    });

    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
  });

  it("stays on the last clip instead of looping back or erroring when it ends", () => {
    render(<GestureVideo glossSequence={["WE"]} />);

    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);

    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
  });

  it("shows PAST as a real gesture with its own video, unlike the 3D avatar which drops it", () => {
    render(<GestureVideo glossSequence={["WE", "PAST"]} />);

    expect(screen.queryByText(/Немає відео/)).not.toBeInTheDocument();
    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);
    expect(video.getAttribute("src")).toBe("/videos/PAST.mp4");
  });

  it("honestly lists a gloss with no uploaded video instead of guessing or substituting one", () => {
    render(<GestureVideo glossSequence={["WE", "YOU"]} />);

    // "YOU" has no uploaded video -- only WE's clip should ever play.
    expect(screen.getByText(/Немає відео для: ти/)).toBeInTheDocument();
    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);
    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
  });

  it("plays a fingerspelled word letter by letter with real per-letter clips and progress", () => {
    render(<GestureVideo glossSequence={["FS_О", "FS_К", "FS_О"]} />);

    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/О.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("Око");
    expect(screen.getByTestId("fingerspell-letters")).toHaveTextContent("[О] К О");

    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.letterToLetter);
    expect(video.getAttribute("src")).toBe("/videos/К.mp4");
    expect(screen.getByTestId("fingerspell-letters")).toHaveTextContent("О [К] О");
  });

  it("skips a letter with no uploaded video gracefully instead of crashing, and reports it honestly", () => {
    // "Настя" -> Н-А-С-Т-Я; "А" has no uploaded clip yet.
    render(<GestureVideo glossSequence={["FS_Н", "FS_А", "FS_С", "FS_Т", "FS_Я"]} />);

    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/Н.mp4");
    expect(screen.getByTestId("partial-fingerspell-notice")).toHaveTextContent("Настя (А)");

    // Ending Н's clip should skip straight to С (the next letter with a
    // real video), never a fabricated clip for А.
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.letterToLetter);
    expect(video.getAttribute("src")).toBe("/videos/С.mp4");
  });

  it("keeps a mixed sentence as four words in the playback progress, not eight unrelated items", () => {
    render(<GestureVideo glossSequence={["I", "WANT", "WATER", "FS_Н", "FS_А", "FS_С", "FS_Т", "FS_Я"]} />);

    expect(screen.getByTestId("playback-progress")).toHaveTextContent("1/4");
  });

  it("honestly reports a fully unsupported fingerspelled word (no letter has a video) as having no video", () => {
    // A single-letter word made only of "А" (unmapped, see dactylVideos.ts)
    // has zero playable letters -- the whole word must fall into the
    // "no video for this translation" state, not render a broken player.
    render(<GestureVideo glossSequence={["FS_А"]} />);

    expect(screen.getByText(/Немає відео для жодного слова/)).toBeInTheDocument();
    expect(document.querySelector("video")).toBeNull();
  });

  it("replaying a single gesture word narrows playback to just that clip", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);

    fireEvent.click(screen.getByTitle("Повторити «вода»"));

    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/WATER.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("1/1");
  });

  it("replaying a fingerspelled word plays its whole letter sequence, not just one letter", () => {
    render(<GestureVideo glossSequence={["WE", "FS_О", "FS_К", "FS_О"]} />);

    fireEvent.click(screen.getByTitle("Повторити «Око»"));

    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/О.mp4");
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.letterToLetter);
    expect(video.getAttribute("src")).toBe("/videos/К.mp4");
  });

  it("restarts playback from the beginning", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);
    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);
    expect(video.getAttribute("src")).toBe("/videos/WATER.mp4");

    fireEvent.click(screen.getByTitle("Спочатку"));

    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
  });

  it("restarts (rewinds + replays) even when already on the first clip, not just when the index changes", () => {
    render(<GestureVideo glossSequence={["WE", "WATER"]} />);
    const video = document.querySelector("video")! as HTMLVideoElement;
    const playSpy = vi.mocked(HTMLMediaElement.prototype.play);
    video.currentTime = 5;
    playSpy.mockClear();

    fireEvent.click(screen.getByTitle("Спочатку"));

    expect(video.currentTime).toBe(0);
    expect(playSpy).toHaveBeenCalled();
  });

  it("repeats (rewinds + replays) even when already at the start of the current word", () => {
    render(<GestureVideo glossSequence={["WE"]} />);
    const video = document.querySelector("video")! as HTMLVideoElement;
    const playSpy = vi.mocked(HTMLMediaElement.prototype.play);
    video.currentTime = 3;
    playSpy.mockClear();

    fireEvent.click(screen.getByTitle("Повторити слово"));

    expect(video.currentTime).toBe(0);
    expect(playSpy).toHaveBeenCalled();
  });

  it("navigates to the next and previous word by semantic word, not by letter", () => {
    render(<GestureVideo glossSequence={["WE", "FS_О", "FS_К", "FS_О"]} />);
    const video = document.querySelector("video")!;
    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");

    fireEvent.click(screen.getByTitle("Наступне слово"));
    expect(video.getAttribute("src")).toBe("/videos/О.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("2/2");

    fireEvent.click(screen.getByTitle("Попереднє слово"));
    expect(video.getAttribute("src")).toBe("/videos/WE.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("1/2");
  });

  it("repeats the current word from its own start", () => {
    render(<GestureVideo glossSequence={["FS_О", "FS_К", "FS_О"]} />);
    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.letterToLetter);
    expect(video.getAttribute("src")).toBe("/videos/К.mp4");

    fireEvent.click(screen.getByTitle("Повторити слово"));

    expect(video.getAttribute("src")).toBe("/videos/О.mp4");
  });

  it("applies the selected playback speed to the video element", () => {
    render(<GestureVideo glossSequence={["WE"]} />);
    const video = document.querySelector("video")! as HTMLVideoElement;

    fireEvent.change(screen.getByTestId("playback-speed"), { target: { value: "1.5" } });

    expect(video.playbackRate).toBe(1.5);
  });

  it("honestly reports a fingerspelled-only translation with real per-letter videos as playable, unlike before", () => {
    render(<GestureVideo glossSequence={["FS_О", "FS_К", "FS_О"]} />);

    expect(screen.queryByText(/Немає відео для жодного слова/)).not.toBeInTheDocument();
    expect(document.querySelector("video")).not.toBeNull();
  });

  it("resets to the full sequence from the start on a new translation", () => {
    const { rerender } = render(<GestureVideo glossSequence={["WE", "WATER"]} />);
    const video = document.querySelector("video")!;
    endAndAdvance(video, PLAYBACK_TRANSITIONS_MS.wordToWord);
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("2/2");

    rerender(<GestureVideo glossSequence={["HAVE"]} />);

    expect(video.getAttribute("src")).toBe("/videos/HAVE.mp4");
    expect(screen.getByTestId("playback-progress")).toHaveTextContent("1/1");
  });

  it("preserves original casing/word text for copy/history purposes via originalWord, not just uppercase letters", () => {
    render(<GestureVideo glossSequence={["FS_Н", "FS_А", "FS_С", "FS_Т", "FS_Я"]} />);

    expect(screen.getByTestId("playback-progress")).toHaveTextContent("Настя");
  });
});
