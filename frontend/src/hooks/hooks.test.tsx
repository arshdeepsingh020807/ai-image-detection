import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import { fixtures } from "../test/fixtures";
import { JPEG_BYTES, makeFile } from "../test/files";
import { useAsync } from "./useAsync";
import { detectReducer, INITIAL_STATE, type DetectState } from "./useDetector";
import { useElementWidth } from "./useElementWidth";

const source = { file: makeFile(JPEG_BYTES), url: "blob:x" };
const error = { code: "timeout", title: "t", detail: "d", retryable: true };
const result = fixtures.prediction.ai();

describe("detectReducer", () => {
  const states: Record<string, DetectState> = {
    empty: INITIAL_STATE,
    ready: { phase: "ready", source },
    uploading: { phase: "uploading", source },
    analysing: { phase: "analysing", source },
    done: { phase: "done", source, result },
    failed: { phase: "failed", source, error },
  };

  it.each([
    ["ready", { type: "start" }, "uploading"],
    ["failed", { type: "start" }, "uploading"],
    ["done", { type: "start" }, "uploading"],
    ["uploading", { type: "uploaded" }, "analysing"],
    ["uploading", { type: "succeed", result }, "done"],
    ["analysing", { type: "succeed", result }, "done"],
    ["analysing", { type: "fail", error }, "failed"],
    ["uploading", { type: "cancel" }, "ready"],
    ["done", { type: "reset" }, "empty"],
    ["ready", { type: "check" }, "checking"],
  ] as const)("%s + %o -> %s", (from, action, to) => {
    expect(detectReducer(states[from] as DetectState, action).phase).toBe(to);
  });

  it.each([
    ["empty", { type: "start" }],
    ["uploading", { type: "start" }],
    ["analysing", { type: "uploaded" }],
    ["ready", { type: "succeed", result }],
    ["done", { type: "fail", error }],
    ["ready", { type: "cancel" }],
  ] as const)("ignores %s + %o (stale or invalid)", (from, action) => {
    const state = states[from] as DetectState;
    expect(detectReducer(state, action)).toBe(state);
  });

  it("keeps the source through the analysis and records rejections", () => {
    const done = detectReducer({ phase: "analysing", source }, { type: "succeed", result });
    expect(done).toEqual({ phase: "done", source, result });
    expect(detectReducer(INITIAL_STATE, { type: "reject", error })).toEqual({ phase: "empty", rejection: error });
    expect(detectReducer(INITIAL_STATE, { type: "accept", source })).toEqual({ phase: "ready", source });
  });
});

describe("useAsync", () => {
  it("loads, reports errors and reloads", async () => {
    const load = vi.fn().mockRejectedValueOnce(new ApiError("rate_limited", "x")).mockResolvedValueOnce(42);
    const { result: hook } = renderHook(() => useAsync(load));
    expect(hook.current.status).toBe("loading");
    await waitFor(() => { expect(hook.current).toMatchObject({ status: "error", error: { code: "rate_limited" } }); });
    act(() => { hook.current.reload(); });
    await waitFor(() => { expect(hook.current).toMatchObject({ status: "ready", data: 42 }); });
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("aborts on unmount and ignores the late failure", () => {
    let signal: AbortSignal | undefined;
    const load = vi.fn((s: AbortSignal) => { signal = s; return new Promise<number>((_, reject) => { s.addEventListener("abort", () => { reject(new Error("aborted")); }); }); });
    const { unmount } = renderHook(() => useAsync(load));
    unmount();
    expect(signal?.aborted).toBe(true);
  });
});

describe("useElementWidth", () => {
  function Probe() {
    const [ref, width] = useElementWidth<HTMLDivElement>(560);
    return <div ref={ref}>{width}</div>;
  }

  it("uses the fallback without ResizeObserver", () => {
    render(<Probe />);
    expect(screen.getByText("560")).toBeInTheDocument();
  });

  it("tracks the observed width and disconnects", () => {
    let callback: ResizeObserverCallback = () => undefined;
    const disconnect = vi.fn();
    vi.stubGlobal("ResizeObserver", class {
      constructor(cb: ResizeObserverCallback) { callback = cb; }
      observe() { /* driven manually */ }
      disconnect = disconnect;
    });
    const { unmount } = render(<Probe />);
    act(() => { callback([{ contentRect: { width: 343.4 } } as ResizeObserverEntry], {} as ResizeObserver); });
    expect(screen.getByText("343")).toBeInTheDocument();
    act(() => { callback([{ contentRect: { width: 0 } } as ResizeObserverEntry], {} as ResizeObserver); });
    expect(screen.getByText("343")).toBeInTheDocument();
    unmount();
    expect(disconnect).toHaveBeenCalled();
  });
});
