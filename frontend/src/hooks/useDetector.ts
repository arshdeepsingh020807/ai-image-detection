import { useCallback, useEffect, useReducer, useRef } from "react";
import { ApiError, predict } from "../api/client";
import { toFriendlyError, type FriendlyError } from "../api/errors";
import type { Prediction } from "../api/types";
import { validateImageFile } from "../lib/fileValidation";

export interface Source {
  file: File;
  url: string;
}

export type DetectState =
  | { phase: "empty"; rejection: FriendlyError | null }
  | { phase: "checking" }
  | { phase: "ready"; source: Source }
  | { phase: "uploading"; source: Source }
  | { phase: "analysing"; source: Source }
  | { phase: "done"; source: Source; result: Prediction }
  | { phase: "failed"; source: Source; error: FriendlyError };

export type DetectAction =
  | { type: "check" }
  | { type: "accept"; source: Source }
  | { type: "reject"; error: FriendlyError }
  | { type: "start" }
  | { type: "uploaded" }
  | { type: "succeed"; result: Prediction }
  | { type: "fail"; error: FriendlyError }
  | { type: "cancel" }
  | { type: "reset" };

export const INITIAL_STATE: DetectState = { phase: "empty", rejection: null };

const isBusy = (state: DetectState): state is Extract<DetectState, { phase: "uploading" | "analysing" }> =>
  state.phase === "uploading" || state.phase === "analysing";

export function detectReducer(state: DetectState, action: DetectAction): DetectState {
  switch (action.type) {
    case "check":
      return { phase: "checking" };
    case "accept":
      return { phase: "ready", source: action.source };
    case "reject":
      return { phase: "empty", rejection: action.error };
    case "start":
      return "source" in state && !isBusy(state) ? { phase: "uploading", source: state.source } : state;
    case "uploaded":
      return state.phase === "uploading" ? { phase: "analysing", source: state.source } : state;
    case "succeed":
      return isBusy(state) ? { phase: "done", source: state.source, result: action.result } : state;
    case "fail":
      return isBusy(state) ? { phase: "failed", source: state.source, error: action.error } : state;
    case "cancel":
      return isBusy(state) ? { phase: "ready", source: state.source } : state;
    case "reset":
      return INITIAL_STATE;
  }
}

export function useDetector() {
  const [state, dispatch] = useReducer(detectReducer, INITIAL_STATE);
  const request = useRef<AbortController | null>(null);
  const selection = useRef(0);
  const previewUrl = useRef<string | null>(null);

  const cancel = useCallback(() => {
    request.current?.abort();
    request.current = null;
  }, []);

  const replacePreview = useCallback((file: File | null): string | null => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = file ? URL.createObjectURL(file) : null;
    return previewUrl.current;
  }, []);

  useEffect(() => () => {
    cancel();
    replacePreview(null);
  }, [cancel, replacePreview]);

  const clear = useCallback((action: DetectAction) => {
    cancel();
    selection.current++;
    replacePreview(null);
    dispatch(action);
  }, [cancel, replacePreview]);

  const select = useCallback(async (file: File) => {
    clear({ type: "check" });
    const id = selection.current;
    const result = await validateImageFile(file);
    if (id !== selection.current) return;
    if (!result.ok) {
      dispatch({ type: "reject", error: result.error });
      return;
    }
    dispatch({ type: "accept", source: { file, url: replacePreview(file) ?? "" } });
  }, [clear, replacePreview]);

  const analyse = useCallback(async (file: File, explain: boolean) => {
    cancel();
    const controller = new AbortController();
    request.current = controller;
    dispatch({ type: "start" });
    try {
      const result = await predict(file, { explain, signal: controller.signal, onUploaded: () => { dispatch({ type: "uploaded" }); } });
      dispatch({ type: "succeed", result });
    } catch (error) {
      if (error instanceof ApiError && error.code === "aborted") return;
      dispatch({ type: "fail", error: toFriendlyError(error) });
    } finally {
      if (request.current === controller) request.current = null;
    }
  }, [cancel]);

  const stop = useCallback(() => {
    cancel();
    dispatch({ type: "cancel" });
  }, [cancel]);

  const reject = useCallback((error: FriendlyError) => { clear({ type: "reject", error }); }, [clear]);
  const reset = useCallback(() => { clear({ type: "reset" }); }, [clear]);

  return { state, select, analyse, stop, reject, reset };
}
