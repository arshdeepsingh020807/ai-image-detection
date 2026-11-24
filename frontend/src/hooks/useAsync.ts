import { useCallback, useEffect, useState } from "react";
import { toFriendlyError, type FriendlyError } from "../api/errors";

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: FriendlyError }
  | { status: "ready"; data: T };

export function useAsync<T>(load: (signal: AbortSignal) => Promise<T>): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (data) => { setState({ status: "ready", data }); },
      (error: unknown) => { if (!controller.signal.aborted) setState({ status: "error", error: toFriendlyError(error) }); },
    );
    return () => { controller.abort(); };
  }, [load, attempt]);

  const reload = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((n) => n + 1);
  }, []);

  return { ...state, reload };
}
