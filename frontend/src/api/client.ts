import { isApiErrorBody, isHealth, isMetricsResponse, isModelCard, isPrediction, type Guard } from "./guards";
import type { Health, MetricsResponse, ModelCard, Prediction } from "./types";

export const API_BASE = "/api";
export const READ_TIMEOUT_MS = 15_000;
export const PREDICT_TIMEOUT_MS = 60_000;

const TIMEOUT = "timeout";

export class ApiError extends Error {
  constructor(readonly code: string, message: string, readonly status = 0) {
    super(message);
    this.name = "ApiError";
  }
}

const parseJson = (text: string): unknown => {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
};

export function parseResponse<T>(status: number, text: string, guard: Guard<T>, acceptStatus = (s: number) => s >= 200 && s < 300): T {
  const body = parseJson(text);
  if (acceptStatus(status)) {
    if (guard(body)) return body;
    throw new ApiError("bad_response", "The server response did not match the expected format.", status);
  }
  if (isApiErrorBody(body)) throw new ApiError(body.error.code, body.error.message, status);
  throw new ApiError("bad_response", `The server answered with HTTP ${status}.`, status);
}

interface RequestOptions {
  signal?: AbortSignal | undefined;
  timeoutMs?: number;
}

async function getJson<T>(path: string, guard: Guard<T>, options: RequestOptions = {}, acceptStatus?: (status: number) => boolean): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => { controller.abort(TIMEOUT); }, options.timeoutMs ?? READ_TIMEOUT_MS);
  const forwardAbort = () => { controller.abort(); };
  options.signal?.addEventListener("abort", forwardAbort, { once: true });
  try {
    const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" }, signal: controller.signal });
    return parseResponse(response.status, await response.text(), guard, acceptStatus);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (controller.signal.reason === TIMEOUT) throw new ApiError("timeout", "The request timed out.");
    if (controller.signal.aborted) throw new ApiError("aborted", "The request was cancelled.");
    throw new ApiError("network_error", "The server could not be reached.");
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", forwardAbort);
  }
}

export const getHealth = (options?: RequestOptions): Promise<Health> =>
  getJson("/health", isHealth, options, (status) => status === 200 || status === 503);
export const getModel = (options?: RequestOptions): Promise<ModelCard> => getJson("/model", isModelCard, options);
export const getMetrics = (options?: RequestOptions): Promise<MetricsResponse> => getJson("/metrics", isMetricsResponse, options);

export interface PredictOptions extends RequestOptions {
  explain: boolean;
  onUploaded?: () => void;
}

export async function predict(file: File, { explain, signal, timeoutMs = PREDICT_TIMEOUT_MS, onUploaded }: PredictOptions): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  form.append("explain", String(explain));
  const { status, text } = await upload(`${API_BASE}/predict`, form, timeoutMs, signal, onUploaded);
  return parseResponse(status, text, isPrediction);
}

function upload(url: string, body: FormData, timeoutMs: number, signal?: AbortSignal, onUploaded?: () => void): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new ApiError("aborted", "The request was cancelled."));
      return;
    }
    const xhr = new XMLHttpRequest();
    const abort = () => { xhr.abort(); };
    const fail = (code: string, message: string) => () => {
      signal?.removeEventListener("abort", abort);
      reject(new ApiError(code, message));
    };
    xhr.open("POST", url);
    xhr.timeout = timeoutMs;
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onload = () => onUploaded?.();
    xhr.onload = () => {
      signal?.removeEventListener("abort", abort);
      resolve({ status: xhr.status, text: xhr.responseText });
    };
    xhr.onerror = fail("network_error", "The server could not be reached.");
    xhr.ontimeout = fail("timeout", "The analysis timed out.");
    xhr.onabort = fail("aborted", "The request was cancelled.");
    signal?.addEventListener("abort", abort, { once: true });
    xhr.send(body);
  });
}
