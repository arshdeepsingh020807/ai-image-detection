import { METRIC_GUARDS } from "../api/guards";
import type { MetricsFiles, MetricsResponse } from "../api/types";

export type FileState<T> = { status: "missing" } | { status: "invalid" } | { status: "ok"; data: T };
export type MetricKey = keyof MetricsFiles;
export type MetricStates = { [K in MetricKey]: FileState<MetricsFiles[K]> };

export const METRIC_KEYS = Object.keys(METRIC_GUARDS) as MetricKey[];

export function readMetric<K extends MetricKey>(response: MetricsResponse, key: K): FileState<MetricsFiles[K]> {
  if (!(key in response.metrics)) return { status: "missing" };
  const value = response.metrics[key];
  return METRIC_GUARDS[key](value) ? { status: "ok", data: value } : { status: "invalid" };
}

export const readAllMetrics = (response: MetricsResponse): MetricStates =>
  Object.fromEntries(METRIC_KEYS.map((key) => [key, readMetric(response, key)])) as MetricStates;

export const fileName = (key: MetricKey): string => `${key}.json`;
