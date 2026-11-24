import type {
  ApiErrorBody, BandMetrics, ClassificationMetrics, ConfidenceHistogram, Confusion, DatasetReport, External,
  Health, History, Latency, Matrix2x2, MetricsFiles, MetricsResponse, ModelCard, ModelComparison, PrCurve,
  Prediction, Robustness, Roc, Summary,
} from "./types";

export type Guard<T> = (value: unknown) => value is T;
type Shape<T> = { [K in keyof T]-?: Guard<T[K]> };

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
export const isString = (value: unknown): value is string => typeof value === "string";
export const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";
export const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
export const isCount = (value: unknown): value is number => Number.isInteger(value) && (value as number) >= 0;
export const isUnit = (value: unknown): value is number => isNumber(value) && value >= 0 && value <= 1;

export const oneOf = <T extends string>(...options: T[]): Guard<T> =>
  (value): value is T => options.includes(value as T);
export const nullable = <T>(guard: Guard<T>): Guard<T | null> => (value): value is T | null => value === null || guard(value);
export const optional = <T>(guard: Guard<T>): Guard<T | undefined> => (value): value is T | undefined => value === undefined || guard(value);
export const arrayOf = <T>(guard: Guard<T>): Guard<T[]> => (value): value is T[] => Array.isArray(value) && value.every(guard);
export const recordOf = <T>(guard: Guard<T>): Guard<Record<string, T>> =>
  (value): value is Record<string, T> => isRecord(value) && Object.values(value).every(guard);
export const shape = <T>(spec: Shape<T>): Guard<T> =>
  (value): value is T => isRecord(value) && Object.entries<Guard<unknown>>(spec).every(([key, guard]) => guard(value[key]));
export const refine = <T>(guard: Guard<T>, check: (value: T) => boolean): Guard<T> =>
  (value): value is T => guard(value) && check(value);

const isMatrix2x2: Guard<Matrix2x2> = (value): value is Matrix2x2 =>
  Array.isArray(value) && value.length === 2 && value.every((row) => Array.isArray(row) && row.length === 2 && row.every(isCount));
const isBand = refine(shape({ low: isUnit, high: isUnit }), (band) => band.low <= band.high);
const isRegion = shape({ x: isUnit, y: isUnit, width: isUnit, height: isUnit });
const isModelRef = shape({ architecture: isString, version: isString, trained: isBoolean });
const sameLength = (...arrays: unknown[][]) => arrays.every((array) => array.length === arrays[0]?.length);

export const isApiErrorBody: Guard<ApiErrorBody> = shape({ error: shape({ code: isString, message: isString }) });

export const isPrediction: Guard<Prediction> = refine(
  shape<Prediction>({
    label: oneOf("real", "ai_generated", "inconclusive"),
    probabilities: shape({ real: isUnit, ai_generated: isUnit }),
    confidence: isUnit,
    band: isBand,
    image: shape({ width: isCount, height: isCount, format: oneOf("JPEG", "PNG", "WEBP"), size_bytes: isCount }),
    explanation: nullable(shape({ heatmap_png_base64: isString, region: isRegion, method: isString, note: isString })),
    timing_ms: shape({ decode: isNumber, preprocess: isNumber, inference: isNumber, explain: isNumber, total: isNumber }),
    model: isModelRef,
  }),
  (prediction) => Math.abs(prediction.probabilities.real + prediction.probabilities.ai_generated - 1) < 0.01
    && !(prediction.label === "inconclusive" && prediction.band.low === prediction.band.high),
);

export const isHealth: Guard<Health> = shape<Health>({
  status: oneOf("ok", "degraded"),
  model_loaded: isBoolean,
  model_version: optional(nullable(isString)),
  uptime_seconds: optional(isNumber),
});

export const isModelCard = shape({
  architecture: isString,
  version: isString,
  trained: isBoolean,
  input: shape({ short_side: isCount, crop: isCount, mean: arrayOf(isNumber), std: arrayOf(isNumber) }),
  temperature: isNumber,
  band: isBand,
  onnx_size_mb: isNumber,
}) as Guard<ModelCard>;

export const isMetricsResponse: Guard<MetricsResponse> = shape({ available: arrayOf(isString), metrics: isRecord });

const metric = nullable(isNumber);
export const isClassificationMetrics: Guard<ClassificationMetrics> = shape<ClassificationMetrics>({
  n: isCount, threshold: metric, accuracy: metric, precision: metric, recall: metric, f1: metric,
  specificity: metric, false_positive_rate: metric, false_negative_rate: metric, roc_auc: metric, pr_auc: metric,
  log_loss: metric, brier: metric, ece: metric, confusion_matrix: isMatrix2x2,
});

const isBandMetrics: Guard<BandMetrics> = shape<BandMetrics>({
  coverage: isUnit, accuracy_on_decided: isUnit, false_positive_rate_on_real: isUnit, error_rate: isUnit, inconclusive: isCount,
});

const isSummary: Guard<Summary> = shape<Summary>({
  model: isString, created: isString, seed: isNumber, validation: isClassificationMetrics, test: isClassificationMetrics,
  test_with_band: isBandMetrics, temperature: isNumber, band: isBand,
});

const isHistory: Guard<History> = shape<History>({
  epochs: arrayOf(shape({
    epoch: isCount, phase: oneOf("head", "finetune"), train_loss: isNumber, val_loss: isNumber,
    train_accuracy: isUnit, val_accuracy: isUnit, lr: isNumber,
  })),
});

const isConfusion: Guard<Confusion> = shape<Confusion>({ labels: arrayOf(isString), matrix: isMatrix2x2 });

const isRoc: Guard<Roc> = refine(
  shape<Roc>({ fpr: arrayOf(isUnit), tpr: arrayOf(isUnit), auc: isUnit }),
  (roc) => roc.fpr.length > 1 && sameLength(roc.fpr, roc.tpr),
);

const isPr: Guard<PrCurve> = refine(
  shape<PrCurve>({ precision: arrayOf(isUnit), recall: arrayOf(isUnit), average_precision: isUnit }),
  (pr) => pr.recall.length > 1 && sameLength(pr.precision, pr.recall),
);

const isConfidence: Guard<ConfidenceHistogram> = refine(
  shape<ConfidenceHistogram>({ bins: arrayOf(isUnit), real: arrayOf(isCount), ai_generated: arrayOf(isCount) }),
  (hist) => hist.real.length > 0 && sameLength(hist.real, hist.ai_generated) && hist.bins.length === hist.real.length + 1
    && hist.bins.every((edge, i) => i === 0 || edge > (hist.bins[i - 1] ?? 0)),
);

const isExternal: Guard<External> = shape<External>({
  sets: arrayOf(shape({ name: isString, protocol: isString, n: isCount, metrics: isClassificationMetrics })),
  notes: arrayOf(isString),
});

const isRobustness: Guard<Robustness> = shape<Robustness>({
  conditions: arrayOf(shape({
    name: isString, description: isString, accuracy: isUnit, roc_auc: nullable(isUnit), false_positive_rate: isUnit,
  })),
});

const isClassCounts = shape({ real: isCount, ai_generated: isCount });

const isDatasetReport: Guard<DatasetReport> = shape<DatasetReport>({
  original_images: isCount,
  cleaned_images: isCount,
  removed_images: isCount,
  removed_by_reason: recordOf(isCount),
  exact_duplicates_removed: isCount,
  near_duplicate_clusters: isCount,
  images_in_near_duplicate_clusters: isCount,
  class_distribution: isClassCounts,
  by_source: recordOf(shape({ real: isCount, ai_generated: isCount, role: isString, domain: isString })),
  by_generator: recordOf(isCount),
  formats: recordOf(recordOf(isCount)),
  resolution_top: recordOf(isCount),
  splits: recordOf(isClassCounts),
  leakage_check: recordOf(isCount),
  shortcut_probes: recordOf(shape({ auc: isUnit, accuracy: isUnit, n: isCount, features: arrayOf(isString) })),
});

const isLatency: Guard<Latency> = shape<Latency>({
  cpu: shape({ p50_ms: isNumber, p95_ms: isNumber, rss_mb: isNumber }), onnx_size_mb: isNumber, runs: isCount, machine: isString,
});

const isModelComparison: Guard<ModelComparison> = shape<ModelComparison>({
  models: arrayOf(shape({
    name: isString, parameters_millions: isNumber, onnx_size_mb: isNumber, validation_auc: isUnit,
    external_auc: isUnit, false_positive_rate: isUnit, latency_ms: isNumber,
  })),
});

export const METRIC_GUARDS: { [K in keyof MetricsFiles]: Guard<MetricsFiles[K]> } = {
  summary: isSummary,
  history: isHistory,
  confusion: isConfusion,
  roc: isRoc,
  pr: isPr,
  confidence: isConfidence,
  external: isExternal,
  robustness: isRobustness,
  dataset_report: isDatasetReport,
  latency: isLatency,
  model_comparison: isModelComparison,
};
