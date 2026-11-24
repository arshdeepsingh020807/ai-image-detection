export type Label = "real" | "ai_generated" | "inconclusive";
export type ImageFormat = "JPEG" | "PNG" | "WEBP";

export interface Band {
  low: number;
  high: number;
}

export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Explanation {
  heatmap_png_base64: string;
  region: Region;
  method: string;
  note: string;
}

export interface ModelRef {
  architecture: string;
  version: string;
  trained: boolean;
}

export interface Prediction {
  label: Label;
  probabilities: { real: number; ai_generated: number };
  confidence: number;
  band: Band;
  image: { width: number; height: number; format: ImageFormat; size_bytes: number };
  explanation: Explanation | null;
  timing_ms: { decode: number; preprocess: number; inference: number; explain: number; total: number };
  model: ModelRef;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version?: string | null;
  uptime_seconds?: number;
}

export interface ModelCard extends ModelRef {
  input: { short_side: number; crop: number; mean: number[]; std: number[] };
  temperature: number;
  band: Band;
  onnx_size_mb: number;
  [metadata: string]: unknown;
}

export type Matrix2x2 = [[number, number], [number, number]];

export interface ClassificationMetrics {
  n: number;
  threshold: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  specificity: number | null;
  false_positive_rate: number | null;
  false_negative_rate: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  log_loss: number | null;
  brier: number | null;
  ece: number | null;
  confusion_matrix: Matrix2x2;
}

export interface BandMetrics {
  coverage: number;
  accuracy_on_decided: number;
  false_positive_rate_on_real: number;
  error_rate: number;
  inconclusive: number;
}

export interface Summary {
  model: string;
  created: string;
  seed: number;
  validation: ClassificationMetrics;
  test: ClassificationMetrics;
  test_with_band: BandMetrics;
  temperature: number;
  band: Band;
}

export interface HistoryEpoch {
  epoch: number;
  phase: "head" | "finetune";
  train_loss: number;
  val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
  lr: number;
}

export interface History {
  epochs: HistoryEpoch[];
}

export interface Confusion {
  labels: string[];
  matrix: Matrix2x2;
}

export interface Roc {
  fpr: number[];
  tpr: number[];
  auc: number;
}

export interface PrCurve {
  precision: number[];
  recall: number[];
  average_precision: number;
}

export interface ConfidenceHistogram {
  bins: number[];
  real: number[];
  ai_generated: number[];
}

export interface ExternalSet {
  name: string;
  protocol: string;
  n: number;
  metrics: ClassificationMetrics;
}

export interface External {
  sets: ExternalSet[];
  notes: string[];
}

export interface RobustnessCondition {
  name: string;
  description: string;
  accuracy: number;
  roc_auc: number | null;
  false_positive_rate: number;
}

export interface Robustness {
  conditions: RobustnessCondition[];
}

export interface ClassCounts {
  real: number;
  ai_generated: number;
}

export interface ShortcutProbe {
  auc: number;
  accuracy: number;
  n: number;
  features: string[];
}

export interface DatasetReport {
  original_images: number;
  cleaned_images: number;
  removed_images: number;
  removed_by_reason: Record<string, number>;
  exact_duplicates_removed: number;
  near_duplicate_clusters: number;
  images_in_near_duplicate_clusters: number;
  class_distribution: ClassCounts;
  by_source: Record<string, ClassCounts & { role: string; domain: string }>;
  by_generator: Record<string, number>;
  formats: Record<string, Record<string, number>>;
  resolution_top: Record<string, number>;
  splits: Record<string, ClassCounts>;
  leakage_check: Record<string, number>;
  shortcut_probes: Record<string, ShortcutProbe>;
}

export interface Latency {
  cpu: { p50_ms: number; p95_ms: number; rss_mb: number };
  onnx_size_mb: number;
  runs: number;
  machine: string;
}

export interface ComparedModel {
  name: string;
  parameters_millions: number;
  onnx_size_mb: number;
  validation_auc: number;
  external_auc: number;
  false_positive_rate: number;
  latency_ms: number;
}

export interface ModelComparison {
  models: ComparedModel[];
}

export interface MetricsFiles {
  summary: Summary;
  history: History;
  confusion: Confusion;
  roc: Roc;
  pr: PrCurve;
  confidence: ConfidenceHistogram;
  external: External;
  robustness: Robustness;
  dataset_report: DatasetReport;
  latency: Latency;
  model_comparison: ModelComparison;
}

export interface MetricsResponse {
  available: string[];
  metrics: Record<string, unknown>;
}
