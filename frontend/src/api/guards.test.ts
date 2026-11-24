import { describe, expect, it } from "vitest";
import { fixtures } from "../test/fixtures";
import {
  arrayOf, isApiErrorBody, isClassificationMetrics, isCount, isHealth, isMetricsResponse, isModelCard, isNumber, isPrediction,
  isUnit, METRIC_GUARDS, nullable, oneOf, optional, recordOf, refine, shape,
} from "./guards";
import type { MetricsFiles } from "./types";

type Mutation = [name: string, mutate: (value: Record<string, any>) => void];

const expectRejects = (guard: (value: unknown) => boolean, make: () => Record<string, any>, mutations: Mutation[]) => {
  it.each(mutations)("rejects: %s", (_, mutate) => {
    const value = make();
    mutate(value);
    expect(guard(value)).toBe(false);
  });
};

describe("primitive guards", () => {
  it("accept only finite numbers", () => {
    expect([1, 0, -2.5].every(isNumber)).toBe(true);
    expect([NaN, Infinity, "1", null].some(isNumber)).toBe(false);
  });

  it("bound probabilities to [0, 1] and counts to non-negative integers", () => {
    expect([0, 0.5, 1].every(isUnit)).toBe(true);
    expect([-0.01, 1.01].some(isUnit)).toBe(false);
    expect([0, 3].every(isCount)).toBe(true);
    expect([-1, 1.5, "2"].some(isCount)).toBe(false);
  });

  it("compose", () => {
    const guard = shape({ kind: oneOf("a", "b"), list: arrayOf(isNumber), maybe: nullable(isNumber), opt: optional(isNumber), map: recordOf(isCount) });
    expect(guard({ kind: "a", list: [1], maybe: null, map: { x: 1 } })).toBe(true);
    expect(guard({ kind: "c", list: [1], maybe: null, map: {} })).toBe(false);
    expect(guard({ kind: "a", list: [1], maybe: null, opt: "x", map: {} })).toBe(false);
    expect(guard([])).toBe(false);
    expect(refine(isNumber, (n) => n > 2)(3)).toBe(true);
    expect(refine(isNumber, (n) => n > 2)(1)).toBe(false);
  });
});

describe("POST /api/predict contract", () => {
  it.each(Object.entries(fixtures.prediction))("accepts the %s example", (_, make) => {
    expect(isPrediction(make())).toBe(true);
  });

  expectRejects(isPrediction, fixtures.prediction.ai, [
    ["unknown label", (p) => { p.label = "fake"; }],
    ["probabilities that do not sum to 1", (p) => { p.probabilities.real = 0.5; }],
    ["probability above 1", (p) => { p.probabilities = { real: -0.5, ai_generated: 1.5 }; }],
    ["band with low above high", (p) => { p.band = { low: 0.9, high: 0.1 }; }],
    ["region outside the image", (p) => { p.explanation.region.width = 1.2; }],
    ["unsupported image format", (p) => { p.image.format = "GIF"; }],
    ["fractional width", (p) => { p.image.width = 10.5; }],
    ["missing timing", (p) => { delete p.timing_ms.total; }],
    ["trained as a string", (p) => { p.model.trained = "false"; }],
    ["missing explanation key", (p) => { delete p.explanation; }],
    ["inconclusive without a band (low == high)", (p) => { p.label = "inconclusive"; p.band = { low: 0.5, high: 0.5 }; }],
  ]);
});

describe("error body contract", () => {
  it("accepts {error: {code, message}}", () => {
    expect(isApiErrorBody({ error: { code: "rate_limited", message: "slow down" } })).toBe(true);
  });
  it("rejects other shapes", () => {
    expect(isApiErrorBody({ detail: "Not Found" })).toBe(false);
    expect(isApiErrorBody({ error: { code: 429 } })).toBe(false);
  });
});

describe("GET /api/health contract", () => {
  it("accepts healthy and degraded bodies", () => {
    expect(isHealth(fixtures.health())).toBe(true);
    expect(isHealth({ status: "degraded", model_loaded: false, model_version: null, uptime_seconds: 3.2 })).toBe(true);
    expect(isHealth({ status: "degraded", model_loaded: false })).toBe(true);
  });
  it("rejects unknown status", () => {
    expect(isHealth({ status: "down", model_loaded: false })).toBe(false);
  });
});

describe("GET /api/model contract", () => {
  it("accepts the model card with extra metadata", () => {
    expect(isModelCard(fixtures.model())).toBe(true);
  });
  expectRejects(isModelCard, () => fixtures.model() as Record<string, any>, [
    ["missing onnx_size_mb", (m) => { delete m.onnx_size_mb; }],
    ["band out of range", (m) => { m.band.high = 2; }],
    ["input without crop", (m) => { delete m.input.crop; }],
  ]);
});

describe("GET /api/metrics contract", () => {
  it("accepts the envelope with any subset of files", () => {
    expect(isMetricsResponse(fixtures.metricsFull())).toBe(true);
    expect(isMetricsResponse(fixtures.metricsEmpty())).toBe(true);
    expect(isMetricsResponse({ available: "summary", metrics: {} })).toBe(false);
  });

  it.each(Object.keys(METRIC_GUARDS) as (keyof MetricsFiles)[])("accepts the example %s.json", (key) => {
    expect(METRIC_GUARDS[key](fixtures.metricsFull().metrics[key])).toBe(true);
  });

  it("allows null AUC when a split has one class", () => {
    const test = fixtures.metricsFull().metrics.summary?.test as Record<string, unknown>;
    expect(isClassificationMetrics({ ...test, roc_auc: null, pr_auc: null })).toBe(true);
  });

  const file = (key: keyof MetricsFiles) => () => fixtures.metricsFull().metrics[key] as Record<string, any>;
  const rejects = (key: keyof MetricsFiles, mutations: Mutation[]) =>
    describe(`${key}.json`, () => { expectRejects(METRIC_GUARDS[key], file(key), mutations); });

  rejects("summary", [["confusion matrix with 3 columns", (s) => { s.test.confusion_matrix[0].push(1); }], ["missing band", (s) => { delete s.band; }]]);
  rejects("history", [["unknown phase", (h) => { h.epochs[0].phase = "warmup"; }]]);
  rejects("confusion", [["negative count", (c) => { c.matrix[0][0] = -1; }]]);
  rejects("roc", [["mismatched arrays", (r) => { r.tpr.pop(); }], ["single point", (r) => { r.fpr = [0]; r.tpr = [0]; }]]);
  rejects("pr", [["precision above 1", (p) => { p.precision[0] = 1.2; }]]);
  rejects("confidence", [
    ["edges not one longer than counts", (c) => { c.bins.pop(); }],
    ["edges not increasing", (c) => { c.bins[2] = 0; }],
    ["class counts of different lengths", (c) => { c.real.pop(); c.bins.pop(); }],
  ]);
  rejects("external", [["missing notes", (e) => { delete e.notes; }]]);
  rejects("robustness", [["accuracy as text", (r) => { r.conditions[0].accuracy = "92%"; }]]);
  rejects("dataset_report", [["splits with missing class", (d) => { delete d.splits.train.real; }]]);
  rejects("latency", [["missing cpu block", (l) => { delete l.cpu; }]]);
  rejects("model_comparison", [["model without name", (m) => { delete m.models[0].name; }]]);
});
