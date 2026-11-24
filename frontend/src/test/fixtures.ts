import health from "../../tests/fixtures/health.json";
import metricsEmpty from "../../tests/fixtures/metrics-empty.json";
import metricsFull from "../../tests/fixtures/metrics-full.json";
import model from "../../tests/fixtures/model.json";
import predictionAi from "../../tests/fixtures/prediction-ai.json";
import predictionInconclusive from "../../tests/fixtures/prediction-inconclusive.json";
import predictionReal from "../../tests/fixtures/prediction-real.json";
import predictionUntrained from "../../tests/fixtures/prediction-untrained.json";
import type { MetricsResponse, ModelCard, Prediction } from "../api/types";

const clone = <T>(value: T): T => structuredClone(value);

export const fixtures = {
  health: () => clone(health),
  model: () => clone(model) as ModelCard,
  metricsFull: () => clone(metricsFull) as MetricsResponse & { metrics: Record<string, Record<string, unknown>> },
  metricsEmpty: () => clone(metricsEmpty) as MetricsResponse,
  prediction: {
    ai: () => clone(predictionAi) as Prediction,
    real: () => clone(predictionReal) as Prediction,
    inconclusive: () => clone(predictionInconclusive) as Prediction,
    untrained: () => clone(predictionUntrained) as Prediction,
  },
};
