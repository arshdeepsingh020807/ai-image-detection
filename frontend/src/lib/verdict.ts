import type { Band, Label, Prediction } from "../api/types";
import { formatPercent } from "./format";

export const LABEL_TEXT: Record<Label, string> = {
  real: "Likely real",
  ai_generated: "Likely AI-generated",
  inconclusive: "Inconclusive",
};

export const CLASS_TEXT = { real: "Real", ai_generated: "AI-generated" } as const;

export const hasBand = (band: Band): boolean => band.high > band.low;

export const describeBand = (band: Band): string =>
  hasBand(band)
    ? `Verdicts need p(AI) ≤ ${formatPercent(band.low)} for “real” or ≥ ${formatPercent(band.high)} for “AI-generated”; anything between is inconclusive.`
    : `Single decision threshold at p(AI) = ${formatPercent(band.low)}. This model has no inconclusive band.`;

export function summarise(prediction: Prediction): string {
  const { band, label } = prediction;
  const pAi = formatPercent(prediction.probabilities.ai_generated);
  if (label === "inconclusive") {
    return `The estimated probability of AI generation (${pAi}) falls inside the uncertainty band of ${formatPercent(band.low)} to ${formatPercent(band.high)}. In this range the model is not reliable enough to decide, so no verdict is given.`;
  }
  return label === "ai_generated"
    ? `The model estimates a ${pAi} probability that this image is AI-generated, at or above the ${formatPercent(band.high)} needed for this verdict.`
    : `The model estimates a ${pAi} probability that this image is AI-generated, at or below the ${formatPercent(band.low)} allowed for a “real” verdict.`;
}
