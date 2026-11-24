import type { Ref } from "react";
import type { DetectState } from "../hooks/useDetector";
import { LABEL_TEXT } from "../lib/verdict";
import { Notice } from "./Notice";
import { ProgressSteps } from "./ProgressSteps";
import { ResultView } from "./ResultView";
import { VerdictMark } from "./VerdictMark";

interface ResultPanelProps {
  state: DetectState;
  headingRef: Ref<HTMLHeadingElement>;
  onRetry: () => void;
  onReset: () => void;
  onCancel: () => void;
}

export function ResultPanel({ state, headingRef, onRetry, onReset, onCancel }: ResultPanelProps) {
  switch (state.phase) {
    case "uploading":
    case "analysing":
      return (
        <div className="result-placeholder" aria-busy="true">
          <h2 className="panel-title">Analysis in progress</h2>
          <ProgressSteps phase={state.phase} onCancel={onCancel} />
        </div>
      );
    case "done":
      return (
        <>
          <ResultView result={state.result} headingRef={headingRef} />
          <div className="result__actions">
            <button type="button" className="button button--secondary" onClick={onReset}>Analyse another image</button>
          </div>
        </>
      );
    case "failed":
      return (
        <div className="result-placeholder">
          <h2 className="panel-title">Result</h2>
          <Notice
            tone="danger"
            role="alert"
            title={state.error.title}
            headingRef={headingRef}
            actions={
              <>
                {state.error.retryable && <button type="button" className="button button--primary" onClick={onRetry}>Try again</button>}
                <button type="button" className="button button--secondary" onClick={onReset}>Choose another image</button>
              </>
            }
          >
            <p>{state.error.detail}</p>
          </Notice>
        </div>
      );
    default:
      return (
        <div className="result-placeholder">
          <h2 className="panel-title">Result</h2>
          <p className="result-placeholder__text">
            {state.phase === "ready"
              ? "Ready. Select “Analyse image” to run the model."
              : "Choose an image to analyse. The verdict, calibrated probabilities and heatmap will appear here."}
          </p>
          <dl className="result-placeholder__legend">
            {(["real", "ai_generated", "inconclusive"] as const).map((label) => (
              <PlaceholderLegendItem key={label} label={label} />
            ))}
          </dl>
        </div>
      );
  }
}

const LEGEND_TEXT = {
  real: "Probability of AI generation is low enough to call the image real.",
  ai_generated: "Probability of AI generation is high enough to call it AI-generated.",
  inconclusive: "The probability falls in the uncertainty band; no verdict is given.",
} as const;

function PlaceholderLegendItem({ label }: { label: keyof typeof LEGEND_TEXT }) {
  return (
    <div>
      <dt><VerdictMark label={label} size={12} />{LABEL_TEXT[label]}</dt>
      <dd>{LEGEND_TEXT[label]}</dd>
    </div>
  );
}
