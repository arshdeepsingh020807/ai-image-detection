import type { Ref } from "react";
import type { Prediction } from "../api/types";
import { formatBytes, formatProbability } from "../lib/format";
import { LABEL_TEXT, summarise } from "../lib/verdict";
import { ProbabilityBar } from "./ProbabilityBar";
import { TimingTable } from "./TimingTable";
import { VerdictMark } from "./VerdictMark";

export function ResultView({ result, headingRef }: { result: Prediction; headingRef: Ref<HTMLHeadingElement> }) {
  const { image, model, explanation } = result;
  return (
    <div className="result">
      <div className={`verdict verdict--${result.label}`}>
        <VerdictMark label={result.label} size={20} />
        <div>
          <p className="verdict__eyebrow">Verdict</p>
          <h2 className="verdict__title" ref={headingRef} tabIndex={-1}>{LABEL_TEXT[result.label]}</h2>
        </div>
        <p className="verdict__confidence">
          <span className="verdict__confidence-value">{formatProbability(result.confidence)}</span>
          <span className="verdict__confidence-label">confidence</span>
        </p>
      </div>
      <p className="result__summary">{summarise(result)}</p>
      {result.label === "inconclusive" && (
        <p className="result__inconclusive">
          Treat this image as undetermined. An inconclusive result is a deliberate output, not a failure: it marks the range where the
          model made too many mistakes during evaluation to be trusted.
        </p>
      )}
      <ProbabilityBar real={result.probabilities.real} aiGenerated={result.probabilities.ai_generated} band={result.band} />
      <p className="result__footnote">Confidence is the larger of the two calibrated probabilities.</p>
      <div className="result__columns">
        <section aria-labelledby="details-title">
          <h3 id="details-title">Details</h3>
          <dl className="kv">
            <div><dt>Dimensions</dt><dd>{image.width} × {image.height} px</dd></div>
            <div><dt>Format</dt><dd>{image.format}, {formatBytes(image.size_bytes)}</dd></div>
            <div><dt>Model</dt><dd>{model.architecture}</dd></div>
            <div><dt>Version</dt><dd><code>{model.version.split("-").map((part, i) => (i ? <span key={i}>-<wbr />{part}</span> : part))}</code></dd></div>
            <div><dt>Heatmap</dt><dd>{explanation ? explanation.method : "Not requested"}</dd></div>
          </dl>
        </section>
        <section aria-labelledby="timing-title">
          <h3 id="timing-title">Timing</h3>
          <TimingTable timing={result.timing_ms} />
        </section>
      </div>
    </div>
  );
}
