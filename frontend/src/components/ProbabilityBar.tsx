import type { Band } from "../api/types";
import { formatProbability } from "../lib/format";
import { CLASS_TEXT, describeBand, hasBand } from "../lib/verdict";
import { VerdictMark } from "./VerdictMark";

interface ProbabilityBarProps {
  real: number;
  aiGenerated: number;
  band: Band;
}

export function ProbabilityBar({ real, aiGenerated, band }: ProbabilityBarProps) {
  const bandWidth = (band.high - band.low) * 100;
  return (
    <figure className="probability">
      <figcaption className="probability__caption">Calibrated probabilities</figcaption>
      <dl className="probability__legend">
        <div>
          <dt><VerdictMark label="real" size={12} />{CLASS_TEXT.real}</dt>
          <dd>{formatProbability(real)}</dd>
        </div>
        <div>
          <dt><VerdictMark label="ai_generated" size={12} />{CLASS_TEXT.ai_generated}</dt>
          <dd>{formatProbability(aiGenerated)}</dd>
        </div>
      </dl>
      <div className="probability__track" aria-hidden="true">
        <span className="probability__real" style={{ width: `${real * 100}%` }} />
        <span className="probability__ai" style={{ width: `${aiGenerated * 100}%` }} />
      </div>
      <div className="probability__scale" aria-hidden="true">
        {hasBand(band) ? (
          <span className="probability__band" style={{ left: `${(1 - band.high) * 100}%`, width: `${bandWidth}%` }}>
            <span>inconclusive</span>
          </span>
        ) : (
          <span className="probability__threshold" style={{ left: `${(1 - band.low) * 100}%` }} />
        )}
      </div>
      <p className="probability__note">{describeBand(band)}</p>
    </figure>
  );
}
