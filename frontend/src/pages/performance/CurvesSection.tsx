import { LineChart } from "../../charts/LineChart";
import type { PrCurve, Roc } from "../../api/types";
import { formatDecimal } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { FromFile, Section } from "./Section";

const unit = [0, 1] as const;
const tick = (value: number) => formatDecimal(value, 1);
const precise = (value: number) => formatDecimal(value, 3);
const zip = (xs: number[], ys: number[]) => xs.map((x, i) => [x, ys[i] ?? 0] as const);

export function CurvesSection({ roc, pr }: { roc: FileState<Roc>; pr: FileState<PrCurve> }) {
  return (
    <Section id="curves" title="ROC and precision–recall curves" intro={<p>Threshold-free views of the test split. Hover a curve to read individual operating points.</p>}>
      <div className="report-grid">
        <div className="subsection">
          <h3>ROC curve</h3>
          <FromFile state={roc} source="roc.json">
            {(data) => (
              <LineChart
                title={`ROC curve, AUC ${precise(data.auc)}`}
                description={`Area under the curve: ${precise(data.auc)}. The diagonal marks a classifier that guesses at random.`}
                series={[{ id: "roc", label: "ROC", tone: "accent", points: zip(data.fpr, data.tpr) }]}
                xDomain={unit} yDomain={unit} square diagonal
                xLabel="False positive rate" yLabel="True positive rate"
                xFormat={tick} yFormat={tick} valueFormat={precise}
              />
            )}
          </FromFile>
        </div>
        <div className="subsection">
          <h3>Precision–recall curve</h3>
          <FromFile state={pr} source="pr.json">
            {(data) => (
              <LineChart
                title={`Precision–recall curve, average precision ${precise(data.average_precision)}`}
                description={`Average precision: ${precise(data.average_precision)}. Precision of AI-generated verdicts as the threshold sweeps through recall.`}
                series={[{ id: "pr", label: "Precision", tone: "accent", points: zip(data.recall, data.precision) }]}
                xDomain={unit} yDomain={unit} square
                xLabel="Recall" yLabel="Precision"
                xFormat={tick} yFormat={tick} valueFormat={precise}
              />
            )}
          </FromFile>
        </div>
      </div>
    </Section>
  );
}
