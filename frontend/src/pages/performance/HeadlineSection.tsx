import type { ClassificationMetrics, Summary } from "../../api/types";
import { formatCount, formatDate, formatDecimal, formatPercent } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { FromFile, Section } from "./Section";
import { StatTiles } from "./StatTiles";

type MetricRow = [label: string, key: Exclude<keyof ClassificationMetrics, "confusion_matrix" | "n" | "threshold">, kind: "percent" | "decimal"];

const HEADLINE: MetricRow[] = [
  ["Accuracy", "accuracy", "percent"],
  ["Precision", "precision", "percent"],
  ["Recall", "recall", "percent"],
  ["F1 score", "f1", "percent"],
  ["ROC-AUC", "roc_auc", "decimal"],
  ["Specificity", "specificity", "percent"],
  ["False positive rate", "false_positive_rate", "percent"],
];

const SECONDARY: MetricRow[] = [
  ["False negative rate", "false_negative_rate", "percent"],
  ["PR-AUC", "pr_auc", "decimal"],
  ["Log loss", "log_loss", "decimal"],
  ["Brier score", "brier", "decimal"],
  ["Expected calibration error", "ece", "decimal"],
];

const show = (metrics: ClassificationMetrics, [, key, kind]: MetricRow) =>
  kind === "percent" ? formatPercent(metrics[key]) : formatDecimal(metrics[key]);

export function HeadlineSection({ state }: { state: FileState<Summary> }) {
  return (
    <Section
      id="headline"
      title="Headline metrics"
      intro={<p>Measured on the held-out test split at the default threshold, before the inconclusive band is applied. The positive class is AI-generated, so a false positive is a real image called AI-generated.</p>}
    >
      <FromFile state={state} source="summary.json">
        {(summary) => (
          <>
            <p className="report-meta">
              Test split · n = {formatCount(summary.test.n)} images · threshold p(AI) ≥ {formatDecimal(summary.test.threshold, 2)} ·
              model <code>{summary.model}</code> · evaluated {formatDate(summary.created)} · seed {summary.seed}
            </p>
            <StatTiles label="Test-split metrics" stats={HEADLINE.map((row) => ({ label: row[0], value: show(summary.test, row) }))} />
            <div className="report-grid">
              <table className="data-table">
                <caption>Validation and test splits</caption>
                <thead>
                  <tr><th scope="col">Metric</th><th scope="col" className="num">Validation</th><th scope="col" className="num">Test</th></tr>
                </thead>
                <tbody>
                  <tr><th scope="row">Images (n)</th><td className="num">{formatCount(summary.validation.n)}</td><td className="num">{formatCount(summary.test.n)}</td></tr>
                  {[...HEADLINE, ...SECONDARY].map((row) => (
                    <tr key={row[1]}>
                      <th scope="row">{row[0]}</th>
                      <td className="num">{show(summary.validation, row)}</td>
                      <td className="num">{show(summary.test, row)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div>
                <table className="data-table">
                  <caption>Test split with the inconclusive band</caption>
                  <tbody>
                    <tr><th scope="row">Band (p(AI))</th><td className="num">{formatDecimal(summary.band.low, 2)} – {formatDecimal(summary.band.high, 2)}</td></tr>
                    <tr><th scope="row">Coverage (decided)</th><td className="num">{formatPercent(summary.test_with_band.coverage)}</td></tr>
                    <tr><th scope="row">Accuracy on decided images</th><td className="num">{formatPercent(summary.test_with_band.accuracy_on_decided)}</td></tr>
                    <tr><th scope="row">Real images called AI</th><td className="num">{formatPercent(summary.test_with_band.false_positive_rate_on_real)}</td></tr>
                    <tr><th scope="row">Overall error rate</th><td className="num">{formatPercent(summary.test_with_band.error_rate)}</td></tr>
                    <tr><th scope="row">Inconclusive images</th><td className="num">{formatCount(summary.test_with_band.inconclusive)}</td></tr>
                    <tr><th scope="row">Calibration temperature</th><td className="num">{formatDecimal(summary.temperature)}</td></tr>
                  </tbody>
                </table>
                <p className="table-note">
                  Coverage is the share of test images that receive a verdict. Errors count only decided images; inconclusive ones are
                  neither right nor wrong.
                </p>
              </div>
            </div>
          </>
        )}
      </FromFile>
    </Section>
  );
}
