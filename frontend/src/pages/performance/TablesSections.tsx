import type { ComparedModel, External, ModelComparison, Robustness } from "../../api/types";
import { formatCount, formatDecimal, formatPercent } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { FromFile, Section } from "./Section";
import { TableScroll } from "../../components/TableScroll";

export function ExternalSection({ state }: { state: FileState<External> }) {
  return (
    <Section
      id="external"
      title="External test sets"
      intro={<p>Sources the model never saw during training. Results here are usually lower than on the in-distribution test split, and that gap is the most honest estimate of real-world performance.</p>}
    >
      <FromFile state={state} source="external.json">
        {({ sets, notes }) => (
          <>
            {sets.length > 0 ? (
              <TableScroll label="External test sets table">
                <table className="data-table">
                  <caption className="visually-hidden">Metrics on external test sets</caption>
                  <thead>
                    <tr>
                      <th scope="col">Set</th><th scope="col">Protocol</th><th scope="col" className="num">n</th><th scope="col" className="num">Accuracy</th>
                      <th scope="col" className="num">ROC-AUC</th><th scope="col" className="num">Recall</th><th scope="col" className="num">False positive rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sets.map((set) => (
                      <tr key={set.name}>
                        <th scope="row">{set.name}</th>
                        <td>{set.protocol}</td>
                        <td className="num">{formatCount(set.n)}</td>
                        <td className="num">{formatPercent(set.metrics.accuracy)}</td>
                        <td className="num">{formatDecimal(set.metrics.roc_auc)}</td>
                        <td className="num">{formatPercent(set.metrics.recall)}</td>
                        <td className="num">{formatPercent(set.metrics.false_positive_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableScroll>
            ) : <p className="report-meta">The file lists no external sets.</p>}
            {notes.length > 0 && (
              <ul className="notes">{notes.map((note) => <li key={note}>{note}</li>)}</ul>
            )}
          </>
        )}
      </FromFile>
    </Section>
  );
}

export function RobustnessSection({ state }: { state: FileState<Robustness> }) {
  return (
    <Section
      id="robustness"
      title="Robustness"
      intro={<p>The same evaluation repeated after everyday transformations such as recompression or resizing. A large drop means the model relies on fragile, easily destroyed traces.</p>}
    >
      <FromFile state={state} source="robustness.json">
        {({ conditions }) => (
          <TableScroll label="Robustness table">
            <table className="data-table">
              <caption className="visually-hidden">Metrics under image transformations</caption>
              <thead>
                <tr><th scope="col">Condition</th><th scope="col">Description</th><th scope="col" className="num">Accuracy</th><th scope="col" className="num">ROC-AUC</th><th scope="col" className="num">False positive rate</th></tr>
              </thead>
              <tbody>
                {conditions.map((c) => (
                  <tr key={c.name}>
                    <th scope="row">{c.name}</th>
                    <td>{c.description}</td>
                    <td className="num">{formatPercent(c.accuracy)}</td>
                    <td className="num">{formatDecimal(c.roc_auc)}</td>
                    <td className="num">{formatPercent(c.false_positive_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
        )}
      </FromFile>
    </Section>
  );
}

const isDeployed = (model: ComparedModel, deployed: string | null) => deployed !== null && model.name === deployed;

export function ComparisonSection({ state, deployed }: { state: FileState<ModelComparison>; deployed: string | null }) {
  return (
    <Section id="comparison" title="Model comparison" intro={<p>Candidate backbones trained under the same protocol. The deployed model trades accuracy against size and speed on a CPU.</p>}>
      <FromFile state={state} source="model_comparison.json">
        {({ models }) => (
          <TableScroll label="Model comparison table">
            <table className="data-table">
              <caption className="visually-hidden">Candidate model comparison</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th><th scope="col" className="num">Parameters</th><th scope="col" className="num">ONNX size</th>
                  <th scope="col" className="num">Validation AUC</th><th scope="col" className="num">External AUC</th>
                  <th scope="col" className="num">False positive rate</th><th scope="col" className="num">Latency</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name} data-highlight={isDeployed(m, deployed) || undefined}>
                    <th scope="row">{m.name}{isDeployed(m, deployed) && <span className="tag">Deployed</span>}</th>
                    <td className="num">{formatDecimal(m.parameters_millions, 1)} M</td>
                    <td className="num">{formatDecimal(m.onnx_size_mb, 1)} MB</td>
                    <td className="num">{formatDecimal(m.validation_auc)}</td>
                    <td className="num">{formatDecimal(m.external_auc)}</td>
                    <td className="num">{formatPercent(m.false_positive_rate)}</td>
                    <td className="num">{formatDecimal(m.latency_ms, 1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
        )}
      </FromFile>
    </Section>
  );
}
