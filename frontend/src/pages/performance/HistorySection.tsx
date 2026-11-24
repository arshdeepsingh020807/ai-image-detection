import { LineChart } from "../../charts/LineChart";
import type { History, HistoryEpoch } from "../../api/types";
import { formatDecimal, formatPercent } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { niceDomain } from "../../lib/scale";
import { FromFile, Section } from "./Section";
import { TableScroll } from "../../components/TableScroll";

const epochTick = (value: number) => String(value);

function phaseSummary(epochs: HistoryEpoch[]): string {
  const ranges = ["head", "finetune"].map((phase) => epochs.filter((e) => e.phase === phase).map((e) => e.epoch));
  const [head = [], finetune = []] = ranges;
  const span = (list: number[]) => (list.length > 1 ? `Epochs ${list[0]}–${list[list.length - 1]}` : `Epoch ${list[0]}`);
  return [
    head.length ? `${span(head)}: train the new classification head on a frozen backbone` : "",
    finetune.length ? `${span(finetune)}: fine-tune the backbone at a lower learning rate` : "",
  ].filter(Boolean).join("; ");
}

function HistoryCharts({ epochs }: { epochs: HistoryEpoch[] }) {
  const xDomain = [epochs[0]?.epoch ?? 0, epochs[epochs.length - 1]?.epoch ?? 1] as const;
  const losses = epochs.flatMap((e) => [e.train_loss, e.val_loss]);
  const series = (train: keyof HistoryEpoch, val: keyof HistoryEpoch, name: string) => [
    { id: `train-${name}`, label: `Training ${name}`, tone: "muted" as const, dashed: true, points: epochs.map((e) => [e.epoch, e[train] as number] as const) },
    { id: `val-${name}`, label: `Validation ${name}`, tone: "accent" as const, points: epochs.map((e) => [e.epoch, e[val] as number] as const) },
  ];
  return (
    <>
      <p className="report-meta">{phaseSummary(epochs)}.</p>
      <div className="report-grid">
        <div className="subsection">
          <h3>Loss</h3>
          <LineChart
            title="Training and validation loss per epoch"
            description="Binary cross-entropy. Validation loss rising while training loss falls would indicate overfitting."
            series={series("train_loss", "val_loss", "loss")}
            xDomain={xDomain} yDomain={niceDomain(Math.min(0, ...losses), Math.max(...losses), 4)}
            xLabel="Epoch" yLabel="Loss" xFormat={epochTick} yFormat={(v) => formatDecimal(v, 2)} valueFormat={(v) => formatDecimal(v, 4)}
            xTickCount={Math.min(10, epochs.length)}
          />
        </div>
        <div className="subsection">
          <h3>Accuracy</h3>
          <LineChart
            title="Training and validation accuracy per epoch"
            description="Accuracy at the 0.5 threshold on the training and validation splits."
            series={series("train_accuracy", "val_accuracy", "accuracy")}
            xDomain={xDomain} yDomain={[0, 1]}
            xLabel="Epoch" yLabel="Accuracy" xFormat={epochTick} yFormat={(v) => formatPercent(v, 0)} valueFormat={(v) => formatPercent(v)}
            xTickCount={Math.min(10, epochs.length)}
          />
        </div>
      </div>
      <details className="data-details">
        <summary>Show training log</summary>
        <TableScroll label="Training log table">
          <table className="data-table data-table--compact">
            <thead>
              <tr>
                <th scope="col">Epoch</th><th scope="col">Phase</th><th scope="col" className="num">Train loss</th><th scope="col" className="num">Val loss</th>
                <th scope="col" className="num">Train acc.</th><th scope="col" className="num">Val acc.</th><th scope="col" className="num">Learning rate</th>
              </tr>
            </thead>
            <tbody>
              {epochs.map((e) => (
                <tr key={e.epoch}>
                  <th scope="row" className="num">{e.epoch}</th>
                  <td>{e.phase === "head" ? "Head" : "Fine-tune"}</td>
                  <td className="num">{formatDecimal(e.train_loss, 4)}</td>
                  <td className="num">{formatDecimal(e.val_loss, 4)}</td>
                  <td className="num">{formatPercent(e.train_accuracy)}</td>
                  <td className="num">{formatPercent(e.val_accuracy)}</td>
                  <td className="num">{e.lr.toExponential(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
      </details>
    </>
  );
}

export function HistorySection({ state }: { state: FileState<History> }) {
  return (
    <Section id="history" title="Training history" intro={<p>Two-phase transfer learning, as logged after every epoch.</p>}>
      <FromFile state={state} source="history.json">
        {({ epochs }) => (epochs.length > 0 ? <HistoryCharts epochs={epochs} /> : <p className="report-meta">The training log contains no epochs.</p>)}
      </FromFile>
    </Section>
  );
}
