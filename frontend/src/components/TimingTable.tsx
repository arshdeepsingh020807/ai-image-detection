import type { Prediction } from "../api/types";
import { formatMs } from "../lib/format";

const STAGES = [
  ["decode", "Decode"],
  ["preprocess", "Preprocess"],
  ["inference", "Inference"],
  ["explain", "Heatmap"],
] as const;

export function TimingTable({ timing }: { timing: Prediction["timing_ms"] }) {
  return (
    <table className="data-table data-table--compact">
      <caption className="visually-hidden">Server-side timing</caption>
      <thead className="visually-hidden">
        <tr><th scope="col">Stage</th><th scope="col" className="num">Time</th></tr>
      </thead>
      <tbody>
        {STAGES.map(([key, label]) => (
          <tr key={key}><th scope="row">{label}</th><td className="num">{formatMs(timing[key])}</td></tr>
        ))}
      </tbody>
      <tfoot>
        <tr><th scope="row">Total</th><td className="num">{formatMs(timing.total)}</td></tr>
      </tfoot>
    </table>
  );
}
