import { BarList, type BarRow } from "../../charts/BarList";
import { Legend } from "../../charts/Legend";
import type { ClassCounts, DatasetReport } from "../../api/types";
import { formatCount, formatDecimal, formatPercent, humanize } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { CLASS_TEXT } from "../../lib/verdict";
import { FromFile, Section } from "./Section";
import { StatTiles } from "./StatTiles";
import { TableScroll } from "../../components/TableScroll";

const byValue = (record: Record<string, number>) => Object.entries(record).sort(([, a], [, b]) => b - a);
const single = (record: Record<string, number>, name: (key: string) => string = humanize): BarRow[] =>
  byValue(record).map(([key, value]) => ({ label: name(key), segments: [{ label: name(key), value, tone: "accent" }] }));
const className = (key: string) => (key in CLASS_TEXT ? CLASS_TEXT[key as keyof typeof CLASS_TEXT] : humanize(key));
const split = (label: string, counts: ClassCounts, note?: string): BarRow => ({
  label,
  ...(note ? { note } : {}),
  segments: [
    { label: "Real", value: counts.real, tone: "real" },
    { label: "AI-generated", value: counts.ai_generated, tone: "ai" },
  ],
});
const CLASS_LEGEND = [
  { label: "Real", tone: "real", mark: "real" },
  { label: "AI-generated", tone: "ai", mark: "ai_generated" },
] as const;

function DatasetBody({ report }: { report: DatasetReport }) {
  const leaks = Object.values(report.leakage_check).reduce((a, b) => a + b, 0);
  return (
    <>
      <StatTiles
        label="Dataset size"
        stats={[
          { label: "Images collected", value: formatCount(report.original_images) },
          { label: "Kept after cleaning", value: formatCount(report.cleaned_images), hint: formatPercent(report.cleaned_images / Math.max(1, report.original_images)) },
          { label: "Removed", value: formatCount(report.removed_images) },
          { label: "Near-duplicate clusters", value: formatCount(report.near_duplicate_clusters), hint: `${formatCount(report.images_in_near_duplicate_clusters)} images` },
        ]}
      />
      <div className="report-grid">
        <div className="subsection">
          <h3>Why images were removed</h3>
          <BarList label="Removed images by reason" rows={single(report.removed_by_reason)} />
        </div>
        <div className="subsection">
          <h3>Splits by class</h3>
          <Legend items={[...CLASS_LEGEND]} />
          <BarList label="Images per split, real and AI-generated" rows={Object.entries(report.splits).map(([name, counts]) => split(humanize(name), counts))} />
          <p className="table-note">
            Leakage check: {leaks === 0 ? "no" : formatCount(leaks)} duplicate or near-duplicate group{leaks === 1 ? "" : "s"} span more than one split.
          </p>
        </div>
        <div className="subsection">
          <h3>Sources</h3>
          <Legend items={[...CLASS_LEGEND]} />
          <BarList
            label="Images per source, real and AI-generated"
            rows={Object.entries(report.by_source).map(([name, s]) => split(name, s, `${humanize(s.domain)} · ${s.role === "external" ? "external test only" : "training pool"}`))}
          />
        </div>
        <div className="subsection">
          <h3>AI-generated images by generator</h3>
          <BarList label="AI-generated images per generator" rows={single(report.by_generator, String)} />
        </div>
      </div>
      <div className="subsection">
        <h3>Shortcut probes</h3>
        <p className="table-note">
          A simple classifier is trained on metadata alone (file format, size, average colour). An AUC near 0.5 means the label cannot be
          guessed without looking at the image; a high AUC warns that a model could learn that shortcut instead of real generation traces.
        </p>
        <TableScroll label="Shortcut probes table">
          <table className="data-table">
            <caption className="visually-hidden">Shortcut probe results</caption>
            <thead><tr><th scope="col">Probe</th><th scope="col">Features</th><th scope="col" className="num">n</th><th scope="col" className="num">AUC</th><th scope="col" className="num">Accuracy</th></tr></thead>
            <tbody>
              {Object.entries(report.shortcut_probes).map(([name, probe]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{probe.features.join(", ")}</td>
                  <td className="num">{formatCount(probe.n)}</td>
                  <td className="num">{formatDecimal(probe.auc)}</td>
                  <td className="num">{formatPercent(probe.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
      </div>
      <div className="report-grid">
        <div className="subsection">
          <h3>File formats</h3>
          <table className="data-table data-table--compact">
            <caption className="visually-hidden">File formats per class</caption>
            <thead><tr><th scope="col">Class</th><th scope="col">Format</th><th scope="col" className="num">Images</th></tr></thead>
            <tbody>
              {Object.entries(report.formats).flatMap(([label, formats]) =>
                byValue(formats).map(([format, count]) => (
                  <tr key={`${label}-${format}`}><th scope="row">{className(label)}</th><td>{format}</td><td className="num">{formatCount(count)}</td></tr>
                )))}
            </tbody>
          </table>
        </div>
        <div className="subsection">
          <h3>Most common resolutions</h3>
          <table className="data-table data-table--compact">
            <caption className="visually-hidden">Most common image resolutions</caption>
            <thead><tr><th scope="col">Resolution</th><th scope="col" className="num">Images</th></tr></thead>
            <tbody>
              {byValue(report.resolution_top).map(([resolution, count]) => (
                <tr key={resolution}><th scope="row">{resolution.replace("x", " × ")}</th><td className="num">{formatCount(count)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

export function DatasetSection({ state }: { state: FileState<DatasetReport> }) {
  return (
    <Section id="dataset" title="Dataset and cleaning" intro={<p>What the model learned from, and what was removed before training to stop duplicates and leaked images inflating the results.</p>}>
      <FromFile state={state} source="dataset_report.json">{(report) => <DatasetBody report={report} />}</FromFile>
    </Section>
  );
}
