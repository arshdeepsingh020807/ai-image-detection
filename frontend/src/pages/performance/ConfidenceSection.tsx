import { Histogram } from "../../charts/Histogram";
import type { Band, ConfidenceHistogram } from "../../api/types";
import { formatCount, formatDecimal } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { hasBand } from "../../lib/verdict";
import { FromFile, Section } from "./Section";

const fmt = (value: number) => formatDecimal(value, 1);

function bandMarkers(band: Band | null) {
  if (!band) return [];
  return hasBand(band)
    ? [{ value: band.low, label: `low ${formatDecimal(band.low, 2)}` }, { value: band.high, label: `high ${formatDecimal(band.high, 2)}` }]
    : [{ value: band.low, label: `threshold ${formatDecimal(band.low, 2)}` }];
}

export function ConfidenceSection({ state, band }: { state: FileState<ConfidenceHistogram>; band: Band | null }) {
  return (
    <Section
      id="confidence"
      title="Confidence distribution"
      intro={<p>Calibrated probability of AI generation for every test image, split by its true class. Well-separated peaks near 0 and 1 mean confident, correct predictions; overlap in the middle is where mistakes and inconclusive results come from.</p>}
    >
      <FromFile state={state} source="confidence.json">
        {(data) => (
          <>
            <Histogram
              title="Histogram of p(AI) by true class"
              description={band ? "Vertical lines mark the decision band from summary.json." : "Decision band not shown: summary.json is not published."}
              edges={data.bins}
              series={[
                { label: "Real images", tone: "real", mark: "real", counts: data.real },
                { label: "AI-generated images", tone: "ai", mark: "ai_generated", counts: data.ai_generated },
              ]}
              markers={bandMarkers(band)}
              xLabel="Probability of AI generation"
              xFormat={fmt}
            />
            <details className="data-details">
              <summary>Show histogram data</summary>
              <table className="data-table data-table--compact">
                <thead><tr><th scope="col">p(AI) bin</th><th scope="col" className="num">Real</th><th scope="col" className="num">AI-generated</th></tr></thead>
                <tbody>
                  {data.real.map((count, i) => (
                    <tr key={i}>
                      <th scope="row" className="num">{formatDecimal(data.bins[i] ?? 0, 2)}–{formatDecimal(data.bins[i + 1] ?? 0, 2)}</th>
                      <td className="num">{formatCount(count)}</td>
                      <td className="num">{formatCount(data.ai_generated[i] ?? 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </>
        )}
      </FromFile>
    </Section>
  );
}
