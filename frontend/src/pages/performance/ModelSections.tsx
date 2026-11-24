import type { Latency, ModelCard } from "../../api/types";
import { Notice } from "../../components/Notice";
import type { AsyncState } from "../../hooks/useAsync";
import { formatDecimal, humanize } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { describeBand } from "../../lib/verdict";
import { FromFile, Section } from "./Section";
import { StatTiles } from "./StatTiles";

export function LatencySection({ state }: { state: FileState<Latency> }) {
  return (
    <Section id="latency" title="Inference cost" intro={<p>Single-image inference on CPU with the exported ONNX model.</p>}>
      <FromFile state={state} source="latency.json">
        {(latency) => (
          <>
            <StatTiles
              label="Latency and memory"
              stats={[
                { label: "Median latency", value: `${formatDecimal(latency.cpu.p50_ms, 1)} ms` },
                { label: "95th percentile", value: `${formatDecimal(latency.cpu.p95_ms, 1)} ms` },
                { label: "Resident memory", value: `${formatDecimal(latency.cpu.rss_mb, 0)} MB` },
                { label: "Model file", value: `${formatDecimal(latency.onnx_size_mb, 1)} MB` },
              ]}
            />
            <p className="report-meta">{latency.runs} timed runs on {latency.machine}.</p>
          </>
        )}
      </FromFile>
    </Section>
  );
}

const KNOWN_KEYS = new Set(["architecture", "version", "trained", "input", "temperature", "band", "onnx_size_mb", "outputs"]);
const isScalar = (value: unknown): value is string | number | boolean => ["string", "number", "boolean"].includes(typeof value);

type ModelState = AsyncState<ModelCard> & { reload: () => void };

export function ModelCardSection({ model }: { model: ModelState }) {
  return (
    <Section id="model-card" title="Model card" intro={<p>The configuration of the model currently served by the API.</p>}>
      {model.status === "loading" && <p className="report-meta">Loading model details…</p>}
      {model.status === "error" && (
        <Notice tone="danger" title={model.error.title} actions={<button type="button" className="button button--secondary" onClick={model.reload}>Try again</button>}>
          <p>{model.error.detail}</p>
        </Notice>
      )}
      {model.status === "ready" && <ModelCardDetails card={model.data} />}
    </Section>
  );
}

function ModelCardDetails({ card }: { card: ModelCard }) {
  const extras = Object.entries(card).filter(([key, value]) => !KNOWN_KEYS.has(key) && isScalar(value)) as [string, string | number | boolean][];
  return (
    <dl className="kv kv--wide">
      <div><dt>Architecture</dt><dd>{card.architecture}</dd></div>
      <div><dt>Version</dt><dd><code>{card.version}</code></dd></div>
      <div><dt>Trained</dt><dd>{card.trained ? "Yes" : "No, untrained stand-in for testing only"}</dd></div>
      <div><dt>Input</dt><dd>Short side resized to {card.input.short_side} px, centre crop {card.input.crop} × {card.input.crop} px</dd></div>
      <div><dt>Calibration temperature</dt><dd className="num">{formatDecimal(card.temperature)}</dd></div>
      <div><dt>Decision band</dt><dd>{describeBand(card.band)}</dd></div>
      <div><dt>ONNX file size</dt><dd className="num">{formatDecimal(card.onnx_size_mb, 1)} MB</dd></div>
      {extras.map(([key, value]) => (
        <div key={key}><dt>{humanize(key)}</dt><dd>{String(value)}</dd></div>
      ))}
    </dl>
  );
}
