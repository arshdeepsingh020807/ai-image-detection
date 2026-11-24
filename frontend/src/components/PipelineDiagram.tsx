import { useId } from "react";

export const STAGES = [
  ["Collect", "Faces + general images"],
  ["Audit and clean", "Decode, drop bad files"],
  ["Deduplicate and split", "Clusters stay together"],
  ["Transfer learning", "Head, then fine-tune"],
  ["Calibrate", "Temperature scaling"],
  ["Inconclusive band", "Abstain when unsure"],
  ["Grad-CAM", "Attention heatmap"],
  ["Serve", "ONNX on CPU"],
] as const;

const HEIGHT = 56;
const GAP_X = 32;
const GAP_Y = 40;

export interface Placement {
  x: number;
  y: number;
}

export function placeStages(columns: number, boxWidth: number): Placement[] {
  return STAGES.map((_, i) => {
    const row = Math.floor(i / columns);
    const step = i % columns;
    const column = row % 2 === 0 ? step : columns - 1 - step;
    return { x: column * (boxWidth + GAP_X), y: row * (HEIGHT + GAP_Y) };
  });
}

export function connector(from: Placement, to: Placement, boxWidth: number): string {
  if (from.y === to.y) {
    const forward = to.x > from.x;
    const start = forward ? from.x + boxWidth : from.x;
    const end = forward ? to.x - 4 : to.x + boxWidth + 4;
    return `M${start},${from.y + HEIGHT / 2}H${end}`;
  }
  return `M${from.x + boxWidth / 2},${from.y + HEIGHT}V${to.y - 4}`;
}

function Diagram({ columns, boxWidth, className }: { columns: number; boxWidth: number; className: string }) {
  const titleId = useId();
  const placements = placeStages(columns, boxWidth);
  const rows = Math.ceil(STAGES.length / columns);
  const width = columns * boxWidth + (columns - 1) * GAP_X;
  const height = rows * HEIGHT + (rows - 1) * GAP_Y;
  const markerId = `${titleId}-arrow`;
  return (
    <svg className={`pipeline ${className}`} viewBox={`-2 -2 ${width + 4} ${height + 4}`} role="img" aria-labelledby={titleId}>
      <title id={titleId}>{`Pipeline: ${STAGES.map(([name]) => name).join(", then ")}.`}</title>
      <defs>
        <marker id={markerId} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M0,0L8,4L0,8z" className="pipeline__arrowhead" />
        </marker>
      </defs>
      {placements.slice(1).map((to, i) => (
        <path key={i} className="pipeline__edge" d={connector(placements[i] ?? to, to, boxWidth)} markerEnd={`url(#${markerId})`} />
      ))}
      {STAGES.map(([name, detail], i) => {
        const { x, y } = placements[i] ?? { x: 0, y: 0 };
        return (
          <g key={name} transform={`translate(${x} ${y})`} className={i >= 4 ? "pipeline__stage pipeline__stage--model" : "pipeline__stage"}>
            <rect width={boxWidth} height={HEIGHT} rx="6" />
            <text x="14" y="23" className="pipeline__name">{`${i + 1}. ${name}`}</text>
            <text x="14" y="42" className="pipeline__detail">{detail}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function PipelineDiagram() {
  return (
    <figure className="pipeline-figure">
      <Diagram columns={4} boxWidth={176} className="pipeline--wide" />
      <Diagram columns={1} boxWidth={300} className="pipeline--narrow" />
      <figcaption>From raw images to a served prediction. Steps 1–4 happen once during training; steps 5–8 shape every prediction.</figcaption>
    </figure>
  );
}
