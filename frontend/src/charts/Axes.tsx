import type { Scale } from "../lib/scale";

interface AxesProps {
  x: Scale;
  y: Scale;
  xTicks: number[];
  yTicks: number[];
  xFormat: (value: number) => string;
  yFormat: (value: number) => string;
  xLabel: string;
  yLabel: string;
  plot: { left: number; top: number; right: number; bottom: number };
}

export function Axes({ x, y, xTicks, yTicks, xFormat, yFormat, xLabel, yLabel, plot }: AxesProps) {
  return (
    <g className="axes" aria-hidden="true">
      {yTicks.map((tick) => (
        <g key={`y${tick}`}>
          <line className="grid" x1={plot.left} x2={plot.right} y1={y(tick)} y2={y(tick)} />
          <text className="tick" x={plot.left - 8} y={y(tick)} dy="0.32em" textAnchor="end">{yFormat(tick)}</text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <text key={`x${tick}`} className="tick" x={x(tick)} y={plot.bottom + 18} textAnchor="middle">{xFormat(tick)}</text>
      ))}
      <line className="baseline" x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
      <text className="axis-label" x={(plot.left + plot.right) / 2} y={plot.bottom + 36} textAnchor="middle">{xLabel}</text>
      <text className="axis-label" transform={`translate(12 ${(plot.top + plot.bottom) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
    </g>
  );
}
