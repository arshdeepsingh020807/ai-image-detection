import { useId } from "react";
import { useElementWidth } from "../hooks/useElementWidth";
import { formatCount } from "../lib/format";
import { groupedBar, linearScale, niceDomain, ticks } from "../lib/scale";
import { Axes } from "./Axes";
import { Legend } from "./Legend";
import { MARGIN } from "./LineChart";
import type { Tone } from "./types";
import type { Label } from "../api/types";

export interface HistogramSeries {
  label: string;
  tone: Tone;
  mark: Label;
  counts: readonly number[];
}

interface HistogramProps {
  title: string;
  description: string;
  edges: readonly number[];
  series: HistogramSeries[];
  markers?: { value: number; label: string }[];
  xLabel: string;
  xFormat: (value: number) => string;
  height?: number;
}

export function Histogram({ title, description, edges, series, markers = [], xLabel, xFormat, height = 220 }: HistogramProps) {
  const [container, width] = useElementWidth<HTMLDivElement>(560);
  const descriptionId = useId();
  const plot = { left: MARGIN.left, top: MARGIN.top + 14, right: width - MARGIN.right, bottom: MARGIN.top + 14 + height };
  const xDomain = [edges[0] ?? 0, edges[edges.length - 1] ?? 1] as const;
  const yDomain = niceDomain(0, Math.max(1, ...series.flatMap((s) => s.counts)), 4);
  const x = linearScale(xDomain, [plot.left, plot.right]);
  const y = linearScale(yDomain, [plot.bottom, plot.top]);
  const gap = (plot.right - plot.left) / (edges.length - 1) > 24 ? 2 : 1;

  return (
    <figure className="chart" ref={container}>
      <Legend items={series.map((s) => ({ label: s.label, tone: s.tone, mark: s.mark }))} />
      <div className="chart__canvas">
        <svg width={width} height={plot.bottom + MARGIN.bottom} role="img" aria-label={title} aria-describedby={descriptionId}>
          <Axes x={x} y={y} xTicks={ticks(xDomain, 5)} yTicks={ticks(yDomain, 4)} xFormat={xFormat} yFormat={formatCount} xLabel={xLabel} yLabel="Images" plot={plot} />
          {series.map((s, slot) =>
            s.counts.map((count, bin) => {
              const bar = groupedBar(x(edges[bin] ?? 0), x(edges[bin + 1] ?? 0), slot, series.length, gap);
              return (
                <rect key={`${s.label}-${bin}`} className={`bar bar--${s.tone}`} x={bar.x} width={bar.width} y={y(count)} height={plot.bottom - y(count)}>
                  <title>{`${s.label}, ${xFormat(edges[bin] ?? 0)}–${xFormat(edges[bin + 1] ?? 0)}: ${formatCount(count)}`}</title>
                </rect>
              );
            }),
          )}
          {markers.map((marker) => (
            <g key={marker.label} className="marker" aria-hidden="true">
              <line x1={x(marker.value)} x2={x(marker.value)} y1={plot.top - 10} y2={plot.bottom} />
              <text x={x(marker.value)} y={plot.top - 14} textAnchor="middle">{marker.label}</text>
            </g>
          ))}
        </svg>
      </div>
      <figcaption id={descriptionId} className="chart__caption">{description}</figcaption>
    </figure>
  );
}
