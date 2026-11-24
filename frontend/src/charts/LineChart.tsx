import { useId, useState, type PointerEvent } from "react";
import { useElementWidth } from "../hooks/useElementWidth";
import { linearScale, linePath, nearestIndex, ticks, type Domain } from "../lib/scale";
import { Axes } from "./Axes";
import { Legend } from "./Legend";
import type { Series } from "./types";

export const MARGIN = { top: 12, right: 16, bottom: 48, left: 56 };

interface LineChartProps {
  title: string;
  description: string;
  series: Series[];
  xDomain: Domain;
  yDomain: Domain;
  xLabel: string;
  yLabel: string;
  xFormat: (value: number) => string;
  yFormat: (value: number) => string;
  valueFormat?: (value: number) => string;
  height?: number;
  square?: boolean;
  diagonal?: boolean;
  xTickCount?: number;
}

export function LineChart(props: LineChartProps) {
  const { title, description, series, xDomain, yDomain, xLabel, yLabel, xFormat, yFormat, diagonal = false, xTickCount = 5 } = props;
  const valueFormat = props.valueFormat ?? yFormat;
  const [container, width] = useElementWidth<HTMLDivElement>(560);
  const [hover, setHover] = useState<number | null>(null);
  const descriptionId = useId();
  const plotWidth = Math.max(120, width - MARGIN.left - MARGIN.right);
  const plotHeight = props.square ? Math.min(plotWidth, 320) : (props.height ?? 220);
  const plot = { left: MARGIN.left, top: MARGIN.top, right: MARGIN.left + plotWidth, bottom: MARGIN.top + plotHeight };
  const x = linearScale(xDomain, [plot.left, plot.right]);
  const y = linearScale(yDomain, [plot.bottom, plot.top]);
  const svgWidth = props.square ? plot.right + MARGIN.right : width;

  const handleMove = (event: PointerEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    setHover(x.invert(plot.left + event.clientX - bounds.left));
  };
  const hovered = hover === null ? [] : series.map((s) => {
    const point = s.points[nearestIndex(s.points.map(([px]) => px), hover)];
    return { series: s, point };
  });
  const anchor = hovered[0]?.point;

  return (
    <figure className="chart" ref={container}>
      {series.length > 1 && <Legend items={series.map((s) => ({ label: s.label, tone: s.tone, dashed: s.dashed }))} />}
      <div className="chart__canvas">
        <svg width={svgWidth} height={plot.bottom + MARGIN.bottom} role="img" aria-label={title} aria-describedby={descriptionId}>
          <Axes x={x} y={y} xTicks={ticks(xDomain, xTickCount)} yTicks={ticks(yDomain, 4)} xFormat={xFormat} yFormat={yFormat} xLabel={xLabel} yLabel={yLabel} plot={plot} />
          {diagonal && <line className="reference" x1={x(xDomain[0])} y1={y(yDomain[0])} x2={x(xDomain[1])} y2={y(yDomain[1])} />}
          {series.map((s) => (
            <path key={s.id} className={`line line--${s.tone}`} d={linePath(s.points, x, y)} strokeDasharray={s.dashed ? "6 4" : undefined} />
          ))}
          {anchor && (
            <g className="hover" aria-hidden="true">
              <line className="crosshair" x1={x(anchor[0])} x2={x(anchor[0])} y1={plot.top} y2={plot.bottom} />
              {hovered.map(({ series: s, point }) => point && (
                <circle key={s.id} className={`dot dot--${s.tone}`} cx={x(point[0])} cy={y(point[1])} r={4} />
              ))}
            </g>
          )}
          <rect
            className="hit-area"
            x={plot.left}
            y={plot.top}
            width={plotWidth}
            height={plotHeight}
            onPointerMove={handleMove}
            onPointerLeave={() => { setHover(null); }}
          />
        </svg>
        {anchor && (
          <div
            className="tooltip"
            style={{ left: Math.min(x(anchor[0]) + 12, svgWidth - 168), top: plot.top + 4 }}
            aria-hidden="true"
          >
            <div className="tooltip__title">{xLabel}: {xFormat(anchor[0])}</div>
            {hovered.map(({ series: s, point }) => point && (
              <div key={s.id} className="tooltip__row"><span>{s.label}</span><span className="num">{valueFormat(point[1])}</span></div>
            ))}
          </div>
        )}
      </div>
      <figcaption id={descriptionId} className="chart__caption">{description}</figcaption>
    </figure>
  );
}
