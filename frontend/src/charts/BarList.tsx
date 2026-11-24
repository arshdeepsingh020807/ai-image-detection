import { formatCount } from "../lib/format";
import type { Tone } from "./types";

export interface BarSegment {
  label: string;
  value: number;
  tone: Tone;
}

export interface BarRow {
  label: string;
  segments: BarSegment[];
  note?: string;
}

export function BarList({ rows, label }: { rows: BarRow[]; label: string }) {
  const max = Math.max(1, ...rows.map((row) => row.segments.reduce((sum, s) => sum + s.value, 0)));
  return (
    <ul className="bar-list" aria-label={label}>
      {rows.map((row) => {
        const total = row.segments.reduce((sum, s) => sum + s.value, 0);
        let offset = 0;
        return (
          <li key={row.label} className="bar-list__row">
            <span className="bar-list__label">
              {row.label}
              {row.note && <span className="bar-list__note">{row.note}</span>}
            </span>
            <svg className="bar-list__bar" width="100%" height="12" aria-hidden="true">
              {row.segments.map((segment) => {
                const x = (offset / max) * 100;
                offset += segment.value;
                return (
                  <rect key={segment.label} className={`bar bar--${segment.tone} bar--stacked`} x={`${x}%`} width={`${(segment.value / max) * 100}%`} height="12">
                    <title>{`${segment.label}: ${formatCount(segment.value)}`}</title>
                  </rect>
                );
              })}
            </svg>
            <span className="bar-list__value num">
              {row.segments.length > 1 ? row.segments.map((s) => formatCount(s.value)).join(" / ") : formatCount(total)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
