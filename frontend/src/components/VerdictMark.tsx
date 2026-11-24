import type { Label } from "../api/types";

const SHAPES: Record<Label, string> = {
  real: "M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2z",
  ai_generated: "M8 2l6.5 11.5h-13z",
  inconclusive: "M8 1.5l6.5 6.5-6.5 6.5L1.5 8zM8 4.3L4.3 8 8 11.7 11.7 8z",
};

export function VerdictMark({ label, size = 16 }: { label: Label; size?: number }) {
  return (
    <svg className={`verdict-mark verdict-mark--${label}`} width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <path d={SHAPES[label]} fillRule="evenodd" />
    </svg>
  );
}
