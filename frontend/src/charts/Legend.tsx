import type { Label } from "../api/types";
import { VerdictMark } from "../components/VerdictMark";
import type { Tone } from "./types";

export interface LegendItem {
  label: string;
  tone: Tone;
  dashed?: boolean | undefined;
  mark?: Label;
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="legend">
      {items.map((item) => (
        <li key={item.label}>
          {item.mark ? (
            <VerdictMark label={item.mark} size={12} />
          ) : (
            <svg width="20" height="10" aria-hidden="true">
              <line x1="1" x2="19" y1="5" y2="5" className={`line line--${item.tone}`} strokeDasharray={item.dashed ? "4 3" : undefined} />
            </svg>
          )}
          {item.label}
        </li>
      ))}
    </ul>
  );
}
