import type { CSSProperties } from "react";
import type { Matrix2x2 } from "../api/types";
import { formatCount, formatPercent } from "../lib/format";

const CLASSES = ["Real", "AI-generated"] as const;
const GROUPS = ["real images", "AI-generated images"] as const;
const OUTCOME = [["True negative", "False positive"], ["False negative", "True positive"]] as const;

export function ConfusionMatrix({ matrix, caption }: { matrix: Matrix2x2; caption: string }) {
  return (
    <table className="confusion">
      <caption>{caption}</caption>
      <thead>
        <tr>
          <td className="confusion__corner" />
          <th scope="colgroup" colSpan={2} className="confusion__axis">Predicted</th>
        </tr>
        <tr>
          <th scope="col" className="confusion__axis">Actual</th>
          {CLASSES.map((name) => <th key={name} scope="col">{name}</th>)}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, actual) => {
          const total = row[0] + row[1];
          const name = CLASSES[actual] ?? "";
          return (
            <tr key={name}>
              <th scope="row">{name}</th>
              {row.map((count, predicted) => {
                const share = total ? count / total : 0;
                return (
                  <td key={predicted} className="confusion__cell" data-correct={actual === predicted || undefined} style={{ "--share": share } as CSSProperties}>
                    <span className="confusion__count num">{formatCount(count)}</span>
                    <span className="confusion__share num">{formatPercent(share)} of {GROUPS[actual]}</span>
                    <span className="confusion__outcome">{OUTCOME[actual]?.[predicted]}</span>
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
