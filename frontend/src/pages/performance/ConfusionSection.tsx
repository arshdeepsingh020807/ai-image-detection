import { ConfusionMatrix } from "../../charts/ConfusionMatrix";
import type { Confusion } from "../../api/types";
import { formatCount } from "../../lib/format";
import type { FileState } from "../../lib/metrics";
import { FromFile, Section } from "./Section";

export function ConfusionSection({ state }: { state: FileState<Confusion> }) {
  return (
    <Section id="confusion" title="Confusion matrix" intro={<p>How test images were classified at the default threshold. Rows are the true class, columns the prediction.</p>}>
      <FromFile state={state} source="confusion.json">
        {({ matrix }) => (
          <ConfusionMatrix matrix={matrix} caption={`${formatCount(matrix.flat().reduce((a, b) => a + b, 0))} images`} />
        )}
      </FromFile>
    </Section>
  );
}
