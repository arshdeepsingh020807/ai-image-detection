import type { ReactNode } from "react";
import type { FileState } from "../../lib/metrics";

interface SectionProps {
  id: string;
  title: string;
  intro?: ReactNode;
  children: ReactNode;
}

export function Section({ id, title, intro, children }: SectionProps) {
  return (
    <section className="report-section" id={id} aria-labelledby={`${id}-title`}>
      <div className="report-section__header">
        <h2 id={`${id}-title`}>{title}</h2>
        {intro && <div className="report-section__intro">{intro}</div>}
      </div>
      {children}
    </section>
  );
}

export function EmptyState({ source, invalid = false }: { source: string; invalid?: boolean }) {
  return (
    <div className="empty-state" data-kind={invalid ? "invalid" : "missing"}>
      <p className="empty-state__title">{invalid ? "Results file could not be read" : "No evaluation results published yet"}</p>
      <p className="empty-state__body">
        {invalid
          ? <>The published <code>{source}</code> does not match the expected format, so nothing from it is shown.</>
          : <>This appears once the evaluation pipeline publishes <code>{source}</code>.</>}
      </p>
    </div>
  );
}

export function FromFile<T>({ state, source, children }: { state: FileState<T>; source: string; children: (data: T) => ReactNode }) {
  return state.status === "ok" ? children(state.data) : <EmptyState source={source} invalid={state.status === "invalid"} />;
}
