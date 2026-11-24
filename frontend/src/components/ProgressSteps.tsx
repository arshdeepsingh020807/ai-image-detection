const STEPS = [
  { phase: "uploading", label: "Upload", status: "Uploading image…" },
  { phase: "analysing", label: "Analyse", status: "Running the model…" },
  { phase: "done", label: "Result", status: "" },
] as const;

export function ProgressSteps({ phase, onCancel }: { phase: "uploading" | "analysing"; onCancel: () => void }) {
  const current = STEPS.findIndex((step) => step.phase === phase);
  return (
    <div className="progress">
      <ol className="progress__steps">
        {STEPS.map((step, i) => (
          <li key={step.phase} data-state={i < current ? "complete" : i === current ? "current" : "pending"} aria-current={i === current ? "step" : undefined}>
            <span className="progress__index" aria-hidden="true">{i + 1}</span>
            {step.label}
            {i < current && <span className="visually-hidden"> (complete)</span>}
          </li>
        ))}
      </ol>
      <p className="progress__status">
        <span className="progress__pulse" aria-hidden="true" />
        {STEPS[current]?.status}
      </p>
      <button type="button" className="button button--secondary" onClick={onCancel}>Cancel</button>
    </div>
  );
}
