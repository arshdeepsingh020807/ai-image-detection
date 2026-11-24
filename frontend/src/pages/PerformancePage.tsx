import { getMetrics, getModel } from "../api/client";
import { Notice } from "../components/Notice";
import { PageHeader } from "../components/PageHeader";
import type { ModelCard } from "../api/types";
import { useAsync, type AsyncState } from "../hooks/useAsync";
import { readAllMetrics, type MetricStates } from "../lib/metrics";
import { ConfidenceSection } from "./performance/ConfidenceSection";
import { ConfusionSection } from "./performance/ConfusionSection";
import { CurvesSection } from "./performance/CurvesSection";
import { DatasetSection } from "./performance/DatasetSection";
import { HeadlineSection } from "./performance/HeadlineSection";
import { HistorySection } from "./performance/HistorySection";
import { LatencySection, ModelCardSection } from "./performance/ModelSections";
import { ComparisonSection, ExternalSection, RobustnessSection } from "./performance/TablesSections";

const loadMetrics = (signal: AbortSignal) => getMetrics({ signal });
const loadModel = (signal: AbortSignal) => getModel({ signal });

const CONTENTS = [
  ["headline", "Headline metrics"],
  ["confusion", "Confusion matrix"],
  ["curves", "ROC and PR curves"],
  ["confidence", "Confidence"],
  ["history", "Training history"],
  ["external", "External sets"],
  ["robustness", "Robustness"],
  ["dataset", "Dataset"],
  ["comparison", "Model comparison"],
  ["latency", "Inference cost"],
  ["model-card", "Model card"],
] as const;

function Report({ files, model }: { files: MetricStates; model: AsyncState<ModelCard> & { reload: () => void } }) {
  const summary = files.summary.status === "ok" ? files.summary.data : null;
  return (
    <>
      <HeadlineSection state={files.summary} />
      <ConfusionSection state={files.confusion} />
      <CurvesSection roc={files.roc} pr={files.pr} />
      <ConfidenceSection state={files.confidence} band={summary?.band ?? null} />
      <HistorySection state={files.history} />
      <ExternalSection state={files.external} />
      <RobustnessSection state={files.robustness} />
      <DatasetSection state={files.dataset_report} />
      <ComparisonSection state={files.model_comparison} deployed={model.status === "ready" ? model.data.architecture : null} />
      <LatencySection state={files.latency} />
      <ModelCardSection model={model} />
    </>
  );
}

export function PerformancePage() {
  const metrics = useAsync(loadMetrics);
  const model = useAsync(loadModel);
  const files = metrics.status === "ready" ? readAllMetrics(metrics.data) : null;
  const published = files ? Object.values(files).filter((file) => file.status === "ok").length : 0;

  return (
    <div className="page performance-page">
      <PageHeader title="Model performance" eyebrow="Evaluation">
        <p>
          Every figure on this page is read from evaluation files published by the training pipeline. Nothing is estimated or filled
          in: a section without a published file says so.
        </p>
      </PageHeader>

      <div className="report-layout">
        <nav className="report-contents" aria-label="On this page">
          <p className="report-contents__title">On this page</p>
          <ul>{CONTENTS.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}</ul>
        </nav>
        <div className="report-body">
          {metrics.status === "loading" && <p className="report-meta" role="status">Loading evaluation results…</p>}
          {metrics.status === "error" && (
            <Notice tone="danger" role="alert" level={2} title={metrics.error.title} actions={<button type="button" className="button button--secondary" onClick={metrics.reload}>Try again</button>}>
              <p>{metrics.error.detail}</p>
            </Notice>
          )}
          {files && (
            <>
              {published === 0 && (
                <Notice tone="info" level={2} title="No evaluation results published yet">
                  <p>The model has not been evaluated, or its results have not been published. Each section below will fill in as its file appears.</p>
                </Notice>
              )}
              <Report files={files} model={model} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
