import { useCallback, useEffect, useRef, useState } from "react";
import { getModel } from "../api/client";
import { DropZone } from "../components/DropZone";
import { HeatmapControls } from "../components/HeatmapControls";
import { ImageStage } from "../components/ImageStage";
import { LimitationsNote } from "../components/LimitationsNote";
import { Notice } from "../components/Notice";
import { PageHeader } from "../components/PageHeader";
import { ResultPanel } from "../components/ResultPanel";
import { UntrainedNotice } from "../components/UntrainedNotice";
import { useAsync } from "../hooks/useAsync";
import { useDetector, type DetectState } from "../hooks/useDetector";
import { friendlyMessage } from "../api/errors";
import { checkDimensions } from "../lib/fileValidation";
import { formatBytes } from "../lib/format";
import { LABEL_TEXT } from "../lib/verdict";

const loadModel = (signal: AbortSignal) => getModel({ signal });

function announcement(state: DetectState): string {
  switch (state.phase) {
    case "checking": return "Checking file…";
    case "ready": return `Image selected: ${state.source.file.name}. Ready to analyse.`;
    case "uploading": return "Uploading image…";
    case "analysing": return "Upload complete. Running the model…";
    case "done": return `Analysis complete. Verdict: ${LABEL_TEXT[state.result.label]}.`;
    default: return "";
  }
}

export function DetectPage() {
  const { state, select, analyse, stop, reject, reset } = useDetector();
  const model = useAsync(loadModel);
  const [explain, setExplain] = useState(true);
  const [heatmapVisible, setHeatmapVisible] = useState(true);
  const [opacity, setOpacity] = useState(0.9)  // the heatmap's own alpha already fades weak regions; 90% keeps strong ones visible;
  const focusTarget = useRef<HTMLHeadingElement>(null);
  const source = "source" in state ? state.source : null;
  const file = source?.file ?? null;
  const busy = state.phase === "uploading" || state.phase === "analysing";
  const rejection = state.phase === "empty" ? state.rejection : null;
  const result = state.phase === "done" ? state.result : null;
  const trained = result?.model.trained ?? (model.status === "ready" ? model.data.trained : true);
  const version = result?.model.version ?? (model.status === "ready" ? model.data.version : "");

  useEffect(() => {
    if (state.phase === "done" || state.phase === "failed") focusTarget.current?.focus();
  }, [state.phase]);

  useEffect(() => {
    if (rejection) focusTarget.current?.focus();
  }, [rejection]);

  const handleDimensions = useCallback((width: number, height: number) => {
    const error = checkDimensions(width, height);
    if (error) reject(error);
  }, [reject]);

  const handleDecodeError = useCallback(() => {
    reject(friendlyMessage("invalid_image", "The browser could not decode this file, so it is probably damaged."));
  }, [reject]);

  const startAnalysis = () => {
    if (file) void analyse(file, explain);
  };

  return (
    <div className="page detect-page">
      <PageHeader title="Detect AI-generated images" eyebrow="Image analysis">
        <p>
          Upload a photograph and an image classifier estimates how likely it is to be AI-generated. You get a calibrated
          probability, an explicit “inconclusive” when the model is unsure, and a heatmap of the areas that most influenced the score.
        </p>
      </PageHeader>

      {!trained && <UntrainedNotice version={version} />}

      <p className="visually-hidden" role="status" aria-live="polite">{announcement(state)}</p>

      <div className="detect-grid">
        <section className="panel" aria-labelledby="image-title">
          <h2 className="panel-title" id="image-title">Image</h2>
          {source ? (
            <ImageStage
              src={source.url}
              alt={`Preview of ${source.file.name}`}
              explanation={result?.explanation ?? null}
              showHeatmap={heatmapVisible}
              opacity={opacity}
              onDimensions={handleDimensions}
              onDecodeError={handleDecodeError}
            />
          ) : (
            <DropZone onFile={(next) => { void select(next); }} busy={state.phase === "checking"} describedBy={rejection ? "rejection" : undefined} />
          )}

          {rejection && (
            <div id="rejection">
              <Notice tone="danger" role="alert" title={rejection.title} headingRef={focusTarget}>
                <p>{rejection.detail}</p>
              </Notice>
            </div>
          )}

          {file && (
            <p className="file-meta">
              <span className="file-meta__name">{file.name}</span>
              <span>{formatBytes(file.size)}</span>
            </p>
          )}

          {result?.explanation && (
            <HeatmapControls note={result.explanation.note} visible={heatmapVisible} opacity={opacity} onVisibleChange={setHeatmapVisible} onOpacityChange={setOpacity} />
          )}

          {file && (
            <div className="analyse-controls">
              <label className="checkbox">
                <input type="checkbox" checked={explain} disabled={busy} onChange={(event) => { setExplain(event.target.checked); }} />
                <span>
                  Include heatmap
                  <span className="checkbox__hint">Grad-CAM explanation, adds a short extra step</span>
                </span>
              </label>
              <div className="analyse-controls__buttons">
                <button type="button" className="button button--primary" onClick={startAnalysis} disabled={busy}>
                  {state.phase === "done" ? "Analyse again" : "Analyse image"}
                </button>
                <button type="button" className="button button--secondary" onClick={reset} disabled={busy}>Remove image</button>
              </div>
            </div>
          )}
        </section>

        <section className="panel panel--result" aria-label="Analysis result">
          <ResultPanel state={state} headingRef={focusTarget} onRetry={startAnalysis} onReset={reset} onCancel={stop} />
        </section>
      </div>

      <LimitationsNote />
    </div>
  );
}
