# API and artifact contract (single source of truth for backend, frontend and ml)

All routes are under `/api`. JSON only. Errors always use `{"error": {"code": str, "message": str}}` with the right HTTP status.

## POST /api/predict
Multipart form: `file` (required), `explain` ("true"/"false", default "false").
200:
```json
{
  "label": "real" | "ai_generated" | "inconclusive",
  "probabilities": {"real": 0.0-1.0, "ai_generated": 0.0-1.0},   // calibrated, sum to 1
  "confidence": 0.0-1.0,                                        // max of the two probabilities
  "band": {"low": 0.0-1.0, "high": 0.0-1.0},                     // p_ai <= low -> real, >= high -> ai_generated, else inconclusive
  "image": {"width": int, "height": int, "format": "JPEG"|"PNG"|"WEBP", "size_bytes": int},
  "explanation": null | {
    "heatmap_png_base64": str,        // 224x224 RGBA PNG, transparent->warm = more influence on the AI-generated score
    "region": {"x": 0-1, "y": 0-1, "width": 0-1, "height": 0-1},   // part of the ORIGINAL image the model saw (centre crop after resize)
    "method": "grad-cam",
    "note": str                        // fixed disclaimer: model attention, not proof
  },
  "timing_ms": {"decode": f, "preprocess": f, "inference": f, "explain": f, "total": f},
  "model": {"architecture": str, "version": str, "trained": bool}
}
```
Rules: p_ai = sigmoid(logit / temperature); label from `band` in model_card.json. Error codes:
`missing_file` 400, `empty_file` 400, `invalid_image` 400, `unsupported_format` 415, `animated_image` 415,
`image_too_large` 413 (pixels, including decompression bombs), `file_too_large` 413 (bytes), `invalid_request` 400 (malformed multipart body, `explain` not true/false, over-long field), `rate_limited` 429, `model_unavailable` 503, `not_found` 404, `method_not_allowed` 405, `internal_error` 500.
When `low == high`, `p_ai <= low` is `real`, so that band never yields `inconclusive`.

## GET /api/health
200 `{"status": "ok", "model_loaded": true, "model_version": str, "uptime_seconds": f}`;
503 `{"status": "degraded", "model_loaded": false, "model_version": null, "uptime_seconds": f}` when the model failed to load (service still answers).

## GET /api/model
`model_card.json` contents minus `cam_weights`/`cam_bias`, plus `{"onnx_size_mb": f}` (MiB = bytes / 2^20, 3 decimals).

## GET /api/metrics
`{"available": ["summary", ...], "metrics": {"summary": {...}, ...}}` - every `*.json` in METRICS_DIR keyed by file stem.
If a file does not exist it is simply absent; nothing is ever synthesised.

## Static site
Non-`/api` GET requests return files from `frontend/dist`, falling back to `index.html` for client routes. Unknown `/api/*` -> JSON 404.

## artifacts/metrics file shapes (written by ml, rendered by frontend)
- `summary.json`: `{"model": str, "created": iso, "seed": int, "validation": M, "test": M, "test_with_band": B, "temperature": f, "band": {"low": f, "high": f}}`
  where M = output of `classification_metrics` (n, threshold, accuracy, precision, recall, f1, specificity, false_positive_rate, false_negative_rate, roc_auc, pr_auc, log_loss, brier, ece, confusion_matrix `[[tn,fp],[fn,tp]]`)
  and B = `{coverage, accuracy_on_decided, false_positive_rate_on_real, error_rate, inconclusive}`.
- `history.json`: `{"epochs": [{"epoch": int, "phase": "head"|"finetune", "train_loss": f, "val_loss": f, "train_accuracy": f, "val_accuracy": f, "lr": f}]}`
- `confusion.json`: `{"labels": ["real","ai_generated"], "matrix": [[tn,fp],[fn,tp]]}`
- `roc.json`: `{"fpr": [f], "tpr": [f], "auc": f}`; `pr.json`: `{"precision": [f], "recall": [f], "average_precision": f}`
- `confidence.json`: `{"bins": [f x 11 edges 0..1], "real": [int x 10], "ai_generated": [int x 10]}` histogram of p_ai on the test split by true class
- `external.json`: `{"sets": [{"name": str, "protocol": str, "n": int, "metrics": M}], "notes": [str]}`
- `robustness.json`: `{"conditions": [{"name": str, "description": str, "accuracy": f, "roc_auc": f|null, "false_positive_rate": f}]}`
- `dataset_report.json`: see `ml/preprocessing/report.py::build_report`
- `latency.json`: `{"cpu": {"p50_ms": f, "p95_ms": f, "rss_mb": f}, "onnx_size_mb": f, "runs": int, "machine": str}`
- `model_comparison.json`: `{"models": [{"name": str, "parameters_millions": f, "onnx_size_mb": f, "validation_auc": f, "external_auc": f, "false_positive_rate": f, "latency_ms": f}]}`

## ONNX model (artifacts/models/detector.onnx + model_card.json)
Input `image` float32 [batch,3,224,224] (ImageNet-normalised by `detector_core.preprocess.prepare`); outputs `logit` [batch,1] (AI-generated) and `features` [batch,C,7,7].
`model_card.json`: `architecture, version, trained, input{short_side,crop,mean,std}, cam_weights[C], cam_bias, temperature, band{low,high}`, plus training metadata.
Heatmap = ReLU(sum_c cam_weights[c] * features[c]) -> normalise to 0..1 -> upsample to 224 -> colour map -> RGBA. This equals Grad-CAM exactly for this head (proved in ml/tests/test_model.py).
`artifacts/models/` currently holds an UNTRAINED STAND-IN (`trained: false`) for plumbing only; the real model replaces it with the same file names.

## Shared preprocessing
Backend MUST import `detector_core.preprocess` (`decode_image`, `prepare`, `ImageRejected`); never re-implement it.
