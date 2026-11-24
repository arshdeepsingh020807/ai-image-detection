import math
import time
from functools import partial
from typing import Any

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.api.deps import DetectorDep, RuntimeDep
from backend.app.api.upload import read_upload
from backend.app.core.errors import ApiError
from backend.app.core.ratelimit import client_ip
from backend.app.schemas.responses import HealthResponse, MetricsResponse, ModelInfo, PredictResponse, Timing
from backend.app.services.inference import analyse, calibrated_probability, label_for, probabilities
from backend.app.services.metrics import read_metrics
from detector_core.preprocess import ImageRejected

router = APIRouter(prefix="/api")


def _parse_explain(value: str | None) -> bool:
    normalised = (value or "false").strip().lower()
    if normalised not in ("true", "false"):
        raise ApiError("invalid_request", "explain must be 'true' or 'false'.")
    return normalised == "true"


@router.post("/predict")
async def predict(request: Request, runtime: RuntimeDep, detector: DetectorDep) -> PredictResponse:
    started = time.perf_counter()
    settings = runtime.settings
    wait = runtime.rate_limiter.acquire(client_ip(request, settings.trust_proxy_headers, settings.client_ip_header))
    if wait:
        raise ApiError("rate_limited", "Too many requests; slow down.", {"Retry-After": str(math.ceil(wait))})
    form = await read_upload(request, settings.max_upload_bytes)
    if form.file is None:
        raise ApiError("missing_file", "Send the image as the multipart field 'file'.")
    explain = _parse_explain(form.fields.get("explain"))
    job = partial(analyse, detector, form.file, explain=explain, max_pixels=settings.max_image_pixels)
    try:
        result = await anyio.to_thread.run_sync(job, limiter=runtime.cpu_limiter)
    except ImageRejected as exc:
        raise ApiError(exc.code, str(exc)) from exc
    card = detector.card
    p_ai = calibrated_probability(result.logit, card.temperature)
    total = round((time.perf_counter() - started) * 1000, 3)
    return PredictResponse(
        label=label_for(p_ai, card.band),
        probabilities=probabilities(p_ai),
        confidence=max(p_ai, 1.0 - p_ai),
        band=card.band,
        image=result.image,
        explanation=result.explanation,
        timing_ms=Timing(**result.timing_ms, total=total),
        model=ModelInfo(architecture=card.architecture, version=card.version, trained=card.trained),
    )


@router.get("/health", response_model=HealthResponse)
async def health(runtime: RuntimeDep) -> JSONResponse:
    loaded = runtime.detector is not None
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=runtime.detector.card.version if runtime.detector else None,
        uptime_seconds=round(time.monotonic() - runtime.started_at, 3),
    )
    return JSONResponse(body.model_dump(), status_code=200 if loaded else 503)


@router.get("/model")
async def model_card(detector: DetectorDep) -> dict[str, Any]:
    return detector.public_card


@router.get("/metrics")
def metrics(runtime: RuntimeDep) -> MetricsResponse:
    return MetricsResponse.model_validate(read_metrics(runtime.settings.metrics_dir))
