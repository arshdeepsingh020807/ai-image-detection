from dataclasses import dataclass
from typing import Annotated

import anyio
from fastapi import Depends, Request

from backend.app.core.errors import ApiError
from backend.app.core.ratelimit import RateLimiter
from backend.app.core.settings import Settings
from backend.app.services.model import Detector


@dataclass(frozen=True)
class Runtime:
    """Process-wide state built once in the lifespan."""

    settings: Settings
    detector: Detector | None
    rate_limiter: RateLimiter
    cpu_limiter: anyio.CapacityLimiter
    started_at: float


def get_runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


def require_detector(runtime: Annotated[Runtime, Depends(get_runtime)]) -> Detector:
    if runtime.detector is None:
        raise ApiError("model_unavailable", "The model is not loaded; try again later.")
    return runtime.detector


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
DetectorDep = Annotated[Detector, Depends(require_detector)]
