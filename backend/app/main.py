import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.deps import Runtime
from backend.app.api.frontend import SinglePageApp
from backend.app.api.routes import router
from backend.app.core.errors import EXCEPTION_HANDLERS
from backend.app.core.logging import configure_logging
from backend.app.core.middleware import AccessMiddleware, InternalErrorMiddleware
from backend.app.core.ratelimit import RateLimiter
from backend.app.core.settings import Settings
from backend.app.services.model import try_load_detector


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = Runtime(
            settings=settings,
            detector=try_load_detector(settings.model_dir, settings.inference_threads),
            rate_limiter=RateLimiter(settings.rate_limit_per_minute),
            cpu_limiter=anyio.CapacityLimiter(settings.max_concurrent_inferences),
            started_at=time.monotonic(),
        )
        yield

    app = FastAPI(
        title="AI-Generated Image Detection",
        lifespan=lifespan,
        exception_handlers=EXCEPTION_HANDLERS,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(router)
    app.router.default = SinglePageApp(settings.frontend_dist, app.router.default)
    # Added innermost first: errors -> CORS -> access log, so every response gets CORS and security headers.
    app.add_middleware(InternalErrorMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Retry-After"],
    )
    app.add_middleware(AccessMiddleware)
    return app


app = create_app()
