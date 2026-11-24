from pathlib import Path

from starlette.responses import FileResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.app.core.errors import ApiError


class SinglePageApp:
    """Router fallback that serves the built React app.

    It only runs when no route matched at all, so a wrong method on a real `/api` route still
    gets 405 and unknown `/api/*` paths get JSON 404 instead of `index.html`.
    """

    def __init__(self, dist: Path, not_found: ASGIApp):
        self.root = dist.resolve()
        self.index = self.root / "index.html"
        self.not_found = not_found

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and self.index.is_file():
            await self._response_for(scope["path"])(scope, receive, send)
        else:
            await self.not_found(scope, receive, send)

    def _response_for(self, path: str) -> FileResponse:
        if path == "/api" or path.startswith("/api/"):
            raise ApiError("not_found", "Not found.")
        if file := self._existing_file(path):
            return FileResponse(file)
        if Path(path).suffix:  # a missing asset, not a client-side route
            raise ApiError("not_found", "Not found.")
        return FileResponse(self.index, headers={"Cache-Control": "no-cache"})

    def _existing_file(self, path: str) -> Path | None:
        try:
            candidate = (self.root / path.lstrip("/")).resolve()
        except (OSError, ValueError):
            return None
        # resolve() collapses ".." and follows symlinks, so anything outside the build folder is refused.
        return candidate if candidate.is_relative_to(self.root) and candidate.is_file() else None
