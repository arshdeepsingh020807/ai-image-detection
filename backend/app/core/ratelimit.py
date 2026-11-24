import time
from collections import deque
from collections.abc import Callable

from starlette.requests import Request


class RateLimiter:
    """Sliding-window limiter keyed by client. In memory, so it is per process, which suits one small instance."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    def acquire(self, key: str) -> float:
        """Record a request; return 0 if it is allowed, otherwise the seconds until a slot frees up."""
        now = self.clock()
        if len(self._hits) >= self.max_keys:
            self._forget_idle(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return hits[0] + self.window - now
        hits.append(now)
        return 0.0

    def _forget_idle(self, now: float) -> None:
        self._hits = {key: hits for key, hits in self._hits.items() if hits and now - hits[-1] < self.window}


def client_ip(request: Request, trust_proxy_headers: bool, client_ip_header: str = "") -> str:
    """Client address for rate limiting.

    Behind a CDN that sets a dedicated header (Cloudflare's True-Client-IP in front of Render) that header
    names the visitor; the rightmost X-Forwarded-For hop would be the CDN edge, shared by many visitors.
    Otherwise the rightmost hop is the one our proxy appended; earlier hops are client-controlled.
    """
    if trust_proxy_headers:
        if client_ip_header and (dedicated := request.headers.get(client_ip_header, "").strip()):
            return dedicated
        hops = [hop.strip() for hop in request.headers.get("x-forwarded-for", "").split(",") if hop.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"
