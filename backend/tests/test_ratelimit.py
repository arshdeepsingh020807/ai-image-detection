import pytest
from hypothesis import given
from hypothesis import strategies as st
from starlette.requests import Request

from backend.app.core.ratelimit import RateLimiter, client_ip
from backend.tests.helpers import app_of, assert_error, encode, noise_image, upload


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_reports_wait():
    clock = FakeClock()
    limiter = RateLimiter(3, window_seconds=60, clock=clock)
    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    clock.now += 10
    assert limiter.acquire("a") == pytest.approx(50.0)  # first hit at t=1000 frees at t=1060


def test_window_slides_one_slot_at_a_time():
    clock = FakeClock()
    limiter = RateLimiter(2, window_seconds=60, clock=clock)
    limiter.acquire("a")
    clock.now += 30
    limiter.acquire("a")
    clock.now += 29.9
    assert limiter.acquire("a") == pytest.approx(0.1)
    clock.now += 0.1  # exactly 60 s after the first hit: it has left the window
    assert limiter.acquire("a") == 0.0
    assert limiter.acquire("a") == pytest.approx(30.0)


def test_rejected_requests_do_not_extend_the_block():
    clock = FakeClock()
    limiter = RateLimiter(1, window_seconds=60, clock=clock)
    limiter.acquire("a")
    for _ in range(5):
        clock.now += 10
        limiter.acquire("a")
    clock.now += 10
    assert limiter.acquire("a") == 0.0


def test_keys_are_independent():
    limiter = RateLimiter(1, clock=FakeClock())
    assert limiter.acquire("a") == 0.0 and limiter.acquire("b") == 0.0
    assert limiter.acquire("a") > 0


def test_idle_keys_are_forgotten_when_the_table_is_full():
    clock = FakeClock()
    limiter = RateLimiter(1, window_seconds=60, clock=clock, max_keys=3)
    for key in "abc":
        limiter.acquire(key)
    clock.now += 61
    limiter.acquire("d")
    assert set(limiter._hits) == {"d"}


def test_busy_keys_survive_pruning():
    clock = FakeClock()
    limiter = RateLimiter(1, window_seconds=60, clock=clock, max_keys=2)
    limiter.acquire("a")
    limiter.acquire("b")
    clock.now += 30
    limiter.acquire("c")
    assert set(limiter._hits) == {"a", "b", "c"} and limiter.acquire("a") > 0


@given(st.integers(1, 20), st.lists(st.floats(0, 200), max_size=60))
def test_never_more_than_limit_in_any_window(limit, gaps):
    clock = FakeClock()
    limiter = RateLimiter(limit, window_seconds=60, clock=clock)
    accepted: list[float] = []
    for gap in gaps:
        clock.now += gap
        if limiter.acquire("k") == 0.0:
            accepted.append(clock.now)
    for i, start in enumerate(accepted):
        assert sum(1 for t in accepted[i:] if t - start < 60) <= limit


# ---------- client_ip ----------


def make_request(headers: dict[str, str], client: tuple[str, int] | None = ("10.0.0.1", 1234)) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "client": client})


@pytest.mark.parametrize(
    "trust, forwarded, expected",
    [
        (False, "203.0.113.9", "10.0.0.1"),
        (True, "203.0.113.9", "203.0.113.9"),
        (True, "6.6.6.6, 203.0.113.9", "203.0.113.9"),  # client-supplied prefix is ignored
        (True, " 198.51.100.2 , ", "198.51.100.2"),
        (True, " , ", "10.0.0.1"),
    ],
)
def test_client_ip(trust, forwarded, expected):
    assert client_ip(make_request({"X-Forwarded-For": forwarded}), trust) == expected


@pytest.mark.parametrize(
    "trust, header_name, headers, expected",
    [
        (True, "true-client-ip", {"True-Client-IP": "203.0.113.9", "X-Forwarded-For": "203.0.113.9, 172.64.0.1"},
         "203.0.113.9"),
        (True, "true-client-ip", {"X-Forwarded-For": "198.51.100.7, 172.64.0.1"}, "172.64.0.1"),  # falls back
        (True, "true-client-ip", {"True-Client-IP": "  ", "X-Forwarded-For": "198.51.100.7"}, "198.51.100.7"),
        (False, "true-client-ip", {"True-Client-IP": "203.0.113.9"}, "10.0.0.1"),  # untrusted: header ignored
        (True, "", {"True-Client-IP": "203.0.113.9", "X-Forwarded-For": "198.51.100.7"}, "198.51.100.7"),
    ],
)
def test_dedicated_client_ip_header(trust, header_name, headers, expected):
    assert client_ip(make_request(headers), trust, header_name) == expected


def test_client_ip_without_header_or_peer():
    assert client_ip(make_request({}), True) == "10.0.0.1"
    assert client_ip(make_request({}, client=None), False) == "unknown"


# ---------- /api/predict integration ----------


def test_predict_returns_429_then_recovers(make_client):
    client = make_client(rate_limit_per_minute=2)
    clock = FakeClock()
    app_of(client).state.runtime.rate_limiter.clock = clock
    data = encode(noise_image(40, 40), "PNG")
    assert [upload(client, data).status_code for _ in range(2)] == [200, 200]
    clock.now += 20.2
    blocked = upload(client, data)
    assert_error(blocked, 429, "rate_limited")
    assert blocked.headers["retry-after"] == "40"  # ceil(39.8)
    clock.now += 40
    assert upload(client, data).status_code == 200


def test_rate_limit_counts_failed_requests_and_spares_other_routes(make_client):
    client = make_client(rate_limit_per_minute=1)
    assert_error(client.post("/api/predict"), 400, "missing_file")
    assert_error(client.post("/api/predict"), 429, "rate_limited")
    assert client.get("/api/health").status_code == 200


def test_forwarded_clients_get_separate_buckets_only_when_trusted(make_client):
    trusted = make_client(rate_limit_per_minute=1, trust_proxy_headers=True)
    assert trusted.post("/api/predict", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 400
    assert trusted.post("/api/predict", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 400
    assert trusted.post("/api/predict", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    untrusted = make_client(rate_limit_per_minute=1)
    assert untrusted.post("/api/predict", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 400
    assert untrusted.post("/api/predict", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_visitors_behind_one_cdn_edge_get_separate_buckets_via_dedicated_header(make_client):
    client = make_client(rate_limit_per_minute=1, trust_proxy_headers=True, client_ip_header="true-client-ip")
    edge = "203.0.113.1, 172.64.0.1"  # same CDN edge (rightmost hop) for both visitors

    def post(visitor: str) -> int:
        return client.post("/api/predict", headers={"X-Forwarded-For": edge, "True-Client-IP": visitor}).status_code

    assert [post("198.51.100.1"), post("198.51.100.2"), post("198.51.100.2")] == [400, 400, 429]
