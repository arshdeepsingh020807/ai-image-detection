import asyncio
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anyio
import httpx
import pytest

from backend.app.core.settings import REPO_ROOT
from backend.app.main import create_app
from backend.app.services import inference
from backend.tests.helpers import Reference, encode, make_settings, noise_image, started_client


def distinct_images(count: int) -> list[bytes]:
    """Different sizes, content and formats, so a mixed-up response cannot pass by accident."""
    formats = ["PNG", "JPEG", "WEBP"]
    return [encode(noise_image(40 + 7 * i, 30 + 5 * i, seed=i), formats[i % 3]) for i in range(count)]


def assert_matches(body: dict, data: bytes, index: int, reference: Reference) -> None:
    assert (body["image"]["width"], body["image"]["height"]) == (40 + 7 * index, 30 + 5 * index)
    assert body["image"]["size_bytes"] == len(data)
    assert body["probabilities"]["ai_generated"] == pytest.approx(reference.p_ai(data), abs=1e-6)


@pytest.mark.anyio
async def test_parallel_requests_are_answered_correctly(tmp_path, reference: Reference):
    app = create_app(make_settings(tmp_path, max_concurrent_inferences=2))
    images = distinct_images(16)
    async with started_client(app) as client:
        responses = await asyncio.gather(
            *(client.post("/api/predict", files={"file": ("x", data)}, data={"explain": "true"}) for data in images)
        )
    for index, (response, data) in enumerate(zip(responses, images, strict=True)):
        assert response.status_code == 200, response.text
        assert_matches(response.json(), data, index, reference)


@pytest.mark.anyio
async def test_cpu_work_never_exceeds_the_concurrency_cap(tmp_path, monkeypatch):
    active = peak = 0
    lock = threading.Lock()
    real_analyse = inference.analyse

    def tracked(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        try:
            return real_analyse(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("backend.app.api.routes.analyse", tracked)
    app = create_app(make_settings(tmp_path, max_concurrent_inferences=2))
    data = encode(noise_image(32, 32), "PNG")
    async with started_client(app) as client:
        responses = await asyncio.gather(*(client.post("/api/predict", files={"file": ("x", data)}) for _ in range(10)))
    assert all(r.status_code == 200 for r in responses)
    assert peak == 2


@pytest.mark.anyio
async def test_event_loop_stays_responsive_during_inference(tmp_path, monkeypatch):
    """Health must answer while a slow prediction occupies the worker thread."""
    started = threading.Event()
    real_analyse = inference.analyse

    def slow(*args, **kwargs):
        started.set()
        time.sleep(0.5)
        return real_analyse(*args, **kwargs)

    monkeypatch.setattr("backend.app.api.routes.analyse", slow)
    app = create_app(make_settings(tmp_path))
    async with started_client(app) as client:
        predict = asyncio.ensure_future(
            client.post("/api/predict", files={"file": ("x", encode(noise_image(20, 20), "PNG"))})
        )
        while not started.is_set():
            await anyio.sleep(0.005)
        began = time.perf_counter()
        health = await client.get("/api/health")
        assert health.status_code == 200 and time.perf_counter() - began < 0.25
        assert not predict.done()
        assert (await predict).status_code == 200


# ---------- real server ----------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """The production start command (`uvicorn backend.app.main:app`) run from the repo root."""
    port = free_port()
    log_path = tmp_path_factory.mktemp("server") / "server.log"
    env = {**os.environ, "RATE_LIMIT_PER_MINUTE": "1000", "MAX_CONCURRENT_INFERENCES": "2"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, log_path.read_text()
            try:
                if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "server did not become healthy"
            time.sleep(0.2)
        yield base, log_path
    finally:
        process.terminate()
        process.wait(timeout=20)


def test_live_server_parallel_predictions(live_server, reference: Reference):
    base, _ = live_server
    images = distinct_images(12)

    def post(data: bytes) -> httpx.Response:
        return httpx.post(f"{base}/api/predict", files={"file": ("x", data)}, timeout=60)

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(post, images))
    for index, (response, data) in enumerate(zip(responses, images, strict=True)):
        assert response.status_code == 200, response.text
        assert_matches(response.json(), data, index, reference)
        assert response.headers["x-frame-options"] == "DENY"


def test_live_server_cuts_off_oversized_uploads(live_server):
    base, _ = live_server
    response = httpx.post(f"{base}/api/predict", files={"file": ("big", b"\x00" * (11 * 1024 * 1024))}, timeout=60)
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"


def test_live_server_writes_one_json_log_line_per_request(live_server):
    base, log_path = live_server
    httpx.get(f"{base}/api/health?marker=1", timeout=5)
    time.sleep(0.2)
    lines = [line for line in log_path.read_text().splitlines() if '"path": "/api/health"' in line]
    assert lines and all('"status": 200' in line and '"duration_ms"' in line for line in lines)
    assert "marker" not in log_path.read_text()


# ---------- memory ----------

MEMORY_SCRIPT = textwrap.dedent(
    """
    import io, resource, sys
    import numpy as np
    from PIL import Image
    from starlette.testclient import TestClient
    from backend.app.main import create_app
    from backend.app.core.settings import Settings

    def peak_mb():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux

    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)).save(buffer, "JPEG")
    data = buffer.getvalue()
    with TestClient(create_app(Settings(rate_limit_per_minute=10**6))) as client:
        for _ in range(20):
            assert client.post("/api/predict", files={"file": ("x", data)}, data={"explain": "true"}).status_code == 200
        before = peak_mb()
        for _ in range(200):
            assert client.post("/api/predict", files={"file": ("x", data)}, data={"explain": "true"}).status_code == 200
        print(before, peak_mb())
    """
)


def test_memory_stays_bounded_over_many_predictions():
    """Fresh interpreter, so the peak-RSS high-water mark reflects only this workload."""
    result = subprocess.run(
        [sys.executable, "-c", MEMORY_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    before, after = map(float, result.stdout.split()[-2:])
    assert after - before < 40, f"peak RSS grew from {before:.1f} MB to {after:.1f} MB"
    assert after < 400
