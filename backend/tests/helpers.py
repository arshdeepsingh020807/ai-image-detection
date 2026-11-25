import io
import json
import math
import struct
import zlib
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image
from starlette.testclient import TestClient

from backend.app.core.settings import REPO_ROOT, Settings
from detector_core.preprocess import decode_image, prepare

MODEL_DIR = REPO_ROOT / "artifacts" / "models"
CARD = json.loads((MODEL_DIR / "model_card.json").read_text())


def encode(img: Image.Image, fmt: str, **kwargs: Any) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def noise_image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "model_dir": MODEL_DIR,
        "metrics_dir": tmp_path / "metrics",
        "frontend_dist": tmp_path / "dist",
        "rate_limit_per_minute": 1_000_000,
        "allowed_origins": "http://allowed.example",
        **overrides,
    }
    return Settings(**values)


def upload(client: TestClient, data: bytes, filename: str = "image.png", **fields: str) -> Any:
    return client.post("/api/predict", files={"file": (filename, data, "application/octet-stream")}, data=fields)


def assert_error(response: Any, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


class Reference:
    """Independent oracle: its own onnxruntime session, never the service's."""

    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # the service's default; other counts change float summation order
        self.session = ort.InferenceSession(str(MODEL_DIR / "detector.onnx"), options, providers=["CPUExecutionProvider"])

    def outputs(self, data: bytes) -> tuple[float, np.ndarray]:
        tensor = prepare(decode_image(data))[None]
        logit, features = self.session.run(None, {"image": tensor})
        return float(logit[0, 0]), features[0]

    def p_ai(self, data: bytes, temperature: float = CARD["temperature"]) -> float:
        return 1 / (1 + math.exp(-self.outputs(data)[0] / temperature))


def app_of(client: TestClient) -> FastAPI:
    app = client.app
    assert isinstance(app, FastAPI)
    return app


def declared_size_png(width: int, height: int) -> bytes:
    """A few-hundred-byte 1-bit PNG whose header claims `width` x `height` pixels (decompression-bomb shape)."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    header = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + chunk(b"IEND", b"")
    )


@dataclass
class Exchange:
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    chunks_read: int = 0

    def json(self) -> Any:
        return json.loads(self.body)


async def call(
    app: Any,
    chunks: Iterable[bytes],
    headers: dict[str, str] | None = None,
    path: str = "/api/predict",
    method: str = "POST",
) -> Exchange:
    """Drive the ASGI app by hand and count how many body chunks it actually pulled."""
    pending = list(chunks)
    exchange = Exchange()
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 5000),
        "server": ("testserver", 80),
        "root_path": "",
    }

    async def receive():
        if exchange.chunks_read < len(pending):
            exchange.chunks_read += 1
            return {
                "type": "http.request",
                "body": pending[exchange.chunks_read - 1],
                "more_body": exchange.chunks_read < len(pending),
            }
        await anyio.sleep_forever()  # like a real server: the client stays connected until the response ends

    async def send(message):
        if message["type"] == "http.response.start":
            exchange.status = message["status"]
            exchange.headers = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            exchange.body += message.get("body", b"")

    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
    return exchange


@asynccontextmanager
async def started_client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """httpx's ASGITransport does not run the lifespan, so start it around the client."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client
