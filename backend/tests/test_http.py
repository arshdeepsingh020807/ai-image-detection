import json
import logging
import sys
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from backend.app.core.errors import STATUS_BY_CODE, ApiError
from backend.app.core.logging import JsonFormatter, configure_logging
from backend.app.core.middleware import CONTENT_SECURITY_POLICY, AccessMiddleware
from backend.app.main import create_app
from backend.tests.helpers import assert_error, encode, make_settings, noise_image, upload

EXPECTED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "content-security-policy": CONTENT_SECURITY_POLICY,
}


def assert_security_headers(response) -> None:
    for name, value in EXPECTED_SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, name


@pytest.fixture
def probe_client(tmp_path) -> Iterator[TestClient]:
    """The real app plus routes that fail in the ways production code could."""
    app = create_app(make_settings(tmp_path))

    async def crash() -> None:
        raise RuntimeError("secret detail at /srv/app/backend/x.py")

    async def typed(n: int) -> dict[str, int]:
        return {"n": n}

    async def teapot() -> None:
        raise HTTPException(418)

    async def upstream() -> None:
        raise HTTPException(502)

    for path, endpoint in [("/crash", crash), ("/typed", typed), ("/teapot", teapot), ("/upstream", upstream)]:
        app.add_api_route(f"/api/test{path}", endpoint, methods=["GET"])
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------- uniform errors ----------


def test_api_error_status_comes_from_the_contract_table():
    assert ApiError("file_too_large", "x").status == 413 and ApiError("rate_limited", "x", {"A": "b"}).headers == {
        "A": "b"
    }
    assert {code: STATUS_BY_CODE[code] for code in ("missing_file", "unsupported_format", "model_unavailable")} == {
        "missing_file": 400,
        "unsupported_format": 415,
        "model_unavailable": 503,
    }
    with pytest.raises(KeyError):
        ApiError("made_up", "x")


def test_unexpected_exception_is_a_uniform_500_without_internals(probe_client, caplog):
    with caplog.at_level(logging.ERROR, logger="backend"):
        response = probe_client.get("/api/test/crash", headers={"Origin": "http://allowed.example"})
    assert_error(response, 500, "internal_error")
    assert "secret" not in response.text and "Traceback" not in response.text and "/srv" not in response.text
    assert_security_headers(response)
    assert response.headers["access-control-allow-origin"] == "http://allowed.example"
    assert any(r.getMessage() == "unhandled error" and r.exc_info for r in caplog.records)


def test_validation_error_is_uniform(probe_client):
    assert_error(probe_client.get("/api/test/typed?n=abc"), 400, "invalid_request")
    assert probe_client.get("/api/test/typed?n=4").json() == {"n": 4}


def test_other_http_exceptions_are_mapped(probe_client):
    assert_error(probe_client.get("/api/test/teapot"), 400, "invalid_request")
    assert_error(probe_client.get("/api/test/upstream"), 500, "internal_error")


def test_wrong_method_is_405_with_allow_header(probe_client):
    response = probe_client.get("/api/predict")
    assert_error(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "POST"


def test_error_after_response_started_is_not_masked(tmp_path, caplog):
    app = create_app(make_settings(tmp_path))

    async def broken_stream() -> StreamingResponse:
        def chunks() -> Iterator[bytes]:
            yield b"partial"
            raise RuntimeError("mid-stream")

        return StreamingResponse(chunks())

    app.add_api_route("/api/test/stream", broken_stream, methods=["GET"])
    with caplog.at_level(logging.INFO, logger="backend"), TestClient(app) as client, pytest.raises(RuntimeError):
        client.get("/api/test/stream")
    access = [r for r in caplog.records if r.getMessage() == "request"]
    assert [r.fields["status"] for r in access] == [200]


# ---------- security headers and CORS ----------


@pytest.mark.parametrize(
    "method, path, status",
    [
        ("GET", "/api/health", 200),
        ("GET", "/api/nope", 404),
        ("GET", "/api/predict", 405),
        ("POST", "/api/predict", 400),
    ],
)
def test_security_headers_on_every_response(shared_client, method, path, status):
    response = shared_client.request(method, path)
    assert response.status_code == status
    assert_security_headers(response)


def test_csp_allows_only_self_scripts_and_data_images():
    directives = dict(part.strip().split(" ", 1) for part in CONTENT_SECURITY_POLICY.split(";"))
    assert directives["script-src"] == "'self'" and "unsafe-inline" not in CONTENT_SECURITY_POLICY
    assert directives["img-src"] == "'self' data: blob:"  # upload previews use object URLs
    assert directives["frame-ancestors"] == "'none'"
    assert directives["object-src"] == "'none'"


def test_cors_allowed_origin(shared_client):
    response = shared_client.get("/api/health", headers={"Origin": "http://allowed.example"})
    assert response.headers["access-control-allow-origin"] == "http://allowed.example"
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()


def test_cors_disallowed_origin_gets_no_grant(shared_client):
    response = shared_client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert response.status_code == 200 and "access-control-allow-origin" not in response.headers


def test_cors_preflight(shared_client):
    headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    allowed = shared_client.options("/api/predict", headers={"Origin": "http://allowed.example", **headers})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://allowed.example"
    assert "POST" in allowed.headers["access-control-allow-methods"]
    denied = shared_client.options("/api/predict", headers={"Origin": "http://evil.example", **headers})
    assert denied.status_code == 400 and "access-control-allow-origin" not in denied.headers


def test_cors_headers_on_error_responses(shared_client):
    response = shared_client.post("/api/predict", headers={"Origin": "http://allowed.example"})
    assert response.status_code == 400
    assert response.headers["access-control-allow-origin"] == "http://allowed.example"


# ---------- logging ----------


def test_one_structured_access_line_per_request(shared_client, caplog):
    with caplog.at_level(logging.INFO, logger="backend"):
        shared_client.get("/api/health")
        shared_client.get("/api/nope?q=1")
    lines = [r for r in caplog.records if r.name == "backend.http"]
    assert [(r.fields["method"], r.fields["path"], r.fields["status"]) for r in lines] == [
        ("GET", "/api/health", 200),
        ("GET", "/api/nope", 404),
    ]
    assert all(isinstance(r.fields["duration_ms"], float) and r.fields["duration_ms"] >= 0 for r in lines)


def test_logs_never_contain_image_bytes(shared_client, caplog):
    data = encode(noise_image(64, 64, seed=99), "PNG")
    with caplog.at_level(logging.DEBUG, logger="backend"):
        assert upload(shared_client, data, explain="true").status_code == 200
        upload(shared_client, b"\x89PNG\r\n\x1a\nGARBAGE-MARKER")
    text = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert "GARBAGE-MARKER" not in text and data[40:80].hex() not in text and len(text) < 2000
    assert len([r for r in caplog.records if r.name == "backend.http"]) == 2


def test_access_log_records_the_decoded_path(shared_client, caplog):
    with caplog.at_level(logging.INFO, logger="backend"):
        shared_client.get("/api/caf%C3%A9")
    assert [r.fields["path"] for r in caplog.records if r.name == "backend.http"] == ["/api/café"]


@pytest.mark.anyio
async def test_non_http_scopes_pass_through_untouched(caplog):
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    with caplog.at_level(logging.INFO, logger="backend"):
        await AccessMiddleware(inner)({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"] and caplog.records == []


def test_json_formatter_output():
    record = logging.LogRecord("backend.x", logging.WARNING, __file__, 1, "hello %s", ("world",), None)
    record.fields = {"status": 404}
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["message"] == "hello world" and parsed["level"] == "WARNING" and parsed["status"] == 404
    assert parsed["logger"] == "backend.x" and "exception" not in parsed and parsed["time"]


def test_json_formatter_includes_exceptions_on_one_line():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("backend", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    line = JsonFormatter().format(record)
    assert "\n" not in line and "ValueError: boom" in json.loads(line)["exception"]


def test_configure_logging_is_idempotent_and_silences_uvicorn_access():
    configure_logging("debug")
    configure_logging("warning")
    logger = logging.getLogger("backend")
    assert len(logger.handlers) == 1 and isinstance(logger.handlers[0].formatter, JsonFormatter)
    assert logger.level == logging.WARNING and logging.getLogger("uvicorn.access").disabled
    configure_logging("INFO")
