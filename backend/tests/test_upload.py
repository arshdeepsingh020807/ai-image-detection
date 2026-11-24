import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.requests import Request

from backend.app.api.upload import MAX_ENVELOPE_BYTES, MAX_FIELD_BYTES, read_upload
from backend.app.core.errors import ApiError
from backend.app.main import create_app
from backend.tests.helpers import call, encode, make_settings, noise_image

BOUNDARY = "testboundary7MA4YWxk"
pytestmark = pytest.mark.anyio


def part(name: str, payload: bytes, filename: str | None = None) -> bytes:
    disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
    return f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\n\r\n".encode() + payload + b"\r\n"


def multipart(*parts: bytes) -> bytes:
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


def chunked(body: bytes, size: int) -> list[bytes]:
    return [body[i : i + size] for i in range(0, len(body), size)] or [b""]


def form_headers(length: int | None = None) -> dict[str, str]:
    headers = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}
    if length is not None:
        headers["content-length"] = str(length)
    return headers


@pytest.fixture
def small_app(tmp_path):
    return create_app(make_settings(tmp_path, max_upload_bytes=100_000))


async def test_oversized_stream_is_cut_off_early(small_app):
    body = multipart(part("file", b"\x00" * 2_000_000, "big.png"))
    chunks = chunked(body, 16_384)
    exchange = await call(small_app, chunks, form_headers())  # no content-length: only the stream can tell
    assert exchange.status == 413 and exchange.json()["error"]["code"] == "file_too_large"
    # The file part crosses 100 000 bytes inside chunk 7 (7 x 16 384 = 114 688); nothing after that is read.
    assert exchange.chunks_read == 7 and len(chunks) == 123


async def test_declared_length_over_limit_is_refused_before_reading(small_app):
    body = multipart(part("file", b"\x00" * 300_000, "big.png"))
    exchange = await call(small_app, chunked(body, 16_384), form_headers(len(body)))
    assert exchange.status == 413 and exchange.chunks_read == 0


async def test_body_limit_also_counts_non_file_parts(small_app):
    junk = b"".join(part(f"field{i}", b"x" * 1000) for i in range(200))  # 200 kB of small fields
    exchange = await call(small_app, chunked(multipart(junk), 8192), form_headers())
    assert exchange.status == 413 and exchange.json()["error"]["code"] == "file_too_large"
    assert exchange.chunks_read * 8192 <= 100_000 + MAX_ENVELOPE_BYTES + 8192


async def test_valid_upload_through_many_tiny_chunks(small_app):
    image = encode(noise_image(32, 32), "PNG")
    body = multipart(part("explain", b"true"), part("file", image, "x.png"))
    exchange = await call(small_app, chunked(body, 7), form_headers(len(body)))
    assert exchange.status == 200, exchange.body
    assert exchange.json()["image"]["size_bytes"] == len(image) and exchange.json()["explanation"] is not None


@pytest.mark.parametrize(
    "body",
    [
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"\r\n\r\nabc'.encode(),  # no closing boundary
        b"--wrongboundary\r\n\r\ngarbage",
        f"--{BOUNDARY}\r\n".encode() + b"X" * 5000 + b"\r\n\r\n",  # header line beyond the parser's limit
    ],
)
async def test_malformed_multipart_is_invalid_request(small_app, body):
    exchange = await call(small_app, [body], form_headers(len(body)))
    assert exchange.status == 400 and exchange.json()["error"]["code"] == "invalid_request"


async def test_first_file_part_wins_and_extra_parts_are_not_buffered(small_app):
    image = encode(noise_image(20, 20), "PNG")
    body = multipart(part("file", image, "a.png"), part("file", b"\x00" * 50_000, "b.png"), part("other", b"x"))
    exchange = await call(small_app, [body], form_headers(len(body)))
    assert exchange.status == 200 and exchange.json()["image"]["size_bytes"] == len(image)


async def test_overlong_text_field_is_rejected(small_app):
    body = multipart(part("explain", b"t" * (MAX_FIELD_BYTES + 1)), part("file", b"x", "a.png"))
    exchange = await call(small_app, [body], form_headers(len(body)))
    assert exchange.status == 400 and exchange.json()["error"]["code"] == "invalid_request"
    assert exchange.json()["error"]["message"] == "Form field 'explain' is too long."


# ---------- read_upload directly ----------


def request_for(body: bytes, chunk_size: int, content_type: str | None = None) -> Request:
    chunks = chunked(body, chunk_size)
    position = 0

    async def receive():
        nonlocal position
        position += 1
        return {"type": "http.request", "body": chunks[position - 1], "more_body": position < len(chunks)}

    ctype = content_type or f"multipart/form-data; boundary={BOUNDARY}"
    return Request({"type": "http", "method": "POST", "headers": [(b"content-type", ctype.encode())]}, receive)


async def test_read_upload_collects_file_and_fields():
    body = multipart(part("explain", b"true"), part("file", b"\x01\x02\x03", "f.bin"), part("note", "é".encode()))
    form = await read_upload(request_for(body, 5), max_file_bytes=10)
    assert form.file == b"\x01\x02\x03" and form.fields == {"explain": "true", "note": "é"}


async def test_read_upload_accepts_file_without_filename_and_ignores_nameless_parts():
    nameless = f"--{BOUNDARY}\r\nContent-Type: text/plain\r\n\r\nzzz\r\n".encode()
    form = await read_upload(request_for(multipart(nameless, part("file", b"abc")), 64), max_file_bytes=10)
    assert form.file == b"abc" and form.fields == {}


@pytest.mark.parametrize("content_type", ["application/json", "multipart/form-data", "text/plain; boundary=x"])
async def test_read_upload_without_multipart_boundary_has_no_file(content_type):
    form = await read_upload(request_for(b"{}", 10, content_type), max_file_bytes=10)
    assert form.file is None and form.fields == {}


@pytest.mark.parametrize("size, accepted", [(10, True), (11, False)])
async def test_read_upload_limit_is_inclusive(size, accepted):
    request = request_for(multipart(part("file", b"z" * size, "f.bin")), 4)
    if accepted:
        assert (await read_upload(request, max_file_bytes=10)).file == b"z" * size
    else:
        with pytest.raises(ApiError):
            await read_upload(request, max_file_bytes=10)


async def test_read_upload_header_names_are_case_insensitive():
    body = f'--{BOUNDARY}\r\ncontent-disposition: form-data; name="file"\r\n\r\nabc\r\n--{BOUNDARY}--\r\n'
    assert (await read_upload(request_for(body.encode(), 64), max_file_bytes=10)).file == b"abc"


async def test_read_upload_empty_file_part_is_empty_bytes():
    form = await read_upload(request_for(multipart(part("file", b"", "e.png")), 64), max_file_bytes=10)
    assert form.file == b""


@settings(max_examples=80, deadline=None)
@given(size=st.integers(0, 600), limit=st.integers(1, 400), chunk=st.integers(1, 700), seed=st.integers(0, 255))
async def test_size_limit_property(size, limit, chunk, seed):
    """Accepted exactly when the file fits, whatever the chunking; accepted bytes are returned intact."""
    payload = bytes((seed + i) % 256 for i in range(size))
    request = request_for(multipart(part("file", payload, "p.bin")), chunk)
    if size <= limit:
        assert (await read_upload(request, max_file_bytes=limit)).file == payload
    else:
        with pytest.raises(ApiError) as exc:
            await read_upload(request, max_file_bytes=limit)
        assert exc.value.code == "file_too_large" and exc.value.status == 413
