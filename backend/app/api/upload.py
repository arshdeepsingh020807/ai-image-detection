"""Streaming multipart reader that keeps the upload in memory and stops as soon as it is too large.

Starlette's `request.form()` spools files above 1 MB to disk and reads the whole body before any
size check, which breaks both rules this service makes about uploads.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python_multipart.multipart import MultipartParseError, MultipartParser, parse_options_header
from starlette.requests import Request

from backend.app.core.errors import ApiError

if TYPE_CHECKING:
    from python_multipart.multipart import MultipartCallbacks

FILE_FIELD = "file"
MAX_FIELD_BYTES = 1024
# Room for part headers, boundaries and the small text fields around the file itself.
MAX_ENVELOPE_BYTES = 64 * 1024


@dataclass
class UploadForm:
    file: bytes | None = None
    fields: dict[str, str] = field(default_factory=dict)


def _too_large(max_file_bytes: int) -> ApiError:
    return ApiError("file_too_large", f"The file exceeds the {max_file_bytes} byte limit.")


class _FormCollector:
    """python-multipart callbacks that keep the first `file` part and small text fields, nothing else."""

    def __init__(self, max_file_bytes: int):
        self.max_file_bytes = max_file_bytes
        self.file: bytearray | None = None
        self.fields: dict[str, bytearray] = {}
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._part_name: str | None = None
        self._target: bytearray | None = None
        self._limit = 0
        self.complete = False

    def callbacks(self) -> "MultipartCallbacks":
        return {
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_end": self.on_end,
        }

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            _, options = parse_options_header(bytes(self._header_value).decode("latin-1"))
            self._part_name = options.get(b"name", b"").decode("latin-1")
        self._header_name.clear()
        self._header_value.clear()

    def on_headers_finished(self) -> None:
        # A part without a name inherits the previous name, which is always already taken, so it is ignored.
        self._target = None
        if self._part_name == FILE_FIELD and self.file is None:
            self.file = self._target = bytearray()
            self._limit = self.max_file_bytes
        elif self._part_name and self._part_name != FILE_FIELD and self._part_name not in self.fields:
            self.fields[self._part_name] = self._target = bytearray()
            self._limit = MAX_FIELD_BYTES

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._target is None:
            return
        if len(self._target) + end - start > self._limit:
            if self._target is self.file:
                raise _too_large(self.max_file_bytes)
            raise ApiError("invalid_request", f"Form field '{self._part_name}' is too long.")
        self._target += data[start:end]

    def on_end(self) -> None:
        self.complete = True


async def read_upload(request: Request, max_file_bytes: int) -> UploadForm:
    content_type, options = parse_options_header(request.headers.get("content-type"))
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        return UploadForm()
    max_body = max_file_bytes + MAX_ENVELOPE_BYTES
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_body:
        raise _too_large(max_file_bytes)
    collector = _FormCollector(max_file_bytes)
    parser = MultipartParser(boundary, collector.callbacks())
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_body:
                raise _too_large(max_file_bytes)
            parser.write(chunk)
    except MultipartParseError as exc:
        raise ApiError("invalid_request", "The multipart body is malformed.") from exc
    if not collector.complete:
        raise ApiError("invalid_request", "The multipart body ended before its closing boundary.")
    fields = {name: value.decode("utf-8", "replace") for name, value in collector.fields.items()}
    return UploadForm(file=None if collector.file is None else bytes(collector.file), fields=fields)
