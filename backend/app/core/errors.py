from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

STATUS_BY_CODE = {
    "missing_file": 400,
    "empty_file": 400,
    "invalid_image": 400,
    "invalid_request": 400,
    "not_found": 404,
    "method_not_allowed": 405,
    "image_too_large": 413,
    "file_too_large": 413,
    "unsupported_format": 415,
    "animated_image": 415,
    "rate_limited": 429,
    "internal_error": 500,
    "model_unavailable": 503,
}


class ApiError(Exception):
    """A failure the client should see as `{"error": {"code", "message"}}` with the code's HTTP status."""

    def __init__(self, code: str, message: str, headers: Mapping[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = STATUS_BY_CODE[code]
        self.headers = dict(headers or {})


def error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message}}, status_code=error.status, headers=error.headers
    )


def _from_http_exception(exc: StarletteHTTPException) -> ApiError:
    if exc.status_code == 405:
        return ApiError("method_not_allowed", "Method not allowed.", exc.headers)
    if exc.status_code == 404:
        return ApiError("not_found", "Not found.", exc.headers)
    if exc.status_code < 500:
        return ApiError("invalid_request", "The request is invalid.", exc.headers)
    return ApiError("internal_error", "Internal server error.")


async def _on_api_error(_: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc)


async def _on_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(_from_http_exception(exc))


async def _on_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    return error_response(ApiError("invalid_request", "The request is invalid."))


EXCEPTION_HANDLERS: dict[int | type[Exception], Callable[[Request, Any], Coroutine[Any, Any, Response]]] = {
    ApiError: _on_api_error,
    StarletteHTTPException: _on_http_exception,
    RequestValidationError: _on_validation_error,
}
