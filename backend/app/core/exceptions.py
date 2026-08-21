"""Application exception hierarchy and their FastAPI handlers.

Business/service code should raise `AppError` subclasses instead of returning
HTTP-shaped errors directly, keeping HTTP concerns at the edge.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "internal_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_type = "not_found"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    error_type = "conflict"


class ValidationError(AppError):
    """Input that is well-formed JSON but violates a business rule.

    Distinct from FastAPI's request-schema validation, which rejects malformed
    payloads before they reach the service layer.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_type = "validation_error"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "unauthorized"


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    error_type = "forbidden"


def _error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.info(
        "app_error",
        path=request.url.path,
        error_type=exc.error_type,
        message=exc.message,
    )
    return _error_response(exc.status_code, exc.error_type, exc.message)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        exc_info=exc,
    )
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
