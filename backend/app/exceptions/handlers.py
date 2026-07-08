from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.exceptions.custom_exceptions import (
    FileTooLargeError,
    InferenceServiceError,
    InvalidCredentialsError,
    InvalidTokenError,
    JobNotFoundError,
    UnsupportedMediaTypeError,
    UserAlreadyExistsError,
)

_ERROR_STATUS_MAP = {
    UserAlreadyExistsError: status.HTTP_409_CONFLICT,
    InvalidCredentialsError: status.HTTP_401_UNAUTHORIZED,
    InvalidTokenError: status.HTTP_401_UNAUTHORIZED,
    JobNotFoundError: status.HTTP_404_NOT_FOUND,
    UnsupportedMediaTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    FileTooLargeError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    InferenceServiceError: status.HTTP_502_BAD_GATEWAY,
}


def register_exception_handlers(app: FastAPI) -> None:
    for exc_class, http_status in _ERROR_STATUS_MAP.items():

        def make_handler(status_code: int):
            async def handler(request: Request, exc: Exception) -> JSONResponse:
                return JSONResponse(status_code=status_code, content={"detail": str(exc)})

            return handler

        app.add_exception_handler(exc_class, make_handler(http_status))
