class AppError(Exception):
    """Base class for all application-level (non-HTTP) errors."""


class UserAlreadyExistsError(AppError):
    pass


class InvalidCredentialsError(AppError):
    pass


class InvalidTokenError(AppError):
    pass


class JobNotFoundError(AppError):
    pass


class UnsupportedMediaTypeError(AppError):
    pass


class FileTooLargeError(AppError):
    pass


class InferenceServiceError(AppError):
    pass
