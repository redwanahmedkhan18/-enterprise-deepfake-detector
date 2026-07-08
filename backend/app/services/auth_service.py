import uuid

from jose import JWTError

from app.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
    verify_token_type,
    TokenError,
)
from app.exceptions.custom_exceptions import (
    InvalidCredentialsError,
    UserAlreadyExistsError,
    InvalidTokenError,
)
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import Token, UserCreate


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    async def register(self, data: UserCreate) -> User:
        existing = await self.user_repo.get_by_email(data.email)
        if existing:
            raise UserAlreadyExistsError(f"A user with email {data.email} already exists.")

        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )
        return await self.user_repo.create(user)

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Incorrect email or password.")
        if not user.is_active:
            raise InvalidCredentialsError("This account has been deactivated.")
        return user

    def issue_tokens(self, user: User) -> Token:
        subject = str(user.id)
        return Token(
            access_token=create_access_token(subject, extra_claims={"role": user.role.value}),
            refresh_token=create_refresh_token(subject),
        )

    async def refresh(self, refresh_token: str) -> Token:
        try:
            payload = decode_token(refresh_token)
            verify_token_type(payload, expected_type="refresh")
        except (JWTError, TokenError):
            raise InvalidTokenError("Invalid or expired refresh token.")

        user_id = payload.get("sub")
        user = await self.user_repo.get_by_id(uuid.UUID(user_id)) if user_id else None
        if not user or not user.is_active:
            raise InvalidTokenError("User not found or inactive.")

        return self.issue_tokens(user)
