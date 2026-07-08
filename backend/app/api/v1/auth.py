from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.repositories.user_repository import UserRepository
from app.schemas.user import Token, TokenRefreshRequest, UserCreate, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(UserRepository(db))


@router.post("/register", response_model=UserRead, status_code=201)
async def register(payload: UserCreate, service: AuthService = Depends(get_auth_service)):
    user = await service.register(payload)
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service),
):
    # OAuth2PasswordRequestForm uses 'username' field; we treat it as email.
    user = await service.authenticate(form_data.username, form_data.password)
    return service.issue_tokens(user)


@router.post("/refresh", response_model=Token)
async def refresh(payload: TokenRefreshRequest, service: AuthService = Depends(get_auth_service)):
    return await service.refresh(payload.refresh_token)
