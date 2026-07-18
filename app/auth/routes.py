import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.jwt import authenticate_user, create_token
from app.config import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
) -> dict:
    # bcrypt.checkpw는 동기 호출이라 이벤트 루프를 블록한다 — to_thread로 회피(#10).
    authenticated = await asyncio.to_thread(
        authenticate_user, form.username, form.password, settings
    )
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_token(form.username, settings)
    return {"access_token": token, "token_type": "bearer"}
