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
    if not authenticate_user(form.username, form.password, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_token(form.username, settings)
    return {"access_token": token, "token_type": "bearer"}
