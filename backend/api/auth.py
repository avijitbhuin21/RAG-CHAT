from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import User
from ..schemas import AdminLoginRequest, SessionResponse, UserSessionInfo
from ..security import (
    clear_admin_session_cookie,
    clear_user_session_cookie,
    create_jwt,
    optional_admin_claims,
    optional_user_claims,
    set_admin_session_cookie,
    set_user_session_cookie,
    verify_admin_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_OAUTH_CLIENT_ID,
    client_secret=settings.GOOGLE_OAUTH_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


@router.get("/google/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, settings.GOOGLE_OAUTH_REDIRECT_URI)


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        raise HTTPException(status_code=400, detail=f"oauth error: {e.error}")

    info = token.get("userinfo") or {}
    sub = info.get("sub")
    email = info.get("email")
    if not sub or not email:
        raise HTTPException(status_code=400, detail="google did not return a valid profile")

    user = db.query(User).filter_by(google_sub=sub).one_or_none()
    if user is None:
        user = User(
            google_sub=sub,
            email=email,
            name=info.get("name"),
            picture_url=info.get("picture"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.email = email
        user.name = info.get("name") or user.name
        user.picture_url = info.get("picture") or user.picture_url
        db.commit()

    jwt_token = create_jwt(
        {"sub": str(user.id), "type": "user", "user_id": str(user.id), "email": user.email}
    )
    redirect = RedirectResponse(url=f"{settings.FRONTEND_URL}/chat")
    set_user_session_cookie(redirect, jwt_token)
    return redirect


@router.post("/admin/login")
def admin_login(body: AdminLoginRequest, response: Response):
    if body.username != settings.ADMIN_USERNAME or not verify_admin_password(body.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    token = create_jwt({"sub": "admin", "type": "admin"})
    set_admin_session_cookie(response, token)
    return {"type": "admin"}


@router.post("/logout")
def logout(
    response: Response,
    which: str | None = None,
):
    """Clear one or both session cookies.

    which=user  -> clear only the user session
    which=admin -> clear only the admin session
    (default)   -> clear both
    """
    if which in (None, "user"):
        clear_user_session_cookie(response)
    if which in (None, "admin"):
        clear_admin_session_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=SessionResponse)
def me(
    user_claims: dict | None = Depends(optional_user_claims),
    admin_claims: dict | None = Depends(optional_admin_claims),
    db: Session = Depends(get_db),
):
    user: UserSessionInfo | None = None
    if user_claims:
        u = db.query(User).filter_by(id=user_claims["user_id"]).one_or_none()
        if u:
            user = UserSessionInfo(
                user_id=u.id,
                email=u.email,
                name=u.name,
                picture_url=u.picture_url,
            )
    admin = {"type": "admin"} if admin_claims else None
    return SessionResponse(user=user, admin=admin)
