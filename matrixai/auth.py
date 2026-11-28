"""Google sign-in, the approval gate, and helpers routes use to get the user."""
import time

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

PUBLIC_PATHS = {"/", "/login/google", "/auth/callback", "/logout", "/api/docs", "/healthz"}
PUBLIC_PREFIXES = ("/static/", "/api/v1/")        # /api/v1 authenticates with X-API-Key


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def wants_html(request: Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("accept", "")


class ApprovalChecker:
    """The users table, not the 30-day session cookie, decides approval, so a
    revoke takes effect on the user's next request. Answers are cached per
    process for `ttl_s`; approve/revoke clear the entry immediately. If the
    database can't be reached the last known answer is used, and a user we
    never got an answer for is denied (fail closed)."""

    def __init__(self, store, settings, ttl_s: float = 30.0):
        self.store, self.settings, self.ttl_s = store, settings, ttl_s
        self._cache: dict[str, tuple[bool, float]] = {}

    def forget(self, email: str):
        self._cache.pop((email or "").lower(), None)

    def clear(self):
        self._cache.clear()

    def is_approved(self, email: str) -> bool:
        email = (email or "").lower()
        if self.settings.is_founder(email):
            return True
        now = time.monotonic()
        cached = self._cache.get(email)
        if cached and now - cached[1] < self.ttl_s:
            return cached[0]
        try:
            approved = self.store.approval(email)
        except Exception as err:
            print(f"Approval lookup failed for {email}: {err}")
            return cached[0] if cached else False
        self._cache[email] = (approved, now)
        return approved


def install_gate(app):
    """Every request passes through this; only PUBLIC paths skip it."""

    @app.middleware("http")
    async def approval_gate(request: Request, call_next):
        if is_public(request.url.path):
            return await call_next(request)
        user = request.session.get("user")
        if not user:
            if wants_html(request):
                return RedirectResponse("/login/google", status_code=303)
            return JSONResponse({"error": "not_authenticated"}, status_code=401)
        if not request.app.state.approvals.is_approved(user.get("email")):
            if wants_html(request):
                return RedirectResponse("/", status_code=303)       # "/" shows the pending page
            return JSONResponse({"error": "not_approved", "message": "Your account is waiting for approval."},
                                status_code=403)
        return await call_next(request)


def current_user(request: Request) -> dict:
    """The signed-in, approved user (the gate has already checked)."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="not_authenticated")
    return user


def require_founder(request: Request) -> dict:
    user = current_user(request)
    if not request.app.state.settings.is_founder(user.get("email")):
        raise HTTPException(status_code=403, detail="founder_only")
    return user


def build_router(settings) -> APIRouter:
    router = APIRouter()
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    @router.get("/login/google")
    async def login_google(request: Request):
        return await oauth.google.authorize_redirect(request, request.url_for("auth_callback"))

    @router.get("/auth/callback")
    async def auth_callback(request: Request):
        try:
            token = await oauth.google.authorize_access_token(request)
            info = token.get("userinfo") or await oauth.google.userinfo(token=token)
        except Exception as err:
            print(f"Google sign-in failed: {err}")
            return RedirectResponse("/login/google", status_code=303)
        if not info or not info.get("email"):
            return RedirectResponse("/login/google", status_code=303)

        email = info["email"]
        is_founder = settings.is_founder(email)
        is_new = False
        try:
            approved, is_new = request.app.state.store.record_login(
                email, info.get("name", "Researcher"), info.get("picture", ""), approve_if_new=is_founder)
        except Exception as err:
            print(f"Could not record login for {email}: {err}")
            approved = is_founder
        request.app.state.approvals.forget(email)

        request.session["user"] = {
            "id": info.get("sub") or email, "email": email, "name": info.get("name", "Researcher"),
            "picture": info.get("picture", ""), "is_approved": approved or is_founder,
        }
        request.session["login_greeting"] = {"is_new": is_new, "given_name": info.get("given_name") or ""}
        return RedirectResponse("/", status_code=303)

    @router.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=303)

    return router
