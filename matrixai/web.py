"""Small helpers shared by the route modules."""
import pathlib

from fastapi import Request
from fastapi.templating import Jinja2Templates

PACKAGE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def format_number(value, places=2):
    """Ledger display: up to `places` decimals, no trailing zeros, '-' when not measured."""
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{places}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return value


templates.env.filters["num"] = format_number


def store(request: Request):
    return request.app.state.store


def settings(request: Request):
    return request.app.state.settings


def render(request: Request, name: str, **context):
    user = request.session.get("user")
    if user:
        # Keep the cookie's flag in step with the DB-backed answer the gate used.
        approved = request.app.state.approvals.is_approved(user.get("email"))
        if user.get("is_approved") != approved:
            user = {**user, "is_approved": approved}
            request.session["user"] = user
    context.setdefault("user", user)
    context.setdefault("is_founder", bool(user) and request.app.state.settings.is_founder(user.get("email")))
    return templates.TemplateResponse(request, name, context)


async def json_body(request: Request) -> dict:
    """Request JSON as a dict; an empty or invalid body is {}."""
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}
