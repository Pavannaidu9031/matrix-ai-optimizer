"""Founder-only user management, plus the X-API-Key REST API."""
import hashlib

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..auth import require_founder
from ..science import advanced_features, optimizer
from ..web import json_body, render, store

router = APIRouter()


@router.get("/admin/users")
def admin_users(request: Request, user: dict = Depends(require_founder)):
    users = store(request).list_users()
    for u in users:
        u["created_at"] = str(u.get("created_at") or "")[:10]
    return render(request, "admin.html", users_list=users)


def _set_approval(request: Request, email: str, approved: bool):
    if not approved and request.app.state.settings.is_founder(email):
        return RedirectResponse("/admin/users", status_code=303)      # never lock the founder out
    store(request).set_approval(email, approved)
    request.app.state.approvals.forget(email)                         # takes effect on their next request
    return RedirectResponse("/admin/users", status_code=303)


@router.post("/admin/users/{email}/approve")
def approve_user(email: str, request: Request, user: dict = Depends(require_founder)):
    return _set_approval(request, email, True)


@router.post("/admin/users/{email}/revoke")
def revoke_user(email: str, request: Request, user: dict = Depends(require_founder)):
    return _set_approval(request, email, False)


# ---- Public REST API (X-API-Key) ---------------------------------------------
def api_key_owner(request: Request, x_api_key: str = Header(...)) -> str:
    owner = store(request).api_key_owner(hashlib.sha256(x_api_key.encode()).hexdigest())
    if not owner:
        raise HTTPException(401, "Invalid or unauthorized API key")
    return owner


@router.post("/api/v1/suggest")
async def api_v1_suggest(request: Request, owner: str = Depends(api_key_owner)):
    body = await json_body(request)
    return optimizer.generate_bayesian_suggestion(body.get("experiments") or [], [],
                                                  target_material=body.get("material") or "Generic")


@router.post("/api/v1/predict")
async def api_v1_predict(request: Request, owner: str = Depends(api_key_owner)):
    body = await json_body(request)
    return {"predicted": advanced_features.predict_sensor_performance(body.get("parameters") or {})}
