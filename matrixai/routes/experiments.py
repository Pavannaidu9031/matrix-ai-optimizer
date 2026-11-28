"""Logging, editing, importing and exporting experiment runs."""
import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from ..auth import current_user, require_founder
from ..models import clean_response_curve, clean_text, parse_experiment, score_run, to_float
from ..science import ml_engine
from ..web import json_body, store

router = APIRouter()
MAX_BULK_ROWS = 2000


def save_runs(request: Request, email: str, records: list[dict]) -> list[int]:
    """Scores each record against the user's runs on its branch, then stores
    them all in one transaction."""
    branch_cache: dict[str, list] = {}
    for rec in records:
        branch = rec["branch_name"]
        if branch not in branch_cache:
            branch_cache[branch] = store(request).experiments(email, branch)
        if rec.get("quality_score") is None:
            rec["quality_score"] = score_run(rec, branch_cache[branch])
        branch_cache[branch].append(rec)
    return store(request).add_experiments(email, records)


@router.post("/add")
async def add_from_ledger_form(request: Request, user: dict = Depends(current_user)):
    form = await request.form()
    record = parse_experiment(dict(form))
    try:
        save_runs(request, user["email"], [record])
    except Exception as err:
        print(f"Saving run failed: {err}")
        return HTMLResponse(f"<h1>Could not save the run</h1><p>{err}</p><p><a href='/'>Back</a></p>", status_code=500)
    return RedirectResponse(f"/?branch={record['branch_name']}", status_code=303)


@router.post("/experiments")
async def add_from_json(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    if to_float(body.get("rf_power")) is None:
        raise HTTPException(422, {"error": "invalid", "message": "rf_power is required."})
    record = parse_experiment(body)
    new_id = save_runs(request, user["email"], [record])[0]
    return {"success": True, "id": new_id, "quality_score": record["quality_score"],
            "message": f"Experiment saved (branch: {record['branch_name']})"}


@router.get("/experiments")
def list_experiments(request: Request, branch: str = "main", user: dict = Depends(current_user)):
    runs = store(request).experiments(user["email"], branch)
    return {"success": True, "experiments": runs, "count": len(runs)}


@router.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, request: Request, user: dict = Depends(current_user)):
    outcome = store(request).delete_experiment(experiment_id, user["email"])
    if outcome == "not_found":
        raise HTTPException(404, "not_found")
    if outcome == "forbidden":
        raise HTTPException(403, "unauthorized")
    return {"success": True, "message": "Experiment deleted"}


@router.post("/admin/experiments/bulk-delete")
async def admin_bulk_delete(request: Request, user: dict = Depends(require_founder)):
    body = await json_body(request)
    ids = [int(i) for i in body.get("experiment_ids", []) if to_float(i) is not None]
    return {"success": True, "deleted": store(request).admin_delete(ids, user["email"])}


@router.post("/experiments/{experiment_id}/measurements")
async def update_measurements(experiment_id: int, request: Request, user: dict = Depends(current_user)):
    """Attach results measured after the run was logged (intensity change,
    response/recovery time, analysed curve) and re-score the run."""
    body = await json_body(request)
    intensity = to_float(body.get("light_intensity_uw"))
    response_t = to_float(body.get("h2_response_time"))
    recovery_t = to_float(body.get("recovery_time_s"))
    curve = clean_response_curve(body.get("response_curve"))
    powers = {k: v for k in ("fiber_power_before_uw", "fiber_power_after_uw")
              if (v := to_float(body.get(k))) is not None}
    if intensity is None and response_t is None and recovery_t is None and not powers:
        raise HTTPException(400, "nothing_to_update")

    run = store(request).get_experiment(experiment_id, user["email"])
    if not run:
        raise HTTPException(404, "not_found")
    updated = dict(run)
    updated["light_intensity_uw"] = intensity if intensity is not None else run.get("light_intensity_uw")
    updated["h2_response_time"] = response_t if response_t is not None else run.get("h2_response_time")
    branch_runs = store(request).experiments(user["email"], run.get("branch_name") or "main")
    quality = score_run(updated, branch_runs)

    values = {"light_intensity_uw": updated["light_intensity_uw"], "h2_response_time": updated["h2_response_time"],
              "quality_score": quality}
    if recovery_t is not None:
        values["recovery_time_s"] = recovery_t
    if curve is not None:
        values["response_curve"] = curve
    values.update(powers)
    store(request).update_measurements(experiment_id, user["email"], values)
    return {"success": True, "quality_score": quality}


@router.post("/api/experiments/bulk")
async def bulk_import(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    rows = body.get("experiments") or []
    if not rows:
        raise HTTPException(400, {"error": "No records provided."})
    if len(rows) > MAX_BULK_ROWS:
        raise HTTPException(400, {"error": f"At most {MAX_BULK_ROWS} rows per import."})
    branch = clean_text(body.get("branch")) or "main"
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec = parse_experiment({"batch_notes": "Batch CSV Import", **row}, overrides={"branch_name": branch})
        rec["quality_score"] = to_float(row.get("quality_score"))      # keep a score the CSV supplies
        records.append(rec)
    try:
        ids = save_runs(request, user["email"], records)
    except Exception as err:
        raise HTTPException(500, {"error": f"Import failed, nothing was saved: {err}"})
    return {"success": True, "inserted_count": len(ids), "message": f"Imported {len(ids)} runs into '{branch}'."}


@router.get("/export/csv")
def export_csv(request: Request, user: dict = Depends(current_user)):
    runs = store(request).experiments(user["email"])
    if not runs:
        return RedirectResponse("/", status_code=303)
    buffer = io.StringIO()
    pd.DataFrame(runs).to_csv(buffer, index=False)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=MatrixAI_Experiments.csv"})


@router.post("/api/comments/add")
async def add_comment(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    exp_id, text = to_float(body.get("experiment_id")), clean_text(body.get("comment_text"))
    if exp_id is None or not text:
        raise HTTPException(422, "experiment_id and comment_text are required")
    if not store(request).get_experiment(int(exp_id), user["email"]) and \
            not request.app.state.settings.is_founder(user["email"]):
        raise HTTPException(404, "not_found")
    store(request).add_comment(int(exp_id), user["email"], text[:5000])
    return {"success": True, "message": "Comment added."}


@router.post("/api/experiments/fingerprint")
async def fingerprint(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    target = body.get("experiment") or {}
    similar = ml_engine.get_dna_fingerprint(target, store(request).experiments(user["email"]))
    return JSONResponse({"success": True, "similar": similar})
