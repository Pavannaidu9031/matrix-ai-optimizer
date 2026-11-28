"""Bayesian optimizer, digital twin, model checks and sensor analysis."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..auth import current_user
from ..models import to_float
from ..science import advanced_features, ml_engine, optimizer, sensor_analysis
from ..web import json_body, store

router = APIRouter()


def _runs(request, user, branch=None):
    return store(request).experiments(user["email"], branch or None)


@router.api_route("/suggest", methods=["GET", "POST"])
async def suggest(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request) if request.method == "POST" else {}
    branch = body.get("branch") or "main"
    runs, recent = _runs(request, user, branch), store(request).recent_suggestions(user["email"], 3)
    try:
        result = optimizer.generate_bayesian_suggestion(
            runs, recent,
            target_material=body.get("target_material") or "Generic",
            model_type=body.get("model_type") or "standard",
            acquisition_strategy=body.get("acquisition") or "ucb",
            measured_core_diameter_um=to_float(body.get("measured_core_diameter_um")),
        )
    except Exception as err:
        print(f"Optimizer failed: {err}")
        return JSONResponse({"message": f"Optimization engine error: {err}"}, status_code=500)
    s, expected = result.get("suggested", {}), result.get("expected", {})
    try:
        result["suggestion_id"] = store(request).add_suggestion(str(user.get("id") or user["email"]), user["email"], {
            "suggested_rf_power": s.get("rf_power"), "suggested_pressure": s.get("working_pressure"),
            "suggested_distance": s.get("target_distance"), "suggested_thickness": s.get("film_thickness"),
            "suggested_rotation": s.get("rotation_speed"), "suggested_ar_flow": s.get("ar_flow"),
            "suggested_pd_thickness": s.get("pd_thickness"),
            "predicted_xrd_score": expected.get("xrd_score_predicted"),
            "predicted_wavelength": expected.get("light_intensity_uw_estimate"),
            "confidence_score": result.get("confidence", {}).get("score"),
            "converged": result.get("convergence", {}).get("converged"),
            "kappa_used": result.get("kappa_used"),
        })
    except Exception as err:
        print(f"Suggestion history insert failed (suggestion still returned, unlinked): {err}")
    return JSONResponse(result)


@router.post("/api/sandbox/simulate")
async def sandbox(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    params = body.get("params") or []
    if not 6 <= len(params) <= 8:
        raise HTTPException(400, "Invalid parameter array length.")
    sim = optimizer.simulate_sandbox_point(_runs(request, user, body.get("branch") or "main"),
                                           body.get("target_material") or "Generic", [float(p) for p in params])
    return {"success": True, "simulation": sim}


@router.post("/api/experiments/whatif")
async def whatif(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    sim = optimizer.simulate_sandbox_point(_runs(request, user, body.get("branch")),
                                           body.get("material") or "Generic", body.get("params") or [])
    return {"success": True, "simulation": sim}


@router.get("/api/model/validation")
def validation(request: Request, branch: str = "main", target_material: str = "Generic",
               user: dict = Depends(current_user)):
    return {"success": True,
            "validation": optimizer.leave_one_out_validation(_runs(request, user, branch), target_material)}


@router.post("/api/optimizer/phase-map")
async def phase_map(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    result = optimizer.generate_phase_map(_runs(request, user, body.get("branch")), body.get("target_material") or "Generic",
                                          body.get("param_x") or "rf_power", body.get("param_y") or "working_pressure")
    return {"success": True, "phase_map": result}


@router.post("/api/optimizer/calibrate-noise")
async def calibrate_noise(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    calib = optimizer.calibrate_noise_variance(_runs(request, user, body.get("branch")),
                                               body.get("target_material") or "Generic")
    return {"success": True, "calibration": calib}


@router.post("/api/optimizer/recipe")
async def recipe(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    return {"success": True,
            "recipe": optimizer.generate_synthesis_recipe(body.get("params") or {}, body.get("target_material") or "Generic")}


@router.post("/api/optimizer/pd-recipe")
async def pd_recipe(request: Request, user: dict = Depends(current_user)):
    """Pd is a separate sputtering run from WO3 with no data of its own yet;
    see suggest_pd_deposition_step for what this estimate is and isn't."""
    body = await json_body(request)
    result = optimizer.suggest_pd_deposition_step(
        target_pd_thickness_nm=to_float(body.get("target_pd_thickness_nm")) or 8.0,
        calibrated_rate_nm_per_min=to_float(body.get("calibrated_rate_nm_per_min")) or None,
        distance_cm=to_float(body.get("distance_cm")) or 5.0,
        ar_flow_sccm=to_float(body.get("ar_flow_sccm")) or 30.0,
    )
    return {"success": True, "pd_recipe": result}


@router.get("/api/suggestions/track-record")
def track_record(request: Request, branch: str = "main", user: dict = Depends(current_user)):
    return {"success": True, "track_record": optimizer.suggestion_track_record(_runs(request, user, branch))}


@router.post("/api/agent/campaign")
async def agent_campaign(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    campaign = ml_engine.generate_agent_campaign(int(to_float(body.get("budget")) or 25),
                                                 body.get("deadline") or "2026-12-31",
                                                 _runs(request, user, body.get("branch")))
    return {"success": True, "campaign": campaign}


@router.post("/api/sensor/predict")
async def sensor_predict(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    return {"success": True, "predictions": advanced_features.predict_sensor_performance(body.get("params") or {})}


@router.get("/api/equipment/health")
def equipment_health(request: Request, user: dict = Depends(current_user)):
    return {"success": True,
            "health": advanced_features.check_equipment_health(store(request).experiments(user["email"], limit=40))}


@router.post("/api/scheduler/plan")
async def scheduler_plan(request: Request, user: dict = Depends(current_user)):
    body = await json_body(request)
    schedule = advanced_features.schedule_experiments([], to_float(body.get("hours")) or 0.0,
                                                      int(to_float(body.get("days")) or 1))
    return {"success": True, "schedule": schedule}


@router.post("/api/sensor/analyze-curve")
async def analyze_curve(
    request: Request,
    file: UploadFile = File(...),
    time_col: str | None = Form(None),
    intensity_col: str | None = Form(None),
    unit: str | None = Form(None),
    time_unit: str | None = Form("s"),
    h2_on_s: str | None = Form(None),
    h2_off_s: str | None = Form(None),
    user: dict = Depends(current_user),
):
    """Analyses an uploaded power-meter export. Nothing is stored here; the
    browser shows the result and sends it along when the run is saved."""
    raw = await file.read(sensor_analysis.MAX_FILE_BYTES + 1)
    as_int = lambda v: int(v) if v not in (None, "") else None   # noqa: E731
    try:
        result = sensor_analysis.analyze_file(
            raw, time_col=as_int(time_col), intensity_col=as_int(intensity_col), unit=unit or None,
            time_unit=time_unit or "s", h2_on_s=to_float(h2_on_s), h2_off_s=to_float(h2_off_s))
    except sensor_analysis.CurveError as err:
        return JSONResponse({"success": False, "message": str(err)}, status_code=422)
    except Exception as err:
        print(f"Curve analysis crashed: {err}")
        return JSONResponse({"success": False, "message": "Could not read that file."}, status_code=422)
    result["filename"] = file.filename
    return {"success": True, "analysis": result}
