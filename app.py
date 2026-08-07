import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import pandas as pd
import io
import sqlite3

import database
import optimizer

app = FastAPI(title="MatrixAI — Intelligent Materials Optimizer")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "matrix-ai-super-secret-key-change-this")
)

templates = Jinja2Templates(directory="templates")
database.init_db()

FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "pavannaidu9031@gmail.com")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user_session = request.session.get("user")
    
    if not user_session:
        return templates.TemplateResponse(request, "index.html", {"user": None, "experiments": [], "stats": {}})

    db_user = database.get_user_by_id(user_session["id"])
    if not db_user:
        request.session.clear()
        return RedirectResponse("/")

    if db_user["status"] == "pending":
        return templates.TemplateResponse(request, "pending.html", {"user": db_user})

    if db_user["status"] == "rejected":
        request.session.clear()
        return HTMLResponse("<h1>Access Restricted</h1>", status_code=403)

    raw_experiments = database.get_experiments_by_user(db_user["id"])
    experiments = [dict(row) for row in raw_experiments] if raw_experiments else []

    total_runs = len(experiments)
    monoclinic_runs = sum(1 for e in experiments if e.get("xrd_phase") == "Monoclinic")
    shifts = [float(e["wavelength_shift_pm"]) for e in experiments if e.get("wavelength_shift_pm") is not None]
    best_shift = max(shifts) if shifts else 0.0
    runs_remaining = max(0, 25 - total_runs)

    stats = {
        "total_runs": total_runs,
        "monoclinic_runs": monoclinic_runs,
        "best_shift": round(best_shift, 1),
        "runs_remaining": runs_remaining
    }

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": db_user,
            "experiments": experiments,
            "stats": stats
        }
    )

@app.get("/suggest")
@app.post("/suggest")
async def get_bayes_suggestion(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        raw_experiments = database.get_experiments_by_user(user_session["id"])
        experiments = [dict(row) for row in raw_experiments] if raw_experiments else []

        recent_suggestions = database.get_recent_suggestions(user_session["id"], limit=3)

        result = optimizer.generate_bayesian_suggestion(experiments, recent_suggestions)
        
        # Save suggestion to history
        database.save_suggestion_history(user_session["id"], user_session["email"], result)

        return JSONResponse(content=result)
    except Exception as e:
        print(f"MatrixAI Engine Error: {e}")
        return JSONResponse(status_code=500, content={"message": f"Optimization engine error: {str(e)}"})

@app.post("/add")
def add_experiment(
    request: Request,
    rf_power_w: float = Form(...),
    working_pressure_mtorr: float = Form(...),
    ar_flow_sccm: float = Form(...),
    o2_flow_sccm: float = Form(...),
    substrate_temp_c: float = Form(...),
    target_substrate_distance_cm: float = Form(...),
    sputtering_time_min: float = Form(...),
    film_thickness_nm: float = Form(None),
    rotation_speed_rpm: float = Form(5.0),
    substrate_type: str = Form("Si Wafer"),
    xrd_phase: str = Form("Amorphous"),
    grain_size_nm: float = Form(None),
    h2_response_time_s: float = Form(...),
    wavelength_shift_pm: float = Form(None),
    notes: str = Form(None)
):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_experiments = database.get_experiments_by_user(user["id"])
    all_experiments = [dict(row) for row in raw_experiments] if raw_experiments else []

    # Calculate Quality Score
    quality_score = optimizer.calculate_quality_score(
        xrd_phase, wavelength_shift_pm, h2_response_time_s, grain_size_nm, all_experiments
    )

    database.add_experiment(
        user_id=user["id"],
        rf_power_w=rf_power_w,
        working_pressure_mtorr=working_pressure_mtorr,
        ar_flow_sccm=ar_flow_sccm,
        o2_flow_sccm=o2_flow_sccm,
        substrate_temp_c=substrate_temp_c,
        target_substrate_distance_cm=target_substrate_distance_cm,
        sputtering_time_min=sputtering_time_min,
        film_thickness_nm=film_thickness_nm,
        rotation_speed_rpm=rotation_speed_rpm,
        substrate_type=substrate_type,
        xrd_phase=xrd_phase,
        grain_size_nm=grain_size_nm,
        h2_response_time_s=h2_response_time_s,
        wavelength_shift_pm=wavelength_shift_pm,
        quality_score=quality_score,
        notes=notes
    )
    return RedirectResponse("/", status_code=303)

@app.post("/delete/{exp_id}")
def delete_experiment(exp_id: int, request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    database.delete_experiment(exp_id, user["id"])
    return RedirectResponse("/", status_code=303)

@app.get("/export/csv")
def export_csv(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_experiments = database.get_experiments_by_user(user["id"])
    if not raw_experiments:
        return RedirectResponse("/")

    df = pd.DataFrame([dict(e) for e in raw_experiments])
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=MatrixAI_Thin_Film_Experiments.csv"
    return response

@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    if not user_info:
        return RedirectResponse("/")

    db_user = database.get_or_create_user(user_info, founder_email=FOUNDER_EMAIL)
    request.session["user"] = {"id": db_user["id"], "email": db_user["email"], "name": db_user["name"]}
    return RedirectResponse("/")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
