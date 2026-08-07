import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
import pandas as pd
import psycopg2
from psycopg2 import pool
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import optimizer

# ==============================================================================
# FASTAPI & PERMANENT SESSION INITIALIZATION
# ==============================================================================
app = FastAPI(title="MatrixAI — Intelligent Materials Optimizer")

# Permanent SECRET_KEY ensures user session cookies survive Render redeployments
SECRET_KEY = os.getenv("SECRET_KEY", "matrix-ai-permanent-production-secret-key-2026")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=86400 * 30,  # 30-day session duration
    same_site="lax",
    https_only=True      # Enforce HTTPS on Render
)

templates = Jinja2Templates(directory="templates")

# ==============================================================================
# SUPABASE POSTGRESQL CONNECTION POOLING
# ==============================================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    connection_pool = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
else:
    connection_pool = None

def get_db_connection():
    if connection_pool:
        return connection_pool.getconn()
    elif DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    else:
        raise HTTPException(
            status_code=500, 
            detail="DATABASE_URL environment variable is missing in Render settings."
        )

def release_db_connection(conn):
    if connection_pool and conn:
        connection_pool.putconn(conn)
    elif conn:
        conn.close()

# ==============================================================================
# OAUTH CONFIGURATION
# ==============================================================================
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

# ==============================================================================
# PYDANTIC DATA MODEL
# ==============================================================================
class ExperimentModel(BaseModel):
    rf_power: float
    working_pressure: float
    ar_flow: float
    o2_flow: float
    substrate_temp: float
    target_distance: float
    sputter_time: float
    film_thickness: Optional[float] = None
    rotation_speed: float = 5.0
    substrate_type: str = "Si Wafer"
    xrd_phase: str = "Amorphous"
    grain_size: Optional[float] = None
    h2_response_time: float
    wavelength_shift: Optional[float] = None
    batch_notes: Optional[str] = None

# ==============================================================================
# DASHBOARD ROUTE (FETCHES PERMANENT SUPABASE DATA)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        return templates.TemplateResponse(request, "index.html", {"user": None, "experiments": [], "stats": {}})

    user_email = user_session.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_email, rf_power, working_pressure, ar_flow, o2_flow,
                   substrate_temp, target_distance, sputter_time, film_thickness,
                   rotation_speed, substrate_type, xrd_phase, grain_size,
                   h2_response_time, wavelength_shift, batch_notes, quality_score, created_at
            FROM experiments
            WHERE user_email = %s
            ORDER BY created_at DESC
        """, (user_email,))
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        experiments = [dict(zip(columns, row)) for row in rows]
        cur.close()

        total_runs = len(experiments)
        monoclinic_runs = sum(1 for e in experiments if e.get("xrd_phase") == "Monoclinic")
        shifts = [float(e["wavelength_shift"]) for e in experiments if e.get("wavelength_shift") is not None]
        best_shift = max(shifts) if shifts else 0.0
        runs_remaining = max(0, 25 - total_runs)

        stats = {
            "total_runs": total_runs,
            "monoclinic_runs": monoclinic_runs,
            "best_shift": round(best_shift, 1),
            "runs_remaining": runs_remaining
        }

        return templates.TemplateResponse(request, "index.html", {
            "user": user_session,
            "experiments": experiments,
            "stats": stats
        })
    finally:
        release_db_connection(conn)

# ==============================================================================
# PERMANENT EXPERIMENT SAVE (WRITE TO SUPABASE)
# ==============================================================================
@app.post("/experiments")
async def save_experiment_json(request: Request, data: ExperimentModel):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Get existing user data to calculate normalized quality score
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user_email,))
        existing_rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        all_exps = [dict(zip(cols, r)) for r in existing_rows]
        
        quality_score = optimizer.calculate_quality_score(
            data.xrd_phase, data.wavelength_shift, data.h2_response_time, data.grain_size, all_exps
        )

        cur.execute("""
            INSERT INTO experiments (
                user_email, rf_power, working_pressure, ar_flow,
                o2_flow, substrate_temp, target_distance, sputter_time,
                film_thickness, rotation_speed, substrate_type, xrd_phase,
                grain_size, h2_response_time, wavelength_shift, batch_notes,
                quality_score, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            ) RETURNING id
        """, (
            user_email, data.rf_power, data.working_pressure, data.ar_flow,
            data.o2_flow, data.substrate_temp, data.target_distance, data.sputter_time,
            data.film_thickness, data.rotation_speed, data.substrate_type, data.xrd_phase,
            data.grain_size, data.h2_response_time, data.wavelength_shift, data.batch_notes,
            quality_score
        ))
        
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        return JSONResponse({
            "success": True,
            "id": new_id,
            "quality_score": quality_score,
            "message": "Experiment saved permanently to Supabase"
        })
    except Exception as e:
        if conn:
            conn.rollback()
        return JSONResponse({"error": "save_failed", "message": str(e)}, status_code=500)
    finally:
        release_db_connection(conn)

# Legacy Form POST endpoint compatibility
@app.post("/add")
def add_experiment_form(
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

    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user_email,))
        existing_rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        all_exps = [dict(zip(cols, r)) for r in existing_rows]

        quality_score = optimizer.calculate_quality_score(
            xrd_phase, wavelength_shift_pm, h2_response_time_s, grain_size_nm, all_exps
        )

        cur.execute("""
            INSERT INTO experiments (
                user_email, rf_power, working_pressure, ar_flow,
                o2_flow, substrate_temp, target_distance, sputter_time,
                film_thickness, rotation_speed, substrate_type, xrd_phase,
                grain_size, h2_response_time, wavelength_shift, batch_notes,
                quality_score, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
        """, (
            user_email, rf_power_w, working_pressure_mtorr, ar_flow_sccm,
            o2_flow_sccm, substrate_temp_c, target_substrate_distance_cm, sputtering_time_min,
            film_thickness_nm, rotation_speed_rpm, substrate_type, xrd_phase,
            grain_size_nm, h2_response_time_s, wavelength_shift_pm, notes,
            quality_score
        ))
        conn.commit()
        cur.close()
        return RedirectResponse("/", status_code=303)
    finally:
        release_db_connection(conn)

# ==============================================================================
# FETCH EXPERIMENTS API
# ==============================================================================
@app.get("/experiments")
async def get_experiments(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_email, rf_power, working_pressure, ar_flow, o2_flow,
                   substrate_temp, target_distance, sputter_time, film_thickness,
                   rotation_speed, substrate_type, xrd_phase, grain_size,
                   h2_response_time, wavelength_shift, batch_notes, quality_score, created_at
            FROM experiments
            WHERE user_email = %s
            ORDER BY created_at DESC
        """, (user_email,))
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        experiments = [dict(zip(columns, row)) for row in rows]
        cur.close()

        return JSONResponse({
            "success": True,
            "experiments": experiments,
            "count": len(experiments)
        })
    except Exception as e:
        return JSONResponse({"error": "fetch_failed", "message": str(e)}, status_code=500)
    finally:
        release_db_connection(conn)

@app.get("/experiments/count")
async def count_experiments(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"count": 0, "user": None})
    
    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM experiments WHERE user_email = %s", (user_email,))
        count = cur.fetchone()[0]
        cur.close()
        return JSONResponse({"count": count, "user": user_email})
    finally:
        release_db_connection(conn)

@app.post("/delete/{exp_id}")
async def delete_experiment(exp_id: int, request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM experiments WHERE id = %s AND user_email = %s", (exp_id, user_email))
        conn.commit()
        cur.close()
        return RedirectResponse("/", status_code=303)
    finally:
        release_db_connection(conn)

# ==============================================================================
# BAYESIAN OPTIMIZER ENDPOINT
# ==============================================================================
@app.get("/suggest")
@app.post("/suggest")
async def get_bayes_suggestion(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_email = user_session.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s ORDER BY created_at DESC", (user_email,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        experiments = [dict(zip(cols, r)) for r in rows]

        cur.execute("SELECT * FROM suggestion_history WHERE user_email = %s ORDER BY id DESC LIMIT 3", (user_email,))
        sug_rows = cur.fetchall()
        sug_cols = [desc[0] for desc in cur.description] if cur.description else []
        recent_suggestions = [dict(zip(sug_cols, r)) for r in sug_rows]
        cur.close()

        result = optimizer.generate_bayesian_suggestion(experiments, recent_suggestions)
        
        # Save suggestion history to Supabase
        cur_hist = conn.cursor()
        s = result["suggested"]
        cur_hist.execute("""
            INSERT INTO suggestion_history (
                user_email, suggested_rf_power, suggested_pressure,
                suggested_distance, suggested_thickness, suggested_rotation,
                suggested_ar_flow, predicted_xrd_score, predicted_wavelength,
                confidence_score, converged, kappa_used, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            user_email, s["rf_power"], s["working_pressure"],
            s["target_distance"], s["film_thickness"], s["rotation_speed"],
            s["ar_flow"], result["expected"]["xrd_score"],
            result["expected"]["wavelength_shift_estimate"],
            result["confidence"]["score"], result["convergence"]["converged"],
            result["kappa_used"]
        ))
        conn.commit()
        cur_hist.close()

        return JSONResponse(content=result)
    except Exception as e:
        print(f"MatrixAI Engine Error: {e}")
        return JSONResponse(status_code=500, content={"message": f"Optimization engine error: {str(e)}"})
    finally:
        release_db_connection(conn)

# ==============================================================================
# CSV EXPORT & AUTHENTICATION
# ==============================================================================
@app.get("/export/csv")
def export_csv(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s ORDER BY created_at DESC", (user_email,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, r)) for r in rows]
        cur.close()

        if not experiments:
            return RedirectResponse("/")

        df = pd.DataFrame(experiments)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=MatrixAI_Experiments.csv"
        return response
    finally:
        release_db_connection(conn)

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

    request.session["user"] = {
        "id": user_info.get("sub") or user_info.get("id"),
        "email": user_info.get("email"),
        "name": user_info.get("name", "Researcher"),
        "picture": user_info.get("picture", "")
    }
    return RedirectResponse("/")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
