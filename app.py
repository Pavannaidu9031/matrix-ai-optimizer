from fastapi import Depends, FastAPI, Form, HTTPException, Request, BackgroundTasks, UploadFile, File, Header
import io
import os
import json
import re
import smtplib
import datetime
import hashlib
import secrets
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from authlib.integrations.starlette_client import OAuth
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import psycopg2
from psycopg2 import pool
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from google import genai
from groq import AsyncGroq

import optimizer
import advanced_features 
import matrix_ml_engine 

# ==============================================================================
# FASTAPI & PERMANENT SESSION INITIALIZATION
# ==============================================================================
app = FastAPI(title="MatrixAI — Intelligent Materials Optimizer")

SECRET_KEY = os.getenv("SECRET_KEY", "matrix-ai-permanent-production-secret-key-2026")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=86400 * 30,
    same_site="lax",
    https_only=True
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ==============================================================================
# DYNAMIC SUPABASE POSTGRESQL CONNECTION 
# ==============================================================================
connection_pool = None

def get_db_connection():
    global connection_pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL environment variable is missing in Render settings.")
    
    if "sslmode=" not in db_url:
        db_url += "?sslmode=require" if "?" not in db_url else "&sslmode=require"

    if connection_pool is None:
        try:
            connection_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=db_url)
        except Exception as e:
            print(f"Pool creation error: {e}. Opening direct connection.")
            return psycopg2.connect(db_url)
            
    try:
        conn = connection_pool.getconn()
        if conn.closed != 0:
            return psycopg2.connect(db_url)
        return conn
    except Exception as pool_err:
        print(f"Pool getconn error: {pool_err}. Opening direct connection.")
        return psycopg2.connect(db_url)

def release_db_connection(conn):
    global connection_pool
    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    elif conn:
        try:
            conn.close()
        except Exception:
            pass

def safe_calculate_quality_score(xrd_phase, wavelength_shift, h2_response_time, grain_size, existing_exps):
    try:
        formatted_exps = []
        for e in existing_exps:
            item = dict(e)
            item["h2_response_time_s"] = item.get("h2_response_time") or item.get("h2_response_time_s")
            item["wavelength_shift_pm"] = item.get("wavelength_shift") or item.get("wavelength_shift_pm")
            item["grain_size_nm"] = item.get("grain_size") or item.get("grain_size_nm")
            formatted_exps.append(item)

        return optimizer.calculate_quality_score(
            xrd_phase, wavelength_shift, h2_response_time, grain_size, formatted_exps
        )
    except Exception as err:
        print(f"Quality score fallback: {err}")
        return 50.0

def ensure_material_column(cur):
    try:
        cur.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS target_material VARCHAR DEFAULT 'Generic'")
    except Exception:
        pass

def ensure_branch_column(cur):
    try:
        cur.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS branch_name VARCHAR DEFAULT 'main'")
    except Exception:
        pass

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

class ExperimentModel(BaseModel):
    target_material: str = "Generic"
    rf_power: float
    working_pressure: float
    ar_flow: float
    o2_flow: float
    substrate_temp: float
    target_distance: float
    sputter_time_s: float
    film_thickness: Optional[float] = None
    rotation_speed: float = 5.0
    substrate_type: str = "Si Wafer"
    xrd_phase: str = "Amorphous"
    grain_size: Optional[float] = None
    h2_response_time: float
    wavelength_shift: Optional[float] = None
    batch_notes: Optional[str] = None
    branch_name: str = "main"

class ChatRequest(BaseModel):
    message: str
    provider: str  

# ==============================================================================
# DASHBOARD ROUTE (COMMAND CENTER)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request, branch: str = 'main'):
    user_session = request.session.get("user")
    if not user_session:
        return templates.TemplateResponse(request, "index.html", {"user": None, "experiments": [], "stats": {}, "branches": [], "current_branch": "main"})

    user_email = user_session.get("email", "")
    is_approved = user_session.get("is_approved", False)
    
    if not is_approved and user_email.lower() != FOUNDER_EMAIL.lower():
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT is_approved FROM users WHERE email = %s", (user_email,))
            row = cur.fetchone()
            if row and row[0]:
                is_approved = True
                user_session["is_approved"] = True
                request.session["user"] = user_session
            cur.close()
        except Exception:
            pass
        finally:
            release_db_connection(conn)
            
        if not is_approved:
            return templates.TemplateResponse(request, "pending.html", {"user": user_session})

    experiments = []
    branches = ['main']
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            ensure_material_column(cur)
            ensure_branch_column(cur)
            conn.commit()

            try:
                cur.execute("SELECT DISTINCT branch_name FROM experiments WHERE user_email = %s", (user_email,))
                fetched_branches = [r[0] for r in cur.fetchall() if r[0]]
                if 'main' not in fetched_branches:
                    fetched_branches.insert(0, 'main')
                branches = fetched_branches
            except Exception as e:
                print(f"Error fetching branches: {e}")

            cur.execute("""
                SELECT id, user_email, target_material, rf_power, working_pressure, ar_flow, o2_flow,
                       substrate_temp, target_distance, sputter_time_s, film_thickness,
                       rotation_speed, substrate_type, xrd_phase, grain_size,
                       h2_response_time, wavelength_shift, batch_notes, quality_score, created_at, branch_name
                FROM experiments
                WHERE user_email = %s AND branch_name = %s
                ORDER BY created_at DESC
            """, (user_email, branch))
            
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                if isinstance(row_dict.get("created_at"), datetime.datetime):
                    row_dict["created_at"] = row_dict["created_at"].isoformat()
                experiments.append(row_dict)
                
            cur.close()
        finally:
            release_db_connection(conn)
    except Exception as err:
        print(f"Error loading index page experiments: {err}")
        experiments = []

    total_runs = len(experiments)
    monoclinic_runs = sum(1 for e in experiments if str(e.get("xrd_phase", "")).strip() == "Monoclinic")
    
    shifts = []
    for e in experiments:
        w_val = e.get("wavelength_shift") if e.get("wavelength_shift") is not None else e.get("wavelength_shift_pm")
        if w_val is not None:
            try:
                shifts.append(float(w_val))
            except (ValueError, TypeError):
                pass

    best_shift = max(shifts) if shifts else 0.0
    runs_remaining = max(0, 25 - total_runs)

    stats = {
        "total_runs": total_runs,
        "monoclinic_runs": monoclinic_runs,
        "best_shift": round(best_shift, 1),
        "runs_remaining": runs_remaining
    }

    latest_failure = None
    if experiments:
        latest_failure = advanced_features.analyze_failure(experiments, experiments[0])

    return templates.TemplateResponse(request, "index.html", {
        "user": user_session,
        "experiments": experiments,
        "stats": stats,
        "branches": branches,
        "current_branch": branch,
        "failure_analysis": latest_failure
    })

# ==============================================================================
# MULTI-PAGE VIEWS (EXPERIMENTS & OPTIMIZER)
# ==============================================================================
@app.get("/experiments-view", response_class=HTMLResponse)
def experiments_page(request: Request, branch: str = 'main'):
    user = request.session.get("user")
    if not user: return RedirectResponse("/login/google", status_code=303)
    user_email = user.get("email")

    experiments = []
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_email, target_material, rf_power, working_pressure, ar_flow, o2_flow,
                   substrate_temp, target_distance, sputter_time_s, film_thickness,
                   rotation_speed, substrate_type, xrd_phase, grain_size,
                   h2_response_time, wavelength_shift, batch_notes, quality_score, created_at, branch_name
            FROM experiments
            WHERE user_email = %s AND branch_name = %s
            ORDER BY created_at DESC
        """, (user_email, branch))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        for row in rows:
            row_dict = dict(zip(cols, row))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            experiments.append(row_dict)
        cur.close()
    except Exception as e:
        print(f"Error fetching experiments for view: {e}")
    finally:
        release_db_connection(conn)

    return templates.TemplateResponse(request, "experiments.html", {
        "user": user, "experiments": experiments, "current_branch": branch
    })

@app.get("/optimizer-view", response_class=HTMLResponse)
def optimizer_page(request: Request, branch: str = 'main'):
    user = request.session.get("user")
    if not user: return RedirectResponse("/login/google", status_code=303)
    return templates.TemplateResponse(request, "optimizer.html", {"user": user, "current_branch": branch})

# ==============================================================================
# AUTH ROUTING
# ==============================================================================
@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        if not user_info: user_info = await oauth.google.userinfo(token=token)
        if not user_info: return RedirectResponse("/login/google")

        email = user_info.get("email")
        name = user_info.get("name", "Researcher")
        picture = user_info.get("picture", "")
        sub_id = user_info.get("sub") or user_info.get("id", "user_1")

        is_approved = (email.lower() == FOUNDER_EMAIL.lower())

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    picture VARCHAR,
                    is_approved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

            cur.execute("SELECT is_approved FROM users WHERE email = %s", (email,))
            user_row = cur.fetchone()

            if not user_row:
                cur.execute("INSERT INTO users (email, name, picture, is_approved) VALUES (%s, %s, %s, %s)",
                            (email, name, picture, is_approved))
                conn.commit()
            else:
                is_approved = user_row[0]
            cur.close()
        except Exception as db_err:
            if email.lower() == FOUNDER_EMAIL.lower(): is_approved = True
        finally:
            release_db_connection(conn)

        request.session["user"] = {"id": sub_id, "email": email, "name": name, "picture": picture, "is_approved": is_approved}
        return RedirectResponse("/", status_code=303)
    except Exception as auth_err:
        return RedirectResponse("/login/google")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

# ==============================================================================
# DIGITAL TWIN SANDBOX SIMULATION ENDPOINT
# ==============================================================================
@app.post("/api/sandbox/simulate")
async def simulate_sandbox(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
        
    try:
        body = await request.json()
        target_material = body.get("target_material", "Generic")
        params = body.get("params", []) 
        branch = body.get("branch", "main")
        
        if len(params) != 6: return JSONResponse({"error": "Invalid parameter array length."}, status_code=400)
            
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            ensure_branch_column(cur)
            cur.execute("SELECT * FROM experiments WHERE user_email = %s AND branch_name = %s", (user.get("email"), branch))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            experiments = [dict(zip(cols, r)) for r in rows]
            cur.close()
        finally:
            release_db_connection(conn)
            
        sim_result = optimizer.simulate_sandbox_point(experiments, target_material, [float(p) for p in params])
        return JSONResponse({"success": True, "simulation": sim_result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================================================================
# SECURE EXPERIMENT DELETION ENDPOINT
# ==============================================================================
@app.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int, request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    user_email = user.get("email")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Verify ownership
        cur.execute("SELECT user_email FROM experiments WHERE id = %s", (experiment_id,))
        row = cur.fetchone()
        
        if not row:
            return JSONResponse({"error": "not_found"}, status_code=404)
        if row[0] != user_email:
            return JSONResponse({"error": "unauthorized"}, status_code=403)
            
        # Log to Audit Table
        cur.execute("INSERT INTO audit_log (action, experiment_id, user_email) VALUES (%s, %s, %s)", 
                    ("DELETE", experiment_id, user_email))
        
        # Execute Delete
        cur.execute("DELETE FROM experiments WHERE id = %s AND user_email = %s", (experiment_id, user_email))
        conn.commit()
        cur.close()
        
        return JSONResponse({"success": True, "message": "Experiment deleted"})
    except Exception as e:
        if conn: conn.rollback()
        return JSONResponse({"error": "delete_failed", "message": str(e)}, status_code=500)
    finally:
        release_db_connection(conn)

@app.post("/admin/experiments/bulk-delete")
async def bulk_delete_experiments(request: Request):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return JSONResponse({"error": "unauthorized"}, status_code=403)
        
    body = await request.json()
    ids = body.get("experiment_ids", [])
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for exp_id in ids:
            cur.execute("INSERT INTO audit_log (action, experiment_id, user_email) VALUES (%s, %s, %s)", 
                        ("ADMIN_BULK_DELETE", exp_id, user.get("email")))
            cur.execute("DELETE FROM experiments WHERE id = %s", (exp_id,))
        conn.commit()
        cur.close()
        return JSONResponse({"success": True})
    finally:
        release_db_connection(conn)

# ==============================================================================
# FORM SUBMISSION ROUTE (/add)
# ==============================================================================
@app.post("/add")
async def add_experiment_form(request: Request):
    user = request.session.get("user")
    if not user or (not user.get("is_approved") and user.get("email", "").lower() != FOUNDER_EMAIL.lower()):
        return RedirectResponse("/login/google", status_code=303)

    user_email = user.get("email")
    
    try:
        form = await request.form()
        
        def get_float(keys, default=0.0):
            for k in keys:
                val = form.get(k)
                if val is not None and str(val).strip() != "":
                    try: return float(val)
                    except ValueError: pass
            return default

        def get_str(keys, default=""):
            for k in keys:
                val = form.get(k)
                if val is not None: return str(val).strip()
            return default

        target_material = get_str(["target_material"], "Generic")
        rf_power = get_float(["rf_power_w", "rf_power"], 120.0)
        working_pressure = get_float(["working_pressure_mtorr", "working_pressure", "pressure"], 5.0)
        ar_flow = get_float(["ar_flow_sccm", "ar_flow"], 30.0)
        o2_flow = get_float(["o2_flow_sccm", "o2_flow"], 5.0)
        substrate_temp = get_float(["substrate_temp_c", "substrate_temp"], 300.0)
        target_distance = get_float(["target_substrate_distance_cm", "target_distance"], 7.0)
        
        # Dual time calculation
        sputter_min = get_float(["sputtering_time_min", "sputter_time_min"], 0.0)
        sputter_sec = get_float(["sputtering_time_sec", "sputter_time_sec"], 0.0)
        sputter_time_s = (sputter_min * 60.0) + sputter_sec
        if sputter_time_s <= 0:
            sputter_time_s = 1800.0 # fallback to 30 mins

        film_thickness = get_float(["film_thickness_nm", "film_thickness"], 100.0)
        rotation_speed = get_float(["rotation_speed_rpm", "rotation_speed"], 5.0)
        substrate_type = get_str(["substrate_type"], "Si Wafer")
        xrd_phase = get_str(["xrd_phase"], "Amorphous")
        grain_size = get_float(["grain_size_nm", "grain_size"], 10.0)
        h2_response_time = get_float(["h2_response_time_s", "h2_response_time", "h2_response"], 10.0)
        wavelength_shift = get_float(["wavelength_shift_pm", "wavelength_shift"], 0.0)
        batch_notes = get_str(["notes", "batch_notes"], "Manual Entry")
        branch_name = get_str(["branch_name", "branch"], "main")

    except Exception as parse_err:
        return HTMLResponse(f"<h1>Form Parsing Error</h1><p>{str(parse_err)}</p>", status_code=400)

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        ensure_material_column(cur)
        ensure_branch_column(cur)
        conn.commit()

        existing_rows = []
        try:
            cur.execute("SELECT * FROM experiments WHERE user_email = %s AND branch_name = %s", (user_email, branch_name))
            existing_rows = cur.fetchall()
        except Exception:
            pass 

        cols = [desc[0] for desc in cur.description] if cur.description and existing_rows else []
        all_exps = [dict(zip(cols, r)) for r in existing_rows]

        quality_score = safe_calculate_quality_score(
            xrd_phase, wavelength_shift, h2_response_time, grain_size, all_exps
        )

        cur.execute("""
            INSERT INTO experiments (
                user_email, target_material, rf_power, working_pressure, ar_flow,
                o2_flow, substrate_temp, target_distance, sputter_time_s,
                film_thickness, rotation_speed, substrate_type, xrd_phase,
                grain_size, h2_response_time, wavelength_shift, batch_notes,
                quality_score, branch_name, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
        """, (
            user_email, target_material, rf_power, working_pressure, ar_flow,
            o2_flow, substrate_temp, target_distance, sputter_time_s,
            film_thickness, rotation_speed, substrate_type, xrd_phase,
            grain_size, h2_response_time, wavelength_shift, batch_notes,
            quality_score, branch_name
        ))
        conn.commit()
        cur.close()
        return RedirectResponse(f"/?branch={branch_name}", status_code=303)
    except Exception as db_err:
        if conn:
            try: conn.rollback()
            except Exception: pass
        return HTMLResponse(f"<h1>Database Insert Failed</h1><p>{str(db_err)}</p>", status_code=500)
    finally:
        if conn: release_db_connection(conn)

# ==============================================================================
# PERMANENT EXPERIMENT SAVE (JSON API /experiments)
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
        ensure_branch_column(cur)
        
        cur.execute("SELECT * FROM experiments WHERE user_email = %s AND branch_name = %s", (user_email, data.branch_name))
        existing_rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        all_exps = [dict(zip(cols, r)) for r in existing_rows]
        
        quality_score = safe_calculate_quality_score(
            data.xrd_phase, data.wavelength_shift, data.h2_response_time, data.grain_size, all_exps
        )

        cur.execute("""
            INSERT INTO experiments (
                user_email, rf_power, working_pressure, ar_flow,
                o2_flow, substrate_temp, target_distance, sputter_time_s,
                film_thickness, rotation_speed, substrate_type, xrd_phase,
                grain_size, h2_response_time, wavelength_shift, batch_notes,
                quality_score, branch_name, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            ) RETURNING id
        """, (
            user_email, data.rf_power, data.working_pressure, data.ar_flow,
            data.o2_flow, data.substrate_temp, data.target_distance, data.sputter_time_s,
            data.film_thickness, data.rotation_speed, data.substrate_type, data.xrd_phase,
            data.grain_size, data.h2_response_time, data.wavelength_shift, data.batch_notes,
            quality_score, data.branch_name
        ))
        
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        return JSONResponse({
            "success": True,
            "id": new_id,
            "quality_score": quality_score,
            "message": f"Experiment saved permanently to Supabase (Branch: {data.branch_name})"
        })
    except Exception as e:
        if conn: conn.rollback()
        return JSONResponse({"error": "save_failed", "message": str(e)}, status_code=500)
    finally:
        release_db_connection(conn)

# ==============================================================================
# FETCH EXPERIMENTS API
# ==============================================================================
@app.get("/experiments")
async def get_experiments(request: Request, branch: str = "main"):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    user_email = user.get("email")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        ensure_branch_column(cur)
        cur.execute("""
            SELECT id, user_email, rf_power, working_pressure, ar_flow, o2_flow,
                   substrate_temp, target_distance, sputter_time_s, film_thickness,
                   rotation_speed, substrate_type, xrd_phase, grain_size,
                   h2_response_time, wavelength_shift, batch_notes, quality_score, created_at, branch_name
            FROM experiments
            WHERE user_email = %s AND branch_name = %s
            ORDER BY created_at DESC
        """, (user_email, branch))
        
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        experiments = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            experiments.append(row_dict)
            
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

# ==============================================================================
# AI OPTIMIZER ENDPOINT
# ==============================================================================
@app.get("/suggest")
@app.post("/suggest")
async def get_ai_suggestion(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        return JSONResponse(status_code=401, content={"message": "Unauthorized"})

    user_email = user_session.get("email")
    target_material = "Generic"
    branch = "main"

    if request.method == "POST":
        try:
            body = await request.json()
            target_material = body.get("target_material", "Generic")
            branch = body.get("branch", "main")
        except Exception:
            pass

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ensure_branch_column(cur)
        
        experiments = []
        try:
            cur.execute("SELECT * FROM experiments WHERE user_email = %s AND branch_name = %s ORDER BY created_at DESC", (user_email, branch))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            for row in rows:
                row_dict = dict(zip(cols, row))
                if isinstance(row_dict.get("created_at"), datetime.datetime):
                    row_dict["created_at"] = row_dict["created_at"].isoformat()
                experiments.append(row_dict)
        except Exception:
            pass

        recent_suggestions = []
        try:
            cur.execute("SELECT * FROM suggestion_history WHERE user_email = %s ORDER BY id DESC LIMIT 3", (user_email,))
            sug_rows = cur.fetchall()
            sug_cols = [desc[0] for desc in cur.description] if cur.description else []
            for row in sug_rows:
                row_dict = dict(zip(sug_cols, row))
                if isinstance(row_dict.get("created_at"), datetime.datetime):
                    row_dict["created_at"] = row_dict["created_at"].isoformat()
                recent_suggestions.append(row_dict)
        except Exception:
            pass

        cur.close()

        result = optimizer.generate_bayesian_suggestion(
            experiments, 
            recent_suggestions, 
            target_material=target_material
        )
        
        try:
            cur_hist = conn.cursor()
            s = result.get("suggested", {})
            cur_hist.execute("""
                INSERT INTO suggestion_history (
                    user_email, suggested_rf_power, suggested_pressure,
                    suggested_distance, suggested_thickness, suggested_rotation,
                    suggested_ar_flow, predicted_xrd_score, predicted_wavelength,
                    confidence_score, converged, kappa_used, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                user_email, s.get("rf_power"), s.get("working_pressure"),
                s.get("target_distance"), s.get("film_thickness"), s.get("rotation_speed"),
                s.get("ar_flow"), result.get("expected", {}).get("xrd_score"),
                result.get("expected", {}).get("wavelength_shift_estimate"),
                result.get("confidence", {}).get("score"), result.get("convergence", {}).get("converged"),
                result.get("kappa_used")
            ))
            conn.commit()
            cur_hist.close()
        except Exception:
            pass

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Optimization engine error: {str(e)}"})
    finally:
        if conn:
            release_db_connection(conn)

# ==============================================================================
# MULTIMODAL VISION AI ENDPOINT
# ==============================================================================
@app.post("/api/vision")
async def analyze_image_vision(request: Request, image: UploadFile = File(...), analysis_type: str = Form(...)):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return JSONResponse({"error": "GEMINI_API_KEY not configured."}, status_code=500)
        
        image_bytes = await image.read()
        client = genai.Client(api_key=api_key)
        
        prompt = (f"Analyze this {analysis_type} image for materials science. "
            "If it is an XRD plot, identify the dominant phase (must be exactly: Monoclinic, Partial, or Amorphous). "
            "If it is an SEM/FESEM micrograph, estimate the average grain size in nm (return just a numeric value). "
            "Return strictly a JSON object with a single key 'extracted_value'.")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type=image.content_type)],
        )
        
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return {"success": True, "extracted_value": json.loads(match.group(0)).get("extracted_value")}
        else:
            return {"success": False, "message": "Failed to parse AI output. Try again."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================================================================
# LITERATURE INGESTION ENDPOINT (WITH AUTOMATIC RETRY LOGIC)
# ==============================================================================
@app.post("/api/literature/upload")
async def upload_literature_pdf(request: Request, file: UploadFile = File(...)):
    user = request.session.get("user")
    
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return JSONResponse({"error": "unauthorized"}, status_code=401)
        
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({"error": "GEMINI_API_KEY not configured."}, status_code=500)
            
        pdf_bytes = await file.read()
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "You are an expert Materials Scientist. Read this research paper and extract the optimal or primary "
            "experimental parameters for thin-film deposition. "
            "Return strictly a valid JSON object with the following keys and numerical values (no units): "
            "'target_material' (string), 'rf_power' (float), 'working_pressure' (float), 'ar_flow' (float), "
            "'o2_flow' (float), 'substrate_temp' (float), 'target_distance' (float), 'sputter_time_s' (float in seconds), "
            "'film_thickness' (float), 'rotation_speed' (float), 'xrd_phase' (string: Monoclinic, Partial, or Amorphous), "
            "'grain_size' (float), 'h2_response_time' (float), 'wavelength_shift' (float)."
        )
        
        response = None
        last_exception = None
        
        # Retry loop for transient 503 high-demand errors on gemini-2.5-flash
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        prompt,
                        genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                    ],
                )
                if response and response.text:
                    break
            except Exception as model_err:
                last_exception = model_err
                print(f"Attempt {attempt+1} failed: {model_err}. Retrying in 2 seconds...")
                import asyncio
                await asyncio.sleep(2)
                continue
                
        if not response or not response.text:
            error_msg = str(last_exception) if last_exception else "Gemini endpoint experiencing high demand."
            return JSONResponse({"error": f"Extraction Failed (503): {error_msg}"}, status_code=503)
        
        reply_text = response.text
        match = re.search(r'\{.*\}', reply_text, re.DOTALL)
        if not match:
            return JSONResponse({"error": "Failed to parse AI JSON response structure."}, status_code=500)
            
        cleaned_json_str = re.sub(r'```json|```', '', match.group(0)).strip()
        data = json.loads(cleaned_json_str)
        
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            ensure_material_column(cur)
            
            quality_score = 75.0 
            
            cur.execute("""
                INSERT INTO experiments (
                    user_email, target_material, rf_power, working_pressure, ar_flow,
                    o2_flow, substrate_temp, target_distance, sputter_time_s,
                    film_thickness, rotation_speed, substrate_type, xrd_phase,
                    grain_size, h2_response_time, wavelength_shift, batch_notes,
                    quality_score, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
            """, (
                user.get("email"), 
                data.get("target_material", "Generic"),
                float(data.get("rf_power", 120.0)),
                float(data.get("working_pressure", 5.0)),
                float(data.get("ar_flow", 30.0)),
                float(data.get("o2_flow", 5.0)),
                float(data.get("substrate_temp", 300.0)),
                float(data.get("target_distance", 7.0)),
                float(data.get("sputter_time_s", 1800.0)),
                float(data.get("film_thickness", 100.0)),
                float(data.get("rotation_speed", 5.0)),
                "Si Wafer",
                data.get("xrd_phase", "Monoclinic"),
                float(data.get("grain_size", 15.0)),
                float(data.get("h2_response_time", 10.0)),
                float(data.get("wavelength_shift", 100.0)),
                "Literature Prior (AI Extracted)",
                quality_score
            ))
            conn.commit()
            cur.close()
        finally:
            release_db_connection(conn)
            
        return JSONResponse({"success": True, "extracted_data": data})
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================================================================
# CSV EXPORT
# ==============================================================================
@app.get("/export/csv")
def export_csv(request: Request):
    user = request.session.get("user")
    if not user: raise HTTPException(status_code=401, detail="Unauthorized")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s ORDER BY created_at DESC", (user.get("email"),))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        experiments = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            if "sputter_time" in row_dict:
                row_dict["sputter_time_s"] = row_dict.pop("sputter_time")
            experiments.append(row_dict)
        cur.close()

        if not experiments: return RedirectResponse("/")

        df = pd.DataFrame(experiments)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=MatrixAI_Experiments.csv"
        return response
    finally:
        release_db_connection(conn)

# ==============================================================================
# ADMIN PANEL ROUTES
# ==============================================================================
@app.get("/admin/users", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email VARCHAR PRIMARY KEY,
                name VARCHAR,
                picture VARCHAR,
                is_approved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()

        cur.execute("""
            SELECT u.email, u.name, u.picture, u.is_approved, u.created_at,
                   COUNT(e.id) as exp_count
            FROM users u
            LEFT JOIN experiments e ON u.email = e.user_email
            GROUP BY u.email, u.name, u.picture, u.is_approved, u.created_at
            ORDER BY u.created_at DESC
        """)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        users_list = []
        
        for r in rows:
            row_dict = dict(zip(cols, r))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].strftime("%Y-%m-%d")
            users_list.append(row_dict)
            
        cur.close()
        return templates.TemplateResponse(request, "admin.html", {"user": user, "users_list": users_list})
    except Exception as e:
        return HTMLResponse(f"Error loading admin panel. Details: {e}")
    finally:
        release_db_connection(conn)

@app.post("/admin/users/{email}/approve")
def approve_user(request: Request, email: str, background_tasks: BackgroundTasks):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")
        
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved = TRUE WHERE email = %s", (email,))
        conn.commit()
        cur.close()
    finally:
        release_db_connection(conn)
    return RedirectResponse("/admin/users", status_code=303)

@app.post("/admin/users/{email}/revoke")
def revoke_user(request: Request, email: str):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")
        
    if email.lower() == FOUNDER_EMAIL.lower():
        return RedirectResponse("/admin/users", status_code=303) 
        
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved = FALSE WHERE email = %s", (email,))
        conn.commit()
        cur.close()
    finally:
        release_db_connection(conn)
    return RedirectResponse("/admin/users", status_code=303)

# ==============================================================================
# AI SCIENTIFIC CO-PILOT ENDPOINT
# ==============================================================================
@app.post("/api/chat")
async def chat_with_agent(request: Request, data: ChatRequest):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "not_authenticated"}, status_code=401)
        
    user_email = user.get("email")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        ensure_material_column(cur)
        cur.execute("""
            SELECT target_material, rf_power, working_pressure, target_distance, sputter_time_s, 
                   film_thickness, rotation_speed, ar_flow, xrd_phase, quality_score 
            FROM experiments WHERE user_email = %s ORDER BY created_at DESC LIMIT 5
        """, (user_email,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        recent_exps = [dict(zip(cols, row)) for row in rows]
        cur.close()
    except Exception as e:
        recent_exps = []
        print(f"Error fetching context for AI: {e}")
    finally:
        release_db_connection(conn)
        
    system_context = (
        "You are an expert Materials Science AI Assistant built into the MatrixAI platform. "
        "The user is optimizing thin-film deposition parameters using a PVD/CVD process. "
        "Provide concise, highly scientific, and actionable answers."
    )
    
    if recent_exps:
        system_context += f"\nHere is the data from their most recent experiments: {recent_exps}\n"
    
    full_prompt = f"{system_context}\n\nUser Question: {data.message}"
    
    try:
        if data.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                 return JSONResponse({"error": "GEMINI_API_KEY not found in environment."}, status_code=500)
            
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
            )
            return {"reply": response.text}
            
        elif data.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                 return JSONResponse({"error": "GROQ_API_KEY not found in environment."}, status_code=500)
            
            client = AsyncGroq(api_key=api_key)
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
            )
            return {"reply": chat_completion.choices[0].message.content}
            
        else:
            return JSONResponse({"error": "Invalid AI provider selected."}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==============================================================================
# NEW ADVANCED MODULE ROUTES
# ==============================================================================

@app.get("/thesis", response_class=HTMLResponse)
def thesis_dashboard(request: Request):
    user = request.session.get("user")
    if not user: return RedirectResponse("/")
    return templates.TemplateResponse(request, "thesis.html", {"user": user})

@app.post("/api/thesis/generate")
async def api_generate_thesis(request: Request):
    user = request.session.get("user")
    body = await request.json()
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user.get("email"),))
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)
        
    content = advanced_features.generate_thesis_content(experiments, body.get("section"))
    return JSONResponse({"success": True, "content": content})

@app.get("/scheduler", response_class=HTMLResponse)
def scheduler_dashboard(request: Request):
    return templates.TemplateResponse(request, "scheduler.html", {"user": request.session.get("user")})

@app.post("/api/scheduler/plan")
async def api_schedule_plan(request: Request):
    body = await request.json()
    schedule = advanced_features.schedule_experiments([], float(body.get("hours", 0)), int(body.get("days", 1)))
    return JSONResponse({"success": True, "schedule": schedule})

@app.get("/supervisor", response_class=HTMLResponse)
def supervisor_dashboard(request: Request):
    """Real-Time Collaboration: Read-only view for supervisors."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50")
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)
    return templates.TemplateResponse(request, "supervisor.html", {"user": request.session.get("user"), "experiments": experiments})

@app.get("/api/equipment/health")
async def get_equipment_health(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT film_thickness, sputter_time_s, working_pressure FROM experiments WHERE user_email = %s ORDER BY created_at DESC LIMIT 20", (user.get("email"),))
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
    finally:
        release_db_connection(conn)
        
    health = advanced_features.check_equipment_health(experiments)
    return JSONResponse({"success": True, "health": health})

@app.post("/api/sensor/predict")
async def predict_sensor(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    body = await request.json()
    predictions = advanced_features.predict_sensor_performance(body.get("params", {}))
    return JSONResponse({"success": True, "predictions": predictions})

@app.get("/publication", response_class=HTMLResponse)
def publication_dashboard(request: Request):
    user = request.session.get("user")
    if not user: return RedirectResponse("/")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user.get("email"),))
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
    finally:
        release_db_connection(conn)
        
    readiness = advanced_features.check_publication_readiness(experiments)
    return templates.TemplateResponse(request, "publication.html", {"user": user, "readiness": readiness})

class CommentModel(BaseModel):
    experiment_id: int
    comment_text: str

@app.post("/api/comments/add")
async def add_comment(request: Request, data: CommentModel):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO experiment_comments (experiment_id, user_email, comment_text, created_at)
            VALUES (%s, %s, %s, NOW()) RETURNING id
        """, (data.experiment_id, user.get("email"), data.comment_text))
        conn.commit()
        cur.close()
        return JSONResponse({"success": True, "message": "Comment added."})
    finally:
        release_db_connection(conn)

# ==============================================================================
# FEATURE 6: PUBLIC REST API WITH API KEY AUTH
# ==============================================================================
def verify_api_key(x_api_key: str = Header(...)):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
        cur.execute("SELECT user_email FROM api_keys WHERE api_key_hash = %s", (key_hash,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or unauthorized API Key")
        
        cur.execute("UPDATE api_keys SET usage_count = usage_count + 1 WHERE api_key_hash = %s", (key_hash,))
        conn.commit()
        return user[0]
    finally:
        release_db_connection(conn)

@app.post("/api/v1/suggest")
async def api_v1_suggest(request: Request, body: dict, api_user: str = Depends(verify_api_key)):
    target_material = body.get("material", "Generic")
    experiments = body.get("experiments", [])
    result = optimizer.generate_bayesian_suggestion(experiments, [], target_material=target_material)
    return JSONResponse(result)

@app.post("/api/v1/predict")
async def api_v1_predict(body: dict, api_user: str = Depends(verify_api_key)):
    params = body.get("parameters", {})
    predictions = advanced_features.predict_sensor_performance(params)
    return JSONResponse({"predicted": predictions})

# ==============================================================================
# FEATURE 1: DNA FINGERPRINTING & WHAT-IF ANALYZER
# ==============================================================================
@app.post("/api/experiments/fingerprint")
async def get_fingerprint(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "unauthorized"}, status_code=401)
    
    body = await request.json()
    target_exp = body.get("experiment")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user['email'],))
        cols = [desc[0] for desc in cur.description]
        
        all_exps = []
        for row in cur.fetchall():
            row_dict = dict(zip(cols, row))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            all_exps.append(row_dict)
            
    finally:
        release_db_connection(conn)
        
    similar = matrix_ml_engine.get_dna_fingerprint(target_exp, all_exps)
    return JSONResponse({"success": True, "similar": similar})

@app.post("/api/experiments/whatif")
async def calculate_whatif(request: Request):
    body = await request.json()
    params = body.get("params", [])
    target_material = body.get("material", "Generic")
    
    sim = optimizer.simulate_sandbox_point([], target_material, params)
    return JSONResponse({"success": True, "simulation": sim})

# ==============================================================================
# FEATURE 3: AGENT MODE
# ==============================================================================
@app.get("/agent", response_class=HTMLResponse)
def agent_dashboard(request: Request):
    user = request.session.get("user")
    if not user: return RedirectResponse("/")
    return templates.TemplateResponse(request, "agent.html", {"user": user})

@app.post("/api/agent/campaign")
async def generate_campaign(request: Request):
    body = await request.json()
    budget = int(body.get("budget", 25))
    deadline = body.get("deadline", "2026-12-31")
    
    campaign = matrix_ml_engine.generate_agent_campaign(budget, deadline, [])
    return JSONResponse({"success": True, "campaign": campaign})

# ==============================================================================
# API DOCS PAGE
# ==============================================================================
@app.get("/api/docs", response_class=HTMLResponse)
def apidocs_page(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse(request, "apidocs.html", {"user": user})

# ==============================================================================
# MISSING DASHBOARD API ENDPOINTS (PHASE MAP, NOISE, RECIPE)
# ==============================================================================
@app.post("/api/optimizer/phase-map")
async def api_phase_map(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    body = await request.json()
    target_material = body.get("target_material", "Generic")
    param_x = body.get("param_x", "rf_power")
    param_y = body.get("param_y", "working_pressure")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user.get("email"),))
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)
        
    phase_map = optimizer.generate_phase_map(experiments, target_material, param_x, param_y)
    return JSONResponse({"success": True, "phase_map": phase_map})

@app.post("/api/optimizer/calibrate-noise")
async def api_calibrate_noise(request: Request):
    user = request.session.get("user")
    if not user: return JSONResponse({"error": "not_authenticated"}, status_code=401)
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT wavelength_shift, wavelength_shift_pm FROM experiments WHERE user_email = %s", (user.get("email"),))
        cols = [desc[0] for desc in cur.description]
        experiments = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)
        
    calib = optimizer.calibrate_noise_variance(experiments)
    return JSONResponse({"success": True, "calibration": calib})

@app.post("/api/optimizer/recipe")
async def api_generate_recipe(request: Request):
    body = await request.json()
    target_material = body.get("target_material", "Generic")
    params = body.get("params", {})
    recipe = optimizer.generate_synthesis_recipe(params, target_material)
    return JSONResponse({"success": True, "recipe": recipe})
