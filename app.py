import io
import os
import smtplib
import datetime
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

SECRET_KEY = os.getenv("SECRET_KEY", "matrix-ai-permanent-production-secret-key-2026")

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=86400 * 30,
    same_site="lax",
    https_only=True
)

templates = Jinja2Templates(directory="templates")

# ==============================================================================
# DYNAMIC SUPABASE POSTGRESQL CONNECTION (PORT 6543 POOLER READY)
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
# DASHBOARD ROUTE (With Pending Approval Check)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        return templates.TemplateResponse(request, "index.html", {"user": None, "experiments": [], "stats": {}})

    user_email = user_session.get("email", "")
    
    # Check if user is approved (Founders bypass this)
    is_approved = user_session.get("is_approved", False)
    if not is_approved and user_email.lower() != FOUNDER_EMAIL.lower():
        # Double check DB in case Admin just approved them while they were on the pending page
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

    # Load Dashboard Data
    experiments = []
    try:
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

    return templates.TemplateResponse(request, "index.html", {
        "user": user_session,
        "experiments": experiments,
        "stats": stats
    })

# ==============================================================================
# AUTH ROUTING (Creates Users Table Automatically)
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
        
        if not user_info:
            user_info = await oauth.google.userinfo(token=token)

        if not user_info:
            return RedirectResponse("/login/google")

        email = user_info.get("email")
        name = user_info.get("name", "Researcher")
        picture = user_info.get("picture", "")
        sub_id = user_info.get("sub") or user_info.get("id", "user_1")

        is_approved = (email.lower() == FOUNDER_EMAIL.lower())

        # Sync user with database and manage approvals
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            # Ensure users table exists
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
                cur.execute(
                    "INSERT INTO users (email, name, picture, is_approved) VALUES (%s, %s, %s, %s)",
                    (email, name, picture, is_approved)
                )
                conn.commit()
            else:
                is_approved = user_row[0]

            cur.close()
        except Exception as db_err:
            print(f"Auth DB Sync Error: {db_err}")
            # Fallback allow founder if DB is temporarily down
            if email.lower() == FOUNDER_EMAIL.lower():
                is_approved = True
        finally:
            release_db_connection(conn)

        request.session["user"] = {
            "id": sub_id,
            "email": email,
            "name": name,
            "picture": picture,
            "is_approved": is_approved
        }
        return RedirectResponse("/", status_code=303)
        
    except Exception as auth_err:
        print(f"OAuth Callback Error: {auth_err}")
        return RedirectResponse("/login/google")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

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
                    try:
                        return float(val)
                    except ValueError:
                        pass
            return default

        def get_str(keys, default=""):
            for k in keys:
                val = form.get(k)
                if val is not None:
                    return str(val).strip()
            return default

        rf_power = get_float(["rf_power_w", "rf_power"], 120.0)
        working_pressure = get_float(["working_pressure_mtorr", "working_pressure", "pressure"], 5.0)
        ar_flow = get_float(["ar_flow_sccm", "ar_flow"], 30.0)
        o2_flow = get_float(["o2_flow_sccm", "o2_flow"], 5.0)
        substrate_temp = get_float(["substrate_temp_c", "substrate_temp"], 300.0)
        target_distance = get_float(["target_substrate_distance_cm", "target_distance"], 7.0)
        sputter_time = get_float(["sputtering_time_min", "sputter_time"], 30.0)
        film_thickness = get_float(["film_thickness_nm", "film_thickness"], 100.0)
        rotation_speed = get_float(["rotation_speed_rpm", "rotation_speed"], 5.0)
        substrate_type = get_str(["substrate_type"], "Si Wafer")
        xrd_phase = get_str(["xrd_phase"], "Amorphous")
        grain_size = get_float(["grain_size_nm", "grain_size"], 10.0)
        h2_response_time = get_float(["h2_response_time_s", "h2_response_time", "h2_response"], 10.0)
        wavelength_shift = get_float(["wavelength_shift_pm", "wavelength_shift"], 0.0)
        batch_notes = get_str(["notes", "batch_notes"], "Manual Entry")

    except Exception as parse_err:
        return HTMLResponse(f"<h1>Form Parsing Error</h1><p>{str(parse_err)}</p>", status_code=400)

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        existing_rows = []
        try:
            cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user_email,))
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
                user_email, rf_power, working_pressure, ar_flow,
                o2_flow, substrate_temp, target_distance, sputter_time,
                film_thickness, rotation_speed, substrate_type, xrd_phase,
                grain_size, h2_response_time, wavelength_shift, batch_notes,
                quality_score, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
        """, (
            user_email, rf_power, working_pressure, ar_flow,
            o2_flow, substrate_temp, target_distance, sputter_time,
            film_thickness, rotation_speed, substrate_type, xrd_phase,
            grain_size, h2_response_time, wavelength_shift, batch_notes,
            quality_score
        ))
        conn.commit()
        cur.close()
        return RedirectResponse("/", status_code=303)
    except Exception as db_err:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        error_html = f"""
        <div style="font-family: sans-serif; padding: 2rem; background: #0f172a; color: white; height: 100vh;">
            <h1 style="color: #ef4444;">Database Insert Failed</h1>
            <p>Supabase rejected the experiment data. Exact PostgreSQL Error:</p>
            <div style="background: #1e293b; padding: 1rem; border-radius: 8px; font-family: monospace; color: #f87171; margin-bottom: 2rem;">
                {str(db_err)}
            </div>
            <a href="/" style="color: #3b82f6; text-decoration: none; border: 1px solid #3b82f6; padding: 10px 15px; border-radius: 5px;">&larr; Go Back</a>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)
    finally:
        if conn:
            release_db_connection(conn)

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
        
        cur.execute("SELECT * FROM experiments WHERE user_email = %s", (user_email,))
        existing_rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        all_exps = [dict(zip(cols, r)) for r in existing_rows]
        
        quality_score = safe_calculate_quality_score(
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
# BAYESIAN OPTIMIZER ENDPOINT
# ==============================================================================
@app.get("/suggest")
@app.post("/suggest")
async def get_bayes_suggestion(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        return JSONResponse(status_code=401, content={"message": "Unauthorized"})

    user_email = user_session.get("email")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        experiments = []
        try:
            cur.execute("SELECT * FROM experiments WHERE user_email = %s ORDER BY created_at DESC", (user_email,))
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

        result = optimizer.generate_bayesian_suggestion(experiments, recent_suggestions)
        
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
# CSV EXPORT
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
        experiments = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            if isinstance(row_dict.get("created_at"), datetime.datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            experiments.append(row_dict)
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

# ==============================================================================
# EMAIL NOTIFICATION HELPER
# ==============================================================================
def send_approval_email(recipient_email: str):
    """Sends a welcome email to the newly approved user via SMTP."""
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        print("SMTP_EMAIL or SMTP_PASSWORD not set in environment. Email skipped.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Access Granted: Welcome to MatrixAI"

        body = """
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #1e293b; line-height: 1.6;">
            <h2 style="color: #3b82f6;">Welcome to MatrixAI</h2>
            <p>Hello,</p>
            <p>Your access request for the MatrixAI Research Laboratory has been <strong>approved</strong> by the founder.</p>
            <p>You can now log in, access the dashboard, and begin optimizing your thin-film deposition experiments.</p>
            <br>
            <a href="https://rock-ai-optimizer.onrender.com/" style="display: inline-block; padding: 12px 24px; background-color: #3b82f6; color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: bold;">Access Dashboard</a>
            <br><br>
            <p>Happy experimenting!</p>
        </div>
        """
        msg.attach(MIMEText(body, 'html'))

        # Standard Gmail SMTP configuration
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Approval email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

# ==============================================================================
# ADMIN PANEL ROUTES
# ==============================================================================
@app.get("/admin/users", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    user = request.session.get("user")
    # Strictly enforce that only the Founder can access this route
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Force create the users table if the Admin accesses the panel before anyone logs in
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

        # Fetch all users and calculate how many experiments they have logged
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
def approve_user(request: Request, email: str):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")
        
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved = TRUE WHERE email = %s", (email,))
        conn.commit()
        cur.close()
        
        # Fire off the automated email directly after successful DB update
        send_approval_email(email)
        
    finally:
        release_db_connection(conn)
    return RedirectResponse("/admin/users", status_code=303)

@app.post("/admin/users/{email}/revoke")
def revoke_user(request: Request, email: str):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return RedirectResponse("/")
        
    # Prevent founder from accidentally revoking themselves
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
