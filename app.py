import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import pandas as pd
import io
import sqlite3

import database
import optimizer

# Initialize FastAPI App
app = FastAPI(title="Rock AI — WO3 Sputtering Optimizer")

# Add Session Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "rock-ai-super-secret-key-change-this-in-production")
)

templates = Jinja2Templates(directory="templates")

# Initialize Database
database.init_db()

# ==============================================================================
# CONFIGURATION & CREDENTIALS
# ==============================================================================
FOUNDER_EMAIL = os.getenv("FOUNDER_EMAIL", "pavannaidu9031@gmail.com")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "pavannaidu9031@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def send_approval_notification(user_info, base_url="http://127.0.0.1:8000"):
    approve_link = f"{base_url}/admin/approve/{user_info['id']}"
    reject_link = f"{base_url}/admin/reject/{user_info['id']}"

    print("\n" + "=" * 60)
    print(f"🔔 [NEW ACCESS REQUEST] User: {user_info['name']} ({user_info['email']})")
    print(f"👉 Admin Panel: {base_url}/admin/users")
    print(f"👉 Approve Link: {approve_link}")
    print(f"👉 Reject Link:  {reject_link}")
    print("=" * 60 + "\n")

    if SENDER_PASSWORD and SENDER_PASSWORD != "YOUR_GMAIL_APP_PASSWORD":
        try:
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = FOUNDER_EMAIL
            msg["Subject"] = f"🔔 [Rock AI] Access Approval Needed: {user_info['name']}"

            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #030712; color: #ffffff; padding: 20px;">
                <div style="max-width: 520px; background: #0f172a; padding: 24px; border-radius: 16px; margin: 0 auto; border: 1px solid rgba(255,255,255,0.1);">
                    <h2 style="color: #38bdf8; margin-top: 0;">Rock AI — Founder Access Request</h2>
                    <p style="color: #cbd5e1; font-size: 14px;">A new researcher is requesting access to your WO₃ Sputtering Optimization workspace:</p>
                    
                    <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 12px; margin: 20px 0; border: 1px solid rgba(255,255,255,0.1);">
                        <p style="margin: 6px 0; font-size: 13px;"><strong>Name:</strong> {user_info['name']}</p>
                        <p style="margin: 6px 0; font-size: 13px;"><strong>Email:</strong> {user_info['email']}</p>
                        <p style="margin: 6px 0; font-size: 13px;"><strong>User ID:</strong> {user_info['id']}</p>
                    </div>

                    <div style="margin-top: 24px;">
                        <a href="{approve_link}" style="background: #22c55e; color: #000000; padding: 12px 20px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block; margin-right: 12px;">✅ Approve Access</a>
                        <a href="{reject_link}" style="background: #ef4444; color: #ffffff; padding: 12px 20px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">❌ Reject Request</a>
                    </div>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html_body, "html"))

            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"SMTP Email Error: {e}")

# ==============================================================================
# MAIN ROUTE & GATEKEEPER
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user_session = request.session.get("user")
    
    if not user_session:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"user": None, "experiments": [], "suggestion": None, "chart_html": None}
        )

    db_user = database.get_user_by_id(user_session["id"])
    if not db_user:
        request.session.clear()
        return RedirectResponse("/")

    if db_user["status"] == "pending":
        return templates.TemplateResponse(
            request,
            "pending.html",
            {"user": db_user}
        )

    if db_user["status"] == "rejected":
        request.session.clear()
        return HTMLResponse(
            "<div style='background:#030712; color:#ef4444; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:20px;'>"
            "<h1>Access Restricted</h1><p style='color:#94a3b8;'>Your access request was not approved by the founder.</p>"
            "<a href='/' style='color:#38bdf8;'>Return to Homepage</a></div>",
            status_code=403
        )

    raw_experiments = database.get_experiments_by_user(db_user["id"])
    experiments = [dict(row) for row in raw_experiments] if raw_experiments else []
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": db_user,
            "experiments": experiments,
            "suggestion": None,
            "chart_html": None
        }
    )

# ==============================================================================
# AUTHENTICATION ROUTES
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
            return RedirectResponse("/")

        db_user = database.get_or_create_user(user_info, founder_email=FOUNDER_EMAIL)
        request.session["user"] = {"id": db_user["id"], "email": db_user["email"], "name": db_user["name"]}

        if db_user["status"] == "pending":
            base_url = str(request.base_url).rstrip("/")
            send_approval_notification(db_user, base_url=base_url)

        return RedirectResponse("/")
    except Exception as e:
        print(f"Auth Callback Error: {e}")
        return RedirectResponse("/")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")

# ==============================================================================
# FOUNDER ADMIN MANAGEMENT DASHBOARD
# ==============================================================================
@app.get("/admin/users", response_class=HTMLResponse)
def view_pending_users(request: Request):
    user = request.session.get("user")
    if not user or user.get("email", "").lower() != FOUNDER_EMAIL.lower():
        return HTMLResponse("<h2>Unauthorized: Founder Access Only</h2>", status_code=403)
    
    conn = sqlite3.connect("experiments.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, status, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()

    rows = ""
    for u in users:
        status_color = "#22c55e" if u[3] == "approved" else "#eab308" if u[3] == "pending" else "#ef4444"
        actions = (
            f'<a href="/admin/approve/{u[0]}" style="color:#22c55e; font-weight:bold; margin-right:12px; text-decoration:none;">[✅ Approve]</a> '
            f'<a href="/admin/reject/{u[0]}" style="color:#ef4444; font-weight:bold; text-decoration:none;">[❌ Reject]</a>'
        ) if u[3] == "pending" else u[3].capitalize()
        
        rows += f"""
        <tr style='border-bottom: 1px solid rgba(255,255,255,0.08);'>
            <td style='padding: 12px;'>{u[1]}</td>
            <td style='padding: 12px; color: #94a3b8;'>{u[2]}</td>
            <td style='padding: 12px; color: {status_color}; font-weight: bold;'>{u[3]}</td>
            <td style='padding: 12px;'>{actions}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rock AI — Founder Management</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body style="background:#030712; color:white; font-family:sans-serif; padding:40px;">
        <div style="max-width:850px; margin:0 auto; background:#0f172a; padding:28px; border-radius:20px; border:1px solid rgba(255,255,255,0.1);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <div>
                    <h2 style="color:#38bdf8; margin:0; font-size:22px;">Rock AI — User Control Panel</h2>
                    <p style="color:#64748b; font-size:12px; margin-top:4px;">Logged in as Founder ({FOUNDER_EMAIL})</p>
                </div>
                <a href="/" style="background:#1e293b; color:white; padding:8px 16px; border-radius:10px; text-decoration:none; font-size:12px;">← Workspace Dashboard</a>
            </div>
            <table style="width:100%; text-align:left; border-collapse:collapse; font-size:13px;">
                <thead>
                    <tr style="color:#64748b; border-bottom:1px solid rgba(255,255,255,0.1); text-transform:uppercase; font-size:10px;">
                        <th style="padding:10px;">Name</th>
                        <th style="padding:10px;">Email</th>
                        <th style="padding:10px;">Status</th>
                        <th style="padding:10px;">Quick Action</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/admin/approve/{user_id}", response_class=HTMLResponse)
def approve_user(user_id: str):
    database.update_user_status(user_id, "approved")
    return HTMLResponse(
        "<div style='background:#030712; color:#22c55e; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:20px;'>"
        "<div style='border:1px solid #22c55e; padding:30px; border-radius:16px; background:rgba(34,197,94,0.05);'>"
        "<h1>✅ User Access Approved</h1>"
        "<p style='color:#cbd5e1;'>This researcher can now access the Rock AI WO₃ Sputtering Workspace.</p>"
        "<a href='/admin/users' style='color:#38bdf8; font-weight:bold;'>← Return to User Control Panel</a>"
        "</div></div>"
    )

@app.get("/admin/reject/{user_id}", response_class=HTMLResponse)
def reject_user(user_id: str):
    database.update_user_status(user_id, "rejected")
    return HTMLResponse(
        "<div style='background:#030712; color:#ef4444; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:20px;'>"
        "<div style='border:1px solid #ef4444; padding:30px; border-radius:16px; background:rgba(239,68,68,0.05);'>"
        "<h1>❌ User Access Rejected</h1>"
        "<p style='color:#cbd5e1;'>The access request has been declined.</p>"
        "<a href='/admin/users' style='color:#38bdf8; font-weight:bold;'>← Return to User Control Panel</a>"
        "</div></div>"
    )

# ==============================================================================
# EXPERIMENT ACTIONS
# ==============================================================================
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
        notes=notes
    )
    return RedirectResponse("/", status_code=303)

    from fastapi.responses import JSONResponse

@app.get("/suggest")
@app.post("/suggest")
async def get_bayes_suggestion(request: Request):
    user_session = request.session.get("user")
    if not user_session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_experiments = database.get_experiments_by_user(user_session["id"])
    experiments = [dict(row) for row in raw_experiments] if raw_experiments else []

    result = optimizer.generate_bayesian_suggestion(experiments)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    
    return JSONResponse(content=result)

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
    response.headers["Content-Disposition"] = "attachment; filename=Rock_AI_WO3_Experiments.csv"
    return response
