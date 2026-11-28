"""Smoke tests for the FastAPI app, run against the in-memory store.

Covers page rendering, the JSON endpoints the UI calls, what /add writes,
the login/approval gate, and that the migrations cover every model field.
Run with: python -m pytest tests
"""
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from itsdangerous import TimestampSigner  # noqa: E402

from matrixai.config import Settings  # noqa: E402
from matrixai.db import migration_files  # noqa: E402
from matrixai.main import create_app  # noqa: E402
from matrixai.models import EXPERIMENT_FIELDS  # noqa: E402
from matrixai.store import MemoryStore  # noqa: E402
from test_optimizer import make_runs  # noqa: E402

EMAIL = "tester@example.com"
FOUNDER = "founder@example.com"
SECRET = "test-secret"
SETTINGS = Settings(secret_key=SECRET, database_url=None, founder_email=FOUNDER, google_client_id="x",
                    google_client_secret="y", gemini_api_key=None, groq_api_key=None, https_only=False,
                    run_migrations=False)


def _seed(store: MemoryStore, email=EMAIL, n=8):
    records = []
    for r in make_runs(n, "WO3"):
        records.append(dict(r, o2_flow=5.0, substrate_temp=25.0, sputter_time_s=1500.0, grain_size=None,
                            h2_response_time=12.0, batch_notes="t", branch_name="main", pd_thickness=0.0,
                            sputter_mode="RF", quality_score=50.0))
    store.add_experiments(email, records)


def _login(client, email, user_id="1"):
    session = {"user": {"id": user_id, "email": email, "name": "Tester", "picture": "", "is_approved": True}}
    payload = base64.b64encode(json.dumps(session).encode())
    client.cookies.set("session", TimestampSigner(SECRET).sign(payload).decode())


def _make_client(approved=True, logged_in=True):
    store = MemoryStore()
    store.record_login(EMAIL, "Tester", "", approve_if_new=approved)
    _seed(store)
    app = create_app(SETTINGS, store=store)
    c = TestClient(app, base_url="http://testserver")
    if logged_in:
        _login(c, EMAIL)
    c.store, c.app_ = store, app
    return c


@pytest.fixture
def client():
    return _make_client()


PAGES = ["/", "/experiments-view", "/optimizer-view", "/bulk-import", "/cost-tracker", "/publication",
         "/supervisor", "/thesis", "/scheduler", "/agent", "/api/docs"]


@pytest.mark.parametrize("path", PAGES)
def test_pages_render_and_scripts_parse(client, path):
    res = client.get(path)
    assert res.status_code == 200, res.text[:500]
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed; skipping JS syntax check")
    for i, script in enumerate(re.findall(r"<script(?: [^>]*)?>(.*?)</script>", res.text, re.S)):
        if not script.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(script)
        out = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        os.unlink(f.name)
        assert out.returncode == 0, f"{path} script #{i}: {out.stderr}"


def test_admin_page_for_founder():
    c = _make_client()
    _login(c, FOUNDER, "0")
    res = c.get("/admin/users")
    assert res.status_code == 200 and EMAIL in res.text


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True, "schema": "memory"}


def test_suggest_endpoint(client):
    res = client.post("/suggest", json={"target_material": "WO3", "branch": "main", "acquisition": "ts"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["acquisition_used"] == "ts"
    assert data["training_data"]["real_runs"] == 8
    assert "thickness_curve" in data["digital_twin"]


def test_suggest_records_history_with_user_id(client):
    data = client.post("/suggest", json={"target_material": "WO3"}).json()
    sug = client.store.suggestions[-1]
    assert data["suggestion_id"] == sug["id"]
    assert sug["user_id"] == "1" and sug["user_email"] == EMAIL


def test_sandbox_accepts_ui_payload(client):
    res = client.post("/api/sandbox/simulate", json={"target_material": "WO3", "params": [120, 8, 5, 150, 5, 25, 8], "branch": "main"})
    assert res.status_code == 200, res.text
    assert res.json()["simulation"]["real_runs_used"] == 8
    assert client.post("/api/sandbox/simulate", json={"params": [1, 2]}).status_code == 400


def test_whatif_uses_user_runs(client):
    res = client.post("/api/experiments/whatif", json={"material": "WO3", "params": [120, 8, 5, 150, 5, 25]})
    assert res.status_code == 200
    assert res.json()["simulation"]["real_runs_used"] == 8


def test_validation_phase_map_and_noise(client):
    v = client.get("/api/model/validation?branch=main&target_material=WO3").json()["validation"]
    assert v["n"] == 8 and "coverage_2sigma" in v
    m = client.post("/api/optimizer/phase-map", json={"target_material": "WO3", "param_x": "rf_power", "param_y": "film_thickness"}).json()
    assert len(m["phase_map"]["points"]) == 8
    n = client.post("/api/optimizer/calibrate-noise", json={"target_material": "WO3"}).json()
    assert n["calibration"]["noise_uw"] is not None


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/api/optimizer/recipe", {"params": {"rf_power": 120}, "target_material": "WO3"}),
    ("POST", "/api/optimizer/pd-recipe", {"target_pd_thickness_nm": 8}),
    ("POST", "/api/agent/campaign", {"budget": 10}),
    ("POST", "/api/sensor/predict", {"params": {"rf_power": 120}}),
    ("GET", "/api/equipment/health", None),
    ("POST", "/api/scheduler/plan", {"hours": 8, "days": 2}),
    ("POST", "/api/thesis/generate", {"section": "abstract"}),
    ("POST", "/api/experiments/fingerprint", {"experiment": {"rf_power": 120, "working_pressure": 5}}),
    ("GET", "/experiments?branch=main", None),
])
def test_tool_endpoints_answer(client, method, path, body):
    res = client.request(method, path, json=body)
    assert res.status_code == 200, f"{path}: {res.text[:300]}"
    assert res.json()["success"] is True


FORM = {"target_material": "WO3", "rf_power_w": "120", "working_pressure_mtorr": "8", "ar_flow_sccm": "30",
        "o2_flow_sccm": "5", "substrate_temp_c": "25", "target_substrate_distance_cm": "5",
        "h2_response_time_s": "15", "branch_name": "main"}


def test_add_stores_real_time_and_null_for_unmeasured(client):
    form = dict(FORM, sputtering_time_s="2400", xrd_phase="MONOCLINIC", light_intensity_uw="", grain_size_nm="")
    res = client.post("/add", data=form, follow_redirects=False)
    assert res.status_code == 303
    row = client.store.rows[-1]
    assert row["sputter_time_s"] == 2400.0          # was always stored as 1800
    assert row["xrd_phase"] == "Monoclinic"          # was stored verbatim and scored as amorphous
    assert row["light_intensity_uw"] is None         # was stored as 0.0
    assert row["grain_size"] is None                 # was stored as 10.0
    assert row["film_thickness"] is None             # was stored as 100.0
    assert row["quality_score"] is not None


def test_add_minutes_and_seconds(client):
    client.post("/add", data=dict(FORM, sputtering_time_min="10", sputtering_time_sec="30"), follow_redirects=False)
    assert client.store.rows[-1]["sputter_time_s"] == 630.0


def test_add_stores_suggestion_link(client):
    form = dict(FORM, suggestion_id="555", suggestion_option="Max Quality", predicted_intensity_uw="42.5",
                predicted_intensity_sigma="3.1", predicted_phase="MONOCLINIC", followed_suggestion="1",
                recovery_time_s="70.5", response_curve=json.dumps({"curve": {"t": [0, 1], "y": [5, 4]}, "snr": 9}))
    assert client.post("/add", data=form, follow_redirects=False).status_code == 303
    row = client.store.rows[-1]
    assert row["suggestion_id"] == 555 and row["suggestion_option"] == "Max Quality"
    assert row["predicted_intensity_uw"] == 42.5 and row["predicted_phase"] == "Monoclinic"
    assert row["followed_suggestion"] is True and row["recovery_time_s"] == 70.5
    assert json.loads(row["response_curve"])["snr"] == 9


def test_add_without_suggestion_is_unlinked_and_rejects_bad_curve(client):
    form = dict(FORM, suggestion_option="Max Quality", response_curve="{not json")
    client.post("/add", data=form, follow_redirects=False)
    row = client.store.rows[-1]
    assert row["suggestion_id"] is None and row["suggestion_option"] is None
    assert row["response_curve"] is None


def test_json_add_requires_rf_power(client):
    assert client.post("/experiments", json={"target_material": "WO3"}).status_code == 422
    res = client.post("/experiments", json={"rf_power": 110, "branch": "b2"})
    assert res.status_code == 200 and client.store.rows[-1]["branch_name"] == "b2"


def test_bulk_import_goes_to_chosen_branch(client):
    rows = [{"rf_power": 100 + i, "film_thickness": 150, "xrd_phase": "Partial"} for i in range(3)]
    res = client.post("/api/experiments/bulk", json={"experiments": rows, "branch": "imported"})
    assert res.status_code == 200 and res.json()["inserted_count"] == 3
    assert [r["branch_name"] for r in client.store.rows[-3:]] == ["imported"] * 3
    assert client.post("/api/experiments/bulk", json={"experiments": []}).status_code == 400


def test_export_csv(client):
    res = client.get("/export/csv")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert lines[0].startswith("id,") and len(lines) == 9


def test_delete_own_run_only(client):
    other = client.store.add_experiments("someone@else.com", [{"rf_power": 1, "branch_name": "main"}])[0]
    assert client.delete(f"/experiments/{other}").status_code == 403
    assert client.delete("/experiments/99999").status_code == 404
    own = client.store.rows[0]["id"]
    assert client.delete(f"/experiments/{own}").status_code == 200
    assert ("DELETE", own, EMAIL) in client.store.audit


def test_bulk_delete_is_founder_only(client):
    assert client.post("/admin/experiments/bulk-delete", json={"experiment_ids": [1]}).status_code == 403
    _login(client, FOUNDER, "0")
    res = client.post("/admin/experiments/bulk-delete", json={"experiment_ids": [1, 2]})
    assert res.json()["deleted"] == 2


def test_comments(client):
    own = client.store.rows[0]["id"]
    assert client.post("/api/comments/add", json={"experiment_id": own, "comment_text": "nice"}).status_code == 200
    assert client.post("/api/comments/add", json={"experiment_id": 99999, "comment_text": "x"}).status_code == 404
    assert client.post("/api/comments/add", json={"experiment_id": own}).status_code == 422


def test_analyze_curve_endpoint(client):
    from test_sensor_analysis import synthetic, to_csv, ON, OFF, DROP
    t, y = synthetic()
    res = client.post("/api/sensor/analyze-curve", files={"file": ("run12.csv", to_csv(t, y), "text/csv")},
                      data={"h2_on_s": str(ON), "h2_off_s": str(OFF)})
    assert res.status_code == 200, res.text
    a = res.json()["analysis"]
    assert a["filename"] == "run12.csv"
    assert a["intensity_change_uw"] == pytest.approx(DROP, rel=0.03)
    bad = client.post("/api/sensor/analyze-curve", files={"file": ("x.csv", b"nothing here", "text/csv")})
    assert bad.status_code == 422 and bad.json()["success"] is False


def test_measurements_update_own_run(client):
    run_id = client.store.rows[2]["id"]
    res = client.post(f"/experiments/{run_id}/measurements", json={
        "light_intensity_uw": 7.9, "h2_response_time": 27.5, "recovery_time_s": 66.0,
        "response_curve": {"curve": {"t": [0, 1, 2], "y": [5, 4, 4]}}})
    assert res.status_code == 200, res.text
    row = client.store.rows[2]
    assert (row["light_intensity_uw"], row["h2_response_time"], row["recovery_time_s"]) == (7.9, 27.5, 66.0)
    assert json.loads(row["response_curve"])["curve"]["t"] == [0, 1, 2]
    assert client.post("/experiments/99999/measurements", json={"light_intensity_uw": 1}).status_code == 404
    assert client.post(f"/experiments/{run_id}/measurements", json={}).status_code == 400


def test_track_record_endpoint(client):
    tr = client.get("/api/suggestions/track-record").json()["track_record"]
    assert tr["linked_runs"] == 0 and tr["manual_runs"] == 8 and tr["points"] == []


def test_ai_endpoints_report_missing_keys(client):
    res = client.post("/api/chat", json={"message": "hi", "provider": "gemini"})
    assert res.status_code == 500 and "GEMINI_API_KEY" in res.text
    assert client.post("/api/chat", json={"message": "hi", "provider": "nope"}).status_code == 400
    assert client.post("/api/chat", json={}).status_code == 422


def test_literature_upload_is_founder_only(client):
    res = client.post("/api/literature/upload", files={"file": ("p.pdf", b"%PDF-1.4", "application/pdf")})
    assert res.status_code == 403


def test_supervisor_requires_login():
    anon = _make_client(logged_in=False)
    res = anon.get("/supervisor", headers={"accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303 and "/login/google" in res.headers["location"]


def test_supervisor_scopes_non_founder_to_own_runs(client):
    client.store.add_experiments("someone@else.com", [{"rf_power": 1, "branch_name": "main", "batch_notes": "SECRETRUN"}])
    assert "someone@else.com" not in client.get("/supervisor").text
    _login(client, FOUNDER, "0")
    assert "someone@else.com" in client.get("/supervisor").text


# Every non-public route, with a minimal request for each.
GATED = [("POST", "/api/chat"), ("POST", "/api/vision"), ("POST", "/api/thesis/generate"),
         ("POST", "/api/scheduler/plan"), ("POST", "/api/optimizer/recipe"), ("POST", "/api/agent/campaign"),
         ("POST", "/api/experiments/whatif"), ("POST", "/suggest"), ("GET", "/experiments"),
         ("POST", "/api/sandbox/simulate"), ("GET", "/export/csv"), ("POST", "/api/literature/upload"),
         ("POST", "/api/experiments/bulk"), ("DELETE", "/experiments/1")]


@pytest.mark.parametrize("method,path", GATED)
def test_gate_rejects_anonymous(method, path):
    c = _make_client(logged_in=False)
    assert c.request(method, path).status_code == 401


@pytest.mark.parametrize("method,path", GATED)
def test_gate_rejects_unapproved(method, path):
    c = _make_client(approved=False)
    res = c.request(method, path)
    assert res.status_code == 403 and res.json()["error"] == "not_approved"


def test_unapproved_pages_go_to_pending():
    c = _make_client(approved=False)
    res = c.get("/optimizer-view", headers={"accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == "/"
    home = c.get("/")
    assert home.status_code == 200 and "approval" in home.text.lower()


def test_newly_approved_user_is_let_in_without_relogin():
    c = _make_client(approved=False)
    c.store.set_approval(EMAIL, True)              # approved in the DB after login
    assert c.get("/experiments").status_code == 200


def test_public_routes_stay_open():
    c = _make_client(logged_in=False)
    assert c.get("/").status_code == 200
    assert c.get("/static/manifest.json").status_code == 200
    assert c.post("/api/v1/predict", json={}).status_code == 422   # reaches the API-key check


def test_rest_api_with_key():
    c = _make_client(logged_in=False)
    c.store.api_keys[hashlib.sha256(b"k1").hexdigest()] = {"user_email": EMAIL}
    assert c.post("/api/v1/predict", json={}, headers={"X-API-Key": "bad"}).status_code == 401
    res = c.post("/api/v1/predict", json={"parameters": {"rf_power": 120}}, headers={"X-API-Key": "k1"})
    assert res.status_code == 200 and "predicted" in res.json()


def test_revoked_user_loses_access_despite_approved_cookie():
    c = _make_client(approved=True)
    assert c.get("/experiments").status_code == 200
    c.store.set_approval(EMAIL, False)               # founder revokes in the DB
    c.app_.state.approvals.forget(EMAIL)             # what the revoke route does
    assert c.get("/experiments").status_code == 403
    assert "approval" in c.get("/").text.lower()     # dashboard now shows the pending page


def test_revoke_route_clears_cache():
    c = _make_client(approved=True)
    assert c.get("/experiments").status_code == 200  # caches "approved"
    founder = TestClient(c.app_, base_url="http://testserver")
    _login(founder, FOUNDER, "0")
    assert founder.post(f"/admin/users/{EMAIL}/revoke", follow_redirects=False).status_code == 303
    assert c.store.users[EMAIL]["is_approved"] is False
    assert c.get("/experiments").status_code == 403


def test_founder_cannot_be_revoked():
    c = _make_client()
    _login(c, FOUNDER, "0")
    c.post(f"/admin/users/{FOUNDER}/revoke", follow_redirects=False)
    assert c.get("/experiments").status_code == 200


def test_cached_answer_expires():
    c = _make_client(approved=True)
    assert c.get("/experiments").status_code == 200
    c.store.set_approval(EMAIL, False)               # revoked by some other path, no cache clear
    assert c.get("/experiments").status_code == 200  # still inside the TTL
    c.app_.state.approvals.ttl_s = 0
    assert c.get("/experiments").status_code == 403


def test_db_outage_fails_closed_without_history():
    c = _make_client(approved=True)
    c.store.fail = True
    assert c.get("/api/model/validation").status_code == 403


def test_migrations_cover_every_model_field():
    sql = "\n".join(p.read_text(encoding="utf-8") for p in migration_files()).lower()
    missing = [f.name for f in EXPERIMENT_FIELDS if not re.search(rf"add column if not exists\s+{f.name}\s", sql)]
    assert not missing, f"model fields with no migration: {missing}"


def test_add_stores_fiber_power_before_and_after(client):
    client.post("/add", data=dict(FORM, fiber_power_before_uw="412.5", fiber_power_after_uw="198"), follow_redirects=False)
    row = client.store.rows[-1]
    assert row["fiber_power_before_uw"] == 412.5 and row["fiber_power_after_uw"] == 198.0
    client.post("/add", data=dict(FORM, fiber_power_before_uw="", fiber_power_after_uw=""), follow_redirects=False)
    assert client.store.rows[-1]["fiber_power_before_uw"] is None      # blank = not measured, never 0
    page = client.get("/experiments-view").text
    assert "412.5" in page and "FIBER_P" in page


def test_measurements_can_add_fiber_power_later(client):
    run_id = client.store.rows[0]["id"]
    assert client.post(f"/experiments/{run_id}/measurements", json={"fiber_power_after_uw": 150}).status_code == 200
    assert client.store.rows[0]["fiber_power_after_uw"] == 150.0


def test_legacy_intensity_migration_is_guarded():
    sql = (migration_files()[-1]).read_text(encoding="utf-8")
    assert "light_intensity_uw IS NULL" in sql and "DROP" not in sql.upper()
