"""Smoke tests for the FastAPI app with an in-memory fake database.

Covers page rendering, the JSON endpoints the UI calls, and what /add writes.
Run with: python -m pytest tests
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql://fake")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from itsdangerous import TimestampSigner  # noqa: E402

import app as app_module  # noqa: E402
from test_optimizer import make_runs  # noqa: E402

EMAIL = "tester@example.com"
COLUMNS = ["id", "user_email", "target_material", "rf_power", "working_pressure", "ar_flow", "o2_flow",
           "substrate_temp", "target_distance", "sputter_time_s", "film_thickness", "rotation_speed",
           "substrate_type", "xrd_phase", "grain_size", "h2_response_time", "light_intensity_uw",
           "batch_notes", "quality_score", "created_at", "branch_name", "pd_thickness", "pd_dc_power",
           "pd_distance_cm", "pd_sputter_time_s", "pd_pressure_mtorr", "pd_ar_flow", "pd_o2_flow",
           "core_diameter_um", "sputter_mode", "recovery_time_s", "response_curve", "suggestion_id",
           "suggestion_option", "predicted_intensity_uw", "predicted_intensity_sigma", "predicted_phase",
           "followed_suggestion"]


def _rows():
    rows = []
    for r in make_runs(8, "WO3"):
        r = dict(r, user_email=EMAIL, o2_flow=5.0, substrate_temp=25.0, sputter_time_s=1500.0,
                 grain_size=None, h2_response_time=12.0, batch_notes="t", created_at=None,
                 branch_name="main", pd_thickness=0.0, sputter_mode="RF")
        rows.append(tuple(r.get(c) for c in COLUMNS))
    return rows


class FakeCursor:
    def __init__(self, db):
        self.db, self.description, self._result = db, None, []

    def execute(self, sql, params=None):
        self.db.queries.append((sql, params))
        q = " ".join(sql.split()).upper()
        self.description, self._result = None, []
        if "INFORMATION_SCHEMA.COLUMNS" in q:
            self.description, self._result = [("column_name",)], [(c,) for c in self.db.legacy_columns]
        elif q.startswith("SELECT IS_APPROVED"):
            self.description, self._result = [("is_approved",)], [(self.db.approved,)]
        elif q.startswith("SELECT DISTINCT BRANCH_NAME"):
            self.description, self._result = [("branch_name",)], [("main",)]
        elif "FROM EXPERIMENTS" in q and q.startswith("SELECT"):
            m = re.match(r"SELECT (.*?) FROM EXPERIMENTS", " ".join(sql.split()), re.I)
            wanted = COLUMNS if m.group(1).strip() == "*" else [c.strip() for c in m.group(1).split(",")]
            idx = [COLUMNS.index(c) for c in wanted]
            self.description = [(c,) for c in wanted]
            rows = self.db.rows
            if "WHERE ID = %S" in q:
                rows = [r for r in rows if r[0] == params[0]]
            self._result = [tuple(r[i] for i in idx) for r in rows]
        elif q.startswith("SELECT") and "SUGGESTION_HISTORY" in q:
            self.description, self._result = [("id",)], []
        elif q.startswith("INSERT INTO SUGGESTION_HISTORY"):
            self.db.suggestion_inserts.append((sql, params))
            self.description, self._result = [("id",)], [(555,)]
        elif q.startswith("UPDATE EXPERIMENTS"):
            self.db.updates.append((sql, params))
        elif q.startswith("INSERT INTO EXPERIMENTS"):
            self.db.inserts.append((sql, params))
            self.description, self._result = [("id",)], [(999,)]

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        pass


class FakeConn:
    closed = 0

    def __init__(self, db):
        self.db = db

    def cursor(self):
        return FakeCursor(self.db)

    def commit(self):
        pass

    def rollback(self):
        self.db.rollbacks += 1

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.rows, self.queries, self.inserts = _rows(), [], []
        self.approved = True
        self.suggestion_inserts, self.updates = [], []
        self.legacy_columns = ["wavelength_shift", "light_intensity_uw"]   # what the live table has
        self.rollbacks = 0


def _make_client(monkeypatch, approved=True, logged_in=True):
    db = FakeDB()
    db.approved = approved
    monkeypatch.setattr(app_module, "get_db_connection", lambda: FakeConn(db))
    monkeypatch.setattr(app_module, "release_db_connection", lambda conn: None)
    c = TestClient(app_module.app, base_url="https://testserver")
    if logged_in:
        session = {"user": {"id": "1", "email": EMAIL, "name": "Tester", "picture": "", "is_approved": approved}}
        payload = base64.b64encode(json.dumps(session).encode())
        c.cookies.set("session", TimestampSigner(os.environ["SECRET_KEY"]).sign(payload).decode())
    c.db = db
    return c


@pytest.fixture(autouse=True)
def _fresh_approval_cache():
    app_module._approval_cache.clear()
    yield
    app_module._approval_cache.clear()


@pytest.fixture
def client(monkeypatch):
    return _make_client(monkeypatch)


@pytest.mark.parametrize("path", ["/", "/experiments-view", "/optimizer-view"])
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


def test_suggest_endpoint(client):
    res = client.post("/suggest", json={"target_material": "WO3", "branch": "main", "acquisition": "ts"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["acquisition_used"] == "ts"
    assert data["training_data"]["real_runs"] == 8
    assert "thickness_curve" in data["digital_twin"]


def test_sandbox_accepts_ui_payload(client):
    res = client.post("/api/sandbox/simulate", json={"target_material": "WO3", "params": [120, 8, 5, 150, 5, 25, 8], "branch": "main"})
    assert res.status_code == 200, res.text
    assert res.json()["simulation"]["real_runs_used"] == 8


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


def test_add_stores_real_time_and_null_for_unmeasured(client):
    form = {"target_material": "WO3", "rf_power_w": "120", "working_pressure_mtorr": "8", "ar_flow_sccm": "30",
            "o2_flow_sccm": "5", "substrate_temp_c": "25", "target_substrate_distance_cm": "5",
            "sputtering_time_s": "2400", "xrd_phase": "MONOCLINIC", "h2_response_time_s": "15",
            "light_intensity_uw": "", "grain_size_nm": "", "branch_name": "main"}
    res = client.post("/add", data=form, follow_redirects=False)
    assert res.status_code == 303
    sql, params = client.db.inserts[-1]
    cols = [c.strip() for c in re.search(r"INSERT INTO experiments \((.*?)\) VALUES", sql, re.S).group(1).split(",")]
    row = dict(zip(cols, params))
    assert row["sputter_time_s"] == 2400.0          # was always stored as 1800
    assert row["xrd_phase"] == "Monoclinic"          # was stored verbatim and scored as amorphous
    assert row["light_intensity_uw"] is None         # was stored as 0.0
    assert row["grain_size"] is None                 # was stored as 10.0
    assert row["film_thickness"] is None             # was stored as 100.0


def test_supervisor_requires_login(client):
    anon = TestClient(app_module.app, base_url="https://testserver")
    res = anon.get("/supervisor", headers={"accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303 and "/login/google" in res.headers["location"]


def test_supervisor_scopes_non_founder_to_own_runs(client):
    assert client.get("/supervisor").status_code == 200
    sql, params = next((q, p) for q, p in reversed(client.db.queries) if "FROM experiments" in q)
    assert "user_email = %s" in sql and params == (EMAIL,)


# Every non-public route, with a minimal request for each.
GATED = [("POST", "/api/chat"), ("POST", "/api/vision"), ("POST", "/api/thesis/generate"),
         ("POST", "/api/scheduler/plan"), ("POST", "/api/optimizer/recipe"), ("POST", "/api/agent/campaign"),
         ("POST", "/api/experiments/whatif"), ("POST", "/suggest"), ("GET", "/experiments"),
         ("POST", "/api/sandbox/simulate"), ("GET", "/export/csv"), ("POST", "/api/literature/upload"),
         ("POST", "/api/experiments/bulk"), ("DELETE", "/experiments/1")]


@pytest.mark.parametrize("method,path", GATED)
def test_gate_rejects_anonymous(monkeypatch, method, path):
    c = _make_client(monkeypatch, logged_in=False)
    assert c.request(method, path).status_code == 401


@pytest.mark.parametrize("method,path", GATED)
def test_gate_rejects_unapproved(monkeypatch, method, path):
    c = _make_client(monkeypatch, approved=False)
    res = c.request(method, path)
    assert res.status_code == 403 and res.json()["error"] == "not_approved"


def test_unapproved_pages_go_to_pending(monkeypatch):
    c = _make_client(monkeypatch, approved=False)
    res = c.get("/optimizer-view", headers={"accept": "text/html"}, follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == "/"
    assert c.get("/").status_code == 200          # pending page itself stays reachable


def test_newly_approved_user_is_let_in_without_relogin(monkeypatch):
    c = _make_client(monkeypatch, approved=False)
    c.db.approved = True                           # approved in the DB after login
    assert c.get("/experiments").status_code == 200


def test_public_routes_stay_open(monkeypatch):
    c = _make_client(monkeypatch, logged_in=False)
    assert c.get("/").status_code == 200
    assert c.get("/static/manifest.json").status_code == 200
    assert c.post("/api/v1/predict", json={}).status_code == 422   # reaches the API-key check


def test_revoked_user_loses_access_despite_approved_cookie(monkeypatch):
    c = _make_client(monkeypatch, approved=True)     # cookie says approved
    assert c.get("/experiments").status_code == 200
    c.db.approved = False                            # founder revokes in the DB
    app_module.forget_approval(EMAIL)                # what the revoke route does
    assert c.get("/experiments").status_code == 403
    assert "approval" in c.get("/").text.lower()     # dashboard now shows the pending page


def test_revoke_route_clears_cache(monkeypatch):
    c = _make_client(monkeypatch, approved=True)
    assert c.get("/experiments").status_code == 200  # caches "approved"
    founder = TestClient(app_module.app, base_url="https://testserver")
    session = {"user": {"id": "0", "email": app_module.FOUNDER_EMAIL, "name": "F", "picture": "", "is_approved": True}}
    founder.cookies.set("session", TimestampSigner(os.environ["SECRET_KEY"]).sign(base64.b64encode(json.dumps(session).encode())).decode())
    c.db.approved = False                            # the UPDATE the route runs (fake DB ignores it)
    assert founder.post(f"/admin/users/{EMAIL}/revoke", follow_redirects=False).status_code == 303
    assert c.get("/experiments").status_code == 403


def test_cached_answer_expires(monkeypatch):
    c = _make_client(monkeypatch, approved=True)
    assert c.get("/experiments").status_code == 200
    c.db.approved = False                            # revoked by some other path, no cache clear
    assert c.get("/experiments").status_code == 200  # still inside the TTL
    monkeypatch.setattr(app_module, "APPROVAL_CACHE_TTL_S", 0)
    assert c.get("/experiments").status_code == 403


def test_db_outage_fails_closed_without_history(monkeypatch):
    c = _make_client(monkeypatch, approved=True)
    def broken():
        raise RuntimeError("db down")
    monkeypatch.setattr(app_module, "get_db_connection", broken)
    assert c.get("/api/model/validation").status_code == 403



def _insert_row(db):
    sql, params = db.inserts[-1]
    cols = [c.strip() for c in re.search(r"INSERT INTO experiments \((.*?)\) VALUES", sql, re.S).group(1).split(",")]
    return dict(zip(cols, params))


def test_suggest_records_history_with_user_id(client):
    data = client.post("/suggest", json={"target_material": "WO3"}).json()
    assert data["suggestion_id"] == 555
    sql, params = client.db.suggestion_inserts[-1]
    assert "user_id" in sql and params[0] == "1"      # session user id; was missing (NOT NULL)


def test_add_stores_suggestion_link(client):
    form = {"target_material": "WO3", "rf_power_w": "120", "working_pressure_mtorr": "8", "ar_flow_sccm": "30",
            "o2_flow_sccm": "5", "substrate_temp_c": "25", "target_substrate_distance_cm": "5",
            "h2_response_time_s": "15", "branch_name": "main",
            "suggestion_id": "555", "suggestion_option": "Max Quality", "predicted_intensity_uw": "42.5",
            "predicted_intensity_sigma": "3.1", "predicted_phase": "MONOCLINIC", "followed_suggestion": "1",
            "recovery_time_s": "70.5", "response_curve": json.dumps({"curve": {"t": [0, 1], "y": [5, 4]}, "snr": 9})}
    assert client.post("/add", data=form, follow_redirects=False).status_code == 303
    row = _insert_row(client.db)
    assert row["suggestion_id"] == 555 and row["suggestion_option"] == "Max Quality"
    assert row["predicted_intensity_uw"] == 42.5 and row["predicted_phase"] == "Monoclinic"
    assert row["followed_suggestion"] is True and row["recovery_time_s"] == 70.5
    assert json.loads(row["response_curve"])["snr"] == 9


def test_add_without_suggestion_is_unlinked_and_rejects_bad_curve(client):
    form = {"target_material": "WO3", "rf_power_w": "120", "h2_response_time_s": "15", "branch_name": "main",
            "suggestion_option": "Max Quality", "response_curve": "{not json"}
    client.post("/add", data=form, follow_redirects=False)
    row = _insert_row(client.db)
    assert row["suggestion_id"] is None and row["suggestion_option"] is None
    assert row["response_curve"] is None


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
    res = client.post("/experiments/3/measurements", json={
        "light_intensity_uw": 7.9, "h2_response_time": 27.5, "recovery_time_s": 66.0,
        "response_curve": {"curve": {"t": [0, 1, 2], "y": [5, 4, 4]}}})
    assert res.status_code == 200, res.text
    sql, params = client.db.updates[-1]
    assert params[0] == 7.9 and params[1] == 27.5 and params[2] == 66.0
    assert params[-2:] == (3, EMAIL)                    # scoped to this user's row
    assert client.post("/experiments/99999/measurements", json={"light_intensity_uw": 1}).status_code == 404
    assert client.post("/experiments/3/measurements", json={}).status_code == 400


def test_track_record_endpoint(client):
    tr = client.get("/api/suggestions/track-record").json()["track_record"]
    assert tr["linked_runs"] == 0 and tr["manual_runs"] == 8 and tr["points"] == []



def test_column_migration_survives_table_with_both_intensity_columns(monkeypatch):
    """Regression: the rename used to fail on the live table and its full
    rollback discarded every ADD COLUMN, so new columns never persisted."""
    db = FakeDB()
    cur = FakeConn(db).cursor()
    app_module.ensure_material_column(cur)
    sql = [" ".join(q.split()).upper() for q, _ in db.queries]
    assert not any("RENAME COLUMN" in q for q in sql)
    assert db.rollbacks == 0
    assert any("ADD COLUMN IF NOT EXISTS CORE_DIAMETER_UM" in q for q in sql)
    assert any("ADD COLUMN IF NOT EXISTS SUGGESTION_ID" in q for q in sql)


def test_column_migration_renames_only_legacy_tables(monkeypatch):
    db = FakeDB()
    db.legacy_columns = ["wavelength_shift"]
    app_module.ensure_material_column(FakeConn(db).cursor())
    assert any("RENAME COLUMN WAVELENGTH_SHIFT TO LIGHT_INTENSITY_UW" in " ".join(q.split()).upper() for q, _ in db.queries)
