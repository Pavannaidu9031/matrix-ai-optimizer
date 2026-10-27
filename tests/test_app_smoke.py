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
           "core_diameter_um", "sputter_mode"]


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
        if q.startswith("SELECT IS_APPROVED"):
            self.description, self._result = [("is_approved",)], [(True,)]
        elif q.startswith("SELECT DISTINCT BRANCH_NAME"):
            self.description, self._result = [("branch_name",)], [("main",)]
        elif "FROM EXPERIMENTS" in q and q.startswith("SELECT"):
            m = re.match(r"SELECT (.*?) FROM EXPERIMENTS", " ".join(sql.split()), re.I)
            wanted = COLUMNS if m.group(1).strip() == "*" else [c.strip() for c in m.group(1).split(",")]
            idx = [COLUMNS.index(c) for c in wanted]
            self.description = [(c,) for c in wanted]
            self._result = [tuple(r[i] for i in idx) for r in self.db.rows]
        elif q.startswith("SELECT") and "SUGGESTION_HISTORY" in q:
            self.description, self._result = [("id",)], []
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
        pass

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.rows, self.queries, self.inserts = _rows(), [], []


@pytest.fixture
def client(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(app_module, "get_db_connection", lambda: FakeConn(db))
    monkeypatch.setattr(app_module, "release_db_connection", lambda conn: None)
    c = TestClient(app_module.app, base_url="https://testserver")
    session = {"user": {"id": "1", "email": EMAIL, "name": "Tester", "picture": "", "is_approved": True}}
    payload = base64.b64encode(json.dumps(session).encode())
    c.cookies.set("session", TimestampSigner(os.environ["SECRET_KEY"]).sign(payload).decode())
    c.db = db
    return c


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
    res = anon.get("/supervisor", follow_redirects=False)
    assert res.status_code == 303 and "/login/google" in res.headers["location"]


def test_supervisor_scopes_non_founder_to_own_runs(client):
    assert client.get("/supervisor").status_code == 200
    sql, params = next((q, p) for q, p in reversed(client.db.queries) if "FROM experiments" in q)
    assert "user_email = %s" in sql and params == (EMAIL,)
