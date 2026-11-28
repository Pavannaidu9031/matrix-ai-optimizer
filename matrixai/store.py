"""Every database query in the app lives here.

PgStore is the real one. MemoryStore has the same methods and keeps data in
memory: the tests use it, and the app falls back to it when DATABASE_URL is
not set so the UI can be tried locally without a database.
"""
import datetime
import itertools
import threading

from .db import Database
from .models import EXPERIMENT_FIELDS

EXPERIMENT_COLUMNS = ("id", "user_email", "created_at") + tuple(f.name for f in EXPERIMENT_FIELDS)
SUGGESTION_COLUMNS = ("suggested_rf_power", "suggested_pressure", "suggested_distance", "suggested_thickness",
                      "suggested_rotation", "suggested_ar_flow", "suggested_pd_thickness",
                      "predicted_xrd_score", "predicted_wavelength", "confidence_score", "converged", "kappa_used")


def _iso(row: dict) -> dict:
    row = dict(row)
    if isinstance(row.get("created_at"), (datetime.datetime, datetime.date)):
        row["created_at"] = row["created_at"].isoformat()
    return row


class PgStore:
    def __init__(self, db: Database):
        self.db = db

    # ---- users -------------------------------------------------------------
    def approval(self, email: str) -> bool:
        with self.db.cursor() as cur:
            cur.execute("SELECT is_approved FROM users WHERE LOWER(email) = LOWER(%s)", (email,))
            row = cur.fetchone()
        return bool(row and row["is_approved"])

    def record_login(self, email, name, picture, approve_if_new: bool):
        """Returns (is_approved, is_new_account)."""
        with self.db.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, name, picture, is_approved) VALUES (%s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name, picture = EXCLUDED.picture
                RETURNING is_approved, (xmax = 0) AS inserted
            """, (email, name, picture, approve_if_new))
            row = cur.fetchone()
        return bool(row["is_approved"]), bool(row["inserted"])

    def set_approval(self, email: str, approved: bool):
        with self.db.cursor() as cur:
            cur.execute("UPDATE users SET is_approved = %s WHERE LOWER(email) = LOWER(%s)", (approved, email))

    def list_users(self):
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT u.email, u.name, u.picture, u.is_approved, u.created_at, COUNT(e.id) AS exp_count
                  FROM users u LEFT JOIN experiments e ON e.user_email = u.email
                 GROUP BY u.email ORDER BY u.created_at DESC
            """)
            return [_iso(r) for r in cur.fetchall()]

    # ---- experiments -------------------------------------------------------
    def branches(self, email: str) -> list[str]:
        with self.db.cursor() as cur:
            cur.execute("SELECT DISTINCT branch_name FROM experiments WHERE user_email = %s AND branch_name IS NOT NULL "
                        "ORDER BY branch_name", (email,))
            found = [r["branch_name"] for r in cur.fetchall()]
        return ["main"] + [b for b in found if b != "main"]

    def experiments(self, email: str, branch: str | None = None, limit: int | None = None) -> list[dict]:
        sql = f"SELECT {', '.join(EXPERIMENT_COLUMNS)} FROM experiments WHERE user_email = %s"
        params: list = [email]
        if branch:
            sql += " AND branch_name = %s"
            params.append(branch)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit:
            sql += " LIMIT %s"
            params.append(limit)
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            return [_iso(r) for r in cur.fetchall()]

    def all_experiments(self, limit: int = 50) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {', '.join(EXPERIMENT_COLUMNS)} FROM experiments "
                        "ORDER BY created_at DESC, id DESC LIMIT %s", (limit,))
            return [_iso(r) for r in cur.fetchall()]

    def get_experiment(self, exp_id: int, email: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {', '.join(EXPERIMENT_COLUMNS)} FROM experiments WHERE id = %s AND user_email = %s",
                        (exp_id, email))
            row = cur.fetchone()
        return _iso(row) if row else None

    def add_experiments(self, email: str, records: list[dict]) -> list[int]:
        cols = [f.name for f in EXPERIMENT_FIELDS]
        sql = (f"INSERT INTO experiments (user_email, {', '.join(cols)}) "
               f"VALUES (%s, {', '.join(['%s'] * len(cols))}) RETURNING id")
        ids = []
        with self.db.cursor() as cur:                    # one transaction: all or nothing
            for rec in records:
                cur.execute(sql, [email] + [rec.get(c) for c in cols])
                ids.append(cur.fetchone()["id"])
        return ids

    def update_measurements(self, exp_id: int, email: str, values: dict):
        allowed = {"light_intensity_uw", "h2_response_time", "recovery_time_s", "response_curve", "quality_score",
                   "fiber_power_before_uw", "fiber_power_after_uw"}
        values = {k: v for k, v in values.items() if k in allowed}
        sets = ", ".join(f"{k} = %s" for k in values)
        with self.db.cursor() as cur:
            cur.execute(f"UPDATE experiments SET {sets} WHERE id = %s AND user_email = %s",
                        [*values.values(), exp_id, email])
            cur.execute("INSERT INTO audit_log (action, experiment_id, user_email) VALUES (%s, %s, %s)",
                        ("UPDATE_MEASUREMENTS", exp_id, email))

    def delete_experiment(self, exp_id: int, email: str) -> str:
        """'deleted', 'not_found' or 'forbidden'."""
        with self.db.cursor() as cur:
            cur.execute("SELECT user_email FROM experiments WHERE id = %s", (exp_id,))
            row = cur.fetchone()
            if not row:
                return "not_found"
            if row["user_email"] != email:
                return "forbidden"
            cur.execute("INSERT INTO audit_log (action, experiment_id, user_email) VALUES (%s, %s, %s)",
                        ("DELETE", exp_id, email))
            cur.execute("DELETE FROM experiments WHERE id = %s AND user_email = %s", (exp_id, email))
        return "deleted"

    def admin_delete(self, exp_ids: list[int], actor: str) -> int:
        with self.db.cursor() as cur:
            for exp_id in exp_ids:
                cur.execute("INSERT INTO audit_log (action, experiment_id, user_email) VALUES (%s, %s, %s)",
                            ("ADMIN_BULK_DELETE", exp_id, actor))
            cur.execute("DELETE FROM experiments WHERE id = ANY(%s)", (list(exp_ids),))
            return cur.rowcount

    # ---- suggestions -------------------------------------------------------
    def recent_suggestions(self, email: str, n: int = 3) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT id, created_at, {', '.join(SUGGESTION_COLUMNS)} FROM suggestion_history "
                        "WHERE user_email = %s ORDER BY id DESC LIMIT %s", (email, n))
            return [_iso(r) for r in cur.fetchall()]

    def add_suggestion(self, user_id: str, email: str, values: dict) -> int:
        cols = [c for c in SUGGESTION_COLUMNS if c in values]
        with self.db.cursor() as cur:
            cur.execute(f"INSERT INTO suggestion_history (user_id, user_email, {', '.join(cols)}) "
                        f"VALUES (%s, %s, {', '.join(['%s'] * len(cols))}) RETURNING id",
                        [user_id, email] + [values[c] for c in cols])
            return cur.fetchone()["id"]

    # ---- misc --------------------------------------------------------------
    def add_comment(self, exp_id: int, email: str, text: str) -> int:
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO experiment_comments (experiment_id, user_email, comment_text) "
                        "VALUES (%s, %s, %s) RETURNING id", (exp_id, email, text))
            return cur.fetchone()["id"]

    def api_key_owner(self, key_hash: str) -> str | None:
        with self.db.cursor() as cur:
            cur.execute("UPDATE api_keys SET usage_count = usage_count + 1 WHERE api_key_hash = %s "
                        "RETURNING user_email", (key_hash,))
            row = cur.fetchone()
        return row["user_email"] if row else None

    def schema_version(self) -> str | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT MAX(version) AS v FROM schema_migrations")
            return cur.fetchone()["v"]


class MemoryStore:
    """In-memory stand-in with the same behaviour as PgStore."""

    def __init__(self):
        self.users: dict[str, dict] = {}
        self.rows: list[dict] = []
        self.suggestions: list[dict] = []
        self.comments: list[dict] = []
        self.audit: list[tuple] = []
        self.api_keys: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._sug_ids = itertools.count(1)
        self._lock = threading.Lock()
        self.fail = False            # tests flip this to simulate an outage

    def _check(self):
        if self.fail:
            raise ConnectionError("database unavailable")

    def approval(self, email):
        self._check()
        u = self.users.get(email.lower())
        return bool(u and u["is_approved"])

    def record_login(self, email, name, picture, approve_if_new):
        self._check()
        key = email.lower()
        is_new = key not in self.users
        if is_new:
            self.users[key] = {"email": email, "name": name, "picture": picture, "is_approved": approve_if_new,
                               "created_at": datetime.datetime.now().isoformat()}
        else:
            self.users[key].update(name=name, picture=picture)
        return self.users[key]["is_approved"], is_new

    def set_approval(self, email, approved):
        self._check()
        if email.lower() in self.users:
            self.users[email.lower()]["is_approved"] = approved

    def list_users(self):
        return [dict(u, exp_count=sum(r["user_email"] == u["email"] for r in self.rows)) for u in self.users.values()]

    def branches(self, email):
        found = sorted({r["branch_name"] for r in self.rows if r["user_email"] == email and r.get("branch_name")})
        return ["main"] + [b for b in found if b != "main"]

    def experiments(self, email, branch=None, limit=None):
        self._check()
        rows = [dict(r) for r in reversed(self.rows) if r["user_email"] == email and (not branch or r["branch_name"] == branch)]
        return rows[:limit] if limit else rows

    def all_experiments(self, limit=50):
        return [dict(r) for r in reversed(self.rows)][:limit]

    def get_experiment(self, exp_id, email):
        return next((dict(r) for r in self.rows if r["id"] == exp_id and r["user_email"] == email), None)

    def add_experiments(self, email, records):
        self._check()
        ids = []
        with self._lock:
            for rec in records:
                row = {c: None for c in EXPERIMENT_COLUMNS}
                row.update({k: rec.get(k) for k in row if k in rec})
                row.update(id=next(self._ids), user_email=email, created_at=datetime.datetime.now().isoformat())
                self.rows.append(row)
                ids.append(row["id"])
        return ids

    def update_measurements(self, exp_id, email, values):
        for r in self.rows:
            if r["id"] == exp_id and r["user_email"] == email:
                r.update(values)
        self.audit.append(("UPDATE_MEASUREMENTS", exp_id, email))

    def delete_experiment(self, exp_id, email):
        row = next((r for r in self.rows if r["id"] == exp_id), None)
        if not row:
            return "not_found"
        if row["user_email"] != email:
            return "forbidden"
        self.rows.remove(row)
        self.audit.append(("DELETE", exp_id, email))
        return "deleted"

    def admin_delete(self, exp_ids, actor):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["id"] not in set(exp_ids)]
        self.audit.extend(("ADMIN_BULK_DELETE", i, actor) for i in exp_ids)
        return before - len(self.rows)

    def recent_suggestions(self, email, n=3):
        return [dict(s) for s in reversed(self.suggestions) if s["user_email"] == email][:n]

    def add_suggestion(self, user_id, email, values):
        sid = next(self._sug_ids)
        self.suggestions.append(dict(values, id=sid, user_id=user_id, user_email=email))
        return sid

    def add_comment(self, exp_id, email, text):
        self.comments.append({"experiment_id": exp_id, "user_email": email, "comment_text": text})
        return len(self.comments)

    def api_key_owner(self, key_hash):
        k = self.api_keys.get(key_hash)
        if not k:
            return None
        k["usage_count"] = k.get("usage_count", 0) + 1
        return k["user_email"]

    def schema_version(self):
        return "memory"
