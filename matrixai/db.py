"""Postgres connection pool and the migration runner."""
import contextlib
import pathlib
import threading

import psycopg2
import psycopg2.extras
from psycopg2 import pool

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"
_MIGRATION_LOCK_ID = 7261_0042          # any constant; serializes concurrent app starts


class Database:
    """A thread-safe pool. FastAPI runs sync routes on a thread pool, so the
    old SimpleConnectionPool (not thread-safe) could hand one connection to
    two requests at once."""

    def __init__(self, dsn: str, maxconn: int = 10):
        self.dsn = dsn
        self.maxconn = maxconn
        self._pool = None
        self._lock = threading.Lock()

    def _get_pool(self):
        # Created on first use, so the app still starts (and shows a clear
        # error page) when Supabase has paused the database.
        with self._lock:
            if self._pool is None:
                self._pool = pool.ThreadedConnectionPool(1, self.maxconn, dsn=self.dsn)
            return self._pool

    @contextlib.contextmanager
    def cursor(self):
        """Yields a dict cursor inside one transaction: commit on success,
        rollback on any error, connection always returned to the pool."""
        conn = self._get_pool().getconn()
        broken = False
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
            conn.commit()
        except psycopg2.InterfaceError:
            broken = True
            raise
        except BaseException:
            try:
                conn.rollback()
            except psycopg2.Error:
                broken = True
            raise
        finally:
            self._pool.putconn(conn, close=broken or bool(conn.closed))

    def close(self):
        if self._pool is not None:
            self._pool.closeall()


def migration_files():
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def run_migrations(dsn: str) -> list[str]:
    """Applies every migration not yet recorded in schema_migrations, each in
    its own transaction. A failing migration raises -- it is never skipped
    silently -- and leaves the database exactly as it was before that file."""
    applied_now = []
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    VARCHAR PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT NOW()
                )""")
            cur.execute("ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY")
            cur.execute("SELECT version FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}
        conn.autocommit = False
        for path in migration_files():
            if path.stem in done:
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(path.read_text(encoding="utf-8"))
                    cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
                conn.commit()
            except Exception as err:
                conn.rollback()
                raise RuntimeError(f"Migration {path.name} failed and was rolled back: {err}") from err
            applied_now.append(path.stem)
            print(f"Applied migration {path.name}")
    finally:
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
        except psycopg2.Error:
            pass
        conn.close()
    return applied_now
