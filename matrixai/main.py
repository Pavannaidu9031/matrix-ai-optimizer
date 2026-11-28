"""MatrixAI web app: create_app() wires settings, database, auth and routes."""
import contextlib

import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import auth
from .config import Settings, load_settings
from .db import Database, run_migrations
from .routes import admin, ai, experiments, optimizer, pages
from .store import MemoryStore, PgStore
from .web import PACKAGE_DIR


def create_app(settings: Settings | None = None, store=None) -> FastAPI:
    settings = settings or load_settings()
    database = None
    if store is None:
        if settings.database_url:
            database = Database(settings.database_url)
            store = PgStore(database)
        else:
            print("DATABASE_URL not set: using an in-memory store (data is lost on restart).")
            store = MemoryStore()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if database and settings.run_migrations:
            try:
                applied = run_migrations(settings.database_url)
                print(f"Database schema up to date ({len(applied)} migration(s) applied now).")
            except psycopg2.OperationalError as err:
                # Usually Supabase has paused the project. Start anyway so the
                # app can say so; migrations run on the next restart.
                print(f"Could not reach the database to migrate: {err}")
        yield
        if database:
            database.close()

    app = FastAPI(title="MatrixAI — Intelligent Materials Optimizer", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.approvals = auth.ApprovalChecker(store, settings)

    # Middleware added later runs first: the session must exist before the gate.
    auth.install_gate(app)
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, max_age=86400 * 30,
                       same_site="lax", https_only=settings.https_only)

    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")
    app.include_router(auth.build_router(settings))
    for module in (pages, experiments, optimizer, ai, admin):
        app.include_router(module.router)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if exc.status_code == 401 and auth.wants_html(request):
            return RedirectResponse("/login/google", status_code=303)
        if exc.status_code == 403 and auth.wants_html(request):
            return RedirectResponse("/", status_code=303)
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        return JSONResponse(detail, status_code=exc.status_code)

    @app.exception_handler(psycopg2.OperationalError)
    async def db_down(request: Request, exc):
        print(f"Database unavailable: {exc}")
        return JSONResponse({"error": "database_unavailable",
                             "message": "The database is not reachable (Supabase may have paused it)."},
                            status_code=503)

    return app
