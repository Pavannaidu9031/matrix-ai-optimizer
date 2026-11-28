"""All configuration comes from environment variables (set them in Render)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    secret_key: str
    database_url: str | None
    founder_email: str
    google_client_id: str | None
    google_client_secret: str | None
    gemini_api_key: str | None
    groq_api_key: str | None
    https_only: bool = True
    run_migrations: bool = True

    def is_founder(self, email: str | None) -> bool:
        return bool(email) and email.lower() == self.founder_email.lower()


def load_settings() -> Settings:
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "SECRET_KEY is not set. Generate one with "
            "python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and add it in Render -> Environment."
        )
    db_url = os.getenv("DATABASE_URL")
    if db_url and "sslmode=" not in db_url:
        db_url += ("&" if "?" in db_url else "?") + "sslmode=require"
    return Settings(
        secret_key=secret,
        database_url=db_url,
        founder_email=os.getenv("FOUNDER_EMAIL", "pavannaidu9031@gmail.com"),
        google_client_id=os.getenv("GOOGLE_CLIENT_ID"),
        google_client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        https_only=os.getenv("HTTPS_ONLY", "1") != "0",
        run_migrations=os.getenv("RUN_MIGRATIONS", "1") != "0",
    )
