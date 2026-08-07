import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

# 1. Ensure SSL mode is enabled for Supabase
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require" if "?" not in DATABASE_URL else "&sslmode=require"

# 2. Configure engine with pool_pre_ping to automatically handle drops/reconnections
engine = (
    create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Checks connection validity before executing queries
        pool_size=5,
        max_overflow=10
    )
    if DATABASE_URL
    else None
)

# 3. Create Session Factory
SessionLocal = (
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if engine
    else None
)

def init_db():
    """Stub function to prevent import errors in app.py if init_db() is called."""
    pass

def get_db():
    """Dependency helper to yield database sessions safely."""
    if not SessionLocal:
        raise Exception("DATABASE_URL environment variable is not configured.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
