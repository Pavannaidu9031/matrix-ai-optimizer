"""Entry point for uvicorn / Render: `uvicorn app:app`. The code lives in matrixai/."""
from matrixai.main import create_app

app = create_app()
