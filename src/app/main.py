from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

app = FastAPI(title="Support Agent")
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def demo_widget() -> FileResponse:
    """Serve the demo chat widget for client-facing walkthroughs."""
    return FileResponse(STATIC_DIR / "index.html")
