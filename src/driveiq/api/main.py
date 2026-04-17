from __future__ import annotations

from fastapi import FastAPI

from driveiq.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app.app_name,
    version="0.1.0",
    description="DriveIQ API for multimodal retrieval and summarization workflows.",
)


@app.get("/")
def read_root() -> dict:
    return {
        "app_name": settings.app.app_name,
        "environment": settings.app.environment,
        "status": "ok",
        "message": "DriveIQ API is running.",
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "app_name": settings.app.app_name,
        "environment": settings.app.environment,
    }