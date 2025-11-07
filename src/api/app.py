from fastapi import FastAPI
from src.api.routes.health import router as health_router
from src.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Service d'analyse IA pour extraction d'informations textuelles et generation de prompts image",
)


@app.get("/", tags=["root"])
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs",
        "health_endpoints": {
            "liveness": "/health",
            "readiness": "/ready",
            "metrics": "/metrics",
        },
    }


# Routes healthchecks
app.include_router(health_router, tags=["health"])
