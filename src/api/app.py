from fastapi import FastAPI
from src.api.routes.health import router as health_router
from src.api.routes.preprocessing import router as preprocessing_router
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
        "endpoints": {
            "health": {
                "liveness": "/health",
                "readiness": "/ready",
                "metrics": "/metrics",
            },
            "preprocessing": {
                "preprocess": "/api/v1/preprocessing/preprocess",
                "batch": "/api/v1/preprocessing/preprocess/batch",
                "info": "/api/v1/preprocessing/preprocess/info",
            },
        },
    }


# Routes healthchecks
app.include_router(health_router, tags=["health"])

# Routes preprocessing
app.include_router(
    preprocessing_router,
    prefix="/api/v1/preprocessing",
    tags=["preprocessing"]
)
