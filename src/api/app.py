from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from src.api.routes.health import router as health_router
from src.api.routes.analysis import router as analysis_router
from src.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Service d'analyse IA pour extraction d'informations textuelles et generation de prompts image",
    default_response_class=ORJSONResponse,
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
            "analysis": {
                "analyze": "/api/v1/analyze",
                "analyze_batch": "/api/v1/analyze/batch",
            },
        },
    }


app.include_router(health_router, tags=["health"])
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
