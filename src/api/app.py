from fastapi import FastAPI
from src.api.routes.analysis import router as analysis_router


app = FastAPI(title="ai-analysis-service")

@app.get("/")
def root():
    return {"message": "ai-analysis-service up"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
