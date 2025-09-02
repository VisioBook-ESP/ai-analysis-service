from fastapi import FastAPI

app = FastAPI(title="ai-analysis-service")

@app.get("/")
def root():
    return {"message": "ai-analysis-service up"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
