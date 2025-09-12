from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, constr
from functools import lru_cache
from transformers import pipeline
import os, torch

router = APIRouter()

# ---- Schémas ----
class SummarizeRequest(BaseModel):
    text: constr(min_length=1) = Field(..., description="Texte à résumer")
    max_tokens: int = Field(128, ge=32, le=1024, description="Longueur cible du résumé")

class SummarizeResponse(BaseModel):
    summary: str
    model: str
    device: str

# ---- Utils ----
def _device_index() -> int:
    return 0 if torch.cuda.is_available() else -1

def _device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def _get_summarizer():
    model_id = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    pipe = pipeline("summarization", model=model_id, device=_device_index())
    return pipe, model_id

# ---- Endpoint ----
@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        pipe, model_id = _get_summarizer()

        # Paramètres raisonnables pour éviter les erreurs sur textes très longs
        max_len = req.max_tokens
        min_len = max(32, max_len // 4)

        out = pipe(
            req.text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )[0]["summary_text"]

        return SummarizeResponse(summary=out, model=model_id, device=_device_name())

    except Exception as e:
        # Logguer e si tu as une config logging
        raise HTTPException(status_code=500, detail=f"Summarization failed: {type(e).__name__}")
