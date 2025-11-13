from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field(None)
    remove_links: bool = Field(False)
    mask_pii: bool = Field(True)
    remove_emoji: bool = Field(False)
    lowercase: bool = Field(False)
    max_tokens: int = Field(512, ge=32, le=2048)
    overlap: int = Field(64, ge=0, le=512)


class BatchPreprocessRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    language: Optional[str] = Field(None)
    remove_links: bool = Field(False)
    mask_pii: bool = Field(True)
    remove_emoji: bool = Field(False)
    lowercase: bool = Field(False)
    max_tokens: int = Field(512, ge=32, le=2048)
    overlap: int = Field(64, ge=0, le=512)


class SentenceSegment(BaseModel):
    start: int
    end: int
    text: str


class TextChunk(BaseModel):
    token_start: int
    token_end: int
    start_char: int
    end_char: int
    text: str
    token_count: int


class QualityScore(BaseModel):
    score: float
    non_letter_ratio: float
    punct_ratio: float
    upper_ratio: float
    avg_sent_len: float
    url_ratio: float
    assessment: str


class PreprocessStats(BaseModel):
    original_length: int
    cleaned_length: int
    sentence_count: int
    chunk_count: int


class PreprocessResponse(BaseModel):
    language: str
    text: str
    quality: QualityScore
    sentences: List[SentenceSegment]
    chunks: List[TextChunk]
    masks: Dict[str, List[str]]
    stats: PreprocessStats
    processing_time_ms: float


class BatchPreprocessResponse(BaseModel):
    results: List[PreprocessResponse]
    total_processing_time_ms: float
    success_count: int
    error_count: int
