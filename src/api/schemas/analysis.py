
from pydantic import BaseModel, Field


class AnalysisOptions(BaseModel):
    semantic: bool = Field(True, description="Activer l'analyse semantique")
    scenes: bool = Field(True, description="Activer l'extraction de scenes")
    summarize: bool = Field(True, description="Activer le resume")
    mask_pii: bool = Field(True, description="Masquer les donnees personnelles")
    remove_links: bool = Field(False, description="Supprimer les URLs")
    return_embeddings: bool = Field(False, description="Retourner les embeddings")
    max_summary_length: int = Field(150, ge=50, le=500, description="Longueur max du resume")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte a analyser")
    language: str | None = Field("auto", description="Code langue (fr/en/auto)")
    options: AnalysisOptions = Field(
        default_factory=AnalysisOptions, description="Options d'analyse"
    )


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50, description="Liste de textes")
    language: str | None = Field("auto", description="Code langue")
    options: AnalysisOptions = Field(
        default_factory=AnalysisOptions, description="Options d'analyse"
    )


class TextStats(BaseModel):
    original_length: int
    cleaned_length: int
    sentence_count: int
    word_count: int
    quality_score: float
    quality_assessment: str


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class SentimentScore(BaseModel):
    polarity: float
    subjectivity: float


class SemanticResult(BaseModel):
    entities: list[Entity]
    keywords: list[str]
    topics: list[str]
    sentiment: SentimentScore
    embeddings: list[float] | None = None


class Character(BaseModel):
    name: str
    traits: list[str]
    emotions: list[str]


class Setting(BaseModel):
    location: str
    time_of_day: str
    lighting: list[str]


class VisualAttributes(BaseModel):
    colors: list[str]
    lighting: list[str]


class Scene(BaseModel):
    scene_id: str
    text: str
    characters: list[Character]
    setting: Setting
    atmosphere: str
    objects: list[str]
    actions: list[str]
    visual_attributes: VisualAttributes


class SceneResult(BaseModel):
    scene_count: int
    scenes: list[Scene]


class SummaryResult(BaseModel):
    abstractive_summary: str
    extractive_sentences: list[str]
    key_points: list[str]
    original_length: int
    summary_length: int
    length_reduction_percent: float


class AnalyzeResponse(BaseModel):
    language: str
    text_stats: TextStats
    semantic: SemanticResult | None = None
    scenes: SceneResult | None = None
    summary: SummaryResult | None = None
    processing_time_ms: float


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    total_processing_time_ms: float
    success_count: int
    error_count: int
