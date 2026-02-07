from pydantic import BaseModel, Field
from typing import Optional, List


# ---- Request ----

class AnalysisOptions(BaseModel):
    characters: bool = Field(True, description="Extraire l'analyse des personnages")
    scenes: bool = Field(True, description="Extraire l'analyse des scenes")
    narrative: bool = Field(True, description="Extraire l'analyse narrative")
    summary: bool = Field(True, description="Generer un resume")
    mask_pii: bool = Field(True, description="Masquer les donnees personnelles")
    remove_links: bool = Field(False, description="Supprimer les URLs")
    max_summary_length: int = Field(200, ge=50, le=1000, description="Longueur max du resume")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte a analyser")
    language: Optional[str] = Field("auto", description="Code langue (fr/en/auto)")
    options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="Options d'analyse")


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="Liste de textes")
    language: Optional[str] = Field("auto", description="Code langue")
    options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="Options d'analyse")


# ---- Response: Text Stats ----

class TextStats(BaseModel):
    original_length: int
    cleaned_length: int
    sentence_count: int
    word_count: int
    quality_score: float
    quality_assessment: str


# ---- Response: Characters ----

class CharacterRelationship(BaseModel):
    target: str
    type: str
    description: str


class Character(BaseModel):
    name: str
    role: str
    physical_description: str
    personality_traits: List[str]
    emotions: List[str]
    motivations: List[str]
    actions: List[str]
    relationships: List[CharacterRelationship] = []


# ---- Response: Scenes ----

class SoundTexture(BaseModel):
    sounds: List[str]
    textures: List[str]


class SceneSetting(BaseModel):
    location: str
    time_period: str
    time_of_day: str


class SceneAtmosphere(BaseModel):
    mood: str
    lighting: str
    weather: str
    colors: List[str]
    sounds_textures: SoundTexture


class Scene(BaseModel):
    scene_id: str
    title: str
    text_excerpt: str
    characters_present: List[str]
    setting: SceneSetting
    atmosphere: SceneAtmosphere
    key_events: List[str]
    objects: List[str]


# ---- Response: Narrative ----

class NarrativeAnalysis(BaseModel):
    themes: List[str]
    tone: str
    style: str
    point_of_view: str
    tension_level: str
    pacing: str
    literary_devices: List[str]


# ---- Response: Sentiment ----

class SentimentAnalysis(BaseModel):
    overall: str
    polarity: float
    nuances: List[str]
    emotional_arc: str


# ---- Response: Summary ----

class SummaryResult(BaseModel):
    summary: str
    key_points: List[str]
    original_length: int
    summary_length: int


# ---- Top-level Response ----

class AnalyzeResponse(BaseModel):
    language: str
    text_stats: TextStats
    characters: Optional[List[Character]] = None
    scenes: Optional[List[Scene]] = None
    narrative: Optional[NarrativeAnalysis] = None
    sentiment: Optional[SentimentAnalysis] = None
    summary: Optional[SummaryResult] = None
    processing_time_ms: float


class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]
    total_processing_time_ms: float
    success_count: int
    error_count: int
