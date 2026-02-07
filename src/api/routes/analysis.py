from fastapi import APIRouter, HTTPException, status
import time

from src.services.analysis import Analyzer, AnalysisOptions as ServiceOptions
from src.api.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    TextStats,
    Character,
    CharacterRelationship,
    Scene,
    SceneSetting,
    SceneAtmosphere,
    SoundTexture,
    NarrativeAnalysis,
    SentimentAnalysis,
    SummaryResult,
)


router = APIRouter()
_analyzer = Analyzer()


def _build_response_data(result: dict) -> dict:
    """Build response data dict from analyzer result."""
    response_data = {
        "language": result["language"],
        "text_stats": TextStats(**result["text_stats"]),
        "processing_time_ms": result["processing_time_ms"],
    }

    if "characters" in result:
        response_data["characters"] = [
            Character(
                relationships=[CharacterRelationship(**r) for r in c.get("relationships", [])],
                **{k: v for k, v in c.items() if k != "relationships"},
            )
            for c in result["characters"]
        ]

    if "scenes" in result:
        response_data["scenes"] = [
            Scene(
                setting=SceneSetting(**s["setting"]),
                atmosphere=SceneAtmosphere(
                    sounds_textures=SoundTexture(**s["atmosphere"]["sounds_textures"]),
                    **{k: v for k, v in s["atmosphere"].items() if k != "sounds_textures"},
                ),
                **{k: v for k, v in s.items() if k not in ("setting", "atmosphere")},
            )
            for s in result["scenes"]
        ]

    if "narrative" in result:
        response_data["narrative"] = NarrativeAnalysis(**result["narrative"])

    if "sentiment" in result:
        response_data["sentiment"] = SentimentAnalysis(**result["sentiment"])

    if "summary" in result:
        response_data["summary"] = SummaryResult(**result["summary"])

    return response_data


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        if len(request.text) > 500_000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Text too large. Maximum 500,000 characters.",
            )

        service_options = ServiceOptions(
            characters=request.options.characters,
            scenes=request.options.scenes,
            narrative=request.options.narrative,
            summary=request.options.summary,
            mask_pii=request.options.mask_pii,
            remove_links=request.options.remove_links,
            max_summary_length=request.options.max_summary_length,
        )

        result = await _analyzer.analyze(
            text=request.text, language=request.language, options=service_options
        )

        return AnalyzeResponse(**_build_response_data(result))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    try:
        if len(request.texts) > 50:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Too many texts. Maximum 50 texts per batch.",
            )

        start_time = time.time()

        service_options = ServiceOptions(
            characters=request.options.characters,
            scenes=request.options.scenes,
            narrative=request.options.narrative,
            summary=request.options.summary,
            mask_pii=request.options.mask_pii,
            remove_links=request.options.remove_links,
            max_summary_length=request.options.max_summary_length,
        )

        results = []
        success_count = 0
        error_count = 0

        for text in request.texts:
            try:
                if len(text) > 500_000:
                    error_count += 1
                    continue

                result = await _analyzer.analyze(
                    text=text, language=request.language, options=service_options
                )
                results.append(AnalyzeResponse(**_build_response_data(result)))
                success_count += 1
            except Exception:
                error_count += 1

        total_time = (time.time() - start_time) * 1000

        return BatchAnalyzeResponse(
            results=results,
            total_processing_time_ms=round(total_time, 2),
            success_count=success_count,
            error_count=error_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )
