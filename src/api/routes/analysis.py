import asyncio
import time
import logging

from fastapi import APIRouter, HTTPException, status

from src.services.analysis import Analyzer, AnalysisOptions as ServiceOptions
from src.services.job_store import job_store
from src.api.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    JobSubmittedResponse,
    JobStatusResponse,
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

logger = logging.getLogger(__name__)
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
                relationships=[
                    CharacterRelationship(**r) for r in c.get("relationships", [])
                ],
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
                    **{
                        k: v
                        for k, v in s["atmosphere"].items()
                        if k != "sounds_textures"
                    },
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


async def _run_analysis(job_id: str, request: AnalyzeRequest) -> None:
    """Background task: runs the analysis and updates job state."""
    await job_store.update(job_id, status="processing")

    service_options = ServiceOptions(
        characters=request.options.characters,
        scenes=request.options.scenes,
        narrative=request.options.narrative,
        summary=request.options.summary,
        mask_pii=request.options.mask_pii,
        remove_links=request.options.remove_links,
        max_summary_length=request.options.max_summary_length,
    )

    async def on_step(step: str) -> None:
        await job_store.update(job_id, step=step)

    try:
        result = await _analyzer.analyze(
            text=request.text,
            language=request.language,
            options=service_options,
            on_step=on_step,
        )
        response = AnalyzeResponse(**_build_response_data(result))
        await job_store.update(job_id, status="completed", step=None, result=response)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await job_store.update(job_id, status="failed", step=None, error=str(e))


@router.post(
    "/analyze",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze(request: AnalyzeRequest) -> JobSubmittedResponse:
    if len(request.text) > 500_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Text too large. Maximum 500,000 characters.",
        )

    job = await job_store.create()
    asyncio.create_task(_run_analysis(job.job_id, request))
    return JobSubmittedResponse(job_id=job.job_id, status=job.status)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
)
async def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        step=job.step,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
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
