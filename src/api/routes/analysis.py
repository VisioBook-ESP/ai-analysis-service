from fastapi import APIRouter, HTTPException, status
import time

from src.services.analysis import Analyzer, AnalysisOptions as ServiceOptions
from src.api.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    TextStats,
    SemanticResult,
    SceneResult,
    SummaryResult,
    Entity,
    SentimentScore,
    Character,
    Setting,
    VisualAttributes,
    Scene,
)


router = APIRouter()
_analyzer = Analyzer()


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        if len(request.text) > 500_000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Text too large. Maximum 500,000 characters.",
            )

        service_options = ServiceOptions(
            semantic=request.options.semantic,
            scenes=request.options.scenes,
            summarize=request.options.summarize,
            mask_pii=request.options.mask_pii,
            remove_links=request.options.remove_links,
            return_embeddings=request.options.return_embeddings,
            max_summary_length=request.options.max_summary_length,
        )

        result = _analyzer.analyze(
            text=request.text, language=request.language, options=service_options
        )

        response_data = {
            "language": result["language"],
            "text_stats": TextStats(**result["text_stats"]),
            "processing_time_ms": result["processing_time_ms"],
        }

        if "semantic" in result and result["semantic"]:
            semantic_data = result["semantic"]
            response_data["semantic"] = SemanticResult(
                entities=[Entity(**e) for e in semantic_data["entities"]],
                keywords=semantic_data["keywords"],
                topics=semantic_data["topics"],
                sentiment=SentimentScore(**semantic_data["sentiment"]),
                embeddings=semantic_data.get("embeddings"),
            )

        if "scenes" in result and result["scenes"]:
            scene_data = result["scenes"]
            scenes_list = []
            for scene in scene_data["scenes"]:
                scenes_list.append(
                    Scene(
                        scene_id=scene["scene_id"],
                        text=scene["text"],
                        characters=[Character(**c) for c in scene["characters"]],
                        setting=Setting(**scene["setting"]),
                        atmosphere=scene["atmosphere"],
                        objects=scene["objects"],
                        actions=scene["actions"],
                        visual_attributes=VisualAttributes(**scene["visual_attributes"]),
                    )
                )

            response_data["scenes"] = SceneResult(
                scene_count=scene_data["scene_count"], scenes=scenes_list
            )

        if "summary" in result and result["summary"]:
            response_data["summary"] = SummaryResult(**result["summary"])

        return AnalyzeResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    try:
        if len(request.texts) > 50:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Too many texts. Maximum 50 texts per batch.",
            )

        start_time = time.time()

        service_options = ServiceOptions(
            semantic=request.options.semantic,
            scenes=request.options.scenes,
            summarize=request.options.summarize,
            mask_pii=request.options.mask_pii,
            remove_links=request.options.remove_links,
            return_embeddings=request.options.return_embeddings,
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

                result = _analyzer.analyze(
                    text=text, language=request.language, options=service_options
                )

                response_data = {
                    "language": result["language"],
                    "text_stats": TextStats(**result["text_stats"]),
                    "processing_time_ms": result["processing_time_ms"],
                }

                if "semantic" in result and result["semantic"]:
                    semantic_data = result["semantic"]
                    response_data["semantic"] = SemanticResult(
                        entities=[Entity(**e) for e in semantic_data["entities"]],
                        keywords=semantic_data["keywords"],
                        topics=semantic_data["topics"],
                        sentiment=SentimentScore(**semantic_data["sentiment"]),
                        embeddings=semantic_data.get("embeddings"),
                    )

                if "scenes" in result and result["scenes"]:
                    scene_data = result["scenes"]
                    scenes_list = []
                    for scene in scene_data["scenes"]:
                        scenes_list.append(
                            Scene(
                                scene_id=scene["scene_id"],
                                text=scene["text"],
                                characters=[Character(**c) for c in scene["characters"]],
                                setting=Setting(**scene["setting"]),
                                atmosphere=scene["atmosphere"],
                                objects=scene["objects"],
                                actions=scene["actions"],
                                visual_attributes=VisualAttributes(**scene["visual_attributes"]),
                            )
                        )

                    response_data["scenes"] = SceneResult(
                        scene_count=scene_data["scene_count"], scenes=scenes_list
                    )

                if "summary" in result and result["summary"]:
                    response_data["summary"] = SummaryResult(**result["summary"])

                results.append(AnalyzeResponse(**response_data))
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
