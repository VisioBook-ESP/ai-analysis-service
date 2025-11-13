from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import time

from src.services.preprocessing import TextPreprocessor
from src.api.schemas.preprocessing import (
    PreprocessRequest,
    PreprocessResponse,
    BatchPreprocessRequest,
    BatchPreprocessResponse,
    QualityScore,
    PreprocessStats
)


router = APIRouter()
_preprocessor = TextPreprocessor()


@router.post("/preprocess", response_model=PreprocessResponse, status_code=status.HTTP_200_OK)
def preprocess_text(request: PreprocessRequest) -> PreprocessResponse:
    try:
        if len(request.text) > 500_000:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Text too large. Maximum 500,000 characters allowed."
            )

        result = _preprocessor.preprocess(
            text=request.text,
            language=request.language,
            remove_links=request.remove_links,
            mask_pii=request.mask_pii,
            remove_emoji=request.remove_emoji,
            lowercase=request.lowercase,
            max_tokens=request.max_tokens,
            overlap=request.overlap
        )

        return PreprocessResponse(
            language=result["language"],
            text=result["text"],
            quality=QualityScore(**result["quality"]),
            sentences=result["sentences"],
            chunks=result["chunks"],
            masks=result["masks"],
            stats=PreprocessStats(**result["stats"]),
            processing_time_ms=result["processing_time_ms"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


@router.post("/preprocess/batch", response_model=BatchPreprocessResponse, status_code=status.HTTP_200_OK)
def preprocess_batch(request: BatchPreprocessRequest) -> BatchPreprocessResponse:
    try:
        start_time = time.time()

        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Too many texts. Maximum 100 texts per batch."
            )

        results = []
        success_count = 0
        error_count = 0

        for text in request.texts:
            try:
                if len(text) > 500_000:
                    error_count += 1
                    continue

                result = _preprocessor.preprocess(
                    text=text,
                    language=request.language,
                    remove_links=request.remove_links,
                    mask_pii=request.mask_pii,
                    remove_emoji=request.remove_emoji,
                    lowercase=request.lowercase,
                    max_tokens=request.max_tokens,
                    overlap=request.overlap
                )

                response = PreprocessResponse(
                    language=result["language"],
                    text=result["text"],
                    quality=QualityScore(**result["quality"]),
                    sentences=result["sentences"],
                    chunks=result["chunks"],
                    masks=result["masks"],
                    stats=PreprocessStats(**result["stats"]),
                    processing_time_ms=result["processing_time_ms"]
                )

                results.append(response)
                success_count += 1

            except Exception:
                error_count += 1

        total_time = (time.time() - start_time) * 1000

        return BatchPreprocessResponse(
            results=results,
            total_processing_time_ms=round(total_time, 2),
            success_count=success_count,
            error_count=error_count
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch preprocessing failed: {str(e)}"
        )


@router.get("/preprocess/info")
def preprocessing_info() -> Dict[str, Any]:
    return {
        "supported_languages": ["fr", "en", "auto"],
        "limits": {
            "max_text_length": 500_000,
            "max_batch_size": 100,
            "max_tokens_per_chunk": 2048
        },
        "features": {
            "language_detection": True,
            "pii_masking": True,
            "emoji_removal": True,
            "url_removal": True,
            "sentence_segmentation": True,
            "chunking": True,
            "quality_scoring": True
        }
    }
