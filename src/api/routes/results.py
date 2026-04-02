"""Endpoints for retrieving persisted analysis results."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.clients.database_client import DatabaseClient

router = APIRouter()
_db_client = DatabaseClient()


class AnalysisResultResponse(BaseModel):
    execution_id: str
    project_id: str
    version_id: str
    status: str
    scenes: list | None = None
    characters: list | None = None
    narrative: dict | None = None
    sentiment: dict | None = None
    summary: dict | None = None
    text_stats: dict | None = None
    language: str | None = None
    processing_time_ms: float | None = None
    error: str | None = None

    model_config = {"from_attributes": True}


@router.get("/{execution_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(
    execution_id: str,
    user_id: str = Depends(get_current_user),
):
    """Retrieve a stored analysis result by execution_id."""
    result = await _db_client.get_analysis(execution_id)
    if result is None or result.user_id != user_id:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return result


@router.get("/project/{project_id}", response_model=list[AnalysisResultResponse])
async def get_project_analyses(
    project_id: str,
    user_id: str = Depends(get_current_user),
):
    """List all analysis results for a project."""
    results = await _db_client.get_analyses_by_project(project_id, user_id)
    return results
