"""Client for persisting analysis results to PostgreSQL."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.database.connection import get_session
from src.database.models import AnalysisResult

logger = logging.getLogger(__name__)


class DatabaseClient:
    async def save_analysis(
        self,
        *,
        project_id: str,
        version_id: str,
        execution_id: str,
        user_id: str,
        status: str = "completed",
        scenes: list | None = None,
        characters: list | None = None,
        narrative: dict | None = None,
        sentiment: dict | None = None,
        summary: dict | None = None,
        text_stats: dict | None = None,
        language: str | None = None,
        processing_time_ms: float | None = None,
        error: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Upsert an analysis result by execution_id."""
        try:
            async with get_session() as session:
                now = datetime.now(timezone.utc)
                values = dict(
                    project_id=project_id,
                    version_id=version_id,
                    execution_id=execution_id,
                    user_id=user_id,
                    status=status,
                    scenes=scenes,
                    characters=characters,
                    narrative=narrative,
                    sentiment=sentiment,
                    summary=summary,
                    text_stats=text_stats,
                    language=language,
                    processing_time_ms=processing_time_ms,
                    error=error,
                    correlation_id=correlation_id,
                )
                stmt = (
                    insert(AnalysisResult)
                    .values(**values)
                    .on_conflict_do_update(
                        index_elements=["execution_id"],
                        set_={
                            **{k: v for k, v in values.items() if k != "execution_id"},
                            "updated_at": now,
                        },
                    )
                )
                await session.execute(stmt)
            logger.info("Saved analysis for execution %s (status=%s)", execution_id, status)
            return True
        except Exception as e:
            logger.error("Failed to save analysis for execution %s: %s", execution_id, e)
            return False

    async def get_analysis(self, execution_id: str) -> AnalysisResult | None:
        """Retrieve an analysis result by execution_id."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(AnalysisResult).where(AnalysisResult.execution_id == execution_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get analysis %s: %s", execution_id, e)
            return None

    async def get_analyses_by_project(self, project_id: str, user_id: str) -> list[AnalysisResult]:
        """Retrieve all analyses for a project, scoped to user."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(AnalysisResult)
                    .where(
                        AnalysisResult.project_id == project_id,
                        AnalysisResult.user_id == user_id,
                    )
                    .order_by(AnalysisResult.created_at.desc())
                )
                return list(result.scalars().all())
        except Exception as e:
            logger.error("Failed to get analyses for project %s: %s", project_id, e)
            return []

    async def health_check(self) -> bool:
        try:
            async with get_session() as session:
                await session.execute(select(1))
            return True
        except Exception:
            return False
