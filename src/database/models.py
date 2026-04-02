"""SQLAlchemy models for ai-analysis-service."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisResult(Base):
    """Persisted analysis output from the LLM pipeline."""

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    project_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    execution_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="completed")

    # Analysis output (JSONB)
    scenes: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    characters: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    narrative: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sentiment: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    text_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    processing_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    __table_args__ = (Index("ix_analysis_results_project_version", "project_id", "version_id"),)

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult(id={self.id}, execution_id={self.execution_id}, "
            f"status={self.status})>"
        )
