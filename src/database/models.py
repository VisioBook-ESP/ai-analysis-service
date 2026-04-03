"""SQLAlchemy models for ai-analysis-service."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    execution_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
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

    __table_args__ = (
        Index("ix_analysis_results_project_version", "project_id", "version_id"),
    )

    # Relationships to prompt tables
    scene_prompts: Mapped[list["ScenePrompt"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    character_prompts: Mapped[list["CharacterPrompt"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    location_prompts: Mapped[list["LocationPrompt"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult(id={self.id}, execution_id={self.execution_id}, "
            f"status={self.status})>"
        )


class ScenePrompt(Base):
    """Flux/SDXL-optimized prompt for a scene image."""

    __tablename__ = "scene_prompts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    scene_order: Mapped[int] = mapped_column(Integer, nullable=False)
    image_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    characters_present: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    location_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    analysis: Mapped["AnalysisResult"] = relationship(back_populates="scene_prompts")

    __table_args__ = (Index("ix_scene_prompts_analysis_id", "analysis_id"),)


class CharacterPrompt(Base):
    """Flux/SDXL-optimized prompt for a character portrait reference."""

    __tablename__ = "character_prompts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    physical_description: Mapped[str] = mapped_column(Text, nullable=False)
    portrait_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    portrait_negative_prompt: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    analysis: Mapped["AnalysisResult"] = relationship(
        back_populates="character_prompts"
    )

    __table_args__ = (Index("ix_character_prompts_analysis_id", "analysis_id"),)


class LocationPrompt(Base):
    """Flux/SDXL-optimized prompt for a location reference image."""

    __tablename__ = "location_prompts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    location_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_scene_orders: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    analysis: Mapped["AnalysisResult"] = relationship(back_populates="location_prompts")

    __table_args__ = (Index("ix_location_prompts_analysis_id", "analysis_id"),)
