"""Add prompt tables for Flux/SDXL image generation.

Revision ID: 002
Revises: 001
Create Date: 2026-04-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Scene prompts
    op.create_table(
        "scene_prompts",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analysis_results.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scene_order", sa.Integer, nullable=False),
        sa.Column("image_prompt", sa.Text, nullable=False),
        sa.Column("negative_prompt", sa.Text, nullable=False, server_default=""),
        sa.Column("characters_present", JSONB, nullable=True),
        sa.Column("location_id", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_scene_prompts_analysis_id", "scene_prompts", ["analysis_id"])

    # Character prompts
    op.create_table(
        "character_prompts",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analysis_results.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("physical_description", sa.Text, nullable=False),
        sa.Column("portrait_prompt", sa.Text, nullable=False),
        sa.Column(
            "portrait_negative_prompt", sa.Text, nullable=False, server_default=""
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_character_prompts_analysis_id", "character_prompts", ["analysis_id"]
    )

    # Location prompts
    op.create_table(
        "location_prompts",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analysis_results.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("location_id", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description_prompt", sa.Text, nullable=False),
        sa.Column("negative_prompt", sa.Text, nullable=False, server_default=""),
        sa.Column("source_scene_orders", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_location_prompts_analysis_id", "location_prompts", ["analysis_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_location_prompts_analysis_id")
    op.drop_table("location_prompts")
    op.drop_index("ix_character_prompts_analysis_id")
    op.drop_table("character_prompts")
    op.drop_index("ix_scene_prompts_analysis_id")
    op.drop_table("scene_prompts")
