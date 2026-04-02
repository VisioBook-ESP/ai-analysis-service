"""Create analysis_results table.

Revision ID: 001
Revises: None
Create Date: 2026-04-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_results",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("project_id", sa.String(255), nullable=False),
        sa.Column("version_id", sa.String(255), nullable=False),
        sa.Column("execution_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="completed"),
        sa.Column("scenes", JSONB, nullable=True),
        sa.Column("characters", JSONB, nullable=True),
        sa.Column("narrative", JSONB, nullable=True),
        sa.Column("sentiment", JSONB, nullable=True),
        sa.Column("summary", JSONB, nullable=True),
        sa.Column("text_stats", JSONB, nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("processing_time_ms", sa.Float, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("correlation_id", sa.String(255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_index("ix_analysis_results_project_id", "analysis_results", ["project_id"])
    op.create_index("ix_analysis_results_version_id", "analysis_results", ["version_id"])
    op.create_index(
        "ix_analysis_results_execution_id", "analysis_results", ["execution_id"], unique=True
    )
    op.create_index("ix_analysis_results_user_id", "analysis_results", ["user_id"])
    op.create_index(
        "ix_analysis_results_project_version", "analysis_results", ["project_id", "version_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_results_project_version")
    op.drop_index("ix_analysis_results_user_id")
    op.drop_index("ix_analysis_results_execution_id")
    op.drop_index("ix_analysis_results_project_id")
    op.drop_index("ix_analysis_results_version_id")
    op.drop_table("analysis_results")
