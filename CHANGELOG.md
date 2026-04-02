# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **Database persistence** — analysis results now stored in PostgreSQL (`analysis_results` table on postgres-io/analysis)
  - SQLAlchemy async engine + session factory (`src/database/connection.py`)
  - `AnalysisResult` model with JSONB columns for scenes, characters, narrative, sentiment, summary, text_stats (`src/database/models.py`)
  - `DatabaseClient` with upsert by execution_id, query by project, health check (`src/clients/database_client.py`)
  - Alembic migration `001_create_analysis_results` with indexes on project_id, version_id, execution_id (unique), user_id
- **Results API endpoints** — retrieve persisted analysis output
  - `GET /api/v1/results/{execution_id}` — single result with ownership enforcement
  - `GET /api/v1/results/project/{project_id}` — list all analyses for a project
- **WorkflowHandler persistence** — saves full LLM output (scenes, characters, narrative, sentiment, summary, text_stats, timing) to DB after analysis completes; saves error record on failure
- **Health check** — `/ready` now includes database connectivity check
- **Dependencies** — `sqlalchemy[asyncio]`, `asyncpg`, `alembic`
- **Tests** — unit tests for WorkflowHandler persistence (5 tests) and DatabaseClient CRUD (6 tests)

### Changed

- `docker-compose.yml` — added postgres service (port 5433) with health check
- `Dockerfile` — copies alembic config/migrations, runs `alembic upgrade head` on startup
- `charts/values.yaml` — added `DATABASE_URL` env var pointing to postgres-io cluster
- `src/config/settings.py` — added `DATABASE_URL`, `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`
