"""Tests for DatabaseClient — mocked SQLAlchemy session."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.database_client import DatabaseClient


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def db_client():
    return DatabaseClient()


@pytest.mark.asyncio
async def test_save_analysis_success(db_client, mock_session):
    mock_session.execute = AsyncMock()

    with patch("src.clients.database_client.get_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await db_client.save_analysis(
            project_id="proj-1",
            version_id="ver-1",
            execution_id="exec-1",
            user_id="user-1",
            status="completed",
            scenes=[{"order": 0, "text": "scene text"}],
            characters=[{"name": "Alice"}],
            language="fr",
            processing_time_ms=1500.0,
            correlation_id="corr-1",
        )

    assert result is True
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_save_analysis_failure(db_client):
    with patch("src.clients.database_client.get_session") as mock_get_session:
        mock_get_session.side_effect = RuntimeError("Connection refused")

        result = await db_client.save_analysis(
            project_id="proj-1",
            version_id="ver-1",
            execution_id="exec-1",
            user_id="user-1",
        )

    assert result is False


@pytest.mark.asyncio
async def test_get_analysis_returns_result(db_client, mock_session):
    mock_row = MagicMock()
    mock_row.execution_id = "exec-1"
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_row
    mock_session.execute = AsyncMock(return_value=mock_result)

    with patch("src.clients.database_client.get_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await db_client.get_analysis("exec-1")

    assert result is not None
    assert result.execution_id == "exec-1"


@pytest.mark.asyncio
async def test_get_analysis_returns_none(db_client, mock_session):
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    with patch("src.clients.database_client.get_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await db_client.get_analysis("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_health_check_success(db_client, mock_session):
    mock_session.execute = AsyncMock()

    with patch("src.clients.database_client.get_session") as mock_get_session:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_session.return_value = ctx

        result = await db_client.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_failure(db_client):
    with patch("src.clients.database_client.get_session") as mock_get_session:
        mock_get_session.side_effect = RuntimeError("Connection refused")

        result = await db_client.health_check()

    assert result is False
