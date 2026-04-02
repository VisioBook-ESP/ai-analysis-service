"""Tests for WorkflowHandler database persistence."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.workflow_handler import WorkflowHandler


@pytest.fixture
def mock_nats():
    client = MagicMock()
    client.publish = AsyncMock()
    return client


@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(
        return_value={
            "language": "fr",
            "text_stats": {
                "original_length": 100,
                "cleaned_length": 95,
                "sentence_count": 3,
                "word_count": 20,
                "quality_score": 0.8,
                "quality_assessment": "good",
            },
            "characters": [
                {
                    "name": "Alice",
                    "role": "protagonist",
                    "physical_description": "jeune fille",
                    "personality_traits": ["brave"],
                    "emotions": ["hopeful"],
                    "motivations": [],
                    "actions": ["marche"],
                    "relationships": [],
                }
            ],
            "scenes": [
                {
                    "scene_id": 1,
                    "title": "La forêt",
                    "text_excerpt": "Alice marcha dans la forêt.",
                    "characters_present": ["Alice"],
                    "setting": {
                        "location": "forêt",
                        "time_period": "unknown",
                        "time_of_day": "matin",
                    },
                    "atmosphere": {
                        "mood": "mysterious",
                        "lighting": "dim",
                        "weather": "fog",
                        "colors": ["green", "grey"],
                        "sounds_textures": {"sounds": ["birds"], "textures": ["leaves"]},
                    },
                    "key_events": ["Alice entre dans la forêt"],
                    "objects": ["arbres"],
                }
            ],
            "narrative": {"themes": ["journey"], "tone": "mysterious"},
            "sentiment": {"overall": "neutral", "polarity": 0.1},
            "summary": {"summary": "Alice walks in a forest.", "key_points": ["journey"]},
        }
    )
    return analyzer


@pytest.fixture
def mock_db_client():
    client = MagicMock()
    client.save_analysis = AsyncMock(return_value=True)
    return client


@pytest.fixture
def handler(mock_nats, mock_analyzer, mock_db_client):
    return WorkflowHandler(
        nats_client=mock_nats,
        analyzer=mock_analyzer,
        db_client=mock_db_client,
    )


@pytest.fixture
def workflow_data():
    return {
        "projectId": "proj-123",
        "versionId": "ver-456",
        "executionId": "exec-789",
        "userId": "user-001",
        "correlationId": "corr-abc",
        "contentText": "Alice marcha dans la forêt sombre.",
        "config": {"language": "fr"},
    }


@pytest.mark.asyncio
async def test_persists_successful_analysis(handler, mock_db_client, workflow_data):
    await handler.handle_workflow_started(workflow_data)

    mock_db_client.save_analysis.assert_called_once()
    call_kwargs = mock_db_client.save_analysis.call_args.kwargs
    assert call_kwargs["project_id"] == "proj-123"
    assert call_kwargs["version_id"] == "ver-456"
    assert call_kwargs["execution_id"] == "exec-789"
    assert call_kwargs["user_id"] == "user-001"
    assert call_kwargs["status"] == "completed"
    assert call_kwargs["correlation_id"] == "corr-abc"
    assert isinstance(call_kwargs["scenes"], list)
    assert isinstance(call_kwargs["characters"], list)
    assert call_kwargs["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_persists_failure_on_empty_text(handler, mock_db_client, workflow_data):
    workflow_data["contentText"] = ""
    await handler.handle_workflow_started(workflow_data)

    mock_db_client.save_analysis.assert_called_once()
    call_kwargs = mock_db_client.save_analysis.call_args.kwargs
    assert call_kwargs["status"] == "failed"
    assert "No content text" in call_kwargs["error"]


@pytest.mark.asyncio
async def test_persists_failure_on_analyzer_exception(
    handler, mock_analyzer, mock_db_client, workflow_data
):
    mock_analyzer.analyze.side_effect = RuntimeError("LLM timeout")
    await handler.handle_workflow_started(workflow_data)

    mock_db_client.save_analysis.assert_called_once()
    call_kwargs = mock_db_client.save_analysis.call_args.kwargs
    assert call_kwargs["status"] == "failed"
    assert "LLM timeout" in call_kwargs["error"]


@pytest.mark.asyncio
async def test_publishes_nats_after_persistence(handler, mock_nats, mock_db_client, workflow_data):
    await handler.handle_workflow_started(workflow_data)

    # DB save should have been called before NATS publish
    assert mock_db_client.save_analysis.called
    # analysis.completed should be published
    subjects = [call.args[0] for call in mock_nats.publish.call_args_list]
    assert "visiobook.ai.analysis.completed" in subjects


@pytest.mark.asyncio
async def test_db_failure_does_not_block_nats_publish(
    handler, mock_nats, mock_db_client, workflow_data
):
    mock_db_client.save_analysis.return_value = False
    await handler.handle_workflow_started(workflow_data)

    # NATS should still be published even if DB save fails
    subjects = [call.args[0] for call in mock_nats.publish.call_args_list]
    assert "visiobook.ai.analysis.completed" in subjects
