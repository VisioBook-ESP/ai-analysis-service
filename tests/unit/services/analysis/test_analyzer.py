from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.analysis.analyzer import AnalysisOptions, Analyzer
from src.services.analysis.llm_client import LLMClient
from src.services.preprocessing.preprocessor import TextPreprocessor


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

MOCK_PREPROCESS = {
    "language": "fr",
    "text": "Alice marcha dans la forêt.",
    "quality": {"score": 0.05, "assessment": "excellent"},
    "sentences": [{"start": 0, "end": 27, "text": "Alice marcha dans la forêt."}],
    "chunks": [],
    "masks": {"emails": [], "phones": [], "ibans": []},
    "stats": {"original_length": 28, "cleaned_length": 27, "sentence_count": 1, "chunk_count": 0},
    "processing_time_ms": 2.5,
}

MOCK_LLM = {
    "characters": [
        {
            "name": "Alice",
            "role": "protagonist",
            "physical_description": "",
            "personality_traits": ["brave"],
            "emotions": ["hopeful"],
            "motivations": [],
            "actions": ["walks"],
            "relationships": [],
        }
    ],
    "scenes": [],
    "narrative": {
        "themes": ["journey"],
        "tone": "calm",
        "style": "descriptive",
        "point_of_view": "third person",
        "tension_level": "low",
        "pacing": "slow",
        "literary_devices": [],
    },
    "sentiment": {"overall": "positive", "polarity": 0.5, "nuances": [], "emotional_arc": "stable"},
    "summary": {"summary": "Alice marche dans la forêt.", "key_points": ["Alice marche"]},
}


@pytest.fixture
def analyzer(mock_settings):
    return Analyzer()


# ---------------------------------------------------------------------------
# AnalysisOptions
# ---------------------------------------------------------------------------

class TestAnalysisOptions:
    def test_default_values(self):
        opts = AnalysisOptions()
        assert opts.characters is True
        assert opts.scenes is True
        assert opts.narrative is True
        assert opts.summary is True
        assert opts.mask_pii is True
        assert opts.remove_links is False
        assert opts.max_summary_length == 200

    def test_custom_values(self):
        opts = AnalysisOptions(characters=False, scenes=False, mask_pii=False, max_summary_length=100)
        assert opts.characters is False
        assert opts.scenes is False
        assert opts.mask_pii is False
        assert opts.max_summary_length == 100

    def test_to_dict_all_true(self):
        d = AnalysisOptions().to_dict()
        assert d == {"characters": True, "scenes": True, "narrative": True, "summary": True}

    def test_to_dict_reflects_false_options(self):
        d = AnalysisOptions(characters=False, narrative=False).to_dict()
        assert d["characters"] is False
        assert d["narrative"] is False
        # mask_pii et remove_links ne sont pas dans to_dict
        assert "mask_pii" not in d


# ---------------------------------------------------------------------------
# Analyzer.analyze
# ---------------------------------------------------------------------------

class TestAnalyzerAnalyze:
    async def test_returns_expected_top_level_keys(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert "language" in result
        assert "text_stats" in result
        assert "processing_time_ms" in result

    async def test_language_from_preprocessor(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert result["language"] == "fr"

    async def test_text_stats_structure(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("Alice marcha dans la forêt. x")
        stats = result["text_stats"]
        assert stats["cleaned_length"] == len(MOCK_PREPROCESS["text"])
        assert stats["sentence_count"] == 1
        assert stats["quality_score"] == 0.05
        assert stats["quality_assessment"] == "excellent"

    async def test_summary_enriched_with_lengths(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert "original_length" in result["summary"]
        assert "summary_length" in result["summary"]

    async def test_step_callback_called_in_order(self, analyzer):
        steps = []

        async def on_step(step: str):
            steps.append(step)

        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                await analyzer.analyze("text", on_step=on_step)
        assert steps == ["preprocessing", "llm_call", "parsing"]

    async def test_no_step_callback_does_not_raise(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("text", on_step=None)
        assert "language" in result

    async def test_llm_failure_raises_runtime_error(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(
                LLMClient, "chat_completion", new_callable=AsyncMock, side_effect=Exception("connection failed")
            ):
                with pytest.raises(RuntimeError, match="LLM analysis failed"):
                    await analyzer.analyze("some text")

    async def test_none_options_uses_defaults(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                result = await analyzer.analyze("text", options=None)
        assert "language" in result

    async def test_custom_language_passed_to_preprocessor(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS) as mock_prep:
            with patch.object(LLMClient, "chat_completion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = MOCK_LLM
                await analyzer.analyze("text", language="en")
        mock_prep.assert_called_once()
        call_kwargs = mock_prep.call_args
        assert call_kwargs[1].get("language") == "en" or call_kwargs[0][1] == "en"

    async def test_close_does_not_raise(self, analyzer):
        await analyzer.close()
