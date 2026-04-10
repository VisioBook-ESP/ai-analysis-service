from unittest.mock import AsyncMock, patch

import pytest

from src.services.analysis.analyzer import AnalysisOptions, Analyzer
from src.services.analysis.llm_client import LLMClient
from src.services.preprocessing.chapter_detector import Chapter
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
    "stats": {
        "original_length": 28,
        "cleaned_length": 27,
        "sentence_count": 1,
        "chunk_count": 0,
    },
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
    "sentiment": {
        "overall": "positive",
        "polarity": 0.5,
        "nuances": [],
        "emotional_arc": "stable",
    },
    "summary": {
        "summary": "Alice marche dans la forêt.",
        "key_points": ["Alice marche"],
    },
}

# Single chapter: triggers single-shot path
SINGLE_CHAPTER = [
    Chapter(
        index=0,
        title="Section 1",
        start_char=0,
        end_char=27,
        text="Alice marcha dans la forêt.",
        word_count=5,
        estimated_tokens=7,
    )
]

# Multiple chapters: triggers chunked path
MULTI_CHAPTERS = [
    Chapter(
        index=0,
        title="Chapitre I",
        start_char=0,
        end_char=100,
        text="Premier chapitre. " * 50,
        word_count=100,
        estimated_tokens=140,
    ),
    Chapter(
        index=1,
        title="Chapitre II",
        start_char=100,
        end_char=200,
        text="Deuxième chapitre. " * 50,
        word_count=100,
        estimated_tokens=140,
    ),
    Chapter(
        index=2,
        title="Chapitre III",
        start_char=200,
        end_char=300,
        text="Troisième chapitre. " * 50,
        word_count=100,
        estimated_tokens=140,
    ),
]

# Mock map result (per-chapter)
MOCK_MAP_RESULT = {
    "characters": [
        {
            "name": "Alice",
            "role": "protagonist",
            "physical_description": "",
            "personality_traits": ["brave"],
            "emotions": [],
            "actions": [],
        }
    ],
    "scenes": [
        {
            "scene_id": "scene_001",
            "title": "Une scène",
            "text_excerpt": "texte",
            "characters_present": ["Alice"],
            "setting": {},
            "atmosphere": {},
            "key_events": [],
            "objects": [],
        }
    ],
    "key_events": ["Alice walks"],
    "sentiment_hint": "positive",
}

# Mock reduce result (global)
MOCK_REDUCE_RESULT = {
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
        },
    ],
    "narrative": {
        "themes": ["journey"],
        "tone": "calm",
        "style": "descriptive",
        "point_of_view": "third person",
        "tension_level": "low",
        "pacing": "slow",
        "literary_devices": [],
    },
    "sentiment": {
        "overall": "positive",
        "polarity": 0.5,
        "nuances": [],
        "emotional_arc": "stable",
    },
    "summary": {"summary": "Alice marche.", "key_points": ["Alice marche"]},
}


# Mock Pass 1 result (dialogue & structure extraction)
MOCK_PASS1_RESULT = {
    "dialogues": [
        {
            "order": 1,
            "speaker": "Alice",
            "line": "Où suis-je?",
            "delivery": "trembling",
            "context": "Alice regarda autour d'elle.",
        }
    ],
    "narrative_blocks": [
        {
            "order": 0,
            "type": "description",
            "text": "La forêt était sombre et silencieuse.",
            "characters_mentioned": ["Alice"],
        }
    ],
    "audio_cues": [
        {
            "type": "ambient",
            "description": "wind through trees",
            "source_block_order": 0,
        }
    ],
    "characters_in_chapter": ["Alice"],
    "sentiment_hint": "mixed",
}

# Mock Pass 2 result (visual scene composition)
MOCK_PASS2_RESULT = {
    "scenes": [
        {
            "scene_id": "ch1_s1",
            "scene_type": "establishing",
            "title": "La forêt sombre",
            "text_excerpt": "La forêt était sombre.",
            "characters_present": ["Alice"],
            "setting": {
                "location": "forest",
                "time_period": "",
                "time_of_day": "night",
            },
            "atmosphere": {"mood": "tense", "lighting": "dim"},
            "key_events": [],
            "objects": [],
            "audio_cues": [{"type": "ambient", "description": "wind through trees"}],
            "narration_text": "La forêt était sombre et silencieuse.",
            "dialogues": [
                {"speaker": "Alice", "line": "Où suis-je?", "delivery": "trembling"}
            ],
            "source_block_orders": [0, 1],
        }
    ]
}


@pytest.fixture
def analyzer(mock_settings):
    return Analyzer()


def _patch_detect_chapters(chapters):
    """Patch detect_chapters in the analyzer module to return given chapters."""
    return patch(
        "src.services.analysis.analyzer.detect_chapters",
        new_callable=AsyncMock,
        return_value=chapters,
    )


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
        opts = AnalysisOptions(
            characters=False, scenes=False, mask_pii=False, max_summary_length=100
        )
        assert opts.characters is False
        assert opts.scenes is False
        assert opts.mask_pii is False
        assert opts.max_summary_length == 100

    def test_to_dict_all_true(self):
        d = AnalysisOptions().to_dict()
        assert d == {
            "characters": True,
            "scenes": True,
            "narrative": True,
            "summary": True,
        }

    def test_to_dict_reflects_false_options(self):
        d = AnalysisOptions(characters=False, narrative=False).to_dict()
        assert d["characters"] is False
        assert d["narrative"] is False
        assert "mask_pii" not in d


# ---------------------------------------------------------------------------
# Analyzer.analyze — single-shot path
# ---------------------------------------------------------------------------


class TestAnalyzerSingleShot:
    async def test_returns_expected_top_level_keys(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert "language" in result
        assert "text_stats" in result
        assert "processing_time_ms" in result

    async def test_language_from_preprocessor(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert result["language"] == "fr"

    async def test_text_stats_structure(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("Alice marcha dans la forêt. x")
        stats = result["text_stats"]
        assert stats["cleaned_length"] == len(MOCK_PREPROCESS["text"])
        assert stats["sentence_count"] == 1
        assert stats["quality_score"] == 0.05

    async def test_summary_enriched_with_lengths(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("Alice marcha dans la forêt.")
        assert "original_length" in result["summary"]
        assert "summary_length" in result["summary"]

    async def test_step_callback_single_shot(self, analyzer):
        steps = []

        async def on_step(step: str):
            steps.append(step)

        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    await analyzer.analyze("text", on_step=on_step)
        assert steps == ["preprocessing", "chapter_detection", "llm_call", "parsing"]

    async def test_no_step_callback_does_not_raise(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("text", on_step=None)
        assert "language" in result

    async def test_llm_failure_raises_runtime_error(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient,
                    "chat_completion",
                    new_callable=AsyncMock,
                    side_effect=Exception("connection failed"),
                ):
                    with pytest.raises(RuntimeError, match="LLM analysis failed"):
                        await analyzer.analyze("some text")

    async def test_none_options_uses_defaults(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    result = await analyzer.analyze("text", options=None)
        assert "language" in result

    async def test_custom_language_passed_to_preprocessor(self, analyzer):
        with patch.object(
            TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS
        ) as mock_prep:
            with _patch_detect_chapters(SINGLE_CHAPTER):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = MOCK_LLM
                    await analyzer.analyze("text", language="en")
        mock_prep.assert_called_once()
        call_kwargs = mock_prep.call_args
        assert call_kwargs[1].get("language") == "en" or call_kwargs[0][1] == "en"


# ---------------------------------------------------------------------------
# Analyzer.analyze — chunked path
# ---------------------------------------------------------------------------


class TestAnalyzerChunked:
    """Tests for the two-pass chunked analysis path.

    Per chapter: Pass 1 (dialogue extraction) + Pass 2 (scene composition).
    Then 1 reduce call. Total: 2 * N_chapters + 1 calls.
    """

    def _two_pass_side_effects(self, n_chapters: int, reduce_result=None):
        """Build side_effect list for N chapters with two-pass + reduce."""
        effects = []
        for _ in range(n_chapters):
            effects.append(MOCK_PASS1_RESULT)  # Pass 1
            effects.append(MOCK_PASS2_RESULT)  # Pass 2
        effects.append(reduce_result or MOCK_REDUCE_RESULT)
        return effects

    async def test_step_callback_chunked(self, analyzer):
        steps = []

        async def on_step(step: str):
            steps.append(step)

        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(3)
                    await analyzer.analyze("long text", on_step=on_step)

        assert "preprocessing" in steps
        assert "chapter_detection" in steps
        assert "chapter_1_of_3" in steps
        assert "chapter_2_of_3" in steps
        assert "chapter_3_of_3" in steps
        assert "global_synthesis" in steps
        assert "parsing" in steps

    async def test_scenes_renumbered_globally(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(3)
                    result = await analyzer.analyze("long text")

        scenes = result.get("scenes", [])
        assert len(scenes) == 3  # 1 scene per chapter × 3 chapters
        scene_ids = [s["scene_id"] for s in scenes]
        assert scene_ids == ["scene_001", "scene_002", "scene_003"]

    async def test_characters_from_reduce(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(3)
                    result = await analyzer.analyze("long text")

        chars = result.get("characters", [])
        assert len(chars) == 1
        assert chars[0]["name"] == "Alice"

    async def test_llm_calls_count(self, analyzer):
        """Should make 2N pass calls + 1 reduce call (two-pass pipeline)."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(3)
                    await analyzer.analyze("long text")

        assert mock_llm.call_count == 7  # 3 × (Pass1 + Pass2) + 1 reduce

    async def test_partial_failure_pass1_falls_back_to_legacy_map(self, analyzer):
        """If Pass 1 fails, fall back to legacy single-pass MAP for that chapter."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = [
                        MOCK_PASS1_RESULT,  # ch1 pass1
                        MOCK_PASS2_RESULT,  # ch1 pass2
                        Exception("vLLM timeout"),  # ch2 pass1 fails
                        MOCK_MAP_RESULT,  # ch2 legacy MAP fallback
                        MOCK_PASS1_RESULT,  # ch3 pass1
                        MOCK_PASS2_RESULT,  # ch3 pass2
                        MOCK_REDUCE_RESULT,  # reduce
                    ]
                    result = await analyzer.analyze("long text")

        # 3 chapters all produced scenes (ch2 via legacy fallback)
        scenes = result.get("scenes", [])
        assert len(scenes) == 3

    async def test_partial_failure_both_passes_skips_chapter(self, analyzer):
        """If Pass 1 AND legacy MAP both fail, skip that chapter."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = [
                        MOCK_PASS1_RESULT,  # ch1 pass1
                        MOCK_PASS2_RESULT,  # ch1 pass2
                        Exception("pass1 timeout"),  # ch2 pass1 fails
                        Exception("map timeout"),  # ch2 legacy MAP also fails
                        MOCK_PASS1_RESULT,  # ch3 pass1
                        MOCK_PASS2_RESULT,  # ch3 pass2
                        MOCK_REDUCE_RESULT,  # reduce
                    ]
                    result = await analyzer.analyze("long text")

        # 2 successful chapters × 1 scene each = 2 scenes
        scenes = result.get("scenes", [])
        assert len(scenes) == 2

    async def test_pass2_failure_uses_mechanical_split(self, analyzer):
        """If Pass 2 fails but Pass 1 succeeded, use mechanical scene split."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = [
                        MOCK_PASS1_RESULT,  # ch1 pass1
                        Exception("pass2 failed"),  # ch1 pass2 fails → mechanical
                        MOCK_PASS1_RESULT,  # ch2 pass1
                        MOCK_PASS2_RESULT,  # ch2 pass2
                        MOCK_PASS1_RESULT,  # ch3 pass1
                        MOCK_PASS2_RESULT,  # ch3 pass2
                        MOCK_REDUCE_RESULT,  # reduce
                    ]
                    result = await analyzer.analyze("long text")

        scenes = result.get("scenes", [])
        # ch1: mechanical split produces 2 scenes (1 narrative + 1 dialogue group)
        # ch2, ch3: 1 scene each from Pass 2
        assert len(scenes) == 4

    async def test_reduce_failure_raises(self, analyzer):
        """If the reduce call fails, the whole analysis fails."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    effects = self._two_pass_side_effects(3)
                    effects[-1] = Exception("reduce failed")
                    mock_llm.side_effect = effects
                    with pytest.raises(RuntimeError, match="global synthesis"):
                        await analyzer.analyze("long text")

    async def test_returns_all_top_level_keys(self, analyzer):
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS):
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(3)
                    result = await analyzer.analyze("long text")

        assert "language" in result
        assert "text_stats" in result
        assert "processing_time_ms" in result
        assert "characters" in result
        assert "scenes" in result
        assert "narrative" in result
        assert "sentiment" in result
        assert "summary" in result

    async def test_scene_new_fields_preserved(self, analyzer):
        """Scenes from two-pass pipeline should have scene_type, dialogues, etc."""
        with patch.object(TextPreprocessor, "preprocess", return_value=MOCK_PREPROCESS):
            with _patch_detect_chapters(MULTI_CHAPTERS[:2]):  # 2 chapters
                with patch.object(
                    LLMClient, "chat_completion", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.side_effect = self._two_pass_side_effects(2)
                    result = await analyzer.analyze("long text")

        scenes = result.get("scenes", [])
        assert len(scenes) == 2
        scene = scenes[0]
        assert scene.get("scene_type") == "establishing"
        assert scene.get("narration_text") == "La forêt était sombre et silencieuse."
        assert len(scene.get("dialogues", [])) == 1
        assert scene["dialogues"][0]["speaker"] == "Alice"
        assert scene["dialogues"][0]["line"] == "Où suis-je?"


# ---------------------------------------------------------------------------
# Analyzer.close
# ---------------------------------------------------------------------------


class TestAnalyzerClose:
    async def test_close_does_not_raise(self, analyzer):
        await analyzer.close()
