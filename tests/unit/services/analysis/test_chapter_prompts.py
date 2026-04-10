"""Tests for map and reduce prompt builders."""

from src.services.analysis.chapter_prompts import (
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    build_map_prompt,
    build_reduce_prompt,
)


# ── MAP prompt ────────────────────────────────────────────────────────────


class TestBuildMapPrompt:
    def test_contains_chapter_context(self):
        prompt = build_map_prompt(
            chapter_text="Il faisait beau.",
            chapter_title="Chapitre III",
            chapter_index=2,
            total_chapters=10,
            language="fr",
            options={"characters": True, "scenes": True},
        )
        assert "chapter 3 of 10" in prompt
        assert "Chapitre III" in prompt

    def test_contains_chapter_text(self):
        prompt = build_map_prompt(
            chapter_text="The sun was setting over the castle.",
            chapter_title="Chapter 1",
            chapter_index=0,
            total_chapters=5,
            language="en",
            options={"characters": True, "scenes": True},
        )
        assert "The sun was setting over the castle." in prompt

    def test_includes_characters_schema_when_enabled(self):
        prompt = build_map_prompt(
            chapter_text="text",
            chapter_title="Ch 1",
            chapter_index=0,
            total_chapters=1,
            language="en",
            options={"characters": True, "scenes": False},
        )
        assert '"characters"' in prompt
        assert '"name"' in prompt
        assert '"role"' in prompt

    def test_includes_scenes_schema_when_enabled(self):
        prompt = build_map_prompt(
            chapter_text="text",
            chapter_title="Ch 1",
            chapter_index=0,
            total_chapters=1,
            language="en",
            options={"characters": False, "scenes": True},
        )
        assert '"scenes"' in prompt
        assert '"scene_id"' in prompt
        assert '"atmosphere"' in prompt

    def test_always_includes_key_events_and_sentiment(self):
        prompt = build_map_prompt(
            chapter_text="text",
            chapter_title="Ch 1",
            chapter_index=0,
            total_chapters=1,
            language="en",
            options={"characters": False, "scenes": False},
        )
        assert '"key_events"' in prompt
        assert '"sentiment_hint"' in prompt

    def test_includes_language(self):
        prompt = build_map_prompt(
            chapter_text="text",
            chapter_title="Ch 1",
            chapter_index=0,
            total_chapters=1,
            language="fr",
            options={},
        )
        assert "(fr)" in prompt


# ── REDUCE prompt ─────────────────────────────────────────────────────────


class TestBuildReducePrompt:
    def test_contains_chapter_count(self):
        results = [
            {"characters": [], "key_events": ["event1"], "sentiment_hint": "positive"},
            {"characters": [], "key_events": ["event2"], "sentiment_hint": "negative"},
        ]
        prompt = build_reduce_prompt(
            results, "fr", {"characters": True, "summary": True}
        )
        assert "2 chapters" in prompt

    def test_contains_chapter_results_as_json(self):
        results = [
            {
                "characters": [{"name": "Peter", "role": "protagonist"}],
                "key_events": ["Peter lost his shadow"],
                "sentiment_hint": "negative",
            },
        ]
        prompt = build_reduce_prompt(results, "en", {"characters": True})
        assert "Peter" in prompt
        assert "lost his shadow" in prompt

    def test_includes_characters_schema_when_enabled(self):
        prompt = build_reduce_prompt(
            [{"characters": [], "key_events": [], "sentiment_hint": "neutral"}],
            "en",
            {"characters": True, "narrative": False, "summary": False},
        )
        assert '"characters"' in prompt
        assert '"relationships"' in prompt

    def test_includes_narrative_schema_when_enabled(self):
        prompt = build_reduce_prompt(
            [{"characters": [], "key_events": [], "sentiment_hint": "neutral"}],
            "en",
            {"characters": False, "narrative": True, "summary": False},
        )
        assert '"narrative"' in prompt
        assert '"themes"' in prompt

    def test_includes_summary_schema_when_enabled(self):
        prompt = build_reduce_prompt(
            [{"characters": [], "key_events": [], "sentiment_hint": "neutral"}],
            "en",
            {"characters": False, "narrative": False, "summary": True},
        )
        assert '"summary"' in prompt
        assert '"key_points"' in prompt

    def test_always_includes_sentiment(self):
        prompt = build_reduce_prompt(
            [{"characters": [], "key_events": [], "sentiment_hint": "neutral"}],
            "en",
            {},
        )
        assert '"sentiment"' in prompt
        assert '"polarity"' in prompt

    def test_language_in_prompt(self):
        prompt = build_reduce_prompt(
            [{"characters": [], "key_events": [], "sentiment_hint": "neutral"}],
            "fr",
            {},
        )
        assert "(fr)" in prompt


# ── System prompts ────────────────────────────────────────────────────────


class TestSystemPrompts:
    def test_map_system_prompt_mentions_single_chapter(self):
        assert "SINGLE CHAPTER" in MAP_SYSTEM_PROMPT

    def test_reduce_system_prompt_mentions_synthesis(self):
        assert "synthesiz" in REDUCE_SYSTEM_PROMPT.lower()

    def test_both_require_json(self):
        assert "JSON" in MAP_SYSTEM_PROMPT
        assert "JSON" in REDUCE_SYSTEM_PROMPT
