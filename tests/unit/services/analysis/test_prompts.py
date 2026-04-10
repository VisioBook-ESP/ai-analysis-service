from src.services.analysis.prompts import SYSTEM_PROMPT, build_analysis_prompt


class TestSystemPrompt:
    def test_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_contains_json_instruction(self):
        assert "JSON" in SYSTEM_PROMPT


class TestBuildAnalysisPrompt:
    def test_returns_string(self):
        assert isinstance(build_analysis_prompt("text", "fr", {}), str)

    def test_text_embedded_in_prompt(self):
        prompt = build_analysis_prompt("my unique text content here", "fr", {})
        assert "my unique text content here" in prompt

    def test_language_referenced_in_prompt(self):
        prompt = build_analysis_prompt("text", "fr", {})
        assert "fr" in prompt

    def test_sentiment_always_present(self):
        prompt = build_analysis_prompt("text", "fr", {})
        assert '"sentiment"' in prompt

    def test_no_options_excludes_characters_scenes_narrative_summary(self):
        prompt = build_analysis_prompt("text", "fr", {})
        assert '"characters"' not in prompt
        assert '"scenes"' not in prompt
        assert '"narrative"' not in prompt
        assert '"summary"' not in prompt

    def test_characters_option_true(self):
        prompt = build_analysis_prompt("text", "fr", {"characters": True})
        assert '"characters"' in prompt

    def test_scenes_option_true(self):
        prompt = build_analysis_prompt("text", "fr", {"scenes": True})
        assert '"scenes"' in prompt

    def test_narrative_option_true(self):
        prompt = build_analysis_prompt("text", "fr", {"narrative": True})
        assert '"narrative"' in prompt

    def test_summary_option_true(self):
        prompt = build_analysis_prompt("text", "fr", {"summary": True})
        assert '"summary"' in prompt

    def test_all_options_true_includes_all_sections(self):
        opts = {"characters": True, "scenes": True, "narrative": True, "summary": True}
        prompt = build_analysis_prompt("text", "en", opts)
        for key in [
            '"characters"',
            '"scenes"',
            '"narrative"',
            '"sentiment"',
            '"summary"',
        ]:
            assert key in prompt

    def test_options_false_excludes_sections(self):
        opts = {
            "characters": False,
            "scenes": False,
            "narrative": False,
            "summary": False,
        }
        prompt = build_analysis_prompt("text", "fr", opts)
        assert '"characters"' not in prompt
        assert '"scenes"' not in prompt
        assert '"narrative"' not in prompt
        assert '"summary"' not in prompt
        assert '"sentiment"' in prompt  # toujours présent
