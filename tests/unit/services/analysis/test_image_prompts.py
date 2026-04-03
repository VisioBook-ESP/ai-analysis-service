from src.services.analysis.image_prompts import (
    IMAGE_PROMPT_SYSTEM_PROMPT,
    build_image_prompt_request,
)


class TestImagePromptSystemPrompt:
    def test_system_prompt_requires_english(self):
        assert "English" in IMAGE_PROMPT_SYSTEM_PROMPT

    def test_system_prompt_forbids_style(self):
        assert "Do NOT include the visual style" in IMAGE_PROMPT_SYSTEM_PROMPT

    def test_system_prompt_forbids_names(self):
        assert "never by name" in IMAGE_PROMPT_SYSTEM_PROMPT

    def test_system_prompt_requires_json(self):
        assert "valid JSON only" in IMAGE_PROMPT_SYSTEM_PROMPT


class TestBuildImagePromptRequest:
    def test_includes_visual_style(self):
        result = build_image_prompt_request(
            {"characters": [], "scenes": []}, "watercolor", "fr"
        )
        assert "watercolor" in result
        assert "do not include the style" in result.lower()

    def test_includes_language(self):
        result = build_image_prompt_request(
            {"characters": [], "scenes": []}, "realistic", "fr"
        )
        assert '"fr"' in result

    def test_includes_character_context(self):
        analysis = {
            "characters": [
                {
                    "name": "Pierre",
                    "role": "protagonist",
                    "physical_description": "tall, dark hair",
                    "personality_traits": ["brave"],
                }
            ],
            "scenes": [],
        }
        result = build_image_prompt_request(analysis, "realistic", "fr")
        assert "Pierre" in result
        assert "protagonist" in result
        assert "tall, dark hair" in result

    def test_includes_scene_context(self):
        analysis = {
            "characters": [],
            "scenes": [
                {
                    "title": "The Forest",
                    "text_excerpt": "A dark and mysterious forest...",
                    "characters_present": ["Pierre"],
                    "setting": {
                        "location": "forest",
                        "time_of_day": "dusk",
                    },
                    "atmosphere": {
                        "mood": "mysterious",
                        "lighting": "dim",
                        "colors": ["green", "black"],
                    },
                }
            ],
        }
        result = build_image_prompt_request(analysis, "realistic", "fr")
        assert "The Forest" in result
        assert "forest" in result
        assert "mysterious" in result

    def test_includes_narrative_context(self):
        analysis = {
            "characters": [],
            "scenes": [],
            "narrative": {
                "tone": "dark",
                "themes": ["redemption", "loss"],
            },
        }
        result = build_image_prompt_request(analysis, "realistic", "fr")
        assert "dark" in result
        assert "redemption" in result

    def test_handles_missing_fields_gracefully(self):
        analysis = {"characters": [], "scenes": []}
        result = build_image_prompt_request(analysis, "realistic", "auto")
        assert "scene_prompts" in result
        assert "character_prompts" in result
        assert "location_prompts" in result

    def test_output_requests_json_schema(self):
        result = build_image_prompt_request(
            {"characters": [], "scenes": []}, "realistic", "fr"
        )
        assert "scene_prompts" in result
        assert "character_prompts" in result
        assert "location_prompts" in result
        assert "image_prompt" in result
        assert "portrait_prompt" in result
        assert "description_prompt" in result
