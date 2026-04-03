"""Tests for WorkflowHandler enriched mapping methods."""

import pytest

from src.services.workflow_handler import WorkflowHandler


@pytest.fixture
def handler():
    """Create a WorkflowHandler with mocked dependencies."""
    return WorkflowHandler(
        nats_client=None,
        analyzer=None,
        db_client=None,
    )


# ---------------------------------------------------------------------------
# _map_scenes — enriched with prompt data
# ---------------------------------------------------------------------------


class TestMapScenesEnriched:
    def test_uses_prompt_gen_image_prompt(self, handler):
        raw_scenes = [
            {
                "title": "Forest Scene",
                "text_excerpt": "A dark forest",
                "setting": {"location": "forest", "time_of_day": "night"},
                "atmosphere": {"mood": "dark"},
                "characters_present": ["Alice"],
            }
        ]
        image_prompts = {
            "scene_prompts": [
                {
                    "scene_order": 0,
                    "image_prompt": "dark mysterious forest at night, moonlight",
                    "negative_prompt": "blurry, text",
                    "characters_present": ["Alice"],
                    "location_id": "dark_forest",
                }
            ]
        }
        result = handler._map_scenes(raw_scenes, image_prompts)
        assert len(result) == 1
        assert result[0]["imagePrompt"] == "dark mysterious forest at night, moonlight"
        assert result[0]["negativePrompt"] == "blurry, text"
        assert result[0]["charactersPresent"] == ["Alice"]
        assert result[0]["locationId"] == "dark_forest"

    def test_falls_back_to_basic_prompt_without_prompt_gen(self, handler):
        raw_scenes = [
            {
                "title": "Forest Scene",
                "text_excerpt": "A dark forest with trees",
                "setting": {"location": "forest"},
                "atmosphere": {"mood": "dark"},
            }
        ]
        result = handler._map_scenes(raw_scenes, None)
        assert len(result) == 1
        assert "Forest Scene" in result[0]["imagePrompt"]
        assert "negativePrompt" not in result[0]
        assert "locationId" not in result[0]

    def test_falls_back_when_scene_not_in_prompts(self, handler):
        raw_scenes = [
            {"title": "S1", "text_excerpt": "text", "atmosphere": {"mood": "x"}},
            {"title": "S2", "text_excerpt": "text", "atmosphere": {"mood": "y"}},
        ]
        image_prompts = {
            "scene_prompts": [
                {"scene_order": 0, "image_prompt": "enriched prompt"}
            ]
        }
        result = handler._map_scenes(raw_scenes, image_prompts)
        assert result[0]["imagePrompt"] == "enriched prompt"
        # Scene 1 has no matching prompt, falls back
        assert "S2" in result[1]["imagePrompt"]


# ---------------------------------------------------------------------------
# _map_characters — enriched with prompt data
# ---------------------------------------------------------------------------


class TestMapCharactersEnriched:
    def test_adds_portrait_data(self, handler):
        raw_chars = [
            {
                "name": "Alice",
                "role": "protagonist",
                "physical_description": "jeune femme aux cheveux roux",
                "personality_traits": ["brave"],
            }
        ]
        image_prompts = {
            "character_prompts": [
                {
                    "name": "Alice",
                    "physical_description": "young woman, red hair, green eyes",
                    "portrait_prompt": "young woman with red hair, green eyes, blue cloak",
                    "portrait_negative_prompt": "multiple people, blurry",
                }
            ]
        }
        result = handler._map_characters(raw_chars, image_prompts)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["physicalDescription"] == "young woman, red hair, green eyes"
        assert "blue cloak" in result[0]["portraitPrompt"]
        assert result[0]["portraitNegativePrompt"] == "multiple people, blurry"

    def test_no_portrait_data_without_prompts(self, handler):
        raw_chars = [{"name": "Bob", "role": "secondary", "personality_traits": []}]
        result = handler._map_characters(raw_chars, None)
        assert "physicalDescription" not in result[0]
        assert "portraitPrompt" not in result[0]


# ---------------------------------------------------------------------------
# _map_locations
# ---------------------------------------------------------------------------


class TestMapLocations:
    def test_maps_locations(self):
        image_prompts = {
            "location_prompts": [
                {
                    "location_id": "forest_clearing",
                    "name": "La clairiere",
                    "description_prompt": "ancient forest clearing",
                    "negative_prompt": "people, text",
                    "source_scene_orders": [0, 2],
                }
            ]
        }
        result = WorkflowHandler._map_locations(image_prompts)
        assert len(result) == 1
        assert result[0]["locationId"] == "forest_clearing"
        assert result[0]["name"] == "La clairiere"
        assert result[0]["descriptionPrompt"] == "ancient forest clearing"
        assert result[0]["sourceSceneOrders"] == [0, 2]

    def test_returns_empty_without_prompts(self):
        assert WorkflowHandler._map_locations(None) == []

    def test_returns_empty_with_no_locations(self):
        assert WorkflowHandler._map_locations({"location_prompts": []}) == []
