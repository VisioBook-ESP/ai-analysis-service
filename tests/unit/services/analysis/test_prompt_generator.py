import pytest

from src.services.analysis.prompt_generator import PromptGenerator


class FakeLLMClient:
    """Fake LLM client that returns a configurable response."""

    def __init__(self, response: dict | None = None):
        self.response = response or {}
        self.calls: list[dict] = []

    async def chat_completion(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.response


# ---------------------------------------------------------------------------
# _parse_scene_prompts
# ---------------------------------------------------------------------------


class TestParseScenePrompts:
    def test_valid_scene_prompt(self):
        prompts = [
            {
                "scene_order": 0,
                "image_prompt": "a forest clearing at dawn",
                "negative_prompt": "blurry, text",
                "characters_present": ["Alice"],
                "location_id": "forest_clearing",
            }
        ]
        result = PromptGenerator._parse_scene_prompts(prompts)
        assert len(result) == 1
        assert result[0]["scene_order"] == 0
        assert result[0]["image_prompt"] == "a forest clearing at dawn"
        assert result[0]["negative_prompt"] == "blurry, text"
        assert result[0]["characters_present"] == ["Alice"]
        assert result[0]["location_id"] == "forest_clearing"

    def test_empty_image_prompt_skipped(self):
        prompts = [{"scene_order": 0, "image_prompt": ""}]
        result = PromptGenerator._parse_scene_prompts(prompts)
        assert len(result) == 0

    def test_non_list_returns_empty(self):
        assert PromptGenerator._parse_scene_prompts("not a list") == []
        assert PromptGenerator._parse_scene_prompts(None) == []

    def test_non_dict_entries_skipped(self):
        result = PromptGenerator._parse_scene_prompts(["invalid", 42])
        assert len(result) == 0

    def test_missing_optional_fields_default(self):
        prompts = [{"image_prompt": "a scene"}]
        result = PromptGenerator._parse_scene_prompts(prompts)
        assert result[0]["negative_prompt"] == ""
        assert result[0]["characters_present"] == []
        assert result[0]["location_id"] is None


# ---------------------------------------------------------------------------
# _parse_character_prompts
# ---------------------------------------------------------------------------


class TestParseCharacterPrompts:
    def test_valid_character_prompt(self):
        prompts = [
            {
                "name": "Alice",
                "physical_description": "young woman, red hair",
                "portrait_prompt": "young woman with red hair, green eyes",
                "portrait_negative_prompt": "blurry, bad anatomy",
            }
        ]
        result = PromptGenerator._parse_character_prompts(prompts)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["portrait_prompt"] == "young woman with red hair, green eyes"

    def test_missing_name_skipped(self):
        prompts = [{"portrait_prompt": "some prompt"}]
        result = PromptGenerator._parse_character_prompts(prompts)
        assert len(result) == 0

    def test_missing_portrait_prompt_skipped(self):
        prompts = [{"name": "Alice"}]
        result = PromptGenerator._parse_character_prompts(prompts)
        assert len(result) == 0

    def test_non_list_returns_empty(self):
        assert PromptGenerator._parse_character_prompts(None) == []


# ---------------------------------------------------------------------------
# _parse_location_prompts
# ---------------------------------------------------------------------------


class TestParseLocationPrompts:
    def test_valid_location_prompt(self):
        prompts = [
            {
                "location_id": "forest_clearing",
                "name": "La clairiere",
                "description_prompt": "ancient forest clearing",
                "negative_prompt": "people, text",
                "source_scene_orders": [0, 2],
            }
        ]
        result = PromptGenerator._parse_location_prompts(prompts)
        assert len(result) == 1
        assert result[0]["location_id"] == "forest_clearing"
        assert result[0]["source_scene_orders"] == [0, 2]

    def test_missing_location_id_skipped(self):
        prompts = [{"description_prompt": "a place"}]
        result = PromptGenerator._parse_location_prompts(prompts)
        assert len(result) == 0

    def test_missing_description_skipped(self):
        prompts = [{"location_id": "somewhere"}]
        result = PromptGenerator._parse_location_prompts(prompts)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# generate() — single call
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_single_call_returns_parsed_prompts(self):
        llm_response = {
            "scene_prompts": [
                {
                    "scene_order": 0,
                    "image_prompt": "a forest scene",
                    "negative_prompt": "blurry",
                    "characters_present": ["Alice"],
                    "location_id": "forest",
                }
            ],
            "character_prompts": [
                {
                    "name": "Alice",
                    "physical_description": "young woman",
                    "portrait_prompt": "young woman with red hair",
                    "portrait_negative_prompt": "bad anatomy",
                }
            ],
            "location_prompts": [
                {
                    "location_id": "forest",
                    "name": "Dark Forest",
                    "description_prompt": "ancient dark forest",
                    "negative_prompt": "people",
                    "source_scene_orders": [0],
                }
            ],
        }
        client = FakeLLMClient(response=llm_response)
        gen = PromptGenerator(client)
        result = await gen.generate(
            {"scenes": [{"title": "Scene 1"}], "characters": [{"name": "Alice"}]},
            visual_style="realistic",
            language="fr",
        )

        assert len(result["scene_prompts"]) == 1
        assert len(result["character_prompts"]) == 1
        assert len(result["location_prompts"]) == 1
        assert client.calls[0]["temperature"] == gen.temperature

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_empty_lists(self):
        client = FakeLLMClient(response={})
        gen = PromptGenerator(client)
        result = await gen.generate({"scenes": [], "characters": []})
        assert result == {
            "scene_prompts": [],
            "character_prompts": [],
            "location_prompts": [],
        }
