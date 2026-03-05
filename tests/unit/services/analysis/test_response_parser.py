import pytest

from src.services.analysis.response_parser import ResponseParser


@pytest.fixture
def parser():
    return ResponseParser()


# ---------------------------------------------------------------------------
# parse() — routing
# ---------------------------------------------------------------------------

class TestParse:
    def test_empty_options_empty_raw_returns_empty(self, parser):
        assert parser.parse({}, {}) == {}

    def test_characters_option_true_includes_field(self, parser):
        raw = {"characters": [{"name": "Alice"}]}
        result = parser.parse(raw, {"characters": True})
        assert "characters" in result

    def test_characters_option_false_skips_field(self, parser):
        raw = {"characters": [{"name": "Alice"}]}
        result = parser.parse(raw, {"characters": False})
        assert "characters" not in result

    def test_sentiment_always_parsed_when_present_regardless_of_options(self, parser):
        raw = {"sentiment": {"overall": "positive", "polarity": 0.8}}
        result = parser.parse(raw, {})
        assert "sentiment" in result

    def test_sentiment_absent_from_raw_not_in_result(self, parser):
        result = parser.parse({}, {})
        assert "sentiment" not in result

    def test_scenes_option_false_skips_field(self, parser):
        raw = {"scenes": [{"title": "Scene 1"}]}
        result = parser.parse(raw, {"scenes": False})
        assert "scenes" not in result

    def test_narrative_option_true_includes_field(self, parser):
        raw = {"narrative": {"tone": "dark"}}
        result = parser.parse(raw, {"narrative": True})
        assert "narrative" in result

    def test_scenes_option_true_includes_field(self, parser):
        raw = {"scenes": [{"title": "Forest"}]}
        result = parser.parse(raw, {"scenes": True})
        assert "scenes" in result

    def test_summary_option_true_includes_field(self, parser):
        raw = {"summary": {"summary": "Good story."}}
        result = parser.parse(raw, {"summary": True})
        assert "summary" in result

    def test_summary_option_false_skips_field(self, parser):
        raw = {"summary": {"summary": "Good story."}}
        result = parser.parse(raw, {"summary": False})
        assert "summary" not in result


# ---------------------------------------------------------------------------
# _parse_characters
# ---------------------------------------------------------------------------

class TestParseCharacters:
    def test_full_valid_character(self, parser):
        char = {
            "name": "Alice",
            "role": "protagonist",
            "physical_description": "tall and blonde",
            "personality_traits": ["brave", "curious"],
            "emotions": ["joy", "fear"],
            "motivations": ["save the world"],
            "actions": ["runs", "fights"],
            "relationships": [{"target": "Bob", "type": "friend", "description": "close"}],
        }
        result = parser._parse_characters([char])
        assert len(result) == 1
        c = result[0]
        assert c["name"] == "Alice"
        assert c["role"] == "protagonist"
        assert c["physical_description"] == "tall and blonde"
        assert c["personality_traits"] == ["brave", "curious"]
        assert c["emotions"] == ["joy", "fear"]
        assert c["relationships"][0]["target"] == "Bob"

    def test_missing_fields_use_defaults(self, parser):
        result = parser._parse_characters([{}])
        assert result[0]["name"] == "Unknown"
        assert result[0]["role"] == "secondary"
        assert result[0]["physical_description"] == ""
        assert result[0]["personality_traits"] == []

    def test_not_a_list_returns_empty(self, parser):
        assert parser._parse_characters("not a list") == []
        assert parser._parse_characters(None) == []
        assert parser._parse_characters(42) == []

    def test_non_dict_items_skipped(self, parser):
        result = parser._parse_characters(["string", None, {"name": "Alice"}])
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_multiple_characters(self, parser):
        chars = [{"name": "Alice"}, {"name": "Bob"}]
        result = parser._parse_characters(chars)
        assert [c["name"] for c in result] == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# _parse_relationships
# ---------------------------------------------------------------------------

class TestParseRelationships:
    def test_valid_relationship(self, parser):
        rels = [{"target": "Bob", "type": "enemy", "description": "old rivals"}]
        result = parser._parse_relationships(rels)
        assert result[0] == {"target": "Bob", "type": "enemy", "description": "old rivals"}

    def test_missing_fields_default_empty_string(self, parser):
        result = parser._parse_relationships([{}])
        assert result[0] == {"target": "", "type": "", "description": ""}

    def test_not_a_list_returns_empty(self, parser):
        assert parser._parse_relationships(None) == []
        assert parser._parse_relationships("string") == []

    def test_non_dict_items_skipped(self, parser):
        result = parser._parse_relationships(["bad", {"target": "X"}])
        assert len(result) == 1
        assert result[0]["target"] == "X"


# ---------------------------------------------------------------------------
# _parse_scenes
# ---------------------------------------------------------------------------

class TestParseScenes:
    def test_full_valid_scene(self, parser):
        scene = {
            "scene_id": "scene_001",
            "title": "The Beginning",
            "text_excerpt": "It was a dark and stormy night.",
            "characters_present": ["Alice", "Bob"],
            "key_events": ["Alice enters"],
            "objects": ["sword"],
        }
        result = parser._parse_scenes([scene])
        assert result[0]["scene_id"] == "scene_001"
        assert result[0]["title"] == "The Beginning"
        assert result[0]["characters_present"] == ["Alice", "Bob"]
        assert result[0]["key_events"] == ["Alice enters"]

    def test_scene_id_auto_generated_from_index(self, parser):
        result = parser._parse_scenes([{}, {}, {}])
        assert result[0]["scene_id"] == "scene_001"
        assert result[1]["scene_id"] == "scene_002"
        assert result[2]["scene_id"] == "scene_003"

    def test_scene_id_provided_overrides_auto(self, parser):
        result = parser._parse_scenes([{"scene_id": "custom_id"}])
        assert result[0]["scene_id"] == "custom_id"

    def test_not_a_list_returns_empty(self, parser):
        assert parser._parse_scenes(None) == []
        assert parser._parse_scenes("bad") == []

    def test_non_dict_items_skipped(self, parser):
        result = parser._parse_scenes(["bad", {"title": "Good scene"}])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _parse_setting
# ---------------------------------------------------------------------------

class TestParseSetting:
    def test_valid_setting(self, parser):
        setting = {"location": "forest", "time_period": "medieval", "time_of_day": "night"}
        result = parser._parse_setting(setting)
        assert result == {"location": "forest", "time_period": "medieval", "time_of_day": "night"}

    def test_missing_fields_default_unspecified(self, parser):
        result = parser._parse_setting({})
        assert result["location"] == "unspecified"
        assert result["time_period"] == "unspecified"
        assert result["time_of_day"] == "unspecified"

    def test_not_dict_uses_defaults(self, parser):
        result = parser._parse_setting("forest")
        assert result["location"] == "unspecified"


# ---------------------------------------------------------------------------
# _parse_atmosphere
# ---------------------------------------------------------------------------

class TestParseAtmosphere:
    def test_valid_atmosphere(self, parser):
        atm = {
            "mood": "tense",
            "lighting": "dim",
            "weather": "rainy",
            "colors": ["grey", "black"],
            "sounds_textures": {"sounds": ["rain"], "textures": ["rough"]},
        }
        result = parser._parse_atmosphere(atm)
        assert result["mood"] == "tense"
        assert result["lighting"] == "dim"
        assert result["colors"] == ["grey", "black"]
        assert result["sounds_textures"]["sounds"] == ["rain"]
        assert result["sounds_textures"]["textures"] == ["rough"]

    def test_missing_fields_use_defaults(self, parser):
        result = parser._parse_atmosphere({})
        assert result["mood"] == "neutral"
        assert result["lighting"] == "unspecified"
        assert result["weather"] == "unspecified"
        assert result["colors"] == []
        assert result["sounds_textures"] == {"sounds": [], "textures": []}

    def test_not_dict_uses_defaults(self, parser):
        result = parser._parse_atmosphere("dark")
        assert result["mood"] == "neutral"

    def test_sounds_textures_not_dict_uses_empty(self, parser):
        result = parser._parse_atmosphere({"sounds_textures": "loud"})
        assert result["sounds_textures"] == {"sounds": [], "textures": []}


# ---------------------------------------------------------------------------
# _parse_narrative
# ---------------------------------------------------------------------------

class TestParseNarrative:
    def test_valid_narrative(self, parser):
        narr = {
            "themes": ["love", "loss"],
            "tone": "melancholic",
            "style": "lyrical",
            "point_of_view": "first person",
            "tension_level": "high",
            "pacing": "slow",
            "literary_devices": ["metaphor", "alliteration"],
        }
        result = parser._parse_narrative(narr)
        assert result["themes"] == ["love", "loss"]
        assert result["tone"] == "melancholic"
        assert result["tension_level"] == "high"
        assert result["literary_devices"] == ["metaphor", "alliteration"]

    def test_missing_fields_use_defaults(self, parser):
        result = parser._parse_narrative({})
        assert result["themes"] == []
        assert result["tone"] == "neutral"
        assert result["tension_level"] == "low"
        assert result["pacing"] == ""

    def test_not_dict_returns_full_defaults(self, parser):
        result = parser._parse_narrative("invalid")
        assert result == {
            "themes": [], "tone": "neutral", "style": "",
            "point_of_view": "", "tension_level": "low",
            "pacing": "", "literary_devices": [],
        }


# ---------------------------------------------------------------------------
# _parse_sentiment
# ---------------------------------------------------------------------------

class TestParseSentiment:
    def test_valid_sentiment(self, parser):
        sent = {"overall": "positive", "polarity": 0.8, "nuances": ["hopeful"], "emotional_arc": "rising"}
        result = parser._parse_sentiment(sent)
        assert result["overall"] == "positive"
        assert result["polarity"] == 0.8
        assert result["nuances"] == ["hopeful"]
        assert result["emotional_arc"] == "rising"

    def test_polarity_clamped_above_1(self, parser):
        result = parser._parse_sentiment({"polarity": 999})
        assert result["polarity"] == 1.0

    def test_polarity_clamped_below_minus_1(self, parser):
        result = parser._parse_sentiment({"polarity": -999})
        assert result["polarity"] == -1.0

    def test_polarity_at_boundaries_kept(self, parser):
        assert parser._parse_sentiment({"polarity": 1.0})["polarity"] == 1.0
        assert parser._parse_sentiment({"polarity": -1.0})["polarity"] == -1.0

    def test_polarity_string_defaults_to_zero(self, parser):
        result = parser._parse_sentiment({"polarity": "not-a-number"})
        assert result["polarity"] == 0.0

    def test_polarity_none_defaults_to_zero(self, parser):
        result = parser._parse_sentiment({"polarity": None})
        assert result["polarity"] == 0.0

    def test_not_dict_returns_full_defaults(self, parser):
        result = parser._parse_sentiment("positive")
        assert result == {"overall": "neutral", "polarity": 0.0, "nuances": [], "emotional_arc": ""}

    def test_nuances_not_list_returns_empty(self, parser):
        result = parser._parse_sentiment({"nuances": "hopeful"})
        assert result["nuances"] == []

    def test_missing_fields_use_defaults(self, parser):
        result = parser._parse_sentiment({})
        assert result["overall"] == "neutral"
        assert result["polarity"] == 0.0
        assert result["nuances"] == []
        assert result["emotional_arc"] == ""


# ---------------------------------------------------------------------------
# _parse_summary
# ---------------------------------------------------------------------------

class TestParseSummary:
    def test_valid_summary(self, parser):
        summ = {"summary": "A great story.", "key_points": ["hero wins", "villain falls"]}
        result = parser._parse_summary(summ)
        assert result["summary"] == "A great story."
        assert result["key_points"] == ["hero wins", "villain falls"]

    def test_missing_fields_use_defaults(self, parser):
        result = parser._parse_summary({})
        assert result["summary"] == ""
        assert result["key_points"] == []

    def test_not_dict_returns_defaults(self, parser):
        result = parser._parse_summary(None)
        assert result == {"summary": "", "key_points": []}

    def test_key_points_not_list_returns_empty(self, parser):
        result = parser._parse_summary({"key_points": "just one point"})
        assert result["key_points"] == []

    def test_falsy_key_points_filtered_out(self, parser):
        result = parser._parse_summary({"key_points": ["valid", "", None, "also valid"]})
        assert result["key_points"] == ["valid", "also valid"]


# ---------------------------------------------------------------------------
# _ensure_str_list
# ---------------------------------------------------------------------------

class TestEnsureStrList:
    def test_list_of_strings_unchanged(self, parser):
        assert parser._ensure_str_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_integers_converted_to_strings(self, parser):
        assert parser._ensure_str_list([1, 2, 3]) == ["1", "2", "3"]

    def test_falsy_values_filtered_out(self, parser):
        assert parser._ensure_str_list([None, "", "keep"]) == ["keep"]

    def test_empty_list_returns_empty(self, parser):
        assert parser._ensure_str_list([]) == []

    def test_not_a_list_returns_empty(self, parser):
        assert parser._ensure_str_list("string") == []
        assert parser._ensure_str_list(None) == []
        assert parser._ensure_str_list(42) == []
        assert parser._ensure_str_list({}) == []
