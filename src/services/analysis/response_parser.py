import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse and validate LLM JSON output into normalized dictionaries."""

    def parse(self, raw: Dict[str, Any], options: Dict[str, bool]) -> Dict[str, Any]:
        result = {}

        if options.get("characters") and "characters" in raw:
            result["characters"] = self._parse_characters(raw["characters"])

        if options.get("scenes") and "scenes" in raw:
            result["scenes"] = self._parse_scenes(raw["scenes"])

        if options.get("narrative") and "narrative" in raw:
            result["narrative"] = self._parse_narrative(raw["narrative"])

        if "sentiment" in raw:
            result["sentiment"] = self._parse_sentiment(raw["sentiment"])

        if options.get("summary") and "summary" in raw:
            result["summary"] = self._parse_summary(raw["summary"])

        return result

    def _parse_characters(self, chars: Any) -> List[Dict]:
        if not isinstance(chars, list):
            return []
        return [
            {
                "name": str(c.get("name", "Unknown")),
                "role": str(c.get("role", "secondary")),
                "physical_description": str(c.get("physical_description", "")),
                "personality_traits": self._ensure_str_list(c.get("personality_traits", [])),
                "emotions": self._ensure_str_list(c.get("emotions", [])),
                "motivations": self._ensure_str_list(c.get("motivations", [])),
                "actions": self._ensure_str_list(c.get("actions", [])),
                "relationships": self._parse_relationships(c.get("relationships", [])),
            }
            for c in chars
            if isinstance(c, dict)
        ]

    def _parse_relationships(self, rels: Any) -> List[Dict]:
        if not isinstance(rels, list):
            return []
        return [
            {
                "target": str(r.get("target", "")),
                "type": str(r.get("type", "")),
                "description": str(r.get("description", "")),
            }
            for r in rels
            if isinstance(r, dict)
        ]

    def _parse_scenes(self, scenes: Any) -> List[Dict]:
        if not isinstance(scenes, list):
            return []
        return [
            {
                "scene_id": str(s.get("scene_id", f"scene_{i + 1:03d}")),
                "title": str(s.get("title", "")),
                "text_excerpt": str(s.get("text_excerpt", "")),
                "characters_present": self._ensure_str_list(s.get("characters_present", [])),
                "setting": self._parse_setting(s.get("setting", {})),
                "atmosphere": self._parse_atmosphere(s.get("atmosphere", {})),
                "key_events": self._ensure_str_list(s.get("key_events", [])),
                "objects": self._ensure_str_list(s.get("objects", [])),
            }
            for i, s in enumerate(scenes)
            if isinstance(s, dict)
        ]

    @staticmethod
    def _parse_setting(setting: Any) -> Dict:
        if not isinstance(setting, dict):
            setting = {}
        return {
            "location": str(setting.get("location", "unspecified")),
            "time_period": str(setting.get("time_period", "unspecified")),
            "time_of_day": str(setting.get("time_of_day", "unspecified")),
        }

    def _parse_atmosphere(self, atm: Any) -> Dict:
        if not isinstance(atm, dict):
            atm = {}
        st = atm.get("sounds_textures", {})
        if not isinstance(st, dict):
            st = {}
        return {
            "mood": str(atm.get("mood", "neutral")),
            "lighting": str(atm.get("lighting", "unspecified")),
            "weather": str(atm.get("weather", "unspecified")),
            "colors": self._ensure_str_list(atm.get("colors", [])),
            "sounds_textures": {
                "sounds": self._ensure_str_list(st.get("sounds", [])),
                "textures": self._ensure_str_list(st.get("textures", [])),
            },
        }

    def _parse_narrative(self, narr: Any) -> Dict:
        if not isinstance(narr, dict):
            return {
                "themes": [], "tone": "neutral", "style": "",
                "point_of_view": "", "tension_level": "low",
                "pacing": "", "literary_devices": [],
            }
        return {
            "themes": self._ensure_str_list(narr.get("themes", [])),
            "tone": str(narr.get("tone", "neutral")),
            "style": str(narr.get("style", "")),
            "point_of_view": str(narr.get("point_of_view", "")),
            "tension_level": str(narr.get("tension_level", "low")),
            "pacing": str(narr.get("pacing", "")),
            "literary_devices": self._ensure_str_list(narr.get("literary_devices", [])),
        }

    @staticmethod
    def _parse_sentiment(sent: Any) -> Dict:
        if not isinstance(sent, dict):
            return {"overall": "neutral", "polarity": 0.0, "nuances": [], "emotional_arc": ""}
        polarity = sent.get("polarity", 0.0)
        try:
            polarity = max(-1.0, min(1.0, float(polarity)))
        except (ValueError, TypeError):
            polarity = 0.0
        return {
            "overall": str(sent.get("overall", "neutral")),
            "polarity": polarity,
            "nuances": [str(n) for n in sent.get("nuances", []) if n] if isinstance(sent.get("nuances"), list) else [],
            "emotional_arc": str(sent.get("emotional_arc", "")),
        }

    @staticmethod
    def _parse_summary(summ: Any) -> Dict:
        if not isinstance(summ, dict):
            return {"summary": "", "key_points": []}
        return {
            "summary": str(summ.get("summary", "")),
            "key_points": [str(k) for k in summ.get("key_points", []) if k] if isinstance(summ.get("key_points"), list) else [],
        }

    @staticmethod
    def _ensure_str_list(val: Any) -> List[str]:
        if not isinstance(val, list):
            return []
        return [str(v) for v in val if v]
