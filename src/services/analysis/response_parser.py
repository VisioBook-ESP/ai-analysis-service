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

        if "audio_theme" in raw:
            result["audio_theme"] = self._parse_audio_theme(raw["audio_theme"])

        return result

    def _parse_characters(self, chars: Any) -> List[Dict]:
        if not isinstance(chars, list):
            return []
        return [
            {
                "name": str(c.get("name", "Unknown")),
                "role": str(c.get("role", "secondary")),
                "physical_description": str(c.get("physical_description", "")),
                "personality_traits": self._ensure_str_list(
                    c.get("personality_traits", [])
                ),
                "emotions": self._ensure_str_list(c.get("emotions", [])),
                "motivations": self._ensure_str_list(c.get("motivations", [])),
                "actions": self._ensure_str_list(c.get("actions", [])),
                "relationships": self._parse_relationships(c.get("relationships", [])),
                "voice_description": (
                    str(c.get("voice_description", ""))
                    if c.get("voice_description")
                    else None
                ),
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
        result = []
        for i, s in enumerate(scenes):
            if not isinstance(s, dict):
                continue
            scene: Dict[str, Any] = {
                "scene_id": str(s.get("scene_id", f"scene_{i + 1:03d}")),
                "title": str(s.get("title", "")),
                "text_excerpt": str(s.get("text_excerpt", "")),
                "characters_present": self._ensure_str_list(
                    s.get("characters_present", [])
                ),
                "setting": self._parse_setting(s.get("setting", {})),
                "atmosphere": self._parse_atmosphere(s.get("atmosphere", {})),
                "key_events": self._ensure_str_list(s.get("key_events", [])),
                "objects": self._ensure_str_list(s.get("objects", [])),
                "scene_type": str(s["scene_type"]) if s.get("scene_type") else None,
                "audio_cues": self._parse_scene_audio_cues(s.get("audio_cues", [])),
                "narration_text": str(s.get("narration_text", "")) or "",
                "dialogues": self._parse_scene_dialogues(s.get("dialogues", [])),
            }
            result.append(scene)
        return result

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
                "themes": [],
                "tone": "neutral",
                "style": "",
                "point_of_view": "",
                "tension_level": "low",
                "pacing": "",
                "literary_devices": [],
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
            return {
                "overall": "neutral",
                "polarity": 0.0,
                "nuances": [],
                "emotional_arc": "",
            }
        polarity = sent.get("polarity", 0.0)
        try:
            polarity = max(-1.0, min(1.0, float(polarity)))
        except (ValueError, TypeError):
            polarity = 0.0
        return {
            "overall": str(sent.get("overall", "neutral")),
            "polarity": polarity,
            "nuances": (
                [str(n) for n in sent.get("nuances", []) if n]
                if isinstance(sent.get("nuances"), list)
                else []
            ),
            "emotional_arc": str(sent.get("emotional_arc", "")),
        }

    @staticmethod
    def _parse_summary(summ: Any) -> Dict:
        if not isinstance(summ, dict):
            return {"summary": "", "key_points": []}
        return {
            "summary": str(summ.get("summary", "")),
            "key_points": (
                [str(k) for k in summ.get("key_points", []) if k]
                if isinstance(summ.get("key_points"), list)
                else []
            ),
        }

    @staticmethod
    def _parse_scene_audio_cues(cues: Any) -> List[Dict]:
        """Parse audio cues embedded in a scene."""
        if not isinstance(cues, list):
            return []
        return [
            {
                "type": str(c.get("type", "ambient")),
                "description": str(c.get("description", "")),
            }
            for c in cues
            if isinstance(c, dict) and c.get("description")
        ]

    @staticmethod
    def _parse_scene_dialogues(dialogues: Any) -> List[Dict]:
        """Parse dialogue lines embedded in a scene."""
        if not isinstance(dialogues, list):
            return []
        return [
            {
                "speaker": str(d.get("speaker", "")),
                "line": str(d.get("line", "")),
                "delivery": str(d.get("delivery", "neutral")),
            }
            for d in dialogues
            if isinstance(d, dict) and d.get("line")
        ]

    @staticmethod
    def _parse_dialogues(dialogues: Any) -> List[Dict]:
        """Parse Pass 1 dialogue extraction output."""
        if not isinstance(dialogues, list):
            return []
        result = []
        for d in dialogues:
            if not isinstance(d, dict):
                continue
            line = str(d.get("line", "")).strip()
            if not line:
                continue
            result.append(
                {
                    "order": int(d.get("order", len(result))),
                    "speaker": str(d.get("speaker", "")).strip(),
                    "line": line,
                    "delivery": str(d.get("delivery", "neutral")).strip() or "neutral",
                    "context": str(d.get("context", "")).strip() or None,
                }
            )
        return result

    @staticmethod
    def _parse_narrative_blocks(blocks: Any) -> List[Dict]:
        """Parse Pass 1 narrative block extraction output."""
        _VALID_TYPES = {"description", "action", "inner_monologue", "transition"}
        if not isinstance(blocks, list):
            return []
        result = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            text = str(b.get("text", "")).strip()
            if not text:
                continue
            block_type = str(b.get("type", "description")).strip()
            if block_type not in _VALID_TYPES:
                block_type = "description"
            result.append(
                {
                    "order": int(b.get("order", len(result))),
                    "type": block_type,
                    "text": text,
                    "characters_mentioned": (
                        [str(c) for c in b["characters_mentioned"] if c]
                        if isinstance(b.get("characters_mentioned"), list)
                        else []
                    ),
                }
            )
        return result

    @staticmethod
    def _parse_audio_cues(cues: Any) -> List[Dict]:
        """Parse Pass 1 audio cue extraction output."""
        _VALID_TYPES = {"ambient", "sfx", "music_mood"}
        if not isinstance(cues, list):
            return []
        result = []
        for c in cues:
            if not isinstance(c, dict):
                continue
            desc = str(c.get("description", "")).strip()
            if not desc:
                continue
            cue_type = str(c.get("type", "ambient")).strip()
            if cue_type not in _VALID_TYPES:
                cue_type = "ambient"
            result.append(
                {
                    "type": cue_type,
                    "description": desc,
                    "source_block_order": (
                        int(c["source_block_order"])
                        if c.get("source_block_order") is not None
                        else None
                    ),
                }
            )
        return result

    @staticmethod
    def _parse_audio_theme(theme: Any) -> Dict:
        """Parse the reduce-phase audio_theme output."""
        if not isinstance(theme, dict):
            return {"overall_mood": "", "recurring_sounds": []}
        return {
            "overall_mood": str(theme.get("overall_mood", "")),
            "recurring_sounds": (
                [str(s) for s in theme["recurring_sounds"] if s]
                if isinstance(theme.get("recurring_sounds"), list)
                else []
            ),
        }

    @staticmethod
    def _ensure_str_list(val: Any) -> List[str]:
        if not isinstance(val, list):
            return []
        return [str(v) for v in val if v]
