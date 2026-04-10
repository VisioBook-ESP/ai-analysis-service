"""Generate Flux/SDXL-optimized prompts from structured analysis results.

This is the second LLM call in the two-phase pipeline:
  Phase 1: Text analysis (low temperature, accuracy-focused)
  Phase 2: Prompt generation (higher temperature, creative visual descriptions)
"""

import logging
from typing import Any, Dict, List

from src.config.settings import get_settings

from .image_prompts import IMAGE_PROMPT_SYSTEM_PROMPT, build_image_prompt_request
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Max scenes per LLM call to avoid token overflow
_BATCH_SIZE = 15


class PromptGenerator:
    """Generate Flux/SDXL image prompts from analysis results via a second LLM call."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        settings = get_settings()
        self.temperature = settings.prompt_gen_temperature
        self.max_tokens = settings.prompt_gen_max_tokens

    async def generate(
        self,
        analysis_result: dict,
        visual_style: str = "realistic",
        language: str = "auto",
    ) -> dict:
        """Generate image prompts from analysis results.

        Returns dict with scene_prompts, character_prompts, location_prompts.
        """
        scenes = analysis_result.get("scenes", [])

        # If few scenes, single call
        if len(scenes) <= _BATCH_SIZE:
            return await self._single_call(analysis_result, visual_style, language)

        # Batch for many scenes — split scenes but keep full character context
        return await self._batched_call(analysis_result, visual_style, language)

    async def _single_call(
        self, analysis_result: dict, visual_style: str, language: str
    ) -> dict:
        user_prompt = build_image_prompt_request(
            analysis_result, visual_style, language
        )
        raw = await self.llm_client.chat_completion(
            system_prompt=IMAGE_PROMPT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse_response(raw)

    async def _batched_call(
        self, analysis_result: dict, visual_style: str, language: str
    ) -> dict:
        scenes = analysis_result.get("scenes", [])
        all_scene_prompts: List[Dict] = []
        all_character_prompts: List[Dict] = []
        all_location_prompts: List[Dict] = []
        all_audio_prompts: List[Dict] = []
        seen_characters: set = set()
        seen_locations: set = set()

        for start in range(0, len(scenes), _BATCH_SIZE):
            batch_scenes = scenes[start : start + _BATCH_SIZE]
            batch_result = {
                **analysis_result,
                "scenes": batch_scenes,
            }
            user_prompt = build_image_prompt_request(
                batch_result, visual_style, language
            )
            raw = await self.llm_client.chat_completion(
                system_prompt=IMAGE_PROMPT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parsed = self._parse_response(raw)

            # Offset scene_orders from batch-local to global indices
            for sp in parsed.get("scene_prompts", []):
                sp["scene_order"] = sp.get("scene_order", 0) + start

            # Offset source_scene_orders in location_prompts
            for lp in parsed.get("location_prompts", []):
                lp["source_scene_orders"] = [
                    s + start for s in lp.get("source_scene_orders", [])
                ]

            # Offset audio_prompts scene_order
            for ap in parsed.get("audio_prompts", []):
                ap["scene_order"] = ap.get("scene_order", 0) + start

            all_scene_prompts.extend(parsed.get("scene_prompts", []))
            all_audio_prompts.extend(parsed.get("audio_prompts", []))

            # Deduplicate characters and locations across batches
            for cp in parsed.get("character_prompts", []):
                name = cp.get("name", "")
                if name and name not in seen_characters:
                    seen_characters.add(name)
                    all_character_prompts.append(cp)

            for lp in parsed.get("location_prompts", []):
                loc_id = lp.get("location_id", "")
                if loc_id and loc_id not in seen_locations:
                    seen_locations.add(loc_id)
                    all_location_prompts.append(lp)

        return {
            "scene_prompts": all_scene_prompts,
            "character_prompts": all_character_prompts,
            "location_prompts": all_location_prompts,
            "audio_prompts": all_audio_prompts,
        }

    def _parse_response(self, raw: Dict[str, Any]) -> dict:
        """Validate and normalize the LLM response for image prompts."""
        return {
            "scene_prompts": self._parse_scene_prompts(raw.get("scene_prompts", [])),
            "character_prompts": self._parse_character_prompts(
                raw.get("character_prompts", [])
            ),
            "location_prompts": self._parse_location_prompts(
                raw.get("location_prompts", [])
            ),
            "audio_prompts": self._parse_audio_prompts(raw.get("audio_prompts", [])),
        }

    @staticmethod
    def _parse_scene_prompts(prompts: Any) -> List[Dict]:
        if not isinstance(prompts, list):
            return []
        result = []
        for p in prompts:
            if not isinstance(p, dict):
                continue
            image_prompt = str(p.get("image_prompt", "")).strip()
            if not image_prompt:
                continue
            result.append(
                {
                    "scene_order": int(p.get("scene_order", len(result))),
                    "image_prompt": image_prompt,
                    "negative_prompt": str(p.get("negative_prompt", "")).strip(),
                    "characters_present": (
                        [str(c) for c in p["characters_present"] if c]
                        if isinstance(p.get("characters_present"), list)
                        else []
                    ),
                    "location_id": str(p.get("location_id", "")).strip() or None,
                }
            )
        return result

    @staticmethod
    def _parse_character_prompts(prompts: Any) -> List[Dict]:
        if not isinstance(prompts, list):
            return []
        result = []
        for p in prompts:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "")).strip()
            portrait_prompt = str(p.get("portrait_prompt", "")).strip()
            if not name or not portrait_prompt:
                continue
            result.append(
                {
                    "name": name,
                    "physical_description": str(
                        p.get("physical_description", "")
                    ).strip(),
                    "portrait_prompt": portrait_prompt,
                    "portrait_negative_prompt": str(
                        p.get("portrait_negative_prompt", "")
                    ).strip(),
                }
            )
        return result

    @staticmethod
    def _parse_location_prompts(prompts: Any) -> List[Dict]:
        if not isinstance(prompts, list):
            return []
        result = []
        for p in prompts:
            if not isinstance(p, dict):
                continue
            location_id = str(p.get("location_id", "")).strip()
            description_prompt = str(p.get("description_prompt", "")).strip()
            if not location_id or not description_prompt:
                continue
            result.append(
                {
                    "location_id": location_id,
                    "name": str(p.get("name", "")).strip(),
                    "description_prompt": description_prompt,
                    "negative_prompt": str(p.get("negative_prompt", "")).strip(),
                    "source_scene_orders": (
                        [int(s) for s in p["source_scene_orders"] if s is not None]
                        if isinstance(p.get("source_scene_orders"), list)
                        else []
                    ),
                }
            )
        return result

    @staticmethod
    def _parse_audio_prompts(prompts: Any) -> List[Dict]:
        if not isinstance(prompts, list):
            return []
        result = []
        for p in prompts:
            if not isinstance(p, dict):
                continue
            ambient = str(p.get("ambient_description", "")).strip()
            if not ambient:
                continue
            result.append(
                {
                    "scene_order": int(p.get("scene_order", len(result))),
                    "ambient_description": ambient,
                    "sfx": (
                        [str(s) for s in p["sfx"] if s]
                        if isinstance(p.get("sfx"), list)
                        else []
                    ),
                    "music_mood": str(p.get("music_mood", "")).strip(),
                }
            )
        return result
