import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.config.settings import get_settings
from src.services.preprocessing import TextPreprocessor
from src.services.preprocessing.chapter_detector import Chapter, detect_chapters

from .chapter_prompts import (
    MAP_SYSTEM_PROMPT,
    PASS1_SYSTEM_PROMPT,
    PASS2_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    build_map_prompt,
    build_pass1_prompt,
    build_pass2_prompt,
    build_reduce_prompt,
)
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, build_analysis_prompt
from .prompt_generator import PromptGenerator
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]

# Per-chapter map output budget (tokens) — legacy single-pass fallback
_MAP_MAX_TOKENS = 4096
# Pass 1 (dialogue/structure extraction) output budget
_PASS1_MAX_TOKENS = 4096
# Pass 2 (visual scene composition) output budget
_PASS2_MAX_TOKENS = 4096
# Global reduce output budget (tokens)
_REDUCE_MAX_TOKENS = 8192


class AnalysisOptions:
    def __init__(
        self,
        characters: bool = True,
        scenes: bool = True,
        narrative: bool = True,
        summary: bool = True,
        mask_pii: bool = True,
        remove_links: bool = False,
        max_summary_length: int = 200,
    ):
        self.characters = characters
        self.scenes = scenes
        self.narrative = narrative
        self.summary = summary
        self.mask_pii = mask_pii
        self.remove_links = remove_links
        self.max_summary_length = max_summary_length

    def to_dict(self) -> Dict[str, bool]:
        return {
            "characters": self.characters,
            "scenes": self.scenes,
            "narrative": self.narrative,
            "summary": self.summary,
        }


class Analyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.llm_client = LLMClient()
        self.parser = ResponseParser()
        self.prompt_generator = PromptGenerator(self.llm_client)

    async def analyze(
        self,
        text: str,
        language: str = "auto",
        options: Optional[AnalysisOptions] = None,
        on_step: Optional[StepCallback] = None,
        generate_prompts: bool = False,
        visual_style: str = "realistic",
    ) -> Dict[str, Any]:
        if options is None:
            options = AnalysisOptions()

        settings = get_settings()
        start_time = time.time()

        # Step 1: Preprocessing
        if on_step:
            await on_step("preprocessing")
        preprocessed = self.preprocessor.preprocess(
            text,
            language=language,
            mask_pii=options.mask_pii,
            remove_links=options.remove_links,
        )

        detected_language = preprocessed["language"]
        cleaned_text = preprocessed["text"]

        # Step 2: Text stats from preprocessing
        text_stats = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "sentence_count": len(preprocessed["sentences"]),
            "word_count": len(cleaned_text.split()),
            "quality_score": preprocessed["quality"]["score"],
            "quality_assessment": preprocessed["quality"].get("assessment", "unknown"),
        }

        # Step 3: Chapter detection
        if on_step:
            await on_step("chapter_detection")
        chapters = await detect_chapters(
            cleaned_text, detected_language, self.llm_client
        )

        # Step 4: LLM analysis (single-shot or chunked)
        options_dict = options.to_dict()

        if len(chapters) <= 1:
            if on_step:
                await on_step("llm_call")
            raw_response = await self._single_shot_analyze(
                cleaned_text, detected_language, options_dict
            )
        else:
            raw_response = await self._chunked_analyze(
                chapters, detected_language, options_dict, on_step
            )

        # Step 5: Parse LLM output
        if on_step:
            await on_step("parsing")
        parsed = self.parser.parse(raw_response, options_dict)

        # Step 6: Enrich summary with text stats
        if "summary" in parsed:
            parsed["summary"]["original_length"] = len(text)
            parsed["summary"]["summary_length"] = len(
                parsed["summary"].get("summary", "")
            )

        # Step 7: Generate image prompts (Phase 2 — unchanged)
        image_prompts = None
        if generate_prompts and settings.prompt_gen_enabled:
            if on_step:
                await on_step("prompt_generation")
            try:
                image_prompts = await self.prompt_generator.generate(
                    analysis_result=parsed,
                    visual_style=visual_style,
                    language=detected_language,
                )
                logger.info(
                    "Prompt generation completed: %d scenes, %d characters, %d locations",
                    len(image_prompts.get("scene_prompts", [])),
                    len(image_prompts.get("character_prompts", [])),
                    len(image_prompts.get("location_prompts", [])),
                )
            except Exception as e:
                logger.error(f"Prompt generation failed (non-fatal): {e}")

        # Step 8: Assemble result
        result = {
            "language": detected_language,
            "text_stats": text_stats,
            **parsed,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        if image_prompts is not None:
            result["image_prompts"] = image_prompts
        return result

    async def _single_shot_analyze(
        self,
        text: str,
        language: str,
        options: Dict[str, bool],
    ) -> Dict[str, Any]:
        """Original single-shot analysis for short texts."""
        user_prompt = build_analysis_prompt(text, language, options)
        try:
            return await self.llm_client.chat_completion(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM analysis failed: {e}")

    async def _chunked_analyze(
        self,
        chapters: List[Chapter],
        language: str,
        options: Dict[str, bool],
        on_step: Optional[StepCallback],
    ) -> Dict[str, Any]:
        """Two-pass map-reduce analysis over detected chapters.

        Per chapter:
          Pass 1: Extract dialogues, narrative blocks, audio cues.
          Pass 2: Compose visual scenes from Pass 1 output.
          Fallback: Pass 1 fails → legacy single-pass MAP.
                    Pass 2 fails → mechanical scene split from Pass 1 data.
        """
        total = len(chapters)
        logger.info("Starting two-pass chunked analysis: %d chapters", total)

        # ── Map phase: per-chapter two-pass extraction ──
        chapter_results: List[Dict[str, Any]] = []
        all_scenes: List[Dict[str, Any]] = []
        global_scene_index = 0

        for i, chapter in enumerate(chapters):
            if on_step:
                await on_step(f"chapter_{i + 1}_of_{total}")

            pass1_result = None
            scenes_from_chapter: List[Dict[str, Any]] = []

            # ── Pass 1: dialogue & structure extraction ──
            try:
                pass1_result = await self.llm_client.chat_completion(
                    system_prompt=PASS1_SYSTEM_PROMPT,
                    user_prompt=build_pass1_prompt(
                        chapter.text,
                        chapter.title,
                        chapter.index,
                        total,
                        language,
                    ),
                    max_tokens=_PASS1_MAX_TOKENS,
                )
                logger.info(
                    "Chapter %d/%d Pass 1: %d dialogues, %d narrative blocks, %d audio cues",
                    i + 1,
                    total,
                    len(pass1_result.get("dialogues", [])),
                    len(pass1_result.get("narrative_blocks", [])),
                    len(pass1_result.get("audio_cues", [])),
                )
            except Exception as e:
                logger.error(
                    "Pass 1 failed for chapter %d/%d (%s): %s — falling back to legacy MAP",
                    i + 1,
                    total,
                    chapter.title,
                    e,
                )

            if pass1_result is not None:
                # ── Pass 2: visual scene composition ──
                try:
                    pass2_result = await self.llm_client.chat_completion(
                        system_prompt=PASS2_SYSTEM_PROMPT,
                        user_prompt=build_pass2_prompt(
                            pass1_result,
                            chapter.title,
                            chapter.index,
                            total,
                            language,
                        ),
                        max_tokens=_PASS2_MAX_TOKENS,
                    )
                    scenes_from_chapter = pass2_result.get("scenes", [])
                    logger.info(
                        "Chapter %d/%d Pass 2: %d visual scenes",
                        i + 1,
                        total,
                        len(scenes_from_chapter),
                    )
                except Exception as e:
                    logger.error(
                        "Pass 2 failed for chapter %d/%d (%s): %s — using mechanical split",
                        i + 1,
                        total,
                        chapter.title,
                        e,
                    )
                    scenes_from_chapter = self._mechanical_scene_split(pass1_result)

                # Build chapter result for reduce (from Pass 1 data)
                chapter_results.append(
                    {
                        "characters": [
                            {"name": c, "role": "mentioned"}
                            for c in pass1_result.get("characters_in_chapter", [])
                        ],
                        "characters_in_chapter": pass1_result.get(
                            "characters_in_chapter", []
                        ),
                        "key_events": [
                            nb.get("text", "")
                            for nb in pass1_result.get("narrative_blocks", [])
                            if nb.get("type") == "action"
                        ],
                        "sentiment_hint": pass1_result.get("sentiment_hint", "neutral"),
                    }
                )
            else:
                # ── Fallback: legacy single-pass MAP ──
                try:
                    result = await self.llm_client.chat_completion(
                        system_prompt=MAP_SYSTEM_PROMPT,
                        user_prompt=build_map_prompt(
                            chapter.text,
                            chapter.title,
                            chapter.index,
                            total,
                            language,
                            options,
                        ),
                        max_tokens=_MAP_MAX_TOKENS,
                    )
                    chapter_results.append(result)
                    scenes_from_chapter = result.get("scenes", [])
                except Exception as e:
                    logger.error(
                        "Legacy MAP also failed for chapter %d/%d: %s",
                        i + 1,
                        total,
                        e,
                    )
                    chapter_results.append(
                        {
                            "characters": [],
                            "key_events": [],
                            "sentiment_hint": "neutral",
                        }
                    )
                    continue

            # Collect and renumber scenes globally
            for scene in scenes_from_chapter:
                renumbered = {
                    **scene,
                    "scene_id": f"scene_{global_scene_index + 1:03d}",
                }
                global_scene_index += 1
                all_scenes.append(renumbered)

            logger.info(
                "Chapter %d/%d (%s): %d scenes collected",
                i + 1,
                total,
                chapter.title,
                len(scenes_from_chapter),
            )

        # ── Reduce phase: global synthesis ──
        if on_step:
            await on_step("global_synthesis")

        try:
            reduced = await self.llm_client.chat_completion(
                system_prompt=REDUCE_SYSTEM_PROMPT,
                user_prompt=build_reduce_prompt(chapter_results, language, options),
                max_tokens=_REDUCE_MAX_TOKENS,
            )
        except Exception as e:
            logger.error(f"Reduce call failed: {e}")
            raise RuntimeError(f"LLM analysis failed during global synthesis: {e}")

        # ── Merge: scenes from map + global fields from reduce ──
        merged: Dict[str, Any] = {}
        if options.get("scenes"):
            merged["scenes"] = all_scenes
        if options.get("characters"):
            merged["characters"] = reduced.get("characters", [])
        if options.get("narrative"):
            merged["narrative"] = reduced.get("narrative", {})
        merged["sentiment"] = reduced.get("sentiment", {})
        if options.get("summary"):
            merged["summary"] = reduced.get("summary", {})
        if reduced.get("audio_theme"):
            merged["audio_theme"] = reduced["audio_theme"]

        logger.info(
            "Chunked analysis complete: %d total scenes, %d characters",
            len(all_scenes),
            len(merged.get("characters", [])),
        )

        return merged

    @staticmethod
    def _mechanical_scene_split(
        pass1_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build scenes mechanically from Pass 1 data when Pass 2 fails.

        Groups 2-3 dialogues into one dialogue scene, each narrative block
        into one scene. No LLM involved.
        """
        scenes: List[Dict[str, Any]] = []
        dialogues = pass1_result.get("dialogues", [])
        narrative_blocks = pass1_result.get("narrative_blocks", [])
        audio_cues = pass1_result.get("audio_cues", [])

        # Index audio cues by source_block_order
        cue_by_order: Dict[int, List[Dict]] = {}
        for cue in audio_cues:
            order = cue.get("source_block_order")
            if order is not None:
                cue_by_order.setdefault(order, []).append(
                    {
                        "type": cue.get("type", "ambient"),
                        "description": cue.get("description", ""),
                    }
                )

        # Narrative blocks → one scene each
        for nb in narrative_blocks:
            nb_order = nb.get("order", 0)
            scene_type = "description"
            if nb.get("type") == "action":
                scene_type = "action"
            elif nb.get("type") == "transition":
                scene_type = "establishing"
            scenes.append(
                {
                    "scene_type": scene_type,
                    "title": f"{scene_type.capitalize()} block",
                    "text_excerpt": nb.get("text", ""),
                    "characters_present": nb.get("characters_mentioned", []),
                    "setting": {},
                    "atmosphere": {},
                    "key_events": [],
                    "objects": [],
                    "audio_cues": cue_by_order.get(nb_order, []),
                    "narration_text": nb.get("text", ""),
                    "dialogues": [],
                    "source_block_orders": [nb_order],
                }
            )

        # Dialogues → group 2-3 into one dialogue scene
        for batch_start in range(0, len(dialogues), 3):
            batch = dialogues[batch_start : batch_start + 3]
            speakers = list({d.get("speaker", "") for d in batch})
            orders = [d.get("order", 0) for d in batch]
            batch_cues: List[Dict] = []
            for o in orders:
                batch_cues.extend(cue_by_order.get(o, []))
            scenes.append(
                {
                    "scene_type": "dialogue",
                    "title": f"Dialogue: {', '.join(speakers[:2])}",
                    "text_excerpt": batch[0].get("line", ""),
                    "characters_present": speakers,
                    "setting": {},
                    "atmosphere": {},
                    "key_events": [],
                    "objects": [],
                    "audio_cues": batch_cues,
                    "narration_text": "",
                    "dialogues": [
                        {
                            "speaker": d.get("speaker", ""),
                            "line": d.get("line", ""),
                            "delivery": d.get("delivery", "neutral"),
                        }
                        for d in batch
                    ],
                    "source_block_orders": orders,
                }
            )

        # Sort by first source_block_order to maintain chapter sequence
        scenes.sort(
            key=lambda s: (
                s.get("source_block_orders", [0])[0]
                if s.get("source_block_orders")
                else 0
            )
        )

        return scenes

    async def close(self):
        await self.llm_client.close()
