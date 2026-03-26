import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from src.services.preprocessing import TextPreprocessor

from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, build_analysis_prompt
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]


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

    async def analyze(
        self,
        text: str,
        language: str = "auto",
        options: Optional[AnalysisOptions] = None,
        on_step: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        if options is None:
            options = AnalysisOptions()

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

        # Step 3: Call LLM
        if on_step:
            await on_step("llm_call")
        options_dict = options.to_dict()
        user_prompt = build_analysis_prompt(
            cleaned_text, detected_language, options_dict
        )

        try:
            raw_response = await self.llm_client.chat_completion(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM analysis failed: {e}")

        # Step 4: Parse LLM output
        if on_step:
            await on_step("parsing")
        parsed = self.parser.parse(raw_response, options_dict)

        # Step 5: Enrich summary with text stats
        if "summary" in parsed:
            parsed["summary"]["original_length"] = len(text)
            parsed["summary"]["summary_length"] = len(
                parsed["summary"].get("summary", "")
            )

        # Step 6: Assemble result
        return {
            "language": detected_language,
            "text_stats": text_stats,
            **parsed,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }

    async def close(self):
        await self.llm_client.close()
