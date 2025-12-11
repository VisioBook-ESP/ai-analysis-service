from typing import Dict, Any, Optional
import time

from src.services.preprocessing import TextPreprocessor
from .semantic import SemanticAnalyzer
from .scenes import SceneExtractor
from .summarization import Summarizer


class AnalysisOptions:
    def __init__(
        self,
        semantic: bool = True,
        scenes: bool = True,
        summarize: bool = True,
        mask_pii: bool = True,
        remove_links: bool = False,
        return_embeddings: bool = False,
        max_summary_length: int = 150,
    ):
        self.semantic = semantic
        self.scenes = scenes
        self.summarize = summarize
        self.mask_pii = mask_pii
        self.remove_links = remove_links
        self.return_embeddings = return_embeddings
        self.max_summary_length = max_summary_length


class Analyzer:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.scene_extractor = SceneExtractor()
        self.summarizer = Summarizer()

    def analyze(
        self,
        text: str,
        language: str = "auto",
        options: Optional[AnalysisOptions] = None,
    ) -> Dict[str, Any]:
        if options is None:
            options = AnalysisOptions()

        start_time = time.time()

        preprocessed = self.preprocessor.preprocess(
            text,
            language=language,
            mask_pii=options.mask_pii,
            remove_links=options.remove_links,
        )

        results = {
            "language": preprocessed["language"],
            "text_stats": {
                "original_length": len(text),
                "cleaned_length": len(preprocessed["text"]),
                "sentence_count": len(preprocessed["sentences"]),
                "word_count": len(preprocessed["text"].split()),
                "quality_score": preprocessed["quality"]["score"],
                "quality_assessment": preprocessed["quality"].get("assessment", "unknown"),
            },
        }

        semantic_result = None
        if options.semantic:
            semantic_result = self.semantic_analyzer.analyze(
                preprocessed, return_embeddings=options.return_embeddings
            )
            results["semantic"] = semantic_result

        if options.scenes:
            scene_result = self.scene_extractor.extract(preprocessed, semantic_data=semantic_result)
            results["scenes"] = scene_result

        if options.summarize:
            summary_result = self.summarizer.summarize(
                preprocessed, max_length=options.max_summary_length
            )
            results["summary"] = summary_result

        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return results
