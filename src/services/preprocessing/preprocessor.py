import time
from typing import Any

from .language_detector import detect_language
from .quality_scorer import assess_quality, noise_score
from .segmenter import build_chunks, split_sentences
from .text_cleaner import basic_clean


class TextPreprocessor:

    def __init__(self):
        pass

    def preprocess(
        self,
        text: str,
        *,
        language: str | None = None,
        remove_links: bool = False,
        mask_pii: bool = True,
        remove_emoji: bool = False,
        lowercase: bool = False,
        max_tokens: int = 512,
        overlap: int = 64,
    ) -> dict[str, Any]:
        start_time = time.time()

        if language is None or language == "auto":
            detected_lang = detect_language(text, default="auto")
        else:
            detected_lang = language

        clean_text, masks = basic_clean(
            text,
            remove_links=remove_links,
            do_mask_pii=mask_pii,
            remove_emoji=remove_emoji,
            lowercase=lowercase,
        )

        sentences = split_sentences(clean_text, lang=detected_lang)
        chunks = build_chunks(
            clean_text, lang=detected_lang, max_tokens=max_tokens, overlap=overlap
        )

        quality = noise_score(clean_text)
        quality["assessment"] = assess_quality(clean_text)

        processing_time_ms = (time.time() - start_time) * 1000

        return {
            "language": detected_lang,
            "text": clean_text,
            "quality": quality,
            "sentences": sentences,
            "chunks": chunks,
            "masks": masks,
            "stats": {
                "original_length": len(text),
                "cleaned_length": len(clean_text),
                "sentence_count": len(sentences),
                "chunk_count": len(chunks),
            },
            "processing_time_ms": round(processing_time_ms, 2),
        }

    def preprocess_batch(self, texts: list[str], **kwargs) -> list[dict[str, Any]]:
        results = []
        for text in texts:
            try:
                result = self.preprocess(text, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "text": "", "language": "unknown"})
        return results
