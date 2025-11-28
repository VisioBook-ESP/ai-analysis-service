from typing import Dict, List, Any
from collections import Counter
import re

try:
    from transformers import pipeline
    import torch
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class Summarizer:

    def __init__(self):
        self.model = None

        if _TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if torch.cuda.is_available() else -1
                self.model = pipeline(
                    "summarization",
                    model="csebuetnlp/mT5_multilingual_XLSum",
                    device=device
                )
            except Exception:
                pass

    def summarize(self, preprocessed: Dict, max_length: int = 150) -> Dict[str, Any]:
        text = preprocessed["text"]
        sentences = preprocessed["sentences"]

        extractive_sentences = self._extractive_summary(sentences, top_n=3)

        abstractive_summary = None
        if self.model and len(text) > 100:
            abstractive_summary = self._abstractive_summary(text, max_length)

        key_points = self._extract_key_points(sentences)

        original_length = len(text)
        summary_length = len(abstractive_summary) if abstractive_summary else len(" ".join(extractive_sentences))
        reduction = ((original_length - summary_length) / original_length) if original_length > 0 else 0

        return {
            "abstractive_summary": abstractive_summary or " ".join(extractive_sentences),
            "extractive_sentences": extractive_sentences,
            "key_points": key_points,
            "original_length": original_length,
            "summary_length": summary_length,
            "length_reduction_percent": round(reduction * 100, 1)
        }

    def _extractive_summary(self, sentences: List[Dict], top_n: int = 3) -> List[str]:
        if len(sentences) <= top_n:
            return [s["text"] for s in sentences]

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence["text"], i, len(sentences))
            scored_sentences.append((sentence["text"], score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in scored_sentences[:top_n]]

    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        score = 0.0

        words = re.findall(r'\b\w+\b', sentence.lower())
        score += len(words) * 0.1

        if position == 0:
            score += 2.0
        elif position < 3:
            score += 1.0

        if position == total - 1:
            score += 1.0

        important_words = ["important", "essentiel", "principal", "crucial", "clé",
                          "important", "essential", "main", "crucial", "key"]
        for word in important_words:
            if word in sentence.lower():
                score += 1.5

        return score

    def _abstractive_summary(self, text: str, max_length: int) -> str:
        if not self.model:
            return None

        try:
            min_length = max(30, max_length // 3)

            if len(text) > 1024:
                text = text[:1024]

            result = self.model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            return result[0]["summary_text"]

        except Exception:
            return None

    def _extract_key_points(self, sentences: List[Dict]) -> List[str]:
        all_text = " ".join([s["text"] for s in sentences])

        key_patterns = [
            r'(il est important de|nous devons|il faut|the key is|we must|it is important)',
            r'(premièrement|deuxièmement|ensuite|enfin|first|second|then|finally)',
            r'(en conclusion|pour conclure|en résumé|in conclusion|to conclude|in summary)'
        ]

        key_points = []
        for pattern in key_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = min(start + 150, len(all_text))
                sentence_end = all_text.find('.', start)
                if sentence_end != -1 and sentence_end < end:
                    end = sentence_end + 1

                point = all_text[start:end].strip()
                if point and point not in key_points:
                    key_points.append(point)

        if not key_points and sentences:
            key_points = [sentences[0]["text"]]
            if len(sentences) > 1:
                key_points.append(sentences[-1]["text"])

        return key_points[:5]
