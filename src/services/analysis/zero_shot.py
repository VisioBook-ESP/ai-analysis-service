from typing import List, Dict, Optional
import hashlib

try:
    from transformers import pipeline
    import torch

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class ZeroShotClassifier:

    def __init__(self, cache_size: int = 128):
        self.classifier = None
        self.cache_size = cache_size
        self._emotion_cache = {}
        self._trait_cache = {}
        self._atmosphere_cache = {}

        if _TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if torch.cuda.is_available() else -1
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                    device=device,
                )
            except Exception:
                pass

    def _get_cache_key(self, text: str, labels: List[str], threshold: float) -> str:
        """Generate cache key from text, labels, and threshold."""
        content = f"{text}|{','.join(sorted(labels))}|{threshold}"
        return hashlib.md5(content.encode()).hexdigest()

    def _manage_cache_size(self, cache: Dict) -> None:
        """Keep cache size under limit by removing oldest entries."""
        if len(cache) > self.cache_size:
            # Remove 25% oldest entries
            to_remove = len(cache) - int(self.cache_size * 0.75)
            for _ in range(to_remove):
                cache.pop(next(iter(cache)))

    def _extract_character_context(self, text: str, character_name: str) -> str:
        """Extract sentences from text that mention the specific character."""
        import re

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        # Filter to sentences mentioning this character
        relevant_sentences = []
        name_parts = character_name.lower().split()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if any part of the character name appears
            if any(part in sentence_lower for part in name_parts):
                relevant_sentences.append(sentence.strip())

        # Return relevant context or fallback to full text
        if relevant_sentences:
            return " ".join(relevant_sentences)
        return text[:200]  # Fallback to first 200 chars

    def classify_emotions(
        self, text: str, character_name: Optional[str] = None, threshold: float = 0.5
    ) -> List[str]:
        if not self.classifier:
            return []

        emotion_labels = [
            "happy",
            "sad",
            "nervous",
            "excited",
            "angry",
            "calm",
            "passionate",
            "encouraging",
            "surprised",
            "proud",
            "worried",
            "joyful",
            "fearful",
        ]

        if character_name:
            relevant_text = self._extract_character_context(text, character_name)
            context = f"{character_name}: {relevant_text}"
        else:
            context = text

        # Check cache first
        cache_key = self._get_cache_key(context, emotion_labels, threshold)
        if cache_key in self._emotion_cache:
            return self._emotion_cache[cache_key]

        try:
            result = self.classifier(context, candidate_labels=emotion_labels, multi_label=True)

            detected_emotions = []
            for label, score in zip(result["labels"], result["scores"]):
                if score >= threshold:
                    detected_emotions.append(label)

            detected_emotions = detected_emotions[:3]

            # Store in cache
            self._emotion_cache[cache_key] = detected_emotions
            self._manage_cache_size(self._emotion_cache)

            return detected_emotions

        except Exception:
            return []

    def classify_traits(self, text: str, character_name: str, threshold: float = 0.75) -> List[str]:
        if not self.classifier:
            return []

        trait_labels = [
            "young",
            "old",
            "tall",
            "elegant",
            "slim",
            "beautiful",
            "handsome",
            "well-dressed",
            "casual",
            "professional",
        ]

        relevant_text = self._extract_character_context(text, character_name)
        context = f"Description of {character_name}: {relevant_text}"

        # Check cache first
        cache_key = self._get_cache_key(context, trait_labels, threshold)
        if cache_key in self._trait_cache:
            return self._trait_cache[cache_key]

        try:
            result = self.classifier(context, candidate_labels=trait_labels, multi_label=True)

            detected_traits = []
            for label, score in zip(result["labels"], result["scores"]):
                if score >= threshold:
                    detected_traits.append(label)

            detected_traits = detected_traits[:5]

            # Store in cache
            self._trait_cache[cache_key] = detected_traits
            self._manage_cache_size(self._trait_cache)

            return detected_traits

        except Exception:
            return []

    def classify_atmosphere(self, text: str, threshold: float = 0.35) -> str:
        if not self.classifier:
            return "neutral"

        atmosphere_labels = [
            "bright",
            "dark",
            "warm",
            "cold",
            "peaceful",
            "tense",
            "cheerful",
            "gloomy",
            "mysterious",
            "romantic",
        ]

        # Check cache first
        cache_key = self._get_cache_key(text, atmosphere_labels, threshold)
        if cache_key in self._atmosphere_cache:
            return self._atmosphere_cache[cache_key]

        try:
            result = self.classifier(text, candidate_labels=atmosphere_labels, multi_label=False)

            atmosphere = "neutral"
            if result["scores"][0] >= threshold:
                atmosphere = result["labels"][0]

            # Store in cache
            self._atmosphere_cache[cache_key] = atmosphere
            self._manage_cache_size(self._atmosphere_cache)

            return atmosphere

        except Exception:
            return "neutral"
