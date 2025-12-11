from collections import Counter
from typing import Any

import spacy

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticAnalyzer:

    def __init__(self):
        self.nlp_fr = None
        self.nlp_en = None
        self.embedder = None
        self.sentiment_analyzer = None

        try:
            self.nlp_fr = spacy.load("fr_core_news_lg")
        except Exception:
            try:
                self.nlp_fr = spacy.load("fr_core_news_sm")
            except Exception:
                pass

        try:
            self.nlp_en = spacy.load("en_core_web_lg")
        except Exception:
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except Exception:
                pass

        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            except Exception:
                pass

            try:
                device = 0 if torch.cuda.is_available() else -1
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    device=device,
                )
            except Exception:
                pass

    def analyze(self, preprocessed: dict, return_embeddings: bool = False) -> dict[str, Any]:
        text = preprocessed["text"]
        language = preprocessed["language"]

        nlp = self._get_nlp(language)

        if nlp is None:
            return {
                "entities": [],
                "keywords": [],
                "topics": [],
                "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
                "embeddings": None,
            }

        doc = nlp(text)

        entities = self._extract_entities(doc)
        keywords = self._extract_keywords(doc)
        topics = self._extract_topics(doc)
        sentiment = self._analyze_sentiment(doc)

        result = {
            "entities": entities,
            "keywords": keywords,
            "topics": topics,
            "sentiment": sentiment,
        }

        if return_embeddings and self.embedder:
            result["embeddings"] = self._generate_embeddings(text)
        else:
            result["embeddings"] = None

        return result

    def _get_nlp(self, language: str):
        if language == "fr" and self.nlp_fr:
            return self.nlp_fr
        elif language == "en" and self.nlp_en:
            return self.nlp_en
        elif self.nlp_fr:
            return self.nlp_fr
        elif self.nlp_en:
            return self.nlp_en
        return None

    def _extract_entities(self, doc) -> list[dict]:
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        return entities

    def _extract_keywords(self, doc, top_n: int = 10) -> list[str]:
        words = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
        ]

        counter = Counter(words)
        keywords = [word for word, count in counter.most_common(top_n)]

        return keywords

    def _extract_topics(self, doc) -> list[str]:
        topics = set()

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                topics.add(ent.text.lower())

        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
        topics.update(noun_chunks[:5])

        return list(topics)[:10]

    def _analyze_sentiment(self, doc) -> dict[str, float]:
        if self.sentiment_analyzer:
            try:
                text = doc.text[:512]
                result = self.sentiment_analyzer(text)[0]

                label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

                base_polarity = label_map.get(result["label"].lower(), 0.0)
                confidence = result["score"]

                if base_polarity != 0:
                    polarity = base_polarity * (0.3 + 0.7 * confidence)
                else:
                    polarity = 0.0

                return {
                    "polarity": round(polarity, 3),
                    "subjectivity": round(confidence, 3),
                }
            except Exception:
                pass

        positive_words = {
            "bon",
            "bien",
            "heureux",
            "joie",
            "excellent",
            "super",
            "génial",
            "sourit",
            "sourire",
            "rit",
            "rire",
            "content",
            "joyeux",
            "merveilleux",
            "magnifique",
            "beau",
            "belle",
            "agréable",
            "plaisant",
            "réjoui",
            "good",
            "happy",
            "great",
            "smile",
            "laugh",
            "wonderful",
            "beautiful",
            "pleasant",
            "cheerful",
            "joyful",
        }
        negative_words = {
            "mauvais",
            "mal",
            "triste",
            "terrible",
            "nul",
            "pleure",
            "pleurer",
            "malheureux",
            "déçu",
            "désespéré",
            "horrible",
            "affreux",
            "laid",
            "désagréable",
            "pénible",
            "bad",
            "sad",
            "awful",
            "cry",
            "unhappy",
            "disappointed",
            "ugly",
            "unpleasant",
        }

        words = [token.lemma_.lower() for token in doc if token.is_alpha]

        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)

        total = len(words)
        if total == 0:
            return {"polarity": 0.0, "subjectivity": 0.0}

        polarity = (positive_count - negative_count) / max(1, total) * 10
        polarity = max(-1.0, min(1.0, polarity))

        subjectivity = (positive_count + negative_count) / total
        subjectivity = min(1.0, subjectivity)

        return {"polarity": round(polarity, 3), "subjectivity": round(subjectivity, 3)}

    def _generate_embeddings(self, text: str) -> list[float] | None:
        if not self.embedder:
            return None

        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception:
            return None
