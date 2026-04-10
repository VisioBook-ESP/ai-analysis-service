from unittest.mock import patch

import pytest

from src.services.preprocessing.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


EXPECTED_KEYS = {
    "language",
    "text",
    "quality",
    "sentences",
    "chunks",
    "masks",
    "stats",
    "processing_time_ms",
}


class TestTextPreprocessorInit:
    def test_instantiation(self):
        p = TextPreprocessor()
        assert p is not None


class TestPreprocess:
    def test_returns_all_expected_keys(self, preprocessor):
        result = preprocessor.preprocess(
            "Alice walked through the forest. She was happy."
        )
        assert set(result.keys()) == EXPECTED_KEYS

    def test_language_is_string(self, preprocessor):
        result = preprocessor.preprocess("Alice walked through the forest today.")
        assert isinstance(result["language"], str)

    def test_explicit_language_used(self, preprocessor):
        result = preprocessor.preprocess("Some text here.", language="en")
        assert result["language"] == "en"

    def test_text_is_cleaned(self, preprocessor):
        result = preprocessor.preprocess("Hello   world  ")
        assert isinstance(result["text"], str)
        assert "  " not in result["text"]

    def test_stats_original_length_correct(self, preprocessor):
        text = "Hello world."
        result = preprocessor.preprocess(text)
        assert result["stats"]["original_length"] == len(text)

    def test_stats_cleaned_length_matches_text(self, preprocessor):
        result = preprocessor.preprocess("Hello world.")
        assert result["stats"]["cleaned_length"] == len(result["text"])

    def test_stats_sentence_count_matches_sentences(self, preprocessor):
        result = preprocessor.preprocess("First sentence. Second sentence.")
        assert result["stats"]["sentence_count"] == len(result["sentences"])

    def test_quality_has_score_and_assessment(self, preprocessor):
        result = preprocessor.preprocess("Clean prose text here.")
        assert "score" in result["quality"]
        assert "assessment" in result["quality"]
        assert result["quality"]["assessment"] in {"excellent", "good", "fair", "poor"}

    def test_processing_time_non_negative(self, preprocessor):
        result = preprocessor.preprocess("Some text.")
        assert result["processing_time_ms"] >= 0

    def test_sentences_is_list(self, preprocessor):
        result = preprocessor.preprocess("First sentence. Second sentence.")
        assert isinstance(result["sentences"], list)

    def test_chunks_is_list(self, preprocessor):
        result = preprocessor.preprocess("Some text.")
        assert isinstance(result["chunks"], list)

    def test_masks_keys_present(self, preprocessor):
        result = preprocessor.preprocess("Some text.")
        assert "emails" in result["masks"]
        assert "phones" in result["masks"]
        assert "ibans" in result["masks"]

    def test_pii_masked_when_option_true(self, preprocessor):
        result = preprocessor.preprocess(
            "Contact alice@example.com for help.", mask_pii=True
        )
        assert "alice@example.com" not in result["text"]
        assert "alice@example.com" in result["masks"]["emails"]

    def test_pii_not_masked_when_option_false(self, preprocessor):
        # mask_pii=True par défaut — passer False pour désactiver le masquage
        result = preprocessor.preprocess(
            "Contact alice@example.com for help.", mask_pii=False
        )
        assert result["masks"]["emails"] == []

    def test_remove_links_option(self, preprocessor):
        result = preprocessor.preprocess(
            "Visit https://example.com for more.", remove_links=True
        )
        assert "https://" not in result["text"]

    def test_lowercase_option(self, preprocessor):
        result = preprocessor.preprocess("HELLO WORLD.", lowercase=True)
        assert result["text"] == result["text"].lower()

    def test_language_auto_detection(self, preprocessor):
        result = preprocessor.preprocess(
            "Bonjour, comment allez-vous? Il fait beau aujourd'hui dans la forêt.",
            language="auto",
        )
        assert result["language"] in ("fr", "auto")


class TestPreprocessBatch:
    def test_returns_correct_count(self, preprocessor):
        texts = ["First text.", "Second text.", "Third text."]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 3

    def test_each_result_has_text_key(self, preprocessor):
        results = preprocessor.preprocess_batch(["Hello world.", "Bonjour monde."])
        for r in results:
            assert "text" in r

    def test_empty_list_returns_empty(self, preprocessor):
        assert preprocessor.preprocess_batch([]) == []

    def test_error_caught_per_item(self, preprocessor):
        with patch(
            "src.services.preprocessing.preprocessor.noise_score",
            side_effect=RuntimeError("boom"),
        ):
            results = preprocessor.preprocess_batch(["some text"])
        assert "error" in results[0]
        assert results[0]["error"] == "boom"

    def test_error_does_not_stop_other_items(self, preprocessor):
        call_count = 0

        def failing_noise_score(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first item fails")
            from src.services.preprocessing.quality_scorer import noise_score as real

            return real(text)

        with patch(
            "src.services.preprocessing.preprocessor.noise_score",
            side_effect=failing_noise_score,
        ):
            results = preprocessor.preprocess_batch(["fail text", "good text"])
        assert "error" in results[0]
        assert "text" in results[1]
