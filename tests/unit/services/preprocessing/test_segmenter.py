from unittest.mock import MagicMock, patch

import pytest

from src.services.preprocessing.segmenter import (
    _build_chunks_fallback,
    _get_spacy_model,
    _split_sentences_fallback,
    build_chunks,
    split_sentences,
    tokenize,
)


@pytest.fixture(autouse=True)
def clear_spacy_cache():
    _get_spacy_model.cache_clear()
    yield
    _get_spacy_model.cache_clear()


# ---------------------------------------------------------------------------
# _split_sentences_fallback
# ---------------------------------------------------------------------------

class TestSplitSentencesFallback:
    def test_basic_sentences_split(self):
        result = _split_sentences_fallback("Hello world. How are you?")
        assert len(result) >= 1
        for s in result:
            assert "start" in s and "end" in s and "text" in s

    def test_empty_text_returns_empty(self):
        assert _split_sentences_fallback("") == []

    def test_whitespace_only_returns_empty(self):
        assert _split_sentences_fallback("   ") == []

    def test_no_punctuation_returns_single_entry(self):
        result = _split_sentences_fallback("No punctuation here at all in this text")
        assert len(result) == 1
        assert result[0]["text"] == "No punctuation here at all in this text"

    def test_positions_valid(self):
        text = "First sentence. Second one."
        result = _split_sentences_fallback(text)
        for s in result:
            assert s["start"] >= 0
            assert s["end"] <= len(text)
            assert s["start"] < s["end"]

    def test_text_field_matches_original(self):
        text = "Hello world."
        result = _split_sentences_fallback(text)
        for s in result:
            assert s["text"] in text


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_empty_text_returns_empty(self):
        assert split_sentences("") == []

    def test_whitespace_returns_empty(self):
        assert split_sentences("   ") == []

    def test_returns_list_of_dicts_with_text_key(self):
        result = split_sentences("Hello world. This is a sentence.")
        assert isinstance(result, list)
        assert all("text" in s for s in result)

    def test_fallback_when_spacy_returns_none(self):
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=None):
            result = split_sentences("Hello world. Test sentence here.", lang="fr")
            assert len(result) >= 1

    def test_fallback_when_spacy_raises(self):
        mock_nlp = MagicMock(side_effect=Exception("spacy crash"))
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=mock_nlp):
            result = split_sentences("Hello world. Test sentence.", lang="fr")
            assert len(result) >= 1

    def test_french_lang(self):
        result = split_sentences("Bonjour monde. Comment vas-tu?", lang="fr")
        assert isinstance(result, list)

    def test_english_lang(self):
        result = split_sentences("Hello world. How are you today?", lang="en")
        assert isinstance(result, list)

    def test_unknown_lang_uses_fallback_model(self):
        result = split_sentences("Some text here. Another sentence.", lang="zh")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_empty_text_returns_empty(self):
        assert tokenize("") == []

    def test_returns_list_of_strings(self):
        result = tokenize("Hello world.")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)
        assert len(result) > 0

    def test_fallback_when_spacy_returns_none(self):
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=None):
            result = tokenize("Hello world.", lang="fr")
            assert len(result) > 0

    def test_fallback_when_spacy_raises(self):
        mock_nlp = MagicMock(side_effect=Exception("crash"))
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=mock_nlp):
            result = tokenize("Hello world.", lang="fr")
            assert len(result) > 0


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_empty_text_returns_empty(self):
        assert build_chunks("") == []

    def test_returns_list(self):
        result = build_chunks("Hello world. This is some text for testing.")
        assert isinstance(result, list)

    def test_chunk_has_required_keys(self):
        result = build_chunks("Hello world. This is some text.")
        if result:
            assert "text" in result[0]
            assert "token_count" in result[0]
            assert "token_start" in result[0]
            assert "token_end" in result[0]

    def test_fallback_when_spacy_returns_none(self):
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=None):
            result = build_chunks("Hello world. This is some text.", lang="fr")
            assert isinstance(result, list)

    def test_fallback_when_spacy_raises(self):
        mock_nlp = MagicMock(side_effect=Exception("crash"))
        with patch("src.services.preprocessing.segmenter._get_spacy_model", return_value=mock_nlp):
            result = build_chunks("Hello world. This is some text.", lang="fr")
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _build_chunks_fallback
# ---------------------------------------------------------------------------

class TestBuildChunksFallback:
    def test_empty_text_returns_empty(self):
        assert _build_chunks_fallback("", max_tokens=512, overlap=64) == []

    def test_basic_chunking(self):
        text = "Hello world. This is a test sentence with some words."
        result = _build_chunks_fallback(text, max_tokens=10, overlap=2)
        assert isinstance(result, list)
        for chunk in result:
            assert "text" in chunk
            assert "token_count" in chunk
            assert "token_start" in chunk
            assert "token_end" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk

    def test_small_max_tokens_produces_single_chunk(self):
        text = "Hello world."
        result = _build_chunks_fallback(text, max_tokens=1000, overlap=100)
        assert len(result) == 1

    def test_overlap_creates_multiple_chunks(self):
        text = " ".join(["word"] * 100)
        result = _build_chunks_fallback(text, max_tokens=20, overlap=5)
        assert len(result) > 1

    def test_chunk_text_is_substring_of_original(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = _build_chunks_fallback(text, max_tokens=5, overlap=1)
        for chunk in result:
            assert chunk["text"] in text

    def test_token_counts_correct(self):
        text = " ".join(["word"] * 10)
        result = _build_chunks_fallback(text, max_tokens=10, overlap=0)
        assert result[0]["token_count"] == 10


# ---------------------------------------------------------------------------
# _get_spacy_model
# ---------------------------------------------------------------------------

class TestGetSpacyModel:
    def test_returns_model_or_none_for_fr(self):
        model = _get_spacy_model("fr")
        # retourne un modèle spaCy ou None si spaCy n'est pas dispo
        assert model is None or hasattr(model, "__call__")

    def test_returns_model_or_none_for_en(self):
        model = _get_spacy_model("en")
        assert model is None or hasattr(model, "__call__")

    def test_returns_model_or_none_for_unknown_lang(self):
        model = _get_spacy_model("xx")
        assert model is None or hasattr(model, "__call__")

    def test_cached_result_same_for_same_lang(self):
        model1 = _get_spacy_model("fr")
        model2 = _get_spacy_model("fr")
        assert model1 is model2
