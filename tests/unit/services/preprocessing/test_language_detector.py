from unittest.mock import patch

from src.services.preprocessing.language_detector import detect_language, is_language_supported


class TestDetectLanguage:
    def test_empty_string_returns_default(self):
        assert detect_language("", default="auto") == "auto"

    def test_whitespace_only_returns_default(self):
        assert detect_language("   ", default="auto") == "auto"

    def test_too_short_returns_default(self):
        # < 20 chars → default
        assert detect_language("Bonjour", default="xx") == "xx"

    def test_default_parameter_respected(self):
        assert detect_language("", default="unknown") == "unknown"

    def test_french_text_detected(self):
        text = "Bonjour, comment allez-vous aujourd'hui ? Il fait très beau temps en ce moment."
        assert detect_language(text) == "fr"

    def test_english_text_detected(self):
        text = "Hello, how are you today? The weather is very nice at the moment outside here."
        assert detect_language(text) == "en"

    def test_exception_in_detect_returns_default(self):
        with patch("src.services.preprocessing.language_detector.detect", side_effect=Exception("fail")):
            result = detect_language("some longer text for detection here ok now yes", default="fallback")
            assert result == "fallback"

    def test_unknown_language_returns_first_two_chars(self):
        with patch("src.services.preprocessing.language_detector.detect", return_value="de"):
            result = detect_language("Ein sehr langer Text der auf Deutsch geschrieben wurde hier ja.")
            assert result == "de"

    def test_fr_prefix_returns_fr(self):
        with patch("src.services.preprocessing.language_detector.detect", return_value="fr-CA"):
            result = detect_language("Un texte assez long pour être détecté correctement ici.")
            assert result == "fr"

    def test_en_prefix_returns_en(self):
        with patch("src.services.preprocessing.language_detector.detect", return_value="en-GB"):
            result = detect_language("A text that is long enough to be detected correctly here.")
            assert result == "en"


class TestIsLanguageSupported:
    def test_fr_supported(self):
        assert is_language_supported("fr") is True

    def test_en_supported(self):
        assert is_language_supported("en") is True

    def test_auto_supported(self):
        assert is_language_supported("auto") is True

    def test_de_not_supported(self):
        assert is_language_supported("de") is False

    def test_empty_string_not_supported(self):
        assert is_language_supported("") is False

    def test_unknown_not_supported(self):
        assert is_language_supported("zh") is False
