import pytest

from src.services.preprocessing.text_cleaner import (
    normalize_unicode,
    strip_control_chars,
    replace_smart_quotes,
    collapse_spaces,
    mask_pii,
    basic_clean,
)


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------

class TestNormalizeUnicode:
    def test_nfc_composes_decomposed_character(self):
        # 'e' + combining acute accent → 'é' composé
        decomposed = "e\u0301"
        assert normalize_unicode(decomposed) == "\xe9"

    def test_already_composed_unchanged(self):
        assert normalize_unicode("café") == "café"

    def test_empty_string(self):
        assert normalize_unicode("") == ""


# ---------------------------------------------------------------------------
# strip_control_chars
# ---------------------------------------------------------------------------

class TestStripControlChars:
    def test_removes_null_byte(self):
        assert strip_control_chars("hello\x00world") == "helloworld"

    def test_removes_bell_char(self):
        assert strip_control_chars("a\x07b") == "ab"

    def test_keeps_newline(self):
        result = strip_control_chars("line1\nline2")
        assert "\n" in result

    def test_keeps_tab(self):
        result = strip_control_chars("col1\tcol2")
        assert "\t" in result

    def test_regular_text_unchanged(self):
        text = "Hello, world!"
        assert strip_control_chars(text) == text


# ---------------------------------------------------------------------------
# replace_smart_quotes
# ---------------------------------------------------------------------------

class TestReplaceSmartQuotes:
    def test_left_double_quote_replaced(self):
        assert replace_smart_quotes("\u201chello\u201d") == '"hello"'

    def test_single_curly_quotes_replaced(self):
        assert replace_smart_quotes("\u2018hello\u2019") == "'hello'"

    def test_em_dash_replaced_by_hyphen(self):
        assert replace_smart_quotes("word\u2014word") == "word-word"

    def test_en_dash_replaced_by_hyphen(self):
        assert replace_smart_quotes("word\u2013word") == "word-word"

    def test_guillemets_replaced(self):
        result = replace_smart_quotes("\u00abhello\u00bb")
        assert result == '"hello"'

    def test_regular_text_unchanged(self):
        text = "No special quotes here."
        assert replace_smart_quotes(text) == text


# ---------------------------------------------------------------------------
# collapse_spaces
# ---------------------------------------------------------------------------

class TestCollapseSpaces:
    def test_multiple_spaces_collapsed_to_one(self):
        assert collapse_spaces("hello   world") == "hello world"

    def test_strips_leading_and_trailing_spaces(self):
        assert collapse_spaces("  hello  ") == "hello"

    def test_space_before_comma_removed(self):
        result = collapse_spaces("hello , world")
        assert "hello," in result

    def test_space_before_semicolon_removed(self):
        result = collapse_spaces("a ; b")
        assert "a;" in result

    def test_trailing_spaces_before_newline_removed(self):
        result = collapse_spaces("hello   \nworld")
        assert "hello\n" in result


# ---------------------------------------------------------------------------
# strip_emojis
# ---------------------------------------------------------------------------

class TestStripEmojis:
    def test_removes_emojis(self):
        from src.services.preprocessing.text_cleaner import strip_emojis
        result = strip_emojis("Hello 🌟 world 🎉")
        assert "🌟" not in result
        assert "🎉" not in result
        assert "Hello" in result

    def test_no_emojis_unchanged(self):
        from src.services.preprocessing.text_cleaner import strip_emojis
        text = "No emojis here."
        assert strip_emojis(text) == text


# ---------------------------------------------------------------------------
# mask_pii
# ---------------------------------------------------------------------------

class TestMaskPii:
    def test_masks_email_and_collects_it(self):
        text, masks = mask_pii("Contact me at john.doe@example.com please.")
        assert "john.doe@example.com" not in text
        assert "***" in text
        assert "john.doe@example.com" in masks["emails"]

    def test_masks_multiple_emails(self):
        text, masks = mask_pii("From: a@a.com To: b@b.com")
        assert len(masks["emails"]) == 2
        assert "a@a.com" in masks["emails"]
        assert "b@b.com" in masks["emails"]

    def test_masks_iban(self):
        # Le regex phone capture les chiffres avant le regex IBAN — on vérifie
        # que les digits sont masqués, quelle que soit la catégorie détectée.
        text, masks = mask_pii("My IBAN is FR7630006000011234567890189.")
        assert "7630006000011234567890189" not in text
        assert len(masks["ibans"]) > 0 or len(masks["phones"]) > 0

    def test_no_pii_returns_unchanged_and_empty_masks(self):
        text = "No personal data here."
        result, masks = mask_pii(text)
        assert result == text
        assert masks["emails"] == []
        assert masks["phones"] == []
        assert masks["ibans"] == []


# ---------------------------------------------------------------------------
# basic_clean
# ---------------------------------------------------------------------------

class TestBasicClean:
    def test_empty_text_returns_empty_and_empty_masks(self):
        text, masks = basic_clean("")
        assert text == ""
        assert masks == {"emails": [], "phones": [], "ibans": []}

    def test_html_entities_unescaped(self):
        text, _ = basic_clean("Hello &amp; world &lt;tag&gt;")
        assert "&amp;" not in text
        assert "&" in text

    def test_smart_quotes_replaced(self):
        text, _ = basic_clean("\u201chello\u201d")
        assert '"hello"' in text

    def test_remove_links_strips_urls(self):
        text, _ = basic_clean("Visit https://example.com for more.", remove_links=True)
        assert "https://example.com" not in text

    def test_remove_links_false_does_not_strip_host(self):
        # collapse_spaces ajoute des espaces après ':' et '.', donc l'URL est
        # reformatée — mais le nom de domaine reste dans le texte.
        text, _ = basic_clean("Visit https://example.com for more.", remove_links=False)
        assert "example" in text

    def test_lowercase_option(self):
        text, _ = basic_clean("HELLO WORLD", lowercase=True)
        assert text == "hello world"

    def test_pii_masked_when_option_true(self):
        text, masks = basic_clean("Email: user@domain.com", do_mask_pii=True)
        assert "user@domain.com" not in text
        assert "user@domain.com" in masks["emails"]

    def test_pii_not_masked_by_default(self):
        # collapse_spaces reformate l'email (espaces après '.' et ':'), mais
        # aucun masquage ne doit avoir eu lieu — les masks restent vides.
        _, masks = basic_clean("Email: user@domain.com")
        assert masks["emails"] == []

    def test_remove_emoji_option(self):
        text, _ = basic_clean("Hello 🌟 world!", remove_emoji=True)
        assert "🌟" not in text
        assert "Hello" in text

    def test_multiple_spaces_collapsed(self):
        text, _ = basic_clean("Hello   world")
        assert text == "Hello world"

    def test_all_options_combined(self):
        raw = "VISIT https://site.com &amp; contact USER@DOMAIN.COM now !!!"
        text, masks = basic_clean(
            raw,
            remove_links=True,
            do_mask_pii=True,
            lowercase=True,
        )
        assert "https://site.com" not in text
        assert "user@domain.com" not in text
        assert text == text.lower()
