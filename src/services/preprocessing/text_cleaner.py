import re
import html
import unicodedata
from typing import Tuple, Dict, List

try:
    import regex as re_unicode

    _REGEX_AVAILABLE = True
except ImportError:
    re_unicode = None
    _REGEX_AVAILABLE = False


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_control_chars(text: str) -> str:
    return "".join(
        ch for ch in text if (ch in ["\n", "\t"] or unicodedata.category(ch)[0] != "C")
    )


def replace_smart_quotes(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def collapse_spaces(text: str) -> str:
    text = re.sub(r"[ \t\x0b\x0c\r]+", " ", text)
    text = re.sub(r"\s*([,;:!?])\s*", r"\1 ", text)
    text = re.sub(r"\s*([.])(?!\d)\s*", r"\1 ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
_PHONE_RE = re.compile(r"(\+?\d[\d \-\.\(\)]{7,}\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


def remove_urls_emails(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    return collapse_spaces(text)


def mask_pii(text: str) -> Tuple[str, Dict[str, List[str]]]:
    masks = {"emails": [], "phones": [], "ibans": []}

    def _mask_replace(match):
        return "***"

    def _collect_and_mask(pattern, key, text):
        found = pattern.findall(text)
        if found:
            if isinstance(found[0], tuple):
                found = [item[0] if isinstance(item, tuple) else item for item in found]
            masks[key].extend(found)
        return pattern.sub(_mask_replace, text)

    text = _collect_and_mask(_EMAIL_RE, "emails", text)
    text = _collect_and_mask(_PHONE_RE, "phones", text)
    text = _collect_and_mask(_IBAN_RE, "ibans", text)

    return text, masks


def strip_emojis(text: str) -> str:
    if _REGEX_AVAILABLE:
        return re_unicode.sub(r"\p{Emoji}+", "", text)
    return "".join(ch for ch in text if unicodedata.category(ch) not in {"So", "Sk"})


def basic_clean(
    text: str,
    *,
    remove_links: bool = False,
    do_mask_pii: bool = False,
    remove_emoji: bool = False,
    lowercase: bool = False
) -> Tuple[str, Dict[str, List[str]]]:
    meta = {"emails": [], "phones": [], "ibans": []}

    if not text:
        return "", meta

    t = html.unescape(text)
    t = normalize_unicode(t)
    t = replace_smart_quotes(t)
    t = strip_control_chars(t)

    if remove_emoji:
        t = strip_emojis(t)

    if remove_links:
        t = remove_urls_emails(t)

    if do_mask_pii:
        t, masks = mask_pii(t)
        for k, v in masks.items():
            meta[k] = v

    if lowercase:
        t = t.lower()

    t = collapse_spaces(t)

    return t, meta
