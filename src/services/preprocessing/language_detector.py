try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def detect_language(text: str, default: str = "auto") -> str:
    t = (text or "").strip()

    if not t or len(t) < 20:
        return default

    if not _LANGDETECT_AVAILABLE:
        return default

    try:
        lang = detect(t)
        if lang.startswith("fr"):
            return "fr"
        elif lang.startswith("en"):
            return "en"
        else:
            return lang[:2]
    except Exception:
        return default


def is_language_supported(lang: str) -> bool:
    supported = ["fr", "en", "auto"]
    return lang in supported
