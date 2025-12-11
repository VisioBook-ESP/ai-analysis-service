import re
from typing import Dict

_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def noise_score(text: str) -> Dict[str, float]:
    if not text:
        return {
            "score": 0.0,
            "non_letter_ratio": 0.0,
            "punct_ratio": 0.0,
            "upper_ratio": 0.0,
            "avg_sent_len": 0.0,
            "url_ratio": 0.0,
        }

    n = len(text)

    letters = sum(1 for c in text if c.isalpha())
    non_letter_ratio = 1 - (letters / n)

    punct_chars = ".,;:!?"
    punct_count = sum(1 for c in text if c in punct_chars)
    punct_ratio = punct_count / n

    uppers = sum(1 for c in text if c.isupper())
    upper_ratio = uppers / max(1, letters)

    urls = len(_URL_PATTERN.findall(text))
    url_ratio = urls / max(1, n / 80)

    sentences = re.split(r"[.!?]+\s*", text.strip())
    sentences = [s for s in sentences if s]
    avg_sent_len = sum(len(s) for s in sentences) / max(1, len(sentences))

    f_non = min(1.0, non_letter_ratio * 1.2)
    f_punct = min(1.0, punct_ratio * 8)
    f_upper = min(1.0, upper_ratio * 2)
    f_url = min(1.0, url_ratio * 0.5)

    if avg_sent_len < 30:
        f_sent = (30 - avg_sent_len) / 30 * 0.6
    elif avg_sent_len > 250:
        f_sent = (avg_sent_len - 250) / 250 * 0.6
    else:
        f_sent = 0.0

    score = max(
        0.0,
        min(
            1.0,
            0.35 * f_non + 0.20 * f_punct + 0.20 * f_upper + 0.15 * f_url + 0.10 * f_sent,
        ),
    )

    return {
        "score": round(score, 3),
        "non_letter_ratio": round(f_non, 3),
        "punct_ratio": round(f_punct, 3),
        "upper_ratio": round(f_upper, 3),
        "avg_sent_len": round(avg_sent_len, 1),
        "url_ratio": round(f_url, 3),
    }


def assess_quality(text: str) -> str:
    score = noise_score(text)["score"]

    if score < 0.2:
        return "excellent"
    elif score < 0.4:
        return "good"
    elif score < 0.6:
        return "fair"
    else:
        return "poor"
