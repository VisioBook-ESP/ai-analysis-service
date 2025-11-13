import re
from typing import List, Dict, Any
from functools import lru_cache

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    _SPACY_AVAILABLE = False


@lru_cache(maxsize=4)
def _get_spacy_model(lang: str):
    if not _SPACY_AVAILABLE:
        return None

    nlp = None
    model_map = {
        "fr": "fr_core_news_lg",
        "en": "en_core_web_lg",
    }

    if lang in model_map:
        try:
            nlp = spacy.load(model_map[lang])
        except Exception:
            pass

    if nlp is None:
        try:
            target_lang = lang if lang in {"fr", "en"} else "xx"
            nlp = spacy.blank(target_lang)
            if not nlp.has_pipe("sentencizer"):
                nlp.add_pipe("sentencizer")
        except Exception:
            return None

    return nlp


def split_sentences(text: str, lang: str = "fr") -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    target_lang = "fr" if lang == "fr" else ("en" if lang == "en" else "xx")

    nlp = _get_spacy_model(target_lang)
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                sentences.append({
                    "start": sent.start_char,
                    "end": sent.end_char,
                    "text": sent.text.strip()
                })
            return sentences
        except Exception:
            pass

    return _split_sentences_fallback(text)


def _split_sentences_fallback(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"([^.!?;]+[.!?;])", re.DOTALL)
    sentences = []
    cursor = 0

    for match in pattern.finditer(text):
        segment = match.group(0).strip()
        if segment:
            start = text.find(segment, cursor)
            end = start + len(segment)
            sentences.append({
                "start": start,
                "end": end,
                "text": segment
            })
            cursor = end

    if not sentences and text.strip():
        sentences = [{
            "start": 0,
            "end": len(text),
            "text": text.strip()
        }]

    return sentences


def tokenize(text: str, lang: str = "fr") -> List[str]:
    if not text:
        return []

    target_lang = "fr" if lang == "fr" else ("en" if lang == "en" else "xx")

    nlp = _get_spacy_model(target_lang)
    if nlp is not None:
        try:
            doc = nlp(text)
            return [token.text for token in doc]
        except Exception:
            pass

    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def build_chunks(
    text: str,
    lang: str = "fr",
    max_tokens: int = 512,
    overlap: int = 64
) -> List[Dict[str, Any]]:
    if not text:
        return []

    target_lang = "fr" if lang == "fr" else ("en" if lang == "en" else "xx")

    nlp = _get_spacy_model(target_lang)
    if nlp is not None:
        try:
            return _build_chunks_spacy(text, nlp, max_tokens, overlap)
        except Exception:
            pass

    return _build_chunks_fallback(text, max_tokens, overlap)


def _build_chunks_spacy(text: str, nlp, max_tokens: int, overlap: int) -> List[Dict[str, Any]]:
    doc = nlp(text)
    tokens = [t for t in doc]

    if not tokens:
        return []

    chunks = []
    i = 0

    while i < len(tokens):
        j = min(i + max_tokens, len(tokens))

        start_char = tokens[i].idx
        end_token = tokens[j - 1]
        end_char = end_token.idx + len(end_token)
        chunk_text = doc.text[start_char:end_char]

        chunks.append({
            "token_start": i,
            "token_end": j,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text,
            "token_count": j - i
        })

        if j == len(tokens):
            break

        i = max(i + 1, j - overlap)

    return chunks


def _build_chunks_fallback(text: str, max_tokens: int, overlap: int) -> List[Dict[str, Any]]:
    tokens = tokenize(text, "xx")

    if not tokens:
        return []

    positions = []
    cursor = 0
    for token in tokens:
        pos = text.find(token, cursor)
        if pos < 0:
            pos = cursor
        positions.append((pos, pos + len(token)))
        cursor = pos + len(token)

    chunks = []
    i = 0

    while i < len(tokens):
        j = min(i + max_tokens, len(tokens))

        start_char = positions[i][0]
        end_char = positions[j - 1][1]

        chunks.append({
            "token_start": i,
            "token_end": j,
            "start_char": start_char,
            "end_char": end_char,
            "text": text[start_char:end_char],
            "token_count": j - i
        })

        if j == len(tokens):
            break

        i = max(i + 1, j - overlap)

    return chunks
