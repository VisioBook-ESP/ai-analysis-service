"""Hybrid chapter detection: regex fast-path → LLM fallback → token chunks.

Detects chapter boundaries in book text to enable per-chapter LLM analysis
instead of single-shot whole-book processing.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .segmenter import build_chunks, split_sentences

logger = logging.getLogger(__name__)

# Minimum chapters for a valid split
_MIN_CHAPTERS = 3
# Token bounds for merge/split (estimated LLM subword tokens)
_MIN_CHAPTER_TOKENS = 1500
_MAX_CHAPTER_TOKENS = 10000
# Word-to-LLM-token ratio (conservative for European languages)
_WORD_TOKEN_RATIO = 1.4
# Fallback chunk size in spaCy word tokens (~5600 LLM tokens)
_FALLBACK_CHUNK_WORDS = 4000
# TOC region: skip regex matches in the first N% of the document
_TOC_REGION_RATIO = 0.05

# ── Chapter regex patterns ────────────────────────────────────────────────

# Roman numeral pattern (I through XXXIX covers most books)
_ROMAN = r"[IVXLCDM]+"
_ARABIC = r"[0-9]+"

# Multi-language chapter keyword patterns
_CHAPTER_KEYWORDS = [
    # French
    r"Chapitre",
    r"Chap\.",
    # English
    r"Chapter",
    r"CHAPTER",
    # German
    r"Kapitel",
    # Spanish
    r"Cap[ií]tulo",
    # Italian
    r"Capitolo",
    # Generic structural
    r"Partie",
    r"Part(?:ie)?",
    r"Livre",
    r"Book",
    r"Tome",
    r"Acte",
    r"Act",
]

_CHAPTER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    + "|".join(_CHAPTER_KEYWORDS)
    + r")\s+("
    + _ROMAN
    + r"|"
    + _ARABIC
    + r")\b",
    re.IGNORECASE,
)

# LLM chapter detection prompt
_STRUCTURE_SYSTEM_PROMPT = """You are a document structure analyzer. You identify chapter or section boundaries in books and literary texts.

RULES:
- Respond with valid JSON only.
- Identify ALL chapters, parts, or major sections.
- For each chapter, provide its title and a short text snippet (first ~50 characters of the chapter BODY, not the heading) that can be used to locate it in the full document.
- If the document has no chapter structure, return an empty array."""

_STRUCTURE_USER_TEMPLATE = """Identify all chapter/section boundaries in this document structure.

Return JSON array:
[
  {{"title": "Chapter title or number", "starts_with": "first ~50 chars of chapter body text"}}
]

DOCUMENT STRUCTURE:
\"\"\"
{skeleton}
\"\"\""""


@dataclass
class Chapter:
    """A detected chapter with text boundaries and metadata."""

    index: int
    title: str
    start_char: int
    end_char: int
    text: str
    word_count: int
    estimated_tokens: int


def _estimate_tokens(word_count: int) -> int:
    return int(word_count * _WORD_TOKEN_RATIO)


# ── Regex detection ───────────────────────────────────────────────────────


def _detect_by_regex(text: str) -> List[Chapter]:
    """Detect chapters using multi-language regex patterns.

    Skips matches in the TOC region (first 5% of the document) by keeping
    only the last occurrence of each chapter number (body, not TOC).
    """
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return []

    toc_boundary = int(len(text) * _TOC_REGION_RATIO)

    # Group by chapter number, keep body occurrence (after TOC region)
    by_number: Dict[str, List[re.Match]] = {}
    for m in matches:
        num = m.group(1).upper()
        by_number.setdefault(num, []).append(m)

    body_positions: List[tuple] = []
    for num, group in by_number.items():
        # Prefer matches outside the TOC region
        body_matches = [m for m in group if m.start() > toc_boundary]
        if body_matches:
            body_positions.append((body_matches[0].start(), num, body_matches[0]))
        elif len(group) >= 2:
            # Duplicate matches in TOC region: take the last one (likely body)
            body_positions.append((group[-1].start(), num, group[-1]))
        else:
            # Single match in TOC region: it's a real chapter, not a TOC entry
            body_positions.append((group[0].start(), num, group[0]))

    if not body_positions:
        return []

    # Sort by position in text
    body_positions.sort(key=lambda x: x[0])

    # Build chapter objects
    chapters = []
    for i, (start, num, match) in enumerate(body_positions):
        # Chapter text starts at the match position
        if i + 1 < len(body_positions):
            end = body_positions[i + 1][0]
        else:
            end = len(text)

        title = match.group(0).strip()
        chapter_text = text[start:end]
        word_count = len(chapter_text.split())

        chapters.append(
            Chapter(
                index=i,
                title=title,
                start_char=start,
                end_char=end,
                text=chapter_text,
                word_count=word_count,
                estimated_tokens=_estimate_tokens(word_count),
            )
        )

    return chapters


# ── LLM detection ────────────────────────────────────────────────────────


def _build_structural_skeleton(text: str, max_skeleton_chars: int = 8000) -> str:
    """Build a compressed structural view of the document.

    Extracts short lines, paragraph openings, and structural markers
    to produce a ~1-2k token skeleton for LLM chapter detection.
    """
    lines = text.split("\n")
    skeleton_parts = []
    total_chars = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        include = False
        # Short lines are likely structural (titles, chapter headings, breaks)
        if len(stripped) < 100:
            include = True
        # First 120 chars of longer lines (paragraph openings)
        elif total_chars < max_skeleton_chars:
            stripped = stripped[:120] + "..."
            include = True

        if include and total_chars + len(stripped) < max_skeleton_chars:
            skeleton_parts.append(f"[L{i+1}] {stripped}")
            total_chars += len(stripped)

    return "\n".join(skeleton_parts)


async def _detect_by_llm(text: str, lang: str, llm_client: Any) -> List[Chapter]:
    """Detect chapters by sending a structural skeleton to the LLM."""
    skeleton = _build_structural_skeleton(text)
    if not skeleton.strip():
        return []

    user_prompt = _STRUCTURE_USER_TEMPLATE.format(skeleton=skeleton)

    try:
        raw = await llm_client.chat_completion(
            system_prompt=_STRUCTURE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("LLM chapter detection failed: %s", e)
        return []

    # Parse the LLM response
    chapter_defs = raw if isinstance(raw, list) else raw.get("chapters", [])
    if not isinstance(chapter_defs, list) or len(chapter_defs) < 2:
        return []

    # Locate each chapter in the full text using the starts_with locator
    chapters = []
    for i, cdef in enumerate(chapter_defs):
        if not isinstance(cdef, dict):
            continue
        title = str(cdef.get("title", f"Section {i + 1}"))
        starts_with = str(cdef.get("starts_with", ""))

        if not starts_with or len(starts_with) < 10:
            continue

        # Find the locator string in the text
        pos = text.find(starts_with)
        if pos == -1:
            # Try a fuzzy match: first 30 chars
            pos = text.find(starts_with[:30])
        if pos == -1:
            logger.debug("Could not locate chapter %r in text", title)
            continue

        chapters.append(
            Chapter(
                index=i,
                title=title,
                start_char=pos,
                end_char=0,  # filled below
                text="",
                word_count=0,
                estimated_tokens=0,
            )
        )

    if len(chapters) < 2:
        return []

    # Sort by position and fill in end boundaries + text
    chapters.sort(key=lambda c: c.start_char)
    for i, ch in enumerate(chapters):
        ch.index = i
        ch.end_char = chapters[i + 1].start_char if i + 1 < len(chapters) else len(text)
        ch.text = text[ch.start_char : ch.end_char]
        ch.word_count = len(ch.text.split())
        ch.estimated_tokens = _estimate_tokens(ch.word_count)

    return chapters


# ── Validation and size enforcement ───────────────────────────────────────


def _is_valid_split(chapters: List[Chapter], total_text_len: int) -> bool:
    """Check if the detected split is reasonable."""
    if len(chapters) < _MIN_CHAPTERS:
        return False

    if not chapters:
        return False

    avg_tokens = sum(c.estimated_tokens for c in chapters) / len(chapters)
    if avg_tokens < 200 or avg_tokens > 40000:
        return False

    # No single chapter should dominate (>60% of total text)
    max_chars = max(len(c.text) for c in chapters)
    if max_chars > total_text_len * 0.6:
        return False

    return True


def _enforce_size_bounds(chapters: List[Chapter], lang: str) -> List[Chapter]:
    """Merge small chapters and split large ones."""
    # Pass 1: merge small chapters with the next chapter
    merged = []
    i = 0
    while i < len(chapters):
        ch = chapters[i]
        if ch.estimated_tokens < _MIN_CHAPTER_TOKENS and i + 1 < len(chapters):
            # Merge with next chapter
            next_ch = chapters[i + 1]
            combined_text = ch.text + next_ch.text
            combined_words = len(combined_text.split())
            merged.append(
                Chapter(
                    index=0,
                    title=ch.title,
                    start_char=ch.start_char,
                    end_char=next_ch.end_char,
                    text=combined_text,
                    word_count=combined_words,
                    estimated_tokens=_estimate_tokens(combined_words),
                )
            )
            i += 2
        else:
            merged.append(ch)
            i += 1

    # Pass 2: split large chapters at sentence boundaries
    result = []
    for ch in merged:
        if ch.estimated_tokens > _MAX_CHAPTER_TOKENS:
            sub_chapters = _split_chapter(ch, lang)
            result.extend(sub_chapters)
        else:
            result.append(ch)

    # Re-index
    for i, ch in enumerate(result):
        ch.index = i

    return result


def _split_chapter(chapter: Chapter, lang: str) -> List[Chapter]:
    """Split a chapter that exceeds the max token limit at sentence boundaries."""
    sentences = split_sentences(chapter.text, lang=lang)
    if not sentences:
        return [chapter]

    target_words = int(_MAX_CHAPTER_TOKENS / _WORD_TOKEN_RATIO)
    parts = []
    current_sentences = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent["text"].split())
        if current_words + sent_words > target_words and current_sentences:
            # Flush current part
            part_text = " ".join(s["text"] for s in current_sentences)
            part_words = len(part_text.split())
            part_start = chapter.start_char + current_sentences[0]["start"]
            part_end = chapter.start_char + current_sentences[-1]["end"]
            suffix = f" (part {len(parts) + 1})"
            parts.append(
                Chapter(
                    index=0,
                    title=chapter.title + suffix,
                    start_char=part_start,
                    end_char=part_end,
                    text=part_text,
                    word_count=part_words,
                    estimated_tokens=_estimate_tokens(part_words),
                )
            )
            current_sentences = []
            current_words = 0

        current_sentences.append(sent)
        current_words += sent_words

    # Flush remainder
    if current_sentences:
        part_text = " ".join(s["text"] for s in current_sentences)
        part_words = len(part_text.split())
        part_start = chapter.start_char + current_sentences[0]["start"]
        part_end = chapter.start_char + current_sentences[-1]["end"]
        if parts:
            suffix = f" (part {len(parts) + 1})"
        else:
            suffix = ""
        parts.append(
            Chapter(
                index=0,
                title=chapter.title + suffix,
                start_char=part_start,
                end_char=part_end,
                text=part_text,
                word_count=part_words,
                estimated_tokens=_estimate_tokens(part_words),
            )
        )

    return parts if parts else [chapter]


# ── Fallback: token-based chunks ──────────────────────────────────────────


def _fallback_token_chunks(text: str, lang: str) -> List[Chapter]:
    """Fall back to sentence-aligned token chunks when chapter detection fails."""
    chunks = build_chunks(text, lang=lang, max_tokens=_FALLBACK_CHUNK_WORDS, overlap=0)
    if not chunks:
        word_count = len(text.split())
        return [
            Chapter(
                index=0,
                title="Section 1",
                start_char=0,
                end_char=len(text),
                text=text,
                word_count=word_count,
                estimated_tokens=_estimate_tokens(word_count),
            )
        ]

    chapters = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        word_count = len(chunk_text.split())
        chapters.append(
            Chapter(
                index=i,
                title=f"Section {i + 1}",
                start_char=chunk.get("start_char", 0),
                end_char=chunk.get("end_char", len(text)),
                text=chunk_text,
                word_count=word_count,
                estimated_tokens=_estimate_tokens(word_count),
            )
        )

    return chapters


# ── Public API ────────────────────────────────────────────────────────────


async def detect_chapters(
    text: str,
    lang: str = "fr",
    llm_client: Optional[Any] = None,
) -> List[Chapter]:
    """Detect chapter boundaries using a hybrid strategy.

    1. Regex fast-path (instant, handles "Chapter N" / "Chapitre N")
    2. LLM fallback on structural skeleton (~15s, handles all formats)
    3. Token-based chunks as last resort

    Args:
        text: Full cleaned text of the book.
        lang: Detected language code (e.g. "fr", "en").
        llm_client: Optional LLMClient for the LLM fallback path.

    Returns:
        List of Chapter objects with text boundaries and metadata.
    """
    if not text or not text.strip():
        return []

    total_len = len(text)

    # Step 1: Regex fast-path
    chapters = _detect_by_regex(text)
    if _is_valid_split(chapters, total_len):
        logger.info("Chapter detection (regex): found %d chapters", len(chapters))
        return _enforce_size_bounds(chapters, lang)

    # Step 2: LLM fallback
    if llm_client is not None:
        chapters = await _detect_by_llm(text, lang, llm_client)
        if _is_valid_split(chapters, total_len):
            logger.info("Chapter detection (LLM): found %d chapters", len(chapters))
            return _enforce_size_bounds(chapters, lang)

    # Step 3: Token-based fallback
    chapters = _fallback_token_chunks(text, lang)
    logger.info(
        "Chapter detection (fallback): split into %d token chunks",
        len(chapters),
    )
    return chapters
