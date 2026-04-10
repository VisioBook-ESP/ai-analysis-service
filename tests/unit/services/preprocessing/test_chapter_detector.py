"""Tests for the hybrid chapter detector."""

from src.services.preprocessing.chapter_detector import (
    Chapter,
    _build_structural_skeleton,
    _detect_by_regex,
    _enforce_size_bounds,
    _fallback_token_chunks,
    _is_valid_split,
    detect_chapters,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_french_book(num_chapters: int = 5, words_per_chapter: int = 2000) -> str:
    """Build a synthetic French book with clear chapter markers."""
    parts = []
    # TOC
    parts.append("Table des matieres\n")
    for i in range(1, num_chapters + 1):
        parts.append(f"Chapitre {_to_roman(i)} .... page {i * 10}\n")
    parts.append("\n")

    # Body
    filler = "Le soleil brillait sur la campagne verdoyante et les oiseaux chantaient. "
    for i in range(1, num_chapters + 1):
        parts.append(f"\nChapitre {_to_roman(i)}\n\n")
        # Generate enough words
        repeat = max(1, words_per_chapter // 10)
        parts.append(filler * repeat + "\n")

    return "".join(parts)


def _make_english_book(num_chapters: int = 4, words_per_chapter: int = 2000) -> str:
    """Build a synthetic English book with clear chapter markers."""
    parts = []
    filler = "The sun was shining brightly over the rolling hills and green meadows. "
    for i in range(1, num_chapters + 1):
        parts.append(f"\nChapter {i}\n\n")
        repeat = max(1, words_per_chapter // 11)
        parts.append(filler * repeat + "\n")
    return "".join(parts)


def _to_roman(n: int) -> str:
    vals = [
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = ""
    for val, numeral in vals:
        while n >= val:
            result += numeral
            n -= val
    return result


# ── _detect_by_regex ──────────────────────────────────────────────────────


class TestDetectByRegex:
    def test_detects_french_chapters(self):
        text = _make_french_book(num_chapters=5, words_per_chapter=2000)
        chapters = _detect_by_regex(text)
        assert len(chapters) == 5
        assert "Chapitre I" in chapters[0].title
        assert "Chapitre V" in chapters[4].title

    def test_detects_english_chapters(self):
        text = _make_english_book(num_chapters=4, words_per_chapter=2000)
        chapters = _detect_by_regex(text)
        assert len(chapters) == 4

    def test_skips_toc_entries(self):
        """Chapter markers in the TOC region (first 5%) should not be used
        when body markers exist."""
        text = _make_french_book(num_chapters=5, words_per_chapter=3000)
        chapters = _detect_by_regex(text)
        # Should find body chapters, not TOC entries
        for ch in chapters:
            assert ch.word_count > 100  # Body chapters have substantial text

    def test_returns_empty_for_no_chapters(self):
        text = "This is just a plain text without any chapter markers. " * 100
        chapters = _detect_by_regex(text)
        assert chapters == []

    def test_chapter_text_boundaries(self):
        text = _make_english_book(num_chapters=3, words_per_chapter=1500)
        chapters = _detect_by_regex(text)
        assert len(chapters) == 3
        # Each chapter should have substantial text
        for ch in chapters:
            assert ch.word_count > 500
            assert len(ch.text) > 0
        # Chapters should cover the full text (no big gaps)
        total_covered = sum(len(ch.text) for ch in chapters)
        assert total_covered > len(text) * 0.8


# ── _is_valid_split ───────────────────────────────────────────────────────


class TestIsValidSplit:
    def test_valid_split(self):
        chapters = [
            Chapter(i, f"Ch {i}", 0, 100, "x " * 1500, 1500, 2100) for i in range(5)
        ]
        assert _is_valid_split(chapters, 10000) is True

    def test_too_few_chapters(self):
        chapters = [
            Chapter(0, "Ch 1", 0, 100, "x " * 1500, 1500, 2100),
            Chapter(1, "Ch 2", 100, 200, "x " * 1500, 1500, 2100),
        ]
        assert _is_valid_split(chapters, 10000) is False

    def test_empty_chapters(self):
        assert _is_valid_split([], 10000) is False

    def test_one_dominant_chapter(self):
        chapters = [
            Chapter(0, "Ch 1", 0, 8000, "x " * 4000, 4000, 5600),
            Chapter(1, "Ch 2", 8000, 8500, "x " * 250, 250, 350),
            Chapter(2, "Ch 3", 8500, 9000, "x " * 250, 250, 350),
        ]
        # Chapter 0 is >60% of total
        assert _is_valid_split(chapters, 9000) is False

    def test_avg_tokens_too_small(self):
        chapters = [Chapter(i, f"Ch {i}", 0, 10, "x " * 10, 10, 14) for i in range(5)]
        assert _is_valid_split(chapters, 1000) is False


# ── _enforce_size_bounds ──────────────────────────────────────────────────


class TestEnforceSizeBounds:
    def test_merges_small_chapters(self):
        chapters = [
            Chapter(0, "Ch 1", 0, 100, "small. " * 50, 50, 70),
            Chapter(1, "Ch 2", 100, 5000, "normal. " * 2000, 2000, 2800),
            Chapter(2, "Ch 3", 5000, 10000, "normal. " * 2000, 2000, 2800),
        ]
        result = _enforce_size_bounds(chapters, "en")
        # First small chapter should be merged with the next
        assert len(result) < len(chapters)

    def test_preserves_normal_chapters(self):
        chapters = [
            Chapter(i, f"Ch {i}", i * 5000, (i + 1) * 5000, "text. " * 2000, 2000, 2800)
            for i in range(4)
        ]
        result = _enforce_size_bounds(chapters, "en")
        assert len(result) == 4

    def test_reindexes_after_merge(self):
        chapters = [
            Chapter(0, "Ch 1", 0, 100, "small. " * 50, 50, 70),
            Chapter(1, "Ch 2", 100, 5000, "normal. " * 2000, 2000, 2800),
            Chapter(2, "Ch 3", 5000, 10000, "normal. " * 2000, 2000, 2800),
        ]
        result = _enforce_size_bounds(chapters, "en")
        for i, ch in enumerate(result):
            assert ch.index == i


# ── _build_structural_skeleton ────────────────────────────────────────────


class TestBuildStructuralSkeleton:
    def test_includes_short_lines(self):
        text = "Chapter 1\n\nThis is a very long paragraph. " * 50 + "\n\nChapter 2\n\n"
        skeleton = _build_structural_skeleton(text)
        assert "Chapter 1" in skeleton
        assert "Chapter 2" in skeleton

    def test_truncates_long_lines(self):
        text = "A" * 200 + "\n" + "Short line\n"
        skeleton = _build_structural_skeleton(text)
        assert "Short line" in skeleton
        # Long line should be truncated with ...
        assert "..." in skeleton

    def test_respects_max_chars(self):
        text = "Line\n" * 5000
        skeleton = _build_structural_skeleton(text, max_skeleton_chars=500)
        # Skeleton includes [LN] prefixes, so actual output is larger than max_skeleton_chars
        # but should still be bounded (not include all 5000 lines)
        assert len(skeleton) < 3000
        assert skeleton.count("[L") < 200


# ── _fallback_token_chunks ────────────────────────────────────────────────


class TestFallbackTokenChunks:
    def test_returns_chapters_with_section_titles(self):
        text = "word " * 10000
        chunks = _fallback_token_chunks(text, "en")
        assert len(chunks) >= 2
        assert chunks[0].title == "Section 1"
        assert chunks[1].title == "Section 2"

    def test_single_chunk_for_short_text(self):
        text = "Short text. " * 10
        chunks = _fallback_token_chunks(text, "en")
        assert len(chunks) >= 1
        assert chunks[0].title == "Section 1"


# ── detect_chapters (integration) ────────────────────────────────────────


class TestDetectChapters:
    async def test_regex_path_for_well_formatted_book(self):
        text = _make_french_book(num_chapters=6, words_per_chapter=2000)
        chapters = await detect_chapters(text, "fr", llm_client=None)
        assert len(chapters) >= 5

    async def test_fallback_for_plain_text(self):
        text = "word " * 10000  # No chapter markers
        chapters = await detect_chapters(text, "en", llm_client=None)
        # Should fall back to token chunks
        assert len(chapters) >= 2
        assert "Section" in chapters[0].title

    async def test_returns_empty_for_empty_text(self):
        chapters = await detect_chapters("", "en", llm_client=None)
        assert chapters == []

    async def test_returns_empty_for_whitespace(self):
        chapters = await detect_chapters("   \n\n   ", "en", llm_client=None)
        assert chapters == []

    async def test_chapter_objects_have_required_fields(self):
        text = _make_english_book(num_chapters=4, words_per_chapter=2000)
        chapters = await detect_chapters(text, "en", llm_client=None)
        for ch in chapters:
            assert isinstance(ch.index, int)
            assert isinstance(ch.title, str)
            assert isinstance(ch.text, str)
            assert ch.word_count > 0
            assert ch.estimated_tokens > 0
            assert ch.start_char >= 0
            assert ch.end_char > ch.start_char
