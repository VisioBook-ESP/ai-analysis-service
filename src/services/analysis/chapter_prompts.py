"""Map and reduce prompts for chapter-based chunked analysis.

Map (legacy single-pass): per-chapter extraction of scenes, characters, key events.
Pass 1: per-chapter extraction of dialogues, narrative blocks, audio cues.
Pass 2: compose visual scenes from Pass 1 structured data.
Reduce: global synthesis (character dedup, narrative, sentiment, summary)
from structured chapter results.
"""

import json
from typing import Any, Dict, List

# ── Map phase: per-chapter extraction ─────────────────────────────────────

MAP_SYSTEM_PROMPT = """You are a strict literary text analysis engine. You extract structured information from a SINGLE CHAPTER of a larger work, in JSON format.

CRITICAL RULES:
- You MUST respond with valid JSON only, no other text.
- NEVER invent, fabricate, or hallucinate information that is not explicitly stated or directly inferable from the text.
- If something is not mentioned or cannot be reasonably deduced, use "non specifie" or empty arrays.
- Every piece of information you return must be traceable to a specific passage in the chapter.
- Analyze in the text's original language. Return analysis in the same language as the input text."""


def build_map_prompt(
    chapter_text: str,
    chapter_title: str,
    chapter_index: int,
    total_chapters: int,
    language: str,
    options: dict,
) -> str:
    """Build the user prompt for per-chapter extraction (map phase)."""
    sections = []

    if options.get("characters"):
        sections.append(
            """"characters": [
    {
      "name": "string",
      "role": "protagonist|antagonist|secondary|narrator|mentioned",
      "physical_description": "string or 'non decrit'",
      "personality_traits": ["string"],
      "emotions": ["string"],
      "actions": ["string"]
    }
  ]"""
        )

    if options.get("scenes"):
        sections.append(
            """"scenes": [
    {
      "scene_id": "string",
      "title": "string",
      "text_excerpt": "string (first sentence of the scene)",
      "characters_present": ["string"],
      "setting": {
        "location": "string",
        "time_period": "string",
        "time_of_day": "string or 'unspecified'"
      },
      "atmosphere": {
        "mood": "string",
        "lighting": "string",
        "weather": "string or 'unspecified'",
        "colors": ["string"],
        "sounds_textures": {
          "sounds": ["string"],
          "textures": ["string"]
        }
      },
      "key_events": ["string"],
      "objects": ["string"]
    }
  ]"""
        )

    sections.append('"key_events": ["string (major events in this chapter)"]')
    sections.append('"sentiment_hint": "positive|negative|mixed|neutral"')

    schema = "{\n  " + ",\n  ".join(sections) + "\n}"

    return f"""You are analyzing chapter {chapter_index + 1} of {total_chapters}: "{chapter_title}".

Extract structured information from THIS CHAPTER ONLY. Return a JSON object with this exact structure:

{schema}

STRICT RULES:
- Analyze in the text's language ({language}).
- ONLY extract information from this chapter. Do not invent content from other chapters.
- For physical_description: only describe what the text explicitly mentions. If nothing is described, write "non decrit".
- For emotions: only include those clearly expressed or strongly implied by character actions/dialogue.
- For atmosphere fields: only include what the text describes. Leave as "non specifie" or empty if not mentioned.
- Identify ALL characters mentioned in this chapter, including unnamed ones (describe them by their role).
- Detect scene breaks based on changes in location, time, or narrative shifts within this chapter.

CHAPTER TEXT:
\"\"\"
{chapter_text}
\"\"\"

JSON:"""


# ── Pass 1: Dialogue & structure extraction ──────────────────────────────

PASS1_SYSTEM_PROMPT = """You are a strict literary text extraction engine. You extract dialogues, narrative blocks, and audio cues from a SINGLE CHAPTER in sequential order.

CRITICAL RULES:
- You MUST respond with valid JSON only, no other text.
- Extract EVERY dialogue line with EXACT QUOTED TEXT from the chapter. Do not paraphrase or summarize dialogue.
- Assign sequential `order` numbers to dialogues and narrative_blocks interleaved as they appear in the text.
- For `delivery`: describe HOW the line is spoken (screaming, whispering, laughing, confident, trembling, sarcastic, neutral...).
- For `context`: include the surrounding narration (1 sentence max) that frames the dialogue.
- For `audio_cues`: identify ambient sounds, sound effects, and music/mood cues described in the text.
- Analyze in the text's original language. Return analysis in the same language as the input text."""


def build_pass1_prompt(
    chapter_text: str,
    chapter_title: str,
    chapter_index: int,
    total_chapters: int,
    language: str,
) -> str:
    """Build the user prompt for Pass 1: dialogue & structure extraction."""
    return f"""You are analyzing chapter {chapter_index + 1} of {total_chapters}: "{chapter_title}".

Extract ALL dialogues, narrative blocks, and audio cues from THIS CHAPTER in sequential order. Return a JSON object with this exact structure:

{{
  "dialogues": [
    {{
      "order": 1,
      "speaker": "character name",
      "line": "exact quoted text from the book",
      "delivery": "how the line is spoken (trembling, confident, whispering, screaming, laughing, sarcastic, neutral...)",
      "context": "surrounding narration (1 sentence max)"
    }}
  ],
  "narrative_blocks": [
    {{
      "order": 2,
      "type": "description|action|inner_monologue|transition",
      "text": "first 2-3 sentences of the block",
      "characters_mentioned": ["character names"]
    }}
  ],
  "audio_cues": [
    {{
      "type": "ambient|sfx|music_mood",
      "description": "wind howling through trees",
      "source_block_order": 2
    }}
  ],
  "characters_in_chapter": ["Peter", "Shadow Man"],
  "sentiment_hint": "positive|negative|mixed|neutral"
}}

STRICT RULES:
- Analyze in the text's language ({language}).
- ONLY extract from this chapter. Do not invent content.
- Extract EVERY dialogue line with EXACT QUOTED TEXT. Do not paraphrase.
- `order` numbers must be sequential and interleaved: if a narrative block appears between two dialogues, its order reflects that position.
- `delivery` describes the emotional quality of speech (not the content).
- `type` for narrative_blocks: "description" for setting/environment, "action" for physical events, "inner_monologue" for thoughts, "transition" for time/location shifts.
- `audio_cues` reference the narrative_block `order` where the sound is described.
- `characters_in_chapter` lists ALL characters mentioned, including unnamed ones (describe by role).

CHAPTER TEXT:
\"\"\"
{chapter_text}
\"\"\"

JSON:"""


# ── Pass 2: Visual scene composition ────────────────────────────────────

PASS2_SYSTEM_PROMPT = """You are a visual scene composer for video production. You receive structured dialogue and narrative data from a chapter and compose VISUAL SCENES optimized for video generation.

CRITICAL RULES:
- You MUST respond with valid JSON only, no other text.
- Dialogue exchanges → MULTIPLE short scenes (1 scene per 2-3 dialogue exchanges, close-up/medium shots).
- Long description passages → ONE establishing/wide shot scene.
- Action sequences → short scenes (3-8 seconds each).
- Include ALL dialogue lines from the input — do not drop any.
- narration_text = non-dialogue narrative text for TTS narrator voice.
- Aim for 5-8 visual scenes per chapter.
- Analyze in the text's original language."""


def build_pass2_prompt(
    pass1_result: dict,
    chapter_title: str,
    chapter_index: int,
    total_chapters: int,
    language: str,
) -> str:
    """Build the user prompt for Pass 2: visual scene composition from Pass 1 data."""
    pass1_json = json.dumps(pass1_result, ensure_ascii=False, indent=1)

    return f"""You are composing visual scenes for chapter {chapter_index + 1} of {total_chapters}: "{chapter_title}".

Below is the structured extraction from Pass 1 (dialogues, narrative blocks, audio cues). Compose VISUAL SCENES for video production.

Return a JSON object with this exact structure:

{{
  "scenes": [
    {{
      "scene_id": "ch{chapter_index + 1}_s1",
      "scene_type": "dialogue|action|description|establishing",
      "title": "short descriptive title",
      "text_excerpt": "first sentence of the scene's source text",
      "characters_present": ["character names"],
      "setting": {{
        "location": "string",
        "time_period": "string",
        "time_of_day": "string or 'unspecified'"
      }},
      "atmosphere": {{
        "mood": "string",
        "lighting": "string",
        "weather": "string or 'unspecified'",
        "colors": ["string"],
        "sounds_textures": {{
          "sounds": ["string"],
          "textures": ["string"]
        }}
      }},
      "key_events": ["string"],
      "objects": ["string"],
      "audio_cues": [
        {{
          "type": "ambient|sfx|music_mood",
          "description": "string"
        }}
      ],
      "narration_text": "non-dialogue narrative text for TTS narrator",
      "dialogues": [
        {{
          "speaker": "character name",
          "line": "exact quoted text",
          "delivery": "confident"
        }}
      ],
      "source_block_orders": [1, 2, 3]
    }}
  ]
}}

COMPOSITION RULES:
- Analyze in the text's language ({language}).
- Dialogue exchanges (2-3 lines back-and-forth) → ONE scene with scene_type "dialogue" (close-up/medium shot framing).
- Long description passages → ONE scene with scene_type "description" or "establishing" (wide shot).
- Action sequences → SHORT scenes with scene_type "action" (3-8 seconds, dynamic angles).
- Include ALL dialogues from Pass 1 — assign each to exactly one scene. Do not drop any dialogue line.
- `narration_text` = non-dialogue text suitable for a narrator TTS voice. Can be empty for pure dialogue scenes.
- `audio_cues` = ambient sounds, SFX, music mood for this scene (from Pass 1 audio_cues or inferred from context).
- `source_block_orders` = which Pass 1 orders (dialogue + narrative_block) this scene covers.
- Aim for 5-8 visual scenes per chapter. Dialogue-heavy chapters may have more.

PASS 1 DATA:
\"\"\"
{pass1_json}
\"\"\"

JSON:"""


# ── Reduce phase: global synthesis ────────────────────────────────────────

REDUCE_SYSTEM_PROMPT = """You are a literary analysis synthesizer. You receive per-chapter analysis results from a book and produce a unified global analysis.

CRITICAL RULES:
- You MUST respond with valid JSON only, no other text.
- Deduplicate characters that appear across multiple chapters (same person mentioned under different names or references).
- Merge character traits, emotions, and actions from all chapters into a single consolidated entry per character.
- Add relationships between characters based on their co-occurrences and interactions across chapters.
- Produce a coherent narrative analysis, sentiment analysis, and summary that reflects the ENTIRE work, not individual chapters.
- Analyze in the text's original language."""


def build_reduce_prompt(
    chapter_results: List[Dict[str, Any]],
    language: str,
    options: dict,
) -> str:
    """Build the user prompt for global synthesis (reduce phase).

    Input: structured results from all map calls.
    Output schema: characters (merged), narrative, sentiment, summary.
    Scenes are NOT included — they are concatenated from map outputs in code.
    """
    sections = []

    if options.get("characters"):
        sections.append(
            """"characters": [
    {
      "name": "string (canonical name)",
      "role": "protagonist|antagonist|secondary|narrator|mentioned",
      "physical_description": "string (merged from all chapters)",
      "personality_traits": ["string (consolidated)"],
      "emotions": ["string (major emotions across the work)"],
      "motivations": ["string"],
      "actions": ["string (key actions across the work)"],
      "relationships": [{"target": "string", "type": "string", "description": "string"}],
      "voice_description": "string (describe the character's voice for TTS: pitch, tone, accent, speech patterns, e.g. 'deep gravelly voice with a slow deliberate cadence')"
    }
  ]"""
        )

    if options.get("narrative"):
        sections.append(
            """"narrative": {
    "themes": ["string"],
    "tone": "string",
    "style": "string",
    "point_of_view": "string",
    "tension_level": "low|medium|high|escalating|declining",
    "pacing": "string",
    "literary_devices": ["string"]
  }"""
        )

    sections.append(
        """"sentiment": {
    "overall": "positive|negative|mixed|neutral",
    "polarity": "float between -1.0 and 1.0",
    "nuances": ["string"],
    "emotional_arc": "string (describe the emotional journey across the whole work)"
  }"""
    )

    if options.get("summary"):
        sections.append(
            """"summary": {
    "summary": "string (concise abstractive summary of the ENTIRE work)",
    "key_points": ["string"]
  }"""
        )

    sections.append(
        """"audio_theme": {
    "overall_mood": "string (dominant audio mood across the work)",
    "recurring_sounds": ["string (sounds/ambiences that recur across chapters)"]
  }"""
    )

    schema = "{\n  " + ",\n  ".join(sections) + "\n}"

    # Format chapter results compactly for the reduce input
    chapters_input = []
    for i, cr in enumerate(chapter_results):
        chapter_summary: Dict[str, Any] = {
            "chapter": i + 1,
            "characters": cr.get("characters", []),
            "key_events": cr.get("key_events", []),
            "sentiment_hint": cr.get("sentiment_hint", "neutral"),
        }
        # Include characters_in_chapter from Pass 1 for dedup help
        if cr.get("characters_in_chapter"):
            chapter_summary["characters_in_chapter"] = cr["characters_in_chapter"]
        chapters_input.append(chapter_summary)

    chapters_json = json.dumps(chapters_input, ensure_ascii=False, indent=1)

    return f"""You are synthesizing the analysis of a complete literary work ({len(chapter_results)} chapters) into a unified analysis.

Below are the per-chapter extraction results (characters found, key events, sentiment).

Return a JSON object with this exact structure:

{schema}

STRICT RULES:
- Analyze in the text's language ({language}).
- Deduplicate characters: if "Peter" in chapter 1 and "Schlemihl" in chapter 5 are the same person, merge them under the canonical name.
- Consolidate personality traits, emotions, and actions across all chapters.
- Add relationships between characters based on their interactions.
- The summary should cover the ENTIRE work, not individual chapters.
- The emotional_arc should describe the sentiment progression across the whole work.
- polarity is a float between -1.0 (very negative) and 1.0 (very positive).

PER-CHAPTER RESULTS:
\"\"\"
{chapters_json}
\"\"\"

JSON:"""
