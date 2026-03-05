SYSTEM_PROMPT = """You are a strict literary text analysis engine. You extract structured information from text in JSON format.

CRITICAL RULES:
- You MUST respond with valid JSON only, no other text.
- NEVER invent, fabricate, or hallucinate information that is not explicitly stated or directly inferable from the text.
- You may deduce or infer information only when the text strongly implies it. Always distinguish between what is explicitly stated and what is inferred.
- If something is not mentioned or cannot be reasonably deduced, use "non specifie" or empty arrays. Do NOT fill in plausible but ungrounded information.
- Every piece of information you return must be traceable to a specific passage in the text.
- Analyze in the text's original language. Return analysis in the same language as the input text."""


def build_analysis_prompt(text: str, language: str, options: dict) -> str:
    """Build the user prompt requesting specific analysis sections."""
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
      "motivations": ["string"],
      "actions": ["string"],
      "relationships": [{"target": "string", "type": "string", "description": "string"}]
    }
  ]"""
        )

    if options.get("scenes"):
        sections.append(
            """"scenes": [
    {
      "scene_id": "scene_001",
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
    "emotional_arc": "string"
  }"""
    )

    if options.get("summary"):
        sections.append(
            """"summary": {
    "summary": "string (concise abstractive summary)",
    "key_points": ["string"]
  }"""
        )

    schema = "{\n  " + ",\n  ".join(sections) + "\n}"

    return f"""Analyze the following text and return a JSON object with this exact structure:

{schema}

STRICT RULES:
- Analyze in the text's language ({language}).
- ONLY extract information that is explicitly stated or directly inferable from the text.
- If a field cannot be filled from the text, use "non specifie" for strings or empty arrays for lists. NEVER guess or invent.
- For physical_description: only describe what the text explicitly mentions. If nothing is described, write "non decrit".
- For emotions/motivations: only include those clearly expressed or strongly implied by character actions/dialogue in the text.
- For atmosphere fields (lighting, weather, colors, sounds): only include what the text describes. Leave as "non specifie" or empty if not mentioned.
- polarity is a float between -1.0 (very negative) and 1.0 (very positive).
- Identify ALL characters mentioned in the text, including unnamed ones (describe them by their role).
- Detect scene breaks based on explicit changes in location, time, or narrative shifts in the text.

TEXT:
\"\"\"
{text}
\"\"\"

JSON:"""
