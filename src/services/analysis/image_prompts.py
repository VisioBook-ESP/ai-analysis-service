"""System prompt and builder for Flux/SDXL image generation prompts.

This module generates the second LLM call that transforms structured analysis
results into optimized prompts for ai-media-generation-service.
"""

IMAGE_PROMPT_SYSTEM_PROMPT = """You are a visual prompt engineer specialized in generating prompts for Flux and Stable Diffusion XL image generation models.

You receive a structured literary text analysis (characters, scenes, atmosphere, settings) and produce image generation prompts optimized for Flux/SDXL.

CRITICAL RULES:
1. You MUST respond with valid JSON only, no other text.
2. ALL prompts MUST be in English, regardless of the source text language.
3. Do NOT include the visual style (e.g. "realistic", "watercolor", "manga") in prompts — it will be added separately by the image generation pipeline.
4. Describe characters by their PHYSICAL APPEARANCE, never by name. Flux does not understand character names.
5. Use comma-separated visual descriptors: lighting, composition, camera angle, mood, colors, textures.
6. For character portrait prompts: describe the full body, include clothing, hair, eye color, build, distinguishing features.
7. For location prompts: describe the environment WITHOUT any people or characters present.
8. For scene prompts: describe the visual composition including characters (by appearance), setting, lighting, and mood.
9. Keep prompts concise but visually rich (40-80 words each).
10. Negative prompts should list visual artifacts to avoid (e.g. "text, watermark, blurry, bad anatomy").

EXAMPLES OF GOOD FLUX PROMPTS:

Scene prompt: "elderly man with white beard and weathered face sitting at a wooden desk in a dimly lit study, warm candlelight casting long shadows, leather-bound books on shelves, quill pen in hand, wide shot, volumetric lighting, rich warm tones"

Character portrait prompt: "young woman, long auburn hair reaching mid-back, bright green eyes, fair skin with light freckles, slender athletic build, wearing dark blue traveling cloak with silver clasp over white linen shirt, confident expression"

Location prompt: "ancient stone bridge over a misty river at dawn, moss-covered arches, willow trees on both banks, soft golden light breaking through morning fog, reflections in still water, serene atmosphere"

Negative prompt: "text, watermark, signature, blurry, low quality, duplicate, extra limbs, bad anatomy, cropped"
"""


def build_image_prompt_request(
    analysis_result: dict,
    visual_style: str,
    language: str,
) -> str:
    """Build the user prompt for the image prompt generation LLM call.

    Takes structured analysis results and requests Flux-optimized prompts
    for scenes, characters, and locations.
    """
    # Build character context
    characters_context = ""
    characters = analysis_result.get("characters", [])
    if characters:
        char_lines = []
        for c in characters:
            name = c.get("name", "Unknown")
            role = c.get("role", "")
            physical = c.get("physical_description", "")
            traits = ", ".join(c.get("personality_traits", []))
            char_lines.append(
                f"  - {name} ({role}): physical={physical}, traits={traits}"
            )
        characters_context = "CHARACTERS:\n" + "\n".join(char_lines)

    # Build scene context
    scenes_context = ""
    scenes = analysis_result.get("scenes", [])
    if scenes:
        scene_lines = []
        for i, s in enumerate(scenes):
            title = s.get("title", "")
            excerpt = s.get("text_excerpt", "")[:200]
            chars_present = ", ".join(s.get("characters_present", []))
            setting = s.get("setting", {})
            location = setting.get("location", "") if isinstance(setting, dict) else ""
            time_of_day = (
                setting.get("time_of_day", "") if isinstance(setting, dict) else ""
            )
            atmosphere = s.get("atmosphere", {})
            mood = atmosphere.get("mood", "") if isinstance(atmosphere, dict) else ""
            lighting = (
                atmosphere.get("lighting", "") if isinstance(atmosphere, dict) else ""
            )
            colors = (
                ", ".join(atmosphere.get("colors", []))
                if isinstance(atmosphere, dict)
                else ""
            )
            scene_lines.append(
                f"  Scene {i} - {title}:\n"
                f'    text: "{excerpt}"\n'
                f"    characters: [{chars_present}]\n"
                f"    location: {location}, time: {time_of_day}\n"
                f"    mood: {mood}, lighting: {lighting}, colors: {colors}"
            )
        scenes_context = "SCENES:\n" + "\n".join(scene_lines)

    # Build narrative context
    narrative = analysis_result.get("narrative", {})
    narrative_context = ""
    if narrative and isinstance(narrative, dict):
        tone = narrative.get("tone", "")
        themes = ", ".join(narrative.get("themes", []))
        narrative_context = f"NARRATIVE: tone={tone}, themes={themes}"

    return f"""Based on the following literary text analysis, generate Flux/SDXL-optimized image prompts.

The visual style for this project is "{visual_style}" — but do NOT include the style in the prompts.
The source text language is "{language}" — but ALL prompts must be in English.

{characters_context}

{scenes_context}

{narrative_context}

Generate a JSON object with this exact structure:

{{
  "scene_prompts": [
    {{
      "scene_order": 0,
      "image_prompt": "detailed English visual description for Flux image generation, comma-separated descriptors, 40-80 words",
      "negative_prompt": "visual artifacts to avoid",
      "characters_present": ["character name if present in scene"],
      "location_id": "a_short_snake_case_id for the location (e.g. forest_clearing, old_library)"
    }}
  ],
  "character_prompts": [
    {{
      "name": "character name (original language)",
      "physical_description": "concise English physical description for portrait reference, comma-separated",
      "portrait_prompt": "detailed English full-body character description for Flux portrait generation, include clothing, hair, eyes, build, expression, 40-60 words",
      "portrait_negative_prompt": "visual artifacts to avoid for portraits"
    }}
  ],
  "location_prompts": [
    {{
      "location_id": "matching snake_case id used in scene_prompts",
      "name": "location name (original language from text)",
      "description_prompt": "detailed English environment description for Flux location reference, NO people or characters, include architecture, vegetation, lighting, atmosphere, 40-60 words",
      "negative_prompt": "people, characters, text, modern objects",
      "source_scene_orders": [0, 2]
    }}
  ]
}}

RULES:
- One scene_prompt per scene, in order.
- One character_prompt per named character with a physical description.
- Extract unique locations from scenes. Multiple scenes can share a location_id.
- scene_prompts.characters_present must match character names from character_prompts.
- Describe characters in scene prompts by their physical appearance, not by name.
- location_prompts.source_scene_orders lists which scene orders use this location.

JSON:"""
