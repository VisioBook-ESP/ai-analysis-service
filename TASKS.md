# ai-analysis-service — Tasks

## Epic: Media Generation Prompt Pipeline

> Extend ai-analysis-service to generate Flux/SDXL-optimized prompts for scenes, character portraits, and locations via a second LLM call after text analysis. These enriched prompts flow through NATS to core-project-service, which dispatches them to ai-media-generation-service.

**Branch:** `dev`
**Depends on:** core-project-service changes (see core-project-service/TASKS.md)

---

### Phase 1: Database & Settings

- [ ] **A1 — New structured prompt tables**
  - Create Alembic migration `002_add_prompt_tables.py`
  - Add SQLAlchemy models: `ScenePrompt`, `CharacterPrompt`, `LocationPrompt`
  - All linked to `analysis_results` via FK (`analysis_id`)
  - Tables:
    - `scene_prompts`: analysis_id, scene_order, image_prompt, negative_prompt, characters_present (JSONB), location_id
    - `character_prompts`: analysis_id, name, physical_description, portrait_prompt, portrait_negative_prompt
    - `location_prompts`: analysis_id, location_id, name, description_prompt, negative_prompt, source_scene_orders (JSONB)
  - Files: `src/database/models.py`, `alembic/versions/002_add_prompt_tables.py`

- [ ] **A4 — Settings extension**
  - Add `prompt_gen_temperature`, `prompt_gen_max_tokens`, `prompt_gen_enabled` to settings
  - File: `src/config/settings.py`

### Phase 2: Prompt Generation Core

- [ ] **A2 — Image prompt system prompt & builder**
  - Create `IMAGE_PROMPT_SYSTEM_PROMPT` for Flux/SDXL visual prompt engineering
  - Create `build_image_prompt_request(analysis_result, visual_style, language)` function
  - Key requirements:
    - Always output English regardless of source text language
    - Do NOT include visual style (media service adds it)
    - Describe characters by physical appearance, never by name
    - Comma-separated visual descriptors (Flux style)
    - Include 2-3 few-shot examples
    - Link scenes to characters and locations
  - File: `src/services/analysis/image_prompts.py`

- [ ] **A3 — PromptGenerator class**
  - Create `PromptGenerator(llm_client)` wrapping the second LLM call
  - `generate(analysis_result, visual_style, language) -> dict`
  - Uses `LLMClient.chat_completion()` with configurable temperature/max_tokens
  - Response validation + normalization (follow `ResponseParser` patterns)
  - Handle batching for >15 scenes
  - File: `src/services/analysis/prompt_generator.py`

### Phase 3: Integration

- [ ] **A5 — Modify Analyzer to chain prompt generation**
  - Add `PromptGenerator` dependency
  - Add `generate_prompts: bool`, `visual_style: str` params to `analyze()`
  - Chain prompt gen after analysis, add `"prompt_generation"` step callback
  - Return `image_prompts` in result dict
  - File: `src/services/analysis/analyzer.py`

- [ ] **A6 — Modify WorkflowHandler for enriched output**
  - Extract `visual_style` from config
  - Update `_map_scenes()`: Flux-optimized imagePrompt (with fallback), negativePrompt, charactersPresent, locationId
  - Update `_map_characters()`: physicalDescription, portraitPrompt, portraitNegativePrompt
  - New `_map_locations()`: location prompt data
  - Add `locations` to NATS publish payload
  - Adjust progress percentages: 0% → 5% → 15% → 60% → 80% → 100%
  - Graceful degradation: prompt gen failure → fall back to basic prompts
  - File: `src/services/workflow_handler.py`

- [ ] **A7 — Database client — Persist to new tables**
  - Add methods: `save_scene_prompts()`, `save_character_prompts()`, `save_location_prompts()`
  - Called after successful prompt gen, linked to `analysis_results.id`
  - File: `src/clients/database_client.py`

### Phase 4: API & Tests

- [ ] **A8 — Update API response schemas**
  - Update Pydantic models for GET /results to include prompt data
  - File: `src/api/schemas/analysis.py`

- [ ] **A9 — Tests**
  - Unit: `image_prompts.py` (prompt building, few-shot examples)
  - Unit: `prompt_generator.py` (LLM response parsing, validation, fallbacks, batching)
  - Unit: enriched `_map_scenes`, `_map_characters`, `_map_locations`
  - Integration: full two-phase pipeline
  - Graceful degradation (prompt gen failure → fallback)
  - Files: `tests/unit/test_image_prompts.py`, `tests/unit/test_prompt_generator.py`

---

## Enriched NATS Payload Format (analysis.completed)

```json
{
  "projectId": "uuid",
  "versionId": "uuid",
  "executionId": "uuid",
  "userId": "uuid",
  "scenes": [{
    "order": 0,
    "text": "excerpt (original language)",
    "description": "scene title (original language)",
    "imagePrompt": "English Flux-optimized scene prompt",
    "negativePrompt": "scene-specific negative prompt",
    "duration": 5,
    "sentiment": "mysterious",
    "charactersPresent": ["Alice"],
    "locationId": "forest_clearing"
  }],
  "characters": [{
    "name": "Alice",
    "description": "protagoniste. description originale",
    "aliases": [],
    "traits": ["brave", "curious"],
    "physicalDescription": "young woman, long auburn hair, green eyes, fair skin",
    "portraitPrompt": "young woman, long auburn hair reaching shoulders, bright green eyes, fair freckled skin, dark blue traveling cloak",
    "portraitNegativePrompt": "multiple people, blurry, bad anatomy, extra limbs"
  }],
  "locations": [{
    "locationId": "forest_clearing",
    "name": "La clairiere sombre",
    "descriptionPrompt": "ancient forest clearing, tall oak trees, moss-covered ground, golden sunlight through canopy",
    "negativePrompt": "people, characters, text, modern objects",
    "sourceSceneOrders": [0, 2]
  }],
  "correlationId": "uuid"
}
```
