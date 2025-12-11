import re
from typing import Any

from .zero_shot import ZeroShotClassifier


class SceneExtractor:

    def __init__(self):
        self.zero_shot = ZeroShotClassifier()
        self.visual_keywords = {
            "colors": [
                "rouge",
                "bleu",
                "vert",
                "jaune",
                "noir",
                "blanc",
                "gris",
                "rose",
                "violet",
                "orange",
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "gray",
                "pink",
                "purple",
                "orange",
            ],
            "lighting": [
                "lumineux",
                "lumineuse",
                "lumineuses",
                "sombre",
                "sombres",
                "clair",
                "claire",
                "claires",
                "obscur",
                "obscure",
                "obscures",
                "éclairé",
                "éclairée",
                "éclairées",
                "ombre",
                "soleil",
                "lune",
                "bright",
                "dark",
                "light",
                "shadow",
                "sun",
                "moon",
                "illuminated",
                "dim",
                "radiant",
            ],
            "places": [
                "salle",
                "chambre",
                "jardin",
                "rue",
                "forêt",
                "maison",
                "pièce",
                "dehors",
                "intérieur",
                "room",
                "garden",
                "street",
                "forest",
                "house",
                "outside",
                "inside",
            ],
            "actions": [
                "entre",
                "sort",
                "marche",
                "court",
                "regarde",
                "parle",
                "sourit",
                "pleure",
                "enter",
                "exit",
                "walk",
                "run",
                "look",
                "speak",
                "smile",
                "cry",
            ],
        }

    def extract(self, preprocessed: dict, semantic_data: dict | None = None) -> dict[str, Any]:
        sentences = preprocessed["sentences"]

        scenes = self._detect_scenes(sentences, semantic_data)

        for scene in scenes:
            scene["characters"] = self._extract_characters(scene, semantic_data)
            scene["setting"] = self._extract_setting(scene)
            scene["atmosphere"] = self.zero_shot.classify_atmosphere(scene["text"])
            scene["objects"] = self._extract_objects(scene)
            scene["actions"] = self._extract_actions(scene)
            scene["visual_attributes"] = self._extract_visual_attributes(scene)

        return {"scene_count": len(scenes), "scenes": scenes}

    def _detect_scenes(self, sentences: list[dict], semantic_data: dict | None) -> list[dict]:
        scenes = []
        current_scene = {
            "scene_id": f"scene_{len(scenes)+1:03d}",
            "sentences": [],
            "text": "",
        }

        for i, sentence in enumerate(sentences):
            current_scene["sentences"].append(sentence)

            if self._is_scene_break(sentence, i, sentences):
                current_scene["text"] = " ".join([s["text"] for s in current_scene["sentences"]])
                scenes.append(current_scene)

                current_scene = {
                    "scene_id": f"scene_{len(scenes)+1:03d}",
                    "sentences": [],
                    "text": "",
                }

        if current_scene["sentences"]:
            current_scene["text"] = " ".join([s["text"] for s in current_scene["sentences"]])
            scenes.append(current_scene)

        return scenes

    def _is_scene_break(self, sentence: dict, index: int, all_sentences: list[dict]) -> bool:
        text = sentence["text"].lower()

        location_changes = [
            "entre dans",
            "sort de",
            "arrive à",
            "quitte",
            "va à",
            "enter",
            "leave",
            "arrive at",
            "go to",
        ]
        for change in location_changes:
            if change in text:
                return True

        time_changes = [
            "plus tard",
            "le lendemain",
            "soudain",
            "ensuite",
            "later",
            "next day",
            "suddenly",
            "then",
        ]
        for change in time_changes:
            if change in text:
                return True

        if len(all_sentences) > index + 1:
            next_sentence = all_sentences[index + 1]["text"].lower()
            if next_sentence.startswith(("il", "elle", "ils", "elles", "he", "she", "they")):
                return False

        if index > 0 and index % 3 == 0:
            return True

        return False

    def _extract_characters(self, scene: dict, semantic_data: dict | None) -> list[dict]:
        characters = []

        # Extract generic character descriptions (e.g., "le jeune homme", "une belle femme")
        # This is the primary method - works well for literary texts
        generic_characters = self._extract_generic_characters(scene["text"])
        for generic_char in generic_characters:
            # Check if not already detected
            if not any(
                c["name"].lower() in generic_char["name"].lower()
                or generic_char["name"].lower() in c["name"].lower()
                for c in characters
            ):
                characters.append(generic_char)

        deduplicated = self._deduplicate_characters(characters)

        for char in deduplicated:
            char["traits"] = self._extract_character_traits(char["name"], scene["text"])
            char["emotions"] = self._extract_character_emotions(char["name"], scene["text"])

        return deduplicated[:5]

    def _extract_generic_characters(self, text: str) -> list[dict]:
        """Extract generic character descriptions like 'le jeune homme', 'une belle femme'."""
        generic_chars = []

        # First, extract proper names with descriptions (e.g., "Sophie, une belle jeune femme")
        comma_pattern = r"([A-Z][a-z]+)\s*,\s*(un |une )((jeune |vieux |belle |beau |grand |petit |élégant |élégante )+)?(homme|femme|fille|garçon)"
        comma_matches = list(re.finditer(comma_pattern, text, re.IGNORECASE))

        # Track positions of comma descriptions to skip them later
        comma_description_positions = []
        for match in comma_matches:
            name = match.group(1)
            generic_chars.append({"name": name, "traits": [], "emotions": []})
            # Mark the description part after comma to skip it
            comma_start = match.start() + len(name)
            comma_description_positions.append((comma_start, match.end()))

        # Then extract generic descriptions (but skip those already found after commas)
        generic_patterns = [
            # French patterns
            r"(le |la |l\'|un |une )?((jeune |vieux |vieille |grand |grande |petit |petite |bel |belle )+)?(homme|femme|fille|garçon|enfant|personne)",
            # English patterns
            r"(the |a |an )?((young |old |tall |short |beautiful |handsome )+)?(man|woman|girl|boy|person)",
        ]

        for pattern in generic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if this match overlaps with a comma description
                is_comma_description = any(
                    start <= match.start() < end or start < match.end() <= end
                    for start, end in comma_description_positions
                )

                if not is_comma_description:
                    full_match = match.group(0).strip()
                    clean_name = full_match.strip()
                    if clean_name and len(clean_name) > 3:
                        # Capitalize first letter for consistency
                        clean_name = clean_name[0].upper() + clean_name[1:]
                        generic_chars.append({"name": clean_name, "traits": [], "emotions": []})

        return generic_chars

    def _is_valid_character_name(self, name: str, text: str) -> bool:
        """Check if a capitalized word is truly a character name."""
        name_lower = name.lower()

        # Exclude common words that start with capital letters
        excluded_words = {
            "il",
            "elle",
            "ils",
            "elles",
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "ce",
            "cet",
            "cette",
            "ces",
            "son",
            "sa",
            "ses",
            "leur",
            "leurs",
            "plus",
            "tout",
            "tous",
            "toute",
            "toutes",
            "autre",
            "autres",
            "he",
            "she",
            "they",
            "the",
            "a",
            "an",
            "his",
            "her",
            "their",
            "this",
            "that",
            # Conjonctions et adverbes souvent en début de phrase
            "lorsque",
            "quand",
            "alors",
            "mais",
            "donc",
            "car",
            "puis",
            "ensuite",
            "pourquoi",
            "comment",
            "où",
            "qui",
            "que",
            "quoi",
            "when",
            "then",
            "but",
            "so",
            "because",
            "why",
            "how",
            "where",
            "who",
            "what",
            # Noms propres géographiques communs (doivent être filtrés si pas des personnages)
            "forêt",
            "vierge",
            "histoires",
            "chine",
            "arizona",
            "ça",
            "mon",
            "ma",
            "mes",
        }

        if name_lower in excluded_words or len(name) <= 2:
            return False

        # Exclude words that are part of common French/English expressions
        common_expressions = [
            "intelligence artificielle",
            "artificial intelligence",
            "aujourd'hui",
            "aujourd hui",
            "institut de recherche",
            "research institute",
        ]

        text_lower = text.lower()
        for expr in common_expressions:
            if name_lower in expr and expr in text_lower:
                return False

        # Exclude words that appear at the start of sentences but aren't names
        # Check if the word is followed by common character actions/verbs
        character_verbs = [
            "est",
            "entre",
            "sort",
            "marche",
            "regarde",
            "parle",
            "sourit",
            "dit",
            "prend",
            "is",
            "enters",
            "exits",
            "walks",
            "looks",
            "speaks",
            "smiles",
            "says",
            "takes",
        ]

        # Build pattern to check if name appears with character-like context
        has_character_context = False
        name_pattern = rf"\b{re.escape(name)}\b"
        matches = list(re.finditer(name_pattern, text, re.IGNORECASE))

        for match in matches:
            # Extract context around the name (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].lower()

            # Check if any character verb appears near the name
            for verb in character_verbs:
                if verb in context:
                    has_character_context = True
                    break

            if has_character_context:
                break

        return has_character_context

    def _deduplicate_characters(self, characters: list[dict]) -> list[dict]:
        if not characters:
            return []

        invalid_names = {
            "institut",
            "recherche",
            "intelligence",
            "paris",
            "dubois",
            "martin",
        }

        filtered = []
        for char in characters:
            name_lower = char["name"].lower()
            if name_lower not in invalid_names and len(char["name"]) > 2:
                filtered.append(char)

        # Sort: proper names (capitalized single words) first, then by length
        def sort_key(char):
            name = char["name"]
            # Check if it's a proper name (single capitalized word, no spaces)
            is_proper_name = name[0].isupper() and " " not in name and "," not in name
            # Proper names get priority (0), generic descriptions get (1)
            return (0 if is_proper_name else 1, -len(name))

        sorted_chars = sorted(filtered, key=sort_key)

        deduplicated = []
        seen_names = set()

        for char in sorted_chars:
            name_lower = char["name"].lower()

            is_duplicate = False
            for seen in seen_names:
                # Check if names overlap (one contains the other)
                if name_lower in seen or seen in name_lower:
                    # Keep the shorter proper name over longer generic description
                    if len(name_lower) <= len(seen):
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(char)
                seen_names.add(name_lower)

        return deduplicated

    def _has_physical_description(self, name: str, text: str) -> bool:
        """Check if text contains physical description context for the character."""
        text_lower = text.lower()
        name_lower = name.lower()

        # Description indicators
        description_patterns = [
            rf"{name_lower}[^.]*?(cheveux|hair|yeux|eyes)",
            rf"{name_lower}[^.]*?(jeune|vieux|grand|petit|young|old|tall|short)",
            rf"{name_lower}[^.]*?(élégant|beau|belle|handsome|beautiful|elegant)",
            rf"{name_lower}[^.]*?(porte|vêtu|habillé|wear|dressed)",
            rf"(jeune|vieux|grand|petit|young|old|tall|short)[^.]*?{name_lower}",
            rf"(cheveux|hair|yeux|eyes)[^.]*?{name_lower}",
        ]

        for pattern in description_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _validate_traits(self, ai_traits: list[str], name: str, text: str) -> list[str]:
        """Validate AI-predicted traits have some textual support."""
        validated = []

        # Remove contradictory traits
        contradictions = [
            {"elegant", "casual"},
            {"young", "old"},
            {"tall", "short"},
        ]

        # Filter contradictions - keep first occurrence only
        filtered_traits = []
        seen_groups = set()
        for trait in ai_traits:
            # Check which contradiction group this trait belongs to
            for group in contradictions:
                if trait in group:
                    if not any(g in seen_groups for g in group):
                        filtered_traits.append(trait)
                        seen_groups.update(group)
                    break
            else:
                # Trait not in any contradiction group
                filtered_traits.append(trait)

        # Validate traits have contextual support
        for trait in filtered_traits:
            # Check if trait or related words appear near character name
            trait_context = self._extract_character_context_window(name, text, window=100)

            # Semantic validation: trait should appear in context or have synonyms
            if self._has_semantic_support(trait, trait_context):
                validated.append(trait)

        return validated[:5]  # Limit to top 5

    def _extract_character_context_window(self, name: str, text: str, window: int = 100) -> str:
        """Extract text window around character mentions."""
        import re

        name_lower = name.lower()
        text_lower = text.lower()

        contexts = []
        for match in re.finditer(rf"\b{re.escape(name_lower)}\b", text_lower):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            contexts.append(text[start:end])

        return " ".join(contexts) if contexts else text[:200]

    def _has_semantic_support(self, trait: str, context: str) -> bool:
        """Check if trait has semantic support in context."""
        context_lower = context.lower()

        # Semantic mappings - traits and their textual indicators
        trait_indicators = {
            "young": ["jeune", "young", "adolescent", "teenager", "youthful"],
            "old": ["vieux", "vieille", "âgé", "old", "elderly", "aged"],
            "tall": ["grand", "grande", "tall", "haut"],
            "short": ["petit", "petite", "short", "bas"],
            "elegant": ["élégant", "élégante", "elegant", "raffiné", "chic"],
            "casual": ["décontracté", "casual", "simple", "informel"],
            "beautiful": ["beau", "belle", "beautiful", "joli", "jolie", "magnifique"],
            "handsome": ["beau", "handsome", "séduisant"],
            "slim": ["mince", "slim", "svelte", "fin", "fine"],
            "well-dressed": ["bien habillé", "well-dressed", "élégant", "chic"],
            "professional": ["professionnel", "professional", "sérieux", "formel"],
        }

        indicators = trait_indicators.get(trait, [trait])
        return any(indicator in context_lower for indicator in indicators)

    def _extract_character_traits(self, name: str, text: str) -> list[str]:
        # Hybrid approach: Regex (priority) + Zero-shot with strict validation
        traits = []

        # Extract context around the character name to check for descriptions
        has_description = self._has_physical_description(name, text)

        # Use Zero-shot only if there's clear description context
        if has_description:
            ai_traits = self.zero_shot.classify_traits(text, name)
            if ai_traits:
                # Validate AI traits have textual support
                validated_traits = self._validate_traits(ai_traits, name, text)
                traits.extend(validated_traits)

        clothing_pattern = rf"{name}[^.]*?(porte|vêtu|habillé|wear|dressed|in)[^.]*?(robe|costume|dress|suit|veste|jacket|chemise|shirt)"
        matches = re.finditer(clothing_pattern, text, re.IGNORECASE)
        for match in matches:
            full_match = match.group(0)
            color_pattern = r"\b(rouge|bleu|vert|jaune|noir|blanc|red|blue|green|yellow|black|white|gris|gray)\b"
            colors = re.findall(color_pattern, full_match, re.IGNORECASE)
            if colors:
                for color in colors:
                    traits.append(f"{color} clothing")

        appearance_words = {
            "cheveux": ["cheveux", "hair"],
            "yeux": ["yeux", "eyes"],
            "robe": ["robe", "dress"],
            "costume": ["costume", "suit"],
            "veste": ["veste", "jacket", "veston"],
            "pantalon": ["pantalon", "pants", "trousers"],
            "chemise": ["chemise", "shirt"],
            "chapeau": ["chapeau", "hat"],
            "lunettes": ["lunettes", "glasses"],
        }

        for keywords in appearance_words.values():
            for keyword in keywords:
                if keyword in text.lower():
                    pattern = rf"(\w+)\s+{keyword}"
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if match.lower() in self.visual_keywords["colors"]:
                            traits.append(f"{match} {keyword}")

        physical_traits = [
            "jeune",
            "vieux",
            "grand",
            "petit",
            "mince",
            "fort",
            "élégant",
            "simple",
            "moderne",
            "classique",
            "young",
            "old",
            "tall",
            "short",
            "slim",
            "strong",
            "elegant",
            "simple",
            "modern",
            "classic",
            "passionate",
            "beautiful",
        ]

        text_lower = text.lower()
        name_lower = name.lower()

        for trait in physical_traits:
            if trait in text_lower and name_lower in text_lower:
                pattern = rf"{name_lower}[^.]*?{trait}|{trait}[^.]*?{name_lower}"
                if re.search(pattern, text_lower, re.IGNORECASE):
                    traits.append(trait)

        return list(set(traits[:5]))

    def _extract_character_emotions(self, name: str, text: str) -> list[str]:
        ai_emotions = self.zero_shot.classify_emotions(text, character_name=name)

        if ai_emotions:
            # Filter out contradictory emotions
            filtered = self._filter_contradictory_emotions(ai_emotions, text)
            if filtered:
                return filtered

        emotions = []
        emotion_words = {
            "heureux": [
                "heureux",
                "joyeux",
                "content",
                "sourit",
                "rit",
                "célébr",
                "fier",
                "merveilleux",
                "réjoui",
                "excité",
            ],
            "triste": [
                "triste",
                "pleure",
                "mélancolique",
                "déçu",
                "désespéré",
                "malheureux",
            ],
            "surpris": ["surpris", "étonné", "choqué", "stupéfait", "abasourdi"],
            "nerveux": ["nerveux", "anxieux", "inquiet", "stressé", "tendu"],
            "en colère": ["fâché", "énervé", "furieux", "irrité", "colère"],
            "calme": ["calme", "serein", "paisible", "tranquille", "détendu"],
            "passionné": ["passionné", "enthousiaste", "motivé", "inspiré"],
            "encourageant": ["encourageant", "soutien", "réconfortant", "bienveillant"],
        }

        text_lower = text.lower()
        name_lower = name.lower()

        for emotion, keywords in emotion_words.items():
            for keyword in keywords:
                if keyword in text_lower:
                    pattern = rf"{name_lower}[^.]*?{keyword}|{keyword}[^.]*?{name_lower}"
                    if re.search(pattern, text_lower, re.IGNORECASE) or name_lower in text_lower and keyword in text_lower:
                        emotions.append(emotion)
                        break

        return list(set(emotions[:3]))

    def _filter_contradictory_emotions(self, emotions: list[str], text: str) -> list[str]:
        """Filter out emotions that contradict explicit text cues."""
        text_lower = text.lower()

        # Positive indicators
        positive_cues = [
            "sourit",
            "rit",
            "joyeux",
            "heureux",
            "content",
            "smile",
            "laugh",
            "happy",
            "cheerful",
        ]
        # Negative indicators
        negative_cues = [
            "pleure",
            "triste",
            "malheureux",
            "cry",
            "sad",
            "unhappy",
            "worried",
            "anxious",
        ]

        has_positive = any(cue in text_lower for cue in positive_cues)
        has_negative = any(cue in text_lower for cue in negative_cues)

        # If text has clear positive cues, remove negative emotions
        if has_positive and not has_negative:
            positive_emotions = {
                "happy",
                "joyful",
                "excited",
                "proud",
                "calm",
                "passionate",
                "encouraging",
            }
            emotions = [e for e in emotions if e in positive_emotions]

        # If text has clear negative cues, remove positive emotions
        elif has_negative and not has_positive:
            negative_emotions = {"sad", "worried", "fearful", "angry", "nervous"}
            emotions = [e for e in emotions if e in negative_emotions]

        return emotions[:3]

    def _extract_setting(self, scene: dict) -> dict[str, Any]:
        text = scene["text"].lower()

        location = "unknown"
        for place in self.visual_keywords["places"]:
            if place in text:
                location = place
                break

        time_of_day = self._extract_time(text)

        lighting = []
        for light in self.visual_keywords["lighting"]:
            if light in text:
                lighting.append(light)

        return {
            "location": location,
            "time_of_day": time_of_day,
            "lighting": lighting[:2],  # Keep top 2 lighting descriptions
        }

    def _extract_time(self, text: str) -> str:
        times = {
            "matin": ["matin", "aube", "morning", "dawn"],
            "après-midi": ["après-midi", "midi", "afternoon", "noon"],
            "soir": ["soir", "crépuscule", "evening", "dusk"],
            "nuit": ["nuit", "minuit", "night", "midnight"],
        }

        for time_name, keywords in times.items():
            for keyword in keywords:
                if keyword in text:
                    return time_name

        return "unknown"

    def _extract_objects(self, scene: dict) -> list[str]:
        text = scene["text"]

        object_patterns = [
            r"\b(table|chaise|fenêtre|porte|lit|livre|verre|lampe)\b",
            r"\b(table|chair|window|door|bed|book|glass|lamp)\b",
        ]

        objects = []
        for pattern in object_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objects.extend(matches)

        return list(set(objects))[:5]

    def _extract_actions(self, scene: dict) -> list[str]:
        text = scene["text"].lower()

        actions = []
        for action in self.visual_keywords["actions"]:
            if action in text:
                actions.append(action)

        return actions[:5]

    def _extract_visual_attributes(self, scene: dict) -> dict[str, list[str]]:
        text = scene["text"].lower()

        colors = [color for color in self.visual_keywords["colors"] if color in text]
        lighting = [light for light in self.visual_keywords["lighting"] if light in text]

        return {"colors": colors[:3], "lighting": lighting[:2]}

    def _generate_visual_prompt(self, scene: dict) -> str:
        parts = []

        if scene["characters"]:
            char_desc = []
            for char in scene["characters"][:2]:
                desc = char["name"]
                if char["traits"]:
                    desc += f" with {', '.join(char['traits'][:2])}"
                char_desc.append(desc)
            parts.append(", ".join(char_desc))

        if scene["actions"]:
            parts.append(", ".join(scene["actions"][:2]))

        setting = scene["setting"]
        if setting["location"] != "unknown":
            parts.append(f"in {setting['location']}")

        if setting["atmosphere"] != "neutral":
            parts.append(setting["atmosphere"])

        if scene["visual_attributes"]["colors"]:
            parts.append(f"{scene['visual_attributes']['colors'][0]} tones")

        prompt = ", ".join(parts)
        if not prompt:
            prompt = scene["text"][:100]

        return prompt
