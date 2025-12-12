# AI Analysis Service

> Service FastAPI pour l'analyse sémantique de textes et la génération de prompts d'images — robuste, modulaire, prêt pour la prod.

---

## Sommaire

* [Vue d'ensemble](#vue-densemble)
* [Architecture globale](#architecture-globale)
* [Étape 1 — Preprocessing](#étape-1--preprocessing)
* [Étape 2 — AI Analysis](#étape-2--ai-analysis)

  * [Analyse sémantique](#analyse-sémantique)
  * [Extraction d'entités (NER)](#extraction-dentités-ner)
  * [Extraction de scènes](#extraction-de-scènes)
* [Étape 3 — Postprocessing](#étape-3--postprocessing)
* [API — Endpoints](#api--endpoints)
* [Optimisations GPU & Performance](#optimisations-gpu--performance)
* [Stack technique](#stack-technique)
* [Structure du projet](#structure-du-projet)
* [Workflow d'exécution](#workflow-dexécution)
* [Considérations importantes](#considérations-importantes)
* [Roadmap / Prochaines étapes](#roadmap--prochaines-étapes)
* [Questions ouvertes](#questions-ouvertes)
* [Licence](#licence)

---

## Vue d'ensemble

**AI Analysis Service** ingère un texte, l'analyse en profondeur (thèmes, entités, scènes), puis prépare des **prompts d'images** cohérents pour des générateurs visuels (Stable Diffusion, DALL·E, etc.).

---

## Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVICE (Port 8083)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ PREPROCESSING │    │  AI ANALYSIS   │    │ POSTPROCESS  │
│   (Étape 1)   │───▶│   (Étape 2)    │───▶│  (Étape 3)   │
└───────────────┘    └────────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   Texte nettoyé       Données enrichies      Résultats
   + métadonnées       + embeddings           structurés
                        + entités             pour images
```

---

## Étape 1 — Preprocessing

**Statut :** déjà bien avancé (intégration notebook → service).

**Composants clés**

* Détection de langue (langdetect)
* Nettoyage (unicode, HTML, masquage PII)
* Segmentation (phrases via spaCy)
* Chunking (fenêtres avec overlap pour transformers)
* Score qualité/bruit

**Arborescence suggérée**

```
src/services/preprocessing/
├── text_cleaner.py         # Nettoyage basique
├── language_detector.py    # Détection de langue
├── segmenter.py            # Segmentation phrases/chunks
├── quality_scorer.py       # Score qualité/bruit
└── preprocessor.py         # Orchestrateur
```

**Améliorations**

* Cache Redis pour résultats de preprocessing (hash de texte)
* Streaming des gros fichiers (éviter chargement complet en RAM)
* Validation robuste des encodages

---

## Étape 2 — AI Analysis

Cœur du service. Approche **multi-modèles parallèle** pour extraire le maximum d'information.

### Analyse sémantique

**Modèles FR** : CamemBERT (base/large), BARThez (résumé), FlauBERT.
**Secours multilingue** : XLM-RoBERTa-large, mT5.

**Tâches**

* Embeddings (sentence-transformers) → similarité, clustering
* Thèmes (BERTopic / LDA)
* Résumé (extractif + abstractif)
* Ton & sentiment (polarity/intensité)

### Extraction d'entités (NER)

**À extraire pour la génération d'images**

```
• Personnages (noms, traits physiques)
• Lieux (décors, environnements)
• Objets importants (accessoires)
• Actions/Événements (scènes clés)
• Atmosphère (luminosité, ambiance)
• Temps (époque, moment)
• Émotions dominantes
```

**Modèles**

* Standard (PER/LOC/ORG) : spaCy `fr_core_news_lg`, CamemBERT-NER
* Custom : fine-tuning domaine (personnages littéraires, descriptions)

**Arborescence**

```
src/models/
├── semantic_analysis/
│   ├── embedder.py              # Sentence embeddings
│   ├── topic_extractor.py       # Topics (BERTopic)
│   ├── sentiment_analyzer.py    # Sentiment
│   └── summarizer.py            # Résumé
│
├── entity_extraction/
│   ├── ner_engine.py            # NER standard
│   ├── character_extractor.py   # Personnages
│   ├── scene_descriptor.py      # Descriptions de scènes
│   └── attribute_parser.py      # Attributs visuels
│
└── scene_extraction/            # Présent
    ├── scene_detector.py
    └── entity_extractor.py
```

### Extraction de scènes

**Objectif :** découper en **scènes narratives cohérentes**.

**Ruptures narratives** : changement de lieu/temps/personnages, nouveaux dialogues/actions.
**Scoring de visualisabilité**

```json
{
  "visualizability_score": 0.85,
  "has_visual_descriptions": true,
  "has_actions": true,
  "descriptive_richness": 0.78,
  "ambiguity": 0.12
}
```

**Attributs visuels d'une scène**

```json
{
  "scene_id": "scene_001",
  "text": "...",
  "characters": [
    {"name": "Marie", "physical_traits": ["cheveux bruns", "robe bleue"], "emotions": ["surprise", "joie"], "position": "centre"}
  ],
  "setting": {"location": "jardin fleuri", "time_of_day": "après-midi", "weather": "ensoleillé", "atmosphere": "paisible"},
  "objects": ["banc en bois", "rosiers"],
  "actions": ["Marie s'assoit", "regarde le ciel"],
  "image_prompt_suggestion": "A woman with brown hair in a blue dress sitting on a wooden bench in a sunny flower garden, peaceful afternoon atmosphere, roses in background"
}
```

---

## Étape 3 — Postprocessing

**But :** transformer les sorties d'IA en structures **prêtes pour la génération d'images**.

**Agrégation — Structure finale**

```json
{
  "document_id": "uuid",
  "metadata": {"language": "fr", "total_scenes": 12, "processing_time_ms": 2847, "quality_score": 0.87},
  "global_analysis": {
    "summary": "...",
    "main_themes": ["aventure", "amitié", "découverte"],
    "overall_sentiment": {"polarity": 0.65, "subjectivity": 0.72},
    "key_characters": ["Marie", "Pierre"],
    "key_locations": ["forêt enchantée", "château"]
  },
  "scenes": [
    {
      "scene_id": "scene_001",
      "text_span": {"start": 0, "end": 324},
      "text": "...",
      "visual_data": {"characters": [], "setting": {}, "objects": [], "actions": [], "image_prompt": "..."},
      "metadata": {"duration_estimate_seconds": 15, "complexity": "medium", "shot_suggestions": ["establishing shot", "close-up"]}
    }
  ],
  "character_database": {
    "Marie": {"mentions": 15, "consistent_traits": ["cheveux bruns", "robe bleue"], "emotional_arc": [], "reference_image_prompt": "Young woman with brown hair, blue dress, kind expression..."}
  }
}
```

**Génération de prompts**

```
src/services/postprocessing/
├── scene_aggregator.py          # Fusion des analyses
├── character_consolidator.py    # Cohérence personnages
├── prompt_generator.py          # Prompts images
└── quality_validator.py         # Validation finale
```

**Templates de prompts**

```text
template = """
{character_description}, {action}, {setting}, {atmosphere},
{camera_angle}, {lighting}, {art_style},
high quality, detailed, {additional_tags}
"""

Exemple :
"Young woman with long brown hair and blue dress, sitting on wooden bench,
in a sunny flower garden with roses, peaceful afternoon atmosphere,
medium shot, soft natural lighting, realistic style,
high quality, detailed, 4k, professional photography"
```

---

## API — Endpoints

**Endpoint principal**

```http
POST /api/v1/analysis/full
{
  "text": "...",
  "options": {
    "extract_scenes": true,
    "generate_image_prompts": true,
    "target_image_count": 10,
    "style_hints": "realistic"
  }
}
```

**Endoints spécialisés**

```
POST /api/v1/analysis/semantic
POST /api/v1/analysis/extract-scenes
POST /api/v1/analysis/extract-entities
POST /api/v1/analysis/generate-prompts
```

**Asynchrones (gros textes)**

```
POST /api/v1/analysis/async/submit
GET  /api/v1/analysis/async/status/{job_id}
GET  /api/v1/analysis/async/result/{job_id}
```

**Utilitaires**

```
GET  /api/v1/models/list
POST /api/v1/models/reload
GET  /api/v1/health
```

---

## Optimisations GPU & Performance

**Gestion mémoire GPU — pattern**

```python
class ModelManager:
    """Gère le chargement lazy + cache GPU"""
    def __init__(self):
        self._models = {}
        self._gpu_pool = GPUMemoryPool()
    def get_model(self, model_id: str):
        if model_id not in self._models:
            self._load_to_gpu(model_id)
        return self._models[model_id]
    def _load_to_gpu(self, model_id: str):
        # Vérifie mémoire, éviction LRU, chargement
        ...
```

**Stratégies**

* Quantization (int8/fp16)
* Batch inference (multi-chunks)
* Parallélisation (NER + embeddings en parallèle)
* Distillation (modèles "distilled")

**Caching multi-niveaux**

1. Redis : résultats complets (TTL 1h)
2. GPU : modèles chargés
3. App : embeddings de chunks fréquents

**Asynchronisme (texte > 5k tokens)**

* Celery + Redis, workers GPU, webhooks de notification

---

## Stack technique

**Core** : FastAPI, Pydantic v2, Python 3.11+

**IA/NLP** : PyTorch 2.3+, Transformers, sentence-transformers, spaCy 3.7+, (optionnel : LangChain)

**Outils utiles** : BERTopic, KeyBERT, textstat

**Infra** : Redis, PostgreSQL + pgvector, Celery, Prometheus + Grafana

**Conteneurs** : Docker (+ CUDA), docker-compose (dev), Kubernetes (+ GPU Operator) en prod

---

## Structure du projet

```
ai-analysis-service/
├── src/
│   ├── api/
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   ├── routes/
│   │   │   ├── analysis.py
│   │   │   ├── async_jobs.py
│   │   │   ├── models.py
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── requests.py
│   │       └── responses.py
│   ├── services/
│   │   ├── preprocessing/
│   │   │   ├── text_cleaner.py
│   │   │   ├── segmenter.py
│   │   │   └── preprocessor.py
│   │   ├── analysis/
│   │   │   ├── semantic_analyzer.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── scene_detector.py
│   │   │   └── orchestrator.py
│   │   └── postprocessing/
│   │       ├── scene_aggregator.py
│   │       ├── prompt_generator.py
│   │       └── validator.py
│   ├── models/
│   │   ├── manager.py
│   │   ├── loaders/
│   │   │   ├── bert_loader.py
│   │   │   ├── spacy_loader.py
│   │   │   └── sentence_transformer_loader.py
│   │   └── wrappers/
│   │       ├── ner_wrapper.py
│   │       └── embedder_wrapper.py
│   ├── core/
│   │   ├── config.py
│   │   ├── cache.py
│   │   ├── gpu_utils.py
│   │   └── metrics.py
│   └── workers/
│       └── celery_app.py
├── tests/
├── notebooks/
├── models/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
└── helm/
```

---

## Workflow d'exécution

1. **POST** `/api/v1/analysis/full`
2. **API Gateway (FastAPI)** : validation Pydantic → check Redis (HIT = retour immédiat / MISS = suite)
3. **Preprocessing** : langue → nettoyage/segmentation → chunking → score qualité
4. **Analysis (GPU)** : sémantique (embeddings, topics, sentiment, résumé) + NER (pers./lieux/objets/actions) + scènes (ruptures + score)
5. **Postprocessing** : agrégation → consolidation personnages → prompts images → validation
6. **Réponse** : JSON structuré + cache + métriques Prometheus

---

## Considérations importantes

**Cohérence visuelle**

* Base de personnages : traits physiques et attributs consolidés
* Pondération d'attributs répétés
* Résolution de contradictions (ex. "cheveux bruns" puis "blonds")

**Qualité des prompts**

* Templates par style (réaliste, anime, BD, peinture)
* Mots-clés de composition ("close-up", "wide shot")
* Cohérence inter-scènes (apparence constante)

**Monitoring & Observabilité**

* Temps par étape, utilisation GPU, taux de cache hit
* Scores de qualité d'extraction, erreurs par modèle

---

## Roadmap / Prochaines étapes

**Phase 1 — MVP (2–3 j)** : intégrer preprocessing, endpoint `/analysis/preprocess`, tests basiques
**Phase 2 — Sémantique (3–4 j)** : embeddings, spaCy NER, endpoint `/analysis/semantic`
**Phase 3 — Scènes (4–5 j)** : scene detector, attributs visuels, endpoint `/analysis/extract-scenes`
**Phase 4 — Prompts (2–3 j)** : postprocessing + templates, endpoint `/analysis/generate-prompts`
**Phase 5 — Optims (3–4 j)** : Redis cache, jobs async, optims GPU
**Phase 6 — Prod (3–4 j)** : Docker GPU, monitoring, documentation API
*Total estimé : ~2–3 semaines.*

---

## Questions ouvertes

* Formats d'entrée : uniquement texte brut ou aussi PDF/DOCX/EPUB ?
* Volume typique : 1 page, 1 chapitre, 1 livre ?
* Style visuel cible : réaliste, anime, BD, peinture ?
* Temps réel vs batch : réponse immédiate ou délai accepté ?
* Contraintes GPU : type/mémoire disponibles ?
* Priorités d'approfondissement ?

---

## Licence

À définir (MIT/Apache-2.0/BSD-3-Clause…).
