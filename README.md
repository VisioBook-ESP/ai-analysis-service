# ai-analysis-service

Service d'analyse sémantique de textes via LLM. Extrait personnages, scènes, narrative, sentiment et résumé à partir d'un texte libre.

---

## Stack

- **API** : FastAPI 0.121 — Python 3.12
- **LLM** : vLLM (OpenAI-compatible) — Mistral Ministral-3B-Instruct
- **Preprocessing** : spaCy, langdetect
- **Database** : PostgreSQL (postgres-io / analysis) — SQLAlchemy + asyncpg + Alembic
- **Messaging** : NATS JetStream (workflow integration)
- **Déploiement** : Docker, Kubernetes (Helm + Istio)

---

## Architecture

```
Client / NATS JetStream
  │
  ▼
FastAPI :8083
  │
  ├── HTTP API (/analyze, /results, /jobs)
  │
  ├── NATS Subscriber (workflow.started)
  │     └── WorkflowHandler → analyse + persist + publish
  │
  ├── Preprocessing
  │     ├── Détection de langue (langdetect)
  │     ├── Nettoyage texte (unicode, PII, URLs)
  │     ├── Segmentation (spaCy)
  │     └── Score qualité
  │
  ├── LLM Client (httpx async)
  │     └── POST /v1/chat/completions → vLLM :8000
  │
  ├── Response Parser
  │     └── Validation + normalisation JSON
  │
  └── Database (PostgreSQL postgres-io/analysis)
        └── analysis_results (SQLAlchemy + asyncpg)
```

---

## Endpoints

### Santé

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Liveness — retourne `healthy` |
| GET | `/ready` | Readiness — vérifie vLLM + DB |
| GET | `/metrics` | CPU, RAM, disque |

### Analyse (async)

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/api/v1/analyze` | Soumet une analyse, retourne un `job_id` |
| GET | `/api/v1/jobs/{job_id}` | Statut et résultat du job |
| POST | `/api/v1/analyze/batch` | Analyse synchrone de plusieurs textes (max 50) |

### Résultats persistés (via workflow NATS)

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/api/v1/results/{execution_id}` | Résultat d'analyse persisté |
| GET | `/api/v1/results/project/{project_id}` | Liste des analyses d'un projet |

### NATS JetStream (workflow)

| Sujet | Direction | Description |
|-------|-----------|-------------|
| `visiobook.project.workflow.started` | Subscribe | Déclenche l'analyse (reçu de core-project-service) |
| `visiobook.ai.analysis.completed` | Publish | Résultat (scenes + characters) |
| `visiobook.ai.analysis.failed` | Publish | Erreur |
| `visiobook.ai.progress` | Publish | Progression (0%, 10%, 80%, 100%) |

---

## Flux d'une analyse

```
POST /api/v1/analyze
  → { job_id, status: "pending" }          # immédiat

GET /api/v1/jobs/{job_id}
  → { status: "processing", step: "preprocessing" }
  → { status: "processing", step: "llm_call" }
  → { status: "processing", step: "parsing" }
  → { status: "completed", result: { ... } }
```

**Étapes internes :**
1. `preprocessing` — nettoyage, détection langue, segmentation
2. `llm_call` — envoi au LLM (~20s pour un texte court)
3. `parsing` — validation et structuration de la réponse JSON

---

## Requête

```json
POST /api/v1/analyze
{
  "text": "Pierre et Marie se retrouvèrent dans le vieux café...",
  "language": "auto",
  "options": {
    "characters": true,
    "scenes": true,
    "narrative": true,
    "summary": true,
    "mask_pii": true,
    "remove_links": false,
    "max_summary_length": 200
  }
}
```

## Réponse (via GET /api/v1/jobs/{job_id})

```json
{
  "job_id": "e5ab9c5e-...",
  "status": "completed",
  "step": null,
  "result": {
    "language": "fr",
    "text_stats": {
      "original_length": 160,
      "cleaned_length": 159,
      "sentence_count": 3,
      "word_count": 29,
      "quality_score": 0.14,
      "quality_assessment": "excellent"
    },
    "characters": [...],
    "scenes": [...],
    "narrative": { "tone": "...", "themes": [...], ... },
    "sentiment": { "overall": "mixed", "polarity": -0.7, ... },
    "summary": { "summary": "...", "key_points": [...] },
    "processing_time_ms": 21043
  },
  "created_at": "2026-02-27T12:00:00",
  "updated_at": "2026-02-27T12:00:21"
}
```

---

## Configuration

Variables d'environnement (fichier `.env`) :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000` | URL du serveur vLLM |
| `VLLM_MODEL_NAME` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | Modèle à utiliser |
| `VLLM_API_KEY` | `EMPTY` | Clé API vLLM |
| `VLLM_TIMEOUT` | `120.0` | Timeout requête LLM (secondes) |
| `VLLM_MAX_TOKENS` | `4096` | Tokens max en sortie |
| `VLLM_TEMPERATURE` | `0.1` | Température (déterminisme) |
| `APP_PORT` | `8083` | Port du service |
| `DATABASE_URL` | — | Connection string PostgreSQL |
| `DATABASE_POOL_SIZE` | `5` | Taille pool connexions DB |
| `DATABASE_MAX_OVERFLOW` | `10` | Connexions supplémentaires max |
| `NATS_URL` | `nats://nats:4222` | URL NATS JetStream |
| `NATS_STREAM_NAME` | `VISIOBOOK_PROJECT` | Nom du stream |

---

## Connectivité K8s → vLLM (Tailscale)

Le service tourne sur Kubernetes mais le serveur vLLM est sur une VM sans IP publique. La connexion passe par **Tailscale** (VPN mesh WireGuard, IPs fixes en `100.x.x.x`).

### Installation sur la VM vLLM

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey <auth_key>
tailscale ip -4   # note l'IP Tailscale (ex: 100.120.93.126)
```

### Installation sur le(s) nœud(s) K8s

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey <auth_key>
```

Puis redémarrer : `docker compose -f docker-compose.vllm.yml down && docker compose -f docker-compose.vllm.yml up -d`

### Vérifier la connectivité

Depuis le nœud K8s :
```bash
# Santé vLLM
curl http://100.120.93.126:8000/health

# Test d'inférence
curl http://100.120.93.126:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <VLLM_API_KEY>" \
  -d '{"model": "mistralai/Ministral-3-3B-Instruct-2512-BF16", "messages": [{"role": "user", "content": "Dis bonjour."}], "max_tokens": 20}'
```

Configurer `VLLM_BASE_URL=http://100.120.93.126:8000` dans le secret/configmap K8s (ou `charts/values.yaml`).

---

## Lancer en local

### Prérequis

vLLM doit tourner sur un serveur GPU accessible. Pour y accéder depuis le PC de dev, utiliser un tunnel SSH :

```bash
ssh -L 8000:localhost:8000 <user>@<serveur> -p <port> -N
```

### Sans Docker

```bash
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8083 --reload
```

### Avec Docker

```bash
docker compose up --build
```

---

## Lancer vLLM (serveur GPU)

```bash
docker compose -f docker-compose.vllm.yml up -d
```

**Prérequis serveur :**
- NVIDIA GPU (Blackwell/Ampere/Turin)
- Driver ≥ 525, CUDA 13.0 pour GPU Blackwell (RTX 50xx)
- NVIDIA Container Toolkit

Variables requises dans `.env` côté serveur :
```env
HF_TOKEN=<token_huggingface>
VLLM_API_KEY=<clé_api_générée>
```

---

## CI/CD

| Trigger | Workflow | Action |
|---------|----------|--------|
| PR → `dev` | `developpement.yml` | Lint (Black, Ruff, Pylint) |
| Push → `dev` | `merge-to-dev.yml` | Build + push image `:dev` |
| Push → `main` | `dev-to-main.yml` | Build + push image `:prod` |
