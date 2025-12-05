# Image Python 3.12
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements-prod.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements-prod.txt --no-deps || pip install -r requirements-prod.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --version

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source
COPY src ./src

# Création des répertoires nécessaires
RUN mkdir -p data models && \
    chmod -R 755 /app

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Port exposé
EXPOSE 8083

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# Lancement de l'app sans hot-reload pour la prod
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8083", "--workers", "1"]
