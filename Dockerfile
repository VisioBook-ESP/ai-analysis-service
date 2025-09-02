FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# Requirements (cache-friendly)
COPY requirements/ /tmp/requirements/
RUN pip install --upgrade pip && pip install -r /tmp/requirements/base.txt

# Code
COPY src ./src
ENV PYTHONPATH=/app

EXPOSE 8083
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8083"]
