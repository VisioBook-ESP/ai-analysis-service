# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
import psutil
import torch

from src.config.settings import Settings, get_settings
from src.utils.gpu_utils import get_gpu_info, is_gpu_available


router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime
    checks: Dict[str, bool]
    details: Optional[Dict] = None


class MetricsResponse(BaseModel):
    service: str
    timestamp: datetime
    system: Dict
    gpu: Optional[Dict]
    models: Dict


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health probe",
    description="Vérifie que le service est en vie et répond aux requêtes",
)
def health_check(settings: Settings = Depends(get_settings)):
    return HealthResponse(
        status="healthy", service=settings.app_name, timestamp=datetime.now()
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Vérifie que le service est prêt à traiter des requêtes",
)
def readiness_check(settings: Settings = Depends(get_settings)):
    checks = {
        "api": True,
        "pytorch": _check_pytorch(),
        "transformers": _check_transformers(),
        "gpu": is_gpu_available(),  # Non-bloquant
    }

    # Le service est ready si tous les checks CRITIQUES passent
    all_ready = checks["api"] and checks["pytorch"] and checks["transformers"]

    details = {
        "gpu_available": checks["gpu"],
        "device": "cuda" if checks["gpu"] else "cpu",
        "pytorch_version": torch.__version__,
    }

    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(),
        checks=checks,
        details=details,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métriques détaillées",
    description="Métriques complètes du service pour monitoring",
)
def metrics(settings: Settings = Depends(get_settings)):
    # Métriques système
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    system_metrics = {
        "cpu_percent": round(cpu_percent, 1),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "memory_percent": round(memory.percent, 1),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_percent": round(disk.percent, 1),
    }

    # Metriques GPU (None si pas de GPU)
    gpu_metrics = get_gpu_info()

    # Métriques modèles
    model_metrics = {
        "summarizer_configured": settings.summarizer_model,
        "models_loaded_count": _get_loaded_models_count(),
    }

    return MetricsResponse(
        service=settings.app_name,
        timestamp=datetime.now(),
        system=system_metrics,
        gpu=gpu_metrics,
        models=model_metrics,
    )


# ========================================
# FONCTIONS UTILITAIRES
# ========================================


def _check_pytorch() -> bool:
    """Verifie que PyTorch est disponible et fonctionnel."""
    try:
        import torch

        # Test basique de creation de tensor
        _ = torch.tensor([1.0])
        return True
    except Exception:
        return False


def _check_transformers() -> bool:
    """Verifie que transformers (Hugging Face) est disponible."""
    try:
        from transformers import pipeline

        return True
    except Exception:
        return False


def _get_loaded_models_count() -> int:
    """
    Compte le nombre de modeles charges en memoire.

    TODO: Ameliorer avec un ModelManager global qui track les modeles.
    Pour l'instant retourne 0 (pas de ModelManager encore).
    """
    return 0
