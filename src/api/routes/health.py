from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
import psutil

from src.clients.database_client import DatabaseClient
from src.config.settings import Settings, get_settings
from src.services.analysis.llm_client import LLMClient


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


class MetricsResponse(BaseModel):
    service: str
    timestamp: datetime
    system: Dict


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check(settings: Settings = Depends(get_settings)):
    return HealthResponse(status="healthy", service=settings.app_name, timestamp=datetime.now())


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(settings: Settings = Depends(get_settings)):
    llm_client = LLMClient()
    vllm_ok = await llm_client.health_check()
    await llm_client.close()

    db_client = DatabaseClient()
    db_ok = False
    if settings.DATABASE_URL:
        db_ok = await db_client.health_check()
    else:
        db_ok = True  # DB not configured — skip check

    checks = {"api": True, "vllm": vllm_ok, "database": db_ok}
    all_ready = all(checks.values())

    return ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(),
        checks=checks,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(settings: Settings = Depends(get_settings)):
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

    return MetricsResponse(
        service=settings.app_name,
        timestamp=datetime.now(),
        system=system_metrics,
    )
