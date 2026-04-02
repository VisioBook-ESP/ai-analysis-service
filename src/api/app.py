import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from src.api.routes.analysis import router as analysis_router
from src.api.routes.health import router as health_router
from src.api.routes.results import router as results_router
from src.config.settings import get_settings
from src.services.job_store import job_store
from src.services.nats_client import NatsClient
from src.database.connection import close_engine
from src.services.workflow_handler import WorkflowHandler, SUBJECT_WORKFLOW_STARTED

logger = logging.getLogger(__name__)

settings = get_settings()

# Module-level reference so health checks or shutdown can access it
_nats_client: NatsClient | None = None


async def connect_nats_with_retry(
    nats_client: NatsClient, max_retries: int = 5, delay: int = 3
) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            await nats_client.connect()
            return True
        except Exception as e:
            logger.warning(
                "NATS connection attempt %d/%d failed: %s", attempt, max_retries, e
            )
            if attempt < max_retries:
                await asyncio.sleep(delay * attempt)
    logger.error("Could not connect to NATS after %d attempts", max_retries)
    return False


async def _cleanup_jobs_loop():
    while True:
        await asyncio.sleep(3600)
        await job_store.cleanup(max_age_hours=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _nats_client

    cleanup_task = asyncio.create_task(_cleanup_jobs_loop())
    nats_task = None

    # Connect to NATS in the background so the HTTP API starts immediately
    async def _connect_and_subscribe():
        global _nats_client
        nats_client = NatsClient(
            url=settings.NATS_URL,
            user=settings.NATS_USER,
            password=settings.NATS_PASSWORD,
        )

        connected = await connect_nats_with_retry(nats_client)
        if connected:
            _nats_client = nats_client

            from src.api.routes.analysis import _analyzer

            handler = WorkflowHandler(nats_client=nats_client, analyzer=_analyzer)

            try:
                await nats_client.subscribe(
                    stream=settings.NATS_STREAM_NAME,
                    subject=SUBJECT_WORKFLOW_STARTED,
                    durable="ai-analysis-workflow-started",
                    callback=handler.handle_workflow_started,
                )
            except Exception as e:
                logger.error("NATS subscription failed: %s", e)
        else:
            logger.warning("Starting without NATS — HTTP API is still available")

    nats_task = asyncio.create_task(_connect_and_subscribe())

    yield

    # Shutdown
    cleanup_task.cancel()
    if nats_task:
        nats_task.cancel()
    if _nats_client:
        await _nats_client.close()
        _nats_client = None

    await close_engine()

    from src.api.routes.analysis import _analyzer

    await _analyzer.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Service d'analyse IA via LLM (vLLM backend)",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/", tags=["root"])
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs",
        "endpoints": {
            "health": {
                "liveness": "/health",
                "readiness": "/ready",
                "metrics": "/metrics",
            },
            "analysis": {
                "analyze": "/api/v1/analyze",
                "analyze_batch": "/api/v1/analyze/batch",
            },
        },
    }


app.include_router(health_router, tags=["health"])
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
app.include_router(results_router, prefix="/api/v1/results", tags=["results"])
