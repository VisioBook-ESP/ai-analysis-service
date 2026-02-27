import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = "pending"   # pending | processing | completed | failed
        self.step: Optional[str] = None  # preprocessing | llm_call | parsing
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id)
        async with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)
                job.updated_at = datetime.now()

    async def cleanup(self, max_age_hours: int = 1) -> None:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        async with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j.created_at < cutoff]
            for jid in expired:
                del self._jobs[jid]


job_store = JobStore()
