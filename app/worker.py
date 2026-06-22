"""
Single-threaded background worker that drains the job queue.

Design:
- One Python thread processes jobs sequentially (no parallel GPU work).
- Jobs are stored in a dict keyed by job ID for O(1) status lookups.
- ModelManager ensures the right model is loaded before each job runs,
  unloading the previous one only when the model type changes.
- Clients poll GET /jobs/{id} for status; no websockets required.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from app.models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)

# ── State (module-level singletons) ──────────────────────────────────────────

_q:    queue.Queue = queue.Queue()          # FIFO job queue
_jobs: Dict[str, Job] = {}                  # all jobs ever submitted
_current_job: Optional[Job] = None         # job being processed right now
_stats = {"processed": 0, "started_at": time.time()}
_lock  = threading.Lock()                   # protects _jobs / _current_job reads


# ── Public API ────────────────────────────────────────────────────────────────

def submit(job: Job) -> Job:
    with _lock:
        _jobs[job.id] = job
    _q.put(job)
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def status_snapshot() -> Dict[str, Any]:
    """Return a serialisable snapshot of the queue and running job."""
    with _lock:
        current = _current_job
        pending: List[Job] = list(_q.queue)   # deque peek, no removal

    return {
        "current_job":  current.to_dict() if current else None,
        "queue_length": len(pending),
        "queue":        [j.to_dict(queue_position=i + 1) for i, j in enumerate(pending)],
        "stats": {
            "jobs_processed": _stats["processed"],
            "uptime_s":       round(time.time() - _stats["started_at"]),
        },
    }


def queue_position(job_id: str) -> Optional[int]:
    """Return 1-based queue position, 0 if processing, None if not found."""
    with _lock:
        if _current_job and _current_job.id == job_id:
            return 0
    pending = list(_q.queue)
    for i, j in enumerate(pending):
        if j.id == job_id:
            return i + 1
    return None


# ── Worker thread ─────────────────────────────────────────────────────────────

def _run(model_manager, app_config: Dict[str, Any]) -> None:
    global _current_job

    logger.info("Worker thread started.")

    while True:
        job: Job = _q.get()

        with _lock:
            _current_job = job

        job.status     = JobStatus.PROCESSING
        job.started_at = time.time()
        logger.info(f"Processing job {job.id} ({job.type})")

        try:
            if job.type == JobType.TTS:
                model_manager.ensure("tts")
                engine = model_manager.get_tts()
                language = job.params.get("language") or app_config.get("default_language", "en")
                result   = engine.synthesize(job.params["text"], job.params["voice"], language)

            elif job.type == JobType.MUSIC:
                model_manager.ensure("music")
                engine = model_manager.get_music()
                result = engine.generate(
                    prompt=job.params["prompt"],
                    duration=job.params["duration"],
                    guidance_scale=job.params["guidance_scale"],
                )
            else:
                raise ValueError(f"Unknown job type: {job.type}")

            job.result_path  = str(result)
            job.status       = JobStatus.DONE

        except Exception as exc:
            logger.exception(f"Job {job.id} failed")
            job.status = JobStatus.FAILED
            job.error  = str(exc)

        finally:
            job.completed_at = time.time()
            elapsed          = round(job.completed_at - job.started_at, 1)
            logger.info(f"Job {job.id} {job.status} in {elapsed}s")

            with _lock:
                _current_job = None
            _stats["processed"] += 1
            _q.task_done()


def start(model_manager, app_config: Dict[str, Any]) -> threading.Thread:
    t = threading.Thread(
        target=_run,
        args=(model_manager, app_config),
        daemon=True,
        name="gpu-worker",
    )
    t.start()
    return t
