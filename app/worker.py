"""
Single-threaded background worker that drains the job queue.

Priority rules:
- TTS jobs always run before Music jobs regardless of submission order.
- Within the same type, jobs are processed in FIFO submission order.
- A music job already running is NOT interrupted — priority only applies
  to jobs waiting in the queue.

Design:
- PriorityQueue with (priority, sequence, job) tuples.
  TTS priority=0, Music priority=1.
- One Python thread processes jobs sequentially (no parallel GPU work).
- ModelManager loads/unloads models only when the job type changes.
- Clients poll GET /jobs/{id} for status; no websockets needed.
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import Job, JobStatus, JobType

logger = logging.getLogger(__name__)

# ── Priority mapping ──────────────────────────────────────────────────────────

_PRIORITY = {JobType.TTS: 0, JobType.MUSIC: 1}

# ── State (module-level singletons) ──────────────────────────────────────────

_q:       queue.PriorityQueue = queue.PriorityQueue()
_seq:     itertools.count     = itertools.count()       # tie-breaker within same priority
_jobs:    Dict[str, Job]      = {}
_current_job: Optional[Job]   = None
_stats  = {"processed": 0, "started_at": time.time()}
_lock   = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def submit(job: Job) -> Job:
    priority = _PRIORITY.get(job.type, 99)
    with _lock:
        _jobs[job.id] = job
    _q.put((priority, next(_seq), job))
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def status_snapshot() -> Dict[str, Any]:
    """Return a serialisable snapshot of the queue and running job."""
    with _lock:
        current = _current_job
        # PriorityQueue.queue is a heap of (priority, seq, job) tuples
        raw = sorted(_q.queue)           # sort by (priority, seq) → true processing order

    pending_jobs = [entry[2] for entry in raw]
    return {
        "current_job":  current.to_dict() if current else None,
        "queue_length": len(pending_jobs),
        "queue":        [j.to_dict(queue_position=i + 1) for i, j in enumerate(pending_jobs)],
        "stats": {
            "jobs_processed": _stats["processed"],
            "uptime_s":       round(time.time() - _stats["started_at"]),
        },
    }


def queue_position(job_id: str) -> Optional[int]:
    """Return 1-based queue position (in priority order), 0 if processing, None if not found."""
    with _lock:
        if _current_job and _current_job.id == job_id:
            return 0
    raw = sorted(_q.queue)
    for i, entry in enumerate(raw):
        if entry[2].id == job_id:
            return i + 1
    return None


# ── Worker thread ─────────────────────────────────────────────────────────────

def _run(model_manager, app_config: Dict[str, Any], preset_manager=None) -> None:
    global _current_job

    logger.info("Worker thread started — TTS has priority over Music.")

    while True:
        _priority, _seq_n, job = _q.get()  # blocks; unpacks priority tuple

        with _lock:
            _current_job = job

        job.status     = JobStatus.PROCESSING
        job.started_at = time.time()
        logger.info(f"Processing job {job.id} ({job.type}, priority={_priority})")

        try:
            if job.type == JobType.TTS:
                model_manager.ensure("tts")
                engine = model_manager.get_tts()
                language = job.params.get("language") or app_config.get("default_language", "en")
                result   = engine.synthesize(job.params["text"], job.params["voice"], language)

            elif job.type == JobType.MUSIC:
                model_manager.ensure("music")
                engine = model_manager.get_music()

                def _music_progress(value: float, desc: str = "") -> None:
                    job.progress = float(value)
                    job.progress_desc = str(desc)

                raw_result = engine.generate(
                    prompt=job.params["prompt"],
                    lyrics=job.params.get("lyrics", "[Instrumental]"),
                    duration=job.params["duration"],
                    guidance_scale=job.params["guidance_scale"],
                    bpm=job.params.get("bpm"),
                    thinking=job.params.get("thinking", True),
                    on_progress=_music_progress,
                )
                raw_path = Path(raw_result)
                job.raw_path = str(raw_path)

                # Mastering post-processing
                if preset_manager is not None:
                    from app.mastering import apply_mastering
                    mastering_cfg = preset_manager.mastering_settings()
                    ref_path = Path("data/reference.wav")
                    mastered_path = raw_path.parent / (raw_path.stem + "_mastered.wav")
                    job.progress = 0.99
                    job.progress_desc = "Mastering audio..."
                    apply_mastering(raw_path, mastered_path, mastering_cfg, ref_path)
                    result = mastered_path
                else:
                    result = raw_path
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


def start(model_manager, app_config: Dict[str, Any], preset_manager=None) -> threading.Thread:
    t = threading.Thread(
        target=_run,
        args=(model_manager, app_config, preset_manager),
        daemon=True,
        name="gpu-worker",
    )
    t.start()
    return t
