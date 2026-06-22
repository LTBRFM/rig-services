"""
Unified GPU API — TTS + Music Generation
Both models share a single GPU; the worker processes jobs sequentially,
loading/unloading models only when the job type changes.
"""
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.model_manager import ModelManager
from app.models import (
    Job, JobType,
    MusicRequest, TTSRequest,
    SUPPORTED_LANGUAGES,
)
import app.worker as worker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.json"
with open(CONFIG_PATH) as _f:
    _config = json.load(_f)

# ── App lifecycle ─────────────────────────────────────────────────────────────
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    model_manager = ModelManager(_config)
    worker.start(model_manager, _config)
    logger.info("GPU worker started. API ready.")
    yield


app = FastAPI(
    title="Unified GPU API",
    description=(
        "Single-GPU API for Text-to-Speech (XTTS-v2) and Music Generation (MusicGen).\n\n"
        "Jobs are queued and processed sequentially. The active model stays loaded on GPU "
        "until a different model type is requested, then it is swapped automatically.\n\n"
        "**Workflow:** submit a job → receive `job_id` → poll `GET /jobs/{job_id}` until "
        "`status == done` → download result from `GET /result/{job_id}`."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Status & monitoring
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status", summary="System status — queue, GPU, loaded model", tags=["Monitoring"])
async def get_status():
    """
    Real-time system status. Poll this to monitor the queue.

    Returns:
    - **loaded_model**: which model is currently on GPU (`tts`, `music`, or `null`)
    - **vram**: GPU memory stats (null on CPU-only machines)
    - **current_job**: the job being processed right now
    - **queue**: list of pending jobs with their queue positions
    - **stats**: total jobs processed, server uptime
    """
    snap = worker.status_snapshot()
    snap["loaded_model"] = model_manager.loaded_model if model_manager else None
    snap["vram"]         = model_manager.vram_info()  if model_manager else None
    return snap


@app.get("/health", summary="Health check", tags=["Monitoring"])
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}", summary="Get job status", tags=["Jobs"])
async def get_job(job_id: str):
    """
    Poll this endpoint after submitting a job.

    - `status: pending`    — waiting in queue
    - `status: processing` — running on GPU right now
    - `status: done`       — finished; download result from `GET /result/{job_id}`
    - `status: failed`     — see `error` field for details
    """
    job = worker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    pos = worker.queue_position(job_id)
    return job.to_dict(queue_position=pos)


@app.get(
    "/result/{job_id}",
    summary="Download result audio",
    tags=["Jobs"],
    response_description="WAV audio file",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Result WAV audio"},
        404: {"description": "Job not found or not yet complete"},
    },
)
async def get_result(job_id: str):
    """Download the WAV file produced by a completed job."""
    job = worker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail=f"Result not available (status: {job.status})",
        )
    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing from disk")
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=f"{job.type}_{job_id}.wav",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/tts", summary="Submit a TTS job", tags=["TTS"])
async def submit_tts(request: TTSRequest):
    """
    Submit a text-to-speech job. Returns immediately with a `job_id`.

    Poll `GET /jobs/{job_id}` to check progress, then download from
    `GET /result/{job_id}` when `status == done`.

    **Example:**
    ```json
    { "text": "Hello world!", "voice": "en_female", "language": "en" }
    ```
    """
    job = Job(
        type=JobType.TTS,
        params={
            "text":     request.text,
            "voice":    request.voice,
            "language": request.language,
        },
    )
    worker.submit(job)
    pos = worker.queue_position(job.id)
    return {"job_id": job.id, "status": job.status, "queue_position": pos}


@app.get("/voices", summary="List available TTS voices", tags=["TTS"])
async def list_voices():
    """
    Lists all voice WAV files in the `voices/` directory with their profile settings.
    The `name` field is what you pass as `voice` in POST /tts.
    """
    # Instantiate a lightweight reader — no model needed just for listing
    from app.tts_engine import TTSEngine, PROFILE_DEFAULTS
    from pathlib import Path

    voices_dir = Path(_config["voices_dir"])
    voices_dir.mkdir(exist_ok=True)
    results = []
    for f in sorted(voices_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
            profile_path = voices_dir / f"{f.stem}.json"
            overrides = {}
            if profile_path.exists():
                with open(profile_path) as pf:
                    overrides = {k: v for k, v in json.load(pf).items() if not k.startswith("_")}
            results.append({
                "name":    f.stem,
                "file":    f.name,
                "profile": {**PROFILE_DEFAULTS, **overrides},
            })
    return {"voices": results}


@app.get("/profile/defaults", summary="TTS default synthesis parameters", tags=["TTS"])
async def profile_defaults():
    from app.tts_engine import PROFILE_DEFAULTS
    return PROFILE_DEFAULTS


@app.get("/languages", summary="Supported TTS languages", tags=["TTS"])
async def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}


# ─────────────────────────────────────────────────────────────────────────────
# Music
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/music", summary="Submit a music generation job", tags=["Music"])
async def submit_music(request: MusicRequest):
    """
    Submit a music generation job. Returns immediately with a `job_id`.

    Poll `GET /jobs/{job_id}` for progress, download from `GET /result/{job_id}`
    when done.

    **Duration guidance:**
    - 30 s  → ~1 min GPU time
    - 120 s → ~5 min GPU time
    - 300 s → ~12 min GPU time

    **Example:**
    ```json
    {
      "prompt": "upbeat synthwave track with driving bass and arpeggiated synths",
      "duration": 120,
      "guidance_scale": 3.5
    }
    ```
    """
    job = Job(
        type=JobType.MUSIC,
        params={
            "prompt":         request.prompt,
            "duration":       request.duration,
            "guidance_scale": request.guidance_scale,
        },
    )
    worker.submit(job)
    pos = worker.queue_position(job.id)
    return {"job_id": job.id, "status": job.status, "queue_position": pos}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=_config.get("host", "0.0.0.0"),
        port=_config.get("port", 8000),
        reload=False,
    )
