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

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.model_manager import ModelManager
from app.models import (
    Job, JobType,
    MusicRequest, TTSRequest,
    SUPPORTED_LANGUAGES,
)
from app.preset import PresetManager
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
PRESET_PATH = Path("preset.json")
REFERENCE_PATH = Path("data/reference.wav")

with open(CONFIG_PATH) as _f:
    _config = json.load(_f)

# ── App lifecycle ─────────────────────────────────────────────────────────────
model_manager:  ModelManager  = None
preset_manager: PresetManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, preset_manager
    model_manager  = ModelManager(_config)
    preset_manager = PresetManager(PRESET_PATH)
    worker.start(model_manager, _config, preset_manager)
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
async def get_result(job_id: str, raw: bool = False):
    """
    Download the WAV file produced by a completed job.

    - `?raw=true` — download the pre-mastering original (always available for music jobs)
    - default      — download the mastered version
    """
    job = worker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if raw:
        target = job.raw_path or job.result_path
        suffix = "raw"
    else:
        target = job.result_path
        suffix = "mastered"

    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Result not available (status: {job.status})",
        )
    path = Path(target)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing from disk")
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=f"{job.type}_{job_id}_{suffix}.wav",
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
# Preset
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/preset", summary="Get global music preset", tags=["Preset"])
async def get_preset():
    """
    Returns the current global music preset.  All fields except `lyrics` in
    `POST /music` fall back to this preset when not provided in the request.
    """
    p = preset_manager.get()
    p["reference_uploaded"] = REFERENCE_PATH.exists()
    return p


@app.patch("/preset", summary="Update global music preset", tags=["Preset"])
async def patch_preset(updates: dict):
    """
    Deep-merge *updates* into the current preset and persist to `preset.json`.

    Only send the fields you want to change — everything else is preserved.

    **Examples:**

    Change prompt only:
    ```json
    { "prompt": "dark cinematic orchestral, strings, brass, timpani" }
    ```

    Tweak mastering:
    ```json
    { "mastering": { "target_lufs": -9.0, "matchering_enabled": true } }
    ```

    Enable matchering (upload a reference track first via POST /preset/reference):
    ```json
    { "mastering": { "matchering_enabled": true } }
    ```
    """
    result = preset_manager.patch(updates)
    result["reference_uploaded"] = REFERENCE_PATH.exists()
    return result


@app.post("/preset/reference", summary="Upload mastering reference track", tags=["Preset"])
async def upload_reference(file: UploadFile = File(...)):
    """
    Upload a WAV reference track for matchering-based mastering.

    The file should be a professionally mastered song in the same genre as
    your generated music.  Once uploaded, set `mastering.matchering_enabled`
    to `true` via `PATCH /preset` to activate reference-based mastering.

    Accepts WAV files only.
    """
    if not file.filename.lower().endswith((".wav", ".flac")):
        raise HTTPException(status_code=422, detail="Only WAV or FLAC reference files are accepted")
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    REFERENCE_PATH.write_bytes(content)
    logger.info(f"Reference track uploaded: {file.filename} ({len(content) // 1024} KB)")
    return {"status": "ok", "reference": str(REFERENCE_PATH), "size_kb": len(content) // 1024}


@app.delete("/preset/reference", summary="Remove mastering reference track", tags=["Preset"])
async def delete_reference():
    """Remove the uploaded reference track. Mastering will fall back to the pedalboard DSP chain."""
    if REFERENCE_PATH.exists():
        REFERENCE_PATH.unlink()
        return {"status": "ok", "message": "Reference track removed"}
    return {"status": "ok", "message": "No reference track was present"}


# ─────────────────────────────────────────────────────────────────────────────
# Music
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/music", summary="Submit a music generation job", tags=["Music"])
async def submit_music(request: MusicRequest):
    """
    Submit a music generation job powered by **ACE-Step 1.5 XL SFT**.
    Returns immediately with a `job_id`.

    **Only `lyrics` is required** when a global preset is configured via
    `PATCH /preset`.  Any field you include in the request overrides the preset
    for that one job.

    Poll `GET /jobs/{job_id}` for progress (includes `progress` 0–100 and
    `progress_desc`).  Download from `GET /result/{job_id}` when done.

    - `GET /result/{job_id}`            → mastered WAV (default)
    - `GET /result/{job_id}?raw=true`   → pre-mastering original

    **Example — lyrics only (uses preset for everything else):**
    ```json
    { "lyrics": "[Verse 1]\\nHello world\\n[Chorus]\\nLa la la" }
    ```

    **Example — full override:**
    ```json
    {
      "prompt": "upbeat synthwave with driving bass",
      "lyrics": "[Instrumental]",
      "duration": 120,
      "guidance_scale": 7.0,
      "thinking": true
    }
    ```
    """
    preset = preset_manager.get()

    # Merge: request fields override preset; None means "use preset"
    prompt = request.prompt or preset.get("prompt") or ""
    if not prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="prompt is required — either set it in the request or configure a global preset via PATCH /preset",
        )

    job = Job(
        type=JobType.MUSIC,
        params={
            "prompt":         prompt.strip(),
            "lyrics":         request.lyrics,
            "duration":       request.duration if request.duration is not None else preset.get("duration"),
            "guidance_scale": request.guidance_scale if request.guidance_scale is not None else preset.get("guidance_scale", 7.0),
            "bpm":            request.bpm if request.bpm is not None else preset.get("bpm"),
            "thinking":       request.thinking if request.thinking is not None else preset.get("thinking", True),
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
