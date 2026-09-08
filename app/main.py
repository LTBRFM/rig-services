"""
Unified GPU API — TTS + Music Generation
Both models share a single GPU; the worker processes jobs sequentially,
loading/unloading models only when the job type changes.
"""
import importlib.util
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

import app.logging_config as log_cfg
from app.logging_config import request_id_var, LOG_FILE
from app.model_manager import ModelManager
from app.models import (
    Job, JobType,
    MusicRequest, TTSRequest,
    SUPPORTED_LANGUAGES,
)
from app.preset import PresetManager
import app.worker as worker

# ── Logging — must be set up before any other module creates a logger ─────────
log_cfg.setup()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH    = "config.json"
PRESET_PATH    = Path("preset.json")
REFERENCE_PATH = Path("data/reference.wav")

with open(CONFIG_PATH) as _f:
    _config = json.load(_f)

# ── App lifecycle ─────────────────────────────────────────────────────────────
model_manager:  ModelManager  = None
preset_manager: PresetManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, preset_manager
    logger.info("="*60)
    logger.info("TTS-API starting up")
    logger.info(f"Config: {CONFIG_PATH}")
    logger.info(f"Log file: {LOG_FILE}")
    model_manager  = ModelManager(_config)
    preset_manager = PresetManager(PRESET_PATH)
    worker.start(model_manager, _config, preset_manager)
    logger.info("GPU worker started. API ready.")
    logger.info("="*60)
    yield
    logger.info("TTS-API shutting down")


app = FastAPI(
    title="Unified GPU API",
    description=(
        "Single-GPU API for **Text-to-Speech** (XTTS-v2) and **Music Generation** (ACE-Step 1.5 XL SFT).\n\n"
        "Jobs are queued and processed sequentially — TTS always has priority over Music. "
        "The active model stays loaded on GPU until a different type is requested.\n\n"
        "## Workflow\n"
        "1. `POST /tts` or `POST /music` → receive `job_id`\n"
        "2. Poll `GET /jobs/{job_id}` until `status == done` (music jobs include `progress` 0–100)\n"
        "3. Download `GET /result/{job_id}` (mastered) or `GET /result/{job_id}?raw=true` (pre-mastering)\n\n"
        "## Global Preset\n"
        "Configure a single style via `PATCH /preset` — then `POST /music` only needs `lyrics`.\n"
        "Ideal for scheduled generation (e.g. service-update songs every 30 minutes).\n\n"
        "## Mastering\n"
        "Every music job automatically runs a post-processing chain:\n"
        "1. **Pedalboard** — highpass → compression → high-shelf EQ → limiter\n"
        "2. **Matchering** *(optional)* — tonal matching to an uploaded reference track\n"
        "3. **LUFS normalisation** — final loudness + true-peak ceiling\n\n"
        "Upload a reference track via `POST /preset/reference` and enable via `PATCH /preset`."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# ── Request logging middleware ────────────────────────────────────────────────

class _RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Assigns a short request ID to every HTTP request, logs start/end with
    method, path, status code, and elapsed time.  The request ID is injected
    into all log records generated during that request via request_id_var.
    """
    _skip_paths = {"/health"}  # high-frequency health checks — skip to reduce noise

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._skip_paths:
            return await call_next(request)

        req_id = uuid.uuid4().hex[:8]
        token  = request_id_var.set(req_id)
        start  = time.perf_counter()
        client = request.client.host if request.client else "unknown"
        qs     = f"?{request.url.query}" if request.url.query else ""

        logger.info(f"→ {request.method} {request.url.path}{qs}  client={client}")
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                f"✗ {request.method} {request.url.path}  UNHANDLED ERROR  {elapsed_ms:.1f}ms"
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"← {request.method} {request.url.path}{qs}  "
                f"status={response.status_code}  {elapsed_ms:.1f}ms"
            )
            request_id_var.reset(token)

        response.headers["X-Request-Id"] = req_id
        return response


app.add_middleware(_RequestLoggingMiddleware)
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


@app.get(
    "/logs",
    summary="Fetch recent server logs",
    tags=["Monitoring"],
    response_class=PlainTextResponse,
)
async def get_logs(lines: int = 200, level: str = None):
    """
    Returns the last N lines of the active server log as plain text.

    Useful for diagnosing issues or tracing what happened during a job.

    **Query parameters:**

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `lines`   | `200`   | Number of lines to return (max 2000) |
    | `level`   | *(none)*| Filter to lines containing this level: `ERROR`, `WARNING`, `INFO`, `DEBUG` |

    **Examples:**

    Last 200 lines:
    ```
    GET /logs
    ```

    Last 500 errors only:
    ```
    GET /logs?lines=500&level=ERROR
    ```

    All warnings and errors (last 1000 lines searched):
    ```
    GET /logs?lines=1000&level=WARNING
    ```

    Pipe to `grep` for further filtering:
    ```bash
    curl http://host:8001/logs?lines=500 | grep "job_id=abc123"
    ```
    """
    lines = min(max(lines, 1), 2000)
    log_file = LOG_FILE

    if not log_file.exists():
        return PlainTextResponse("No log file yet.\n")

    with open(log_file, encoding="utf-8", errors="replace") as f:
        tail = deque(f, maxlen=lines)

    if level:
        level_upper = level.upper()
        tail = [ln for ln in tail if level_upper in ln]

    return PlainTextResponse("".join(tail))


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}", summary="Get job status", tags=["Jobs"])
async def get_job(job_id: str):
    """
    Poll this endpoint after submitting a job.

    - `status: pending`    — waiting in queue
    - `status: processing` — running on GPU right now; music jobs include `progress` (0–100) and `progress_desc`
    - `status: done`       — finished; download from `GET /result/{job_id}`
    - `status: failed`     — see `error` field for details

    Music jobs also expose `raw_available: true` once the raw pre-mastering file is ready.
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
    Returns the current global music preset including all mastering parameters.

    All generation fields (`prompt`, `guidance_scale`, `bpm`, `thinking`, `duration`)
    are used as defaults when `POST /music` omits them — only `lyrics` is required.

    The `mastering` object controls the full post-processing chain applied to every
    generated song. See `PATCH /preset` for the complete parameter reference.
    """
    p = preset_manager.get()
    p["reference_uploaded"] = REFERENCE_PATH.exists()
    return p


@app.patch("/preset", summary="Update global music preset", tags=["Preset"])
async def patch_preset(updates: dict):
    """
    Deep-merge *updates* into the current preset and persist to `preset.json`.

    Only send the fields you want to change — everything else is preserved.

    ---

    ## Generation parameters

    | Field | Default | Description |
    |-------|---------|-------------|
    | `prompt` | `""` | Music style description — used for every job unless overridden |
    | `guidance_scale` | `7.0` | Prompt adherence (1–15; higher = stricter) |
    | `bpm` | `null` | BPM hint; `null` = model auto-detects |
    | `thinking` | `true` | LM chain-of-thought reasoning (higher quality, slower) |
    | `duration` | `null` | Length in seconds (10–600); `null` = auto from lyrics |

    ---

    ## Mastering parameters (`mastering` object)

    The mastering chain always runs in this order:
    **Pedalboard DSP → Matchering (optional) → LUFS normalisation**

    ### Dynamics & EQ (Pedalboard)

    | Field | Default | Range | What it does |
    |-------|---------|-------|--------------|
    | `enabled` | `true` | bool | Master switch — disables entire mastering chain if false |
    | `low_cut_hz` | `30.0` | 20–80 Hz | Highpass filter — removes sub-bass rumble |
    | `compression_threshold_db` | `-18.0` | -30 to -10 dB | Level above which compression activates |
    | `compression_ratio` | `4.0` | 2–10 | Compression strength (3:1 gentle, 8:1 heavy) |
    | `compression_attack_ms` | `10.0` | 1–50 ms | How fast compression reacts to loud transients |
    | `compression_release_ms` | `100.0` | 50–500 ms | How fast compression releases after loud section |
    | `high_shelf_gain_db` | `2.0` | 0–4 dB | Brightness boost above `high_shelf_hz` |
    | `high_shelf_hz` | `8000.0` | 6000–12000 Hz | Frequency where brightness boost starts |
    | `limiter_ceiling_db` | `-0.5` | -3 to -0.1 dB | Hard ceiling — nothing exceeds this level |

    ### Loudness (LUFS normalisation)

    | Field | Default | Description |
    |-------|---------|-------------|
    | `target_lufs` | `-12.0` | Final loudness target — matches commercial AI generators (Suno/Udio). Use -14 for streaming platforms, -9 for broadcast radio. |

    ### Reference-based mastering (Matchering)

    | Field | Default | Description |
    |-------|---------|-------------|
    | `matchering_enabled` | `false` | Enable tonal matching to uploaded reference track |

    Upload a reference WAV first via `POST /preset/reference`, then set `matchering_enabled: true`.
    The reference should be a professionally mastered track in the same genre and energy level.

    ---

    ## Ready-made mastering presets

    **Default — commercial AI generator loudness (matches Suno/Udio):**
    ```json
    { "mastering": { "target_lufs": -12.0, "compression_ratio": 4.0, "compression_threshold_db": -18.0, "high_shelf_gain_db": 2.0, "limiter_ceiling_db": -0.5 } }
    ```

    **Streaming (Spotify / Apple Music standard):**
    ```json
    { "mastering": { "target_lufs": -14.0, "compression_ratio": 3.0, "compression_threshold_db": -20.0, "high_shelf_gain_db": 1.5, "limiter_ceiling_db": -1.0 } }
    ```

    **Broadcast radio — maximum loudness:**
    ```json
    { "mastering": { "target_lufs": -9.0, "compression_ratio": 5.0, "compression_threshold_db": -16.0, "high_shelf_gain_db": 2.5, "limiter_ceiling_db": -0.3 } }
    ```

    **Gentle — preserve dynamics:**
    ```json
    { "mastering": { "target_lufs": -16.0, "compression_ratio": 2.0, "compression_threshold_db": -24.0, "high_shelf_gain_db": 1.0, "limiter_ceiling_db": -1.0 } }
    ```

    **Enable matchering (reference track must be uploaded first):**
    ```json
    { "mastering": { "matchering_enabled": true } }
    ```

    **Disable mastering entirely:**
    ```json
    { "mastering": { "enabled": false } }
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
    if importlib.util.find_spec("acestep") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Music generation is not installed on this server (ACE-Step missing). "
                "Re-run setup with INSTALL_MUSIC=1 on a GPU with >=10 GB VRAM."
            ),
        )

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
