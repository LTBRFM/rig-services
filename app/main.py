import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

# ── Logging ──────────────────────────────────────────────────────────────────
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
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    from app.tts_engine import TTSEngine
    engine = TTSEngine(config_path=CONFIG_PATH)
    yield


app = FastAPI(
    title="TTS API",
    description="High-quality Text-to-Speech powered by XTTS-v2. "
                "Drop WAV voice samples into the `voices/` folder, "
                "then POST to /tts with the voice name.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
]


class TTSRequest(BaseModel):
    text: str
    voice: str
    language: Optional[str] = None  # defaults to config default_language

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()

    @field_validator("language")
    @classmethod
    def language_supported(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. Supported: {SUPPORTED_LANGUAGES}"
            )
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": _config.get("model_repo")}


@app.get("/voices", summary="List available voice samples")
async def list_voices():
    """
    Returns every audio file in `voices/` with its active profile settings.
    Drop a `<voice_name>.json` next to the WAV to override any parameter.
    """
    return {"voices": engine.list_voices()}


@app.get("/profile/defaults", summary="Show default profile parameters")
async def profile_defaults():
    """
    Returns the default values for all synthesis parameters.
    Override any of these per-voice by creating `voices/<name>.json`.
    """
    return engine.get_profile_defaults()


@app.get("/languages", summary="List supported languages")
async def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post(
    "/tts",
    summary="Synthesise speech",
    response_description="WAV audio file",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Synthesised WAV audio"},
        404: {"description": "Voice not found"},
        422: {"description": "Validation error"},
        500: {"description": "Synthesis error"},
    },
)
async def synthesise(request: TTSRequest):
    """
    Synthesise `text` using the selected `voice` sample.

    **Example request:**
    ```json
    {
      "text": "Hello, this is a test.",
      "voice": "alice",
      "language": "en"
    }
    ```

    The response is a WAV audio file (streamed directly).
    """
    language = request.language or _config.get("default_language", "en")
    try:
        output_path = engine.synthesize(request.text, request.voice, language)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {exc}")

    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"tts_{request.voice}.wav",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=_config.get("host", "0.0.0.0"),
        port=_config.get("port", 8000),
        reload=False,
    )
