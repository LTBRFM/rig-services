"""
Shared request/response models and Job lifecycle types.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class JobType(str, Enum):
    TTS   = "tts"
    MUSIC = "music"


# ── API request models ────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
]


class TTSRequest(BaseModel):
    text:     str
    voice:    str
    language: Optional[str] = None

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
            raise ValueError(f"Unsupported language '{v}'. Supported: {SUPPORTED_LANGUAGES}")
        return v


class MusicRequest(BaseModel):
    prompt:         Optional[str] = Field(default=None, description="Music style/description — falls back to global preset if omitted")
    lyrics:         str   = Field(default="[Instrumental]",
                                  description="Song lyrics, or '[Instrumental]' for no vocals")
    duration:       Optional[int] = Field(default=None, ge=10, le=600,
                                         description="Duration in seconds (10–600). Omit to let the model auto-determine length from lyrics.")
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=15.0,
                                            description="Prompt adherence strength (1–15) — falls back to preset")
    bpm:            Optional[int] = Field(default=None, ge=30, le=300,
                                          description="BPM hint (optional, model auto-detects if omitted)")
    thinking:       Optional[bool] = Field(default=None,
                                           description="Enable LM Chain-of-Thought reasoning — falls back to preset")


# ── Voice management ──────────────────────────────────────────────────────────

# Voice names become file stems in voices_dir — keep them path-safe.
VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class VoiceInfo(BaseModel):
    name:    str            = Field(description="Voice identifier — pass this as `voice` in POST /tts")
    file:    str            = Field(description="Sample file name inside voices_dir")
    format:  str            = Field(description="Audio container (wav, mp3, flac, ogg)")
    profile: Dict[str, Any] = Field(description="Effective XTTS synthesis parameters (defaults merged with <name>.json)")


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]


class VoiceUploadResponse(VoiceInfo):
    status:      str   = Field(default="ok")
    replaced:    bool  = Field(description="True if an existing sample with this name was overwritten")
    size_kb:     int   = Field(description="Stored sample size in KB")
    duration_s:  float = Field(description="Sample length in seconds")
    sample_rate: int   = Field(description="Sample rate in Hz")


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    type:   JobType
    params: Dict[str, Any]

    id:           str       = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status:       JobStatus = JobStatus.PENDING
    created_at:   float     = field(default_factory=time.time)
    started_at:   Optional[float] = None
    completed_at: Optional[float] = None
    result_path:  Optional[str]   = None
    raw_path:     Optional[str]   = None
    error:        Optional[str]   = None
    # Real-time progress for long-running jobs (0.0–1.0)
    progress:      float          = 0.0
    progress_desc: str            = ""

    def to_dict(self, queue_position: Optional[int] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id":              self.id,
            "type":            self.type,
            "status":          self.status,
            "created_at":      self.created_at,
            "started_at":      self.started_at,
            "completed_at":    self.completed_at,
            "processing_s":    round(self.completed_at - self.started_at, 2)
                               if self.completed_at and self.started_at else None,
            "result_available": self.result_path is not None,
            "raw_available":    self.raw_path is not None,
            "progress":        round(self.progress * 100, 1) if self.status == JobStatus.PROCESSING else None,
            "progress_desc":   self.progress_desc or None,
            "error":           self.error,
        }
        if self.type == JobType.TTS:
            d["params"] = {"voice": self.params.get("voice"), "language": self.params.get("language")}
        else:
            d["params"] = self.params
        if queue_position is not None:
            d["queue_position"] = queue_position
        return d
