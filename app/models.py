"""
Shared request/response models and Job lifecycle types.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

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
    prompt:         str   = Field(..., description="Text description of the music to generate")
    lyrics:         str   = Field(default="[Instrumental]",
                                  description="Song lyrics, or '[Instrumental]' for no vocals")
    duration:       Optional[int] = Field(default=None, ge=10, le=600,
                                         description="Duration in seconds (10–600). Omit to let the model auto-determine length from lyrics.")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=15.0,
                                  description="Prompt adherence strength (1–15, default 7)")
    bpm:            Optional[int] = Field(default=None, ge=30, le=300,
                                          description="BPM hint (optional, model auto-detects if omitted)")
    thinking:       bool  = Field(default=True,
                                  description="Enable LM Chain-of-Thought reasoning for higher quality")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v.strip()


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
    error:        Optional[str]   = None

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
            "error":           self.error,
        }
        if self.type == JobType.TTS:
            d["params"] = {"voice": self.params.get("voice"), "language": self.params.get("language")}
        else:
            d["params"] = self.params
        if queue_position is not None:
            d["queue_position"] = queue_position
        return d
