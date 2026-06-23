"""
Manages loading and unloading of TTS and Music models on the GPU.

Only one model lives on GPU at a time. When a different model is needed the
current one is unloaded (VRAM freed) before the new one is loaded.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._loaded: Optional[str] = None   # "tts" | "music" | None
        self._tts_engine   = None
        self._music_engine = None

    # ── Public interface ──────────────────────────────────────────────────────

    def ensure(self, model_type: str) -> None:
        """Guarantee the requested model is loaded on GPU, unloading the other if necessary."""
        if self._loaded == model_type:
            logger.debug(f"Model already loaded: {model_type}")
            return

        if self._loaded is not None:
            self._unload(self._loaded)

        logger.info(f"Loading model: {model_type}  |  {self._vram_summary()}")
        t0 = time.time()
        if model_type == "tts":
            from app.tts_engine import TTSEngine
            self._tts_engine = TTSEngine(self.config)
        elif model_type == "music":
            from app.music_engine import MusicEngine
            self._music_engine = MusicEngine(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._loaded = model_type
        elapsed = round(time.time() - t0, 1)
        logger.info(f"Model ready: {model_type}  elapsed={elapsed}s  |  {self._vram_summary()}")

    def get_tts(self):
        return self._tts_engine

    def get_music(self):
        return self._music_engine

    @property
    def loaded_model(self) -> Optional[str]:
        return self._loaded

    def vram_info(self) -> Optional[Dict]:
        if not torch.cuda.is_available():
            return None
        props    = torch.cuda.get_device_properties(0)
        total    = props.total_memory
        used     = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return {
            "device":      props.name,
            "total_gb":    round(total    / 1e9, 2),
            "used_gb":     round(used     / 1e9, 2),
            "reserved_gb": round(reserved / 1e9, 2),
            "free_gb":     round((total - reserved) / 1e9, 2),
            "used_pct":    round(used / total * 100, 1),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _unload(self, model_type: str) -> None:
        logger.info(f"Unloading model: {model_type}  |  {self._vram_summary()}")
        t0 = time.time()
        if model_type == "tts" and self._tts_engine:
            self._tts_engine.unload()
            self._tts_engine = None
        elif model_type == "music" and self._music_engine:
            self._music_engine.unload()
            self._music_engine = None
        self._loaded = None
        _free_vram()
        elapsed = round(time.time() - t0, 1)
        logger.info(f"Model unloaded: {model_type}  elapsed={elapsed}s  |  {self._vram_summary()}")

    def _vram_summary(self) -> str:
        info = self.vram_info()
        if not info:
            return "CPU mode"
        return f"VRAM {info['used_gb']:.1f}/{info['total_gb']:.1f} GB"
