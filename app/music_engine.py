"""
MusicGen stereo-large wrapper.

Model: facebook/musicgen-stereo-large
- Stereo 32 kHz output
- Text-conditioned generation
- ~6–8 GB VRAM (float16)
- 50 tokens/second → 1500 tokens ≈ 30 s of audio

For a full song (e.g. 180 s) set duration=180 — generation will take several
minutes on GPU but quality is excellent.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

TOKENS_PER_SECOND = 50  # MusicGen internal rate


class MusicEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config     = config
        self.output_dir = Path(config["output_dir"]) / "music"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        music_cfg  = config["music"]
        self.model_id = music_cfg["model_repo"]
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading MusicGen model '{self.model_id}' on {self.device} ...")

        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model     = MusicgenForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        self.sample_rate: int = self.model.config.audio_encoder.sampling_rate
        logger.info(
            f"MusicGen ready — sample_rate={self.sample_rate} Hz, "
            f"device={self.device}"
        )

    def unload(self) -> None:
        for attr in ("model", "processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                if hasattr(obj, "cpu"):
                    obj.cpu()
                del obj
                setattr(self, attr, None)
        gc.collect()

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, duration: int, guidance_scale: float) -> Path:
        import hashlib, numpy as np, scipy.io.wavfile

        cache_key   = hashlib.sha256(
            f"{prompt}|{duration}|{guidance_scale}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if output_path.exists():
            logger.info(f"Music cache hit: prompt='{prompt[:40]}...'")
            return output_path

        max_new_tokens = int(duration * TOKENS_PER_SECOND)
        logger.info(
            f"Music generating: duration={duration}s, tokens={max_new_tokens}, "
            f"guidance={guidance_scale}, prompt='{prompt[:60]}'"
        )

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
            )

        # audio_values: (batch=1, channels, samples)
        audio_np = audio_values[0].cpu().float().numpy()  # (channels, samples)
        audio_np = audio_np.T                              # (samples, channels) for scipy

        # Normalise to int16
        max_val  = max(abs(audio_np).max(), 1e-6)
        audio_i16 = (audio_np / max_val * 32767).astype("int16")

        scipy.io.wavfile.write(str(output_path), self.sample_rate, audio_i16)
        logger.info(f"Music saved → {output_path} ({duration}s @ {self.sample_rate} Hz)")

        return output_path
