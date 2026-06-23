"""
ACE-Step 1.5 music generation engine.

Checkpoint variants for a 24 GB GeForce:
  acestep-v15-xl-sft   — 4B DiT, ~18-20 GB bf16 + 1.7B LM ~3 GB = ~22 GB total (max quality)
  acestep-v15-sft      — 2B DiT, ~9  GB bf16 + 1.7B LM ~3 GB = ~12 GB (good quality)
  acestep-v15-turbo-shift3 — 2B DiT, 8 steps, ~9 GB + LM ~3 GB (fast)

NEVER loaded at the same time as XTTS-v2 (ModelManager enforces this).
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class MusicEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config     = config
        self.output_dir = Path(config["output_dir"]) / "music"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        music_cfg  = config["music"]
        self.ckpt_variant    = music_cfg.get("checkpoint_variant", "acestep-v15-xl-sft")
        self.lm_model        = music_cfg.get("lm_model", "acestep-5Hz-lm-1.7B")
        self.lm_backend      = music_cfg.get("lm_backend", "pt")
        self.inference_steps = int(music_cfg.get("inference_steps", 50))
        self.prefer_source   = music_cfg.get("prefer_source", None)  # "huggingface"|"modelscope"|null
        self.ckpt_dir        = Path(music_cfg["checkpoint_dir"]).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # shift=3.0 is the correct value for ALL ACE-Step v1.5 variants.
        # The GenerationParams default (1.0) is wrong for this model family —
        # the Gradio demo always sets shift=3.0 regardless of turbo/sft/base.
        self.shift = float(music_cfg.get("shift", 3.0))

        # Resolve device: prefer cuda, fall back to cpu (skip MPS — ACE-Step DiT
        # fails silently on MPS, producing near-silence output)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # ACE-Step reads ACESTEP_CHECKPOINTS_DIR to find / auto-download weights
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(self.ckpt_dir)

        # ── Initialise DiT (auto-downloads if needed) ─────────────────────────
        logger.info(f"Initialising ACE-Step DiT variant='{self.ckpt_variant}' on {self.device} ...")
        from acestep.handler import AceStepHandler

        self._dit = AceStepHandler()
        # Disable MLX on MPS — MLX VAE decoder produces bass-only output (no harmonics)
        # on Apple Silicon; forcing PyTorch gives correct full-spectrum audio.
        use_mlx = self.device == "cuda"
        init_kwargs = dict(
            project_root="",          # ignored when ACESTEP_CHECKPOINTS_DIR is set
            config_path=self.ckpt_variant,
            device=self.device,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            use_mlx_dit=use_mlx,
        )
        if self.prefer_source:
            init_kwargs["prefer_source"] = self.prefer_source
        status, ok = self._dit.initialize_service(**init_kwargs)
        if not ok:
            raise RuntimeError(f"ACE-Step DiT init failed: {status}")

        # MLX VAE always activates on MPS/CPU but produces bass-only audio (no harmonics).
        # Force PyTorch VAE decode path instead.
        if self.device in ("mps", "cpu"):
            self._dit.use_mlx_vae = False
            self._dit.mlx_vae = None
            logger.info("MLX VAE disabled — using PyTorch VAE for correct harmonic output.")

        logger.info(f"DiT ready: {status}")

        # ── Ensure LM weights are downloaded ──────────────────────────────────
        lm_path = self.ckpt_dir / self.lm_model
        if not lm_path.exists():
            logger.info(f"Downloading LM '{self.lm_model}' ...")
            from acestep.model_downloader import ensure_lm_model
            dl_kwargs = dict(model_name=self.lm_model, checkpoints_dir=self.ckpt_dir)
            if self.prefer_source:
                dl_kwargs["prefer_source"] = self.prefer_source
            ok_lm, msg_lm = ensure_lm_model(**dl_kwargs)
            if not ok_lm:
                logger.warning(f"LM download failed: {msg_lm} — running in DiT-only mode")
        
        # ── Initialise LLM handler ─────────────────────────────────────────────
        logger.info(f"Initialising LM '{self.lm_model}' (backend={self.lm_backend}) ...")
        from acestep.llm_inference import LLMHandler

        self._llm: Optional[LLMHandler] = LLMHandler()
        status, ok = self._llm.initialize(
            checkpoint_dir=str(self.ckpt_dir),
            lm_model_path=self.lm_model,
            backend=self.lm_backend,
            device=self.device,
        )
        if not ok:
            logger.warning(f"LLM init skipped: {status} — running in DiT-only mode (no CoT reasoning)")
            self._llm = None
        else:
            logger.info(f"LM ready.")

        logger.info("ACE-Step 1.5 ready.")

    def unload(self) -> None:
        """Free all GPU memory held by DiT and LM handlers."""
        if self._llm is not None:
            try:
                self._llm.unload()
            except Exception:
                pass
            self._llm = None

        if self._dit is not None:
            for attr in ("model", "vae", "text_encoder", "text_tokenizer", "silence_latent"):
                obj = getattr(self._dit, attr, None)
                if obj is not None and hasattr(obj, "cpu"):
                    try:
                        obj.cpu()
                    except Exception:
                        pass
                setattr(self._dit, attr, None)
            self._dit = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("ACE-Step unloaded.")

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        duration: int,
        guidance_scale: float,
        lyrics: str = "[Instrumental]",
        bpm: Optional[int] = None,
        thinking: bool = True,
    ) -> Path:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        is_instrumental = lyrics.strip().upper() in ("[INSTRUMENTAL]", "")
        cache_key = hashlib.sha256(
            f"{prompt}|{duration}|{guidance_scale}|{lyrics}|{bpm}|{thinking}|{self.ckpt_variant}|{self.shift}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if output_path.exists():
            logger.info(f"Music cache hit: {prompt[:50]!r}")
            return output_path

        logger.info(
            f"ACE-Step generating: variant={self.ckpt_variant}, duration={duration}s, "
            f"steps={self.inference_steps}, shift={self.shift}, guidance={guidance_scale}, "
            f"thinking={thinking}, dcw_scaler={dcw_scaler}/{dcw_high_scaler}, "
            f"prompt={prompt[:60]!r}"
        )

        # DCW scalers differ by thinking mode (from Gradio demo source)
        if thinking:
            dcw_scaler, dcw_high_scaler = 0.02, 0.06
        else:
            dcw_scaler, dcw_high_scaler = 0.05, 0.02

        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=lyrics,
            instrumental=is_instrumental,
            duration=float(duration),
            guidance_scale=float(guidance_scale),
            thinking=thinking,
            bpm=bpm,
            seed=-1,
            inference_steps=self.inference_steps,
            enable_normalization=True,
            # shift=3.0 is the correct value for ACE-Step v1.5 (all variants).
            # GenerationParams defaults to 1.0 which is wrong — the Gradio demo
            # always passes 3.0 regardless of model type.
            shift=self.shift,
            # DCW (Discrete Cosine Wavelet) denoising correction — enabled for turbo,
            # scalers are thinking-mode-dependent per the official demo.
            dcw_enabled=True,
            dcw_mode="double",
            dcw_wavelet="haar",
            dcw_scaler=dcw_scaler,
            dcw_high_scaler=dcw_high_scaler,
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=True,
        )

        result = generate_music(
            self._dit,
            self._llm,
            params,
            gen_config,
            save_dir=str(self.output_dir),
        )

        if not result.success:
            raise RuntimeError(f"ACE-Step generation failed: {result.error}")

        if not result.audios:
            raise RuntimeError("ACE-Step returned no audio output")

        generated_path = Path(result.audios[0]["path"])
        # Rename to stable cache-key filename (atomic on same filesystem)
        generated_path.rename(output_path)
        logger.info(f"Music saved → {output_path}")

        return output_path

