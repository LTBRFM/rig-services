"""
ACE-Step 1.5 music generation engine.

Checkpoint variants for a 24 GB GeForce:
  acestep-v15-xl-sft  — 4B DiT, 50 steps, ~9 GB DiT + 3 GB LM + 2 GB VAE/enc = ~14 GB  ← radio quality
  acestep-v15-sft     — 2B DiT, 50 steps, ~5 GB DiT + 3 GB LM + 2 GB VAE/enc = ~10 GB  (good quality)
  acestep-v15-turbo   — 2B DiT,  8 steps, same VRAM as sft                               (fast preview)

Model-dependent settings (derived from Gradio demo source):
  xl-sft / sft:  dcw_enabled=False, guidance_scale active, cfg_interval active, use_adg hidden
  turbo:         dcw_enabled=True,  guidance_scale hidden, cfg_interval hidden

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
        self.prefer_source   = music_cfg.get("prefer_source", None)
        self.ckpt_dir        = Path(music_cfg["checkpoint_dir"]).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # shift=3.0 correct for ALL ACE-Step v1.5 variants (Gradio demo always uses 3.0).
        self.shift = float(music_cfg.get("shift", 3.0))

        # DCW is only appropriate for turbo models; SFT/xl-sft disable it per demo source.
        variant_lower = self.ckpt_variant.lower()
        self._is_turbo = "turbo" in variant_lower and "sft" not in variant_lower
        self.dcw_enabled = music_cfg.get("dcw_enabled", self._is_turbo)

        # Fade durations for clean broadcast output (configurable, defaults radio-safe).
        self.fade_in  = float(music_cfg.get("fade_in_duration",  0.5))
        self.fade_out = float(music_cfg.get("fade_out_duration", 3.0))

        # LM CFG scale — higher = closer adherence to prompt (max 3.0 per demo).
        self.lm_cfg_scale = float(music_cfg.get("lm_cfg_scale", 3.0))

        # Peak normalisation target: -1.0 dBFS is broadcast-safe headroom.
        self.normalization_db = float(music_cfg.get("normalization_db", -1.0))

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
        duration: Optional[int],
        guidance_scale: float,
        lyrics: str = "[Instrumental]",
        bpm: Optional[int] = None,
        thinking: bool = True,
        on_progress=None,
    ) -> Path:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        is_instrumental = lyrics.strip().upper() in ("[INSTRUMENTAL]", "")
        cache_key = hashlib.sha256(
            f"{prompt}|{duration}|{guidance_scale}|{lyrics}|{bpm}|{thinking}"
            f"|{self.ckpt_variant}|{self.shift}|{self.fade_out}|{self.lm_cfg_scale}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if output_path.exists():
            logger.info(f"Music cache hit: {prompt[:50]!r}")
            return output_path

        # DCW: turbo models use it; SFT/xl-sft disable it (per Gradio demo).
        # Scalers are thinking-mode-dependent when DCW is active.
        if self.dcw_enabled:
            dcw_scaler, dcw_high_scaler = (0.02, 0.06) if thinking else (0.05, 0.02)
        else:
            dcw_scaler, dcw_high_scaler = 0.05, 0.02  # values unused when dcw_enabled=False

        logger.info(
            f"ACE-Step generating: variant={self.ckpt_variant}, duration={'auto' if duration is None else f'{duration}s'}, "
            f"steps={self.inference_steps}, shift={self.shift}, guidance={guidance_scale}, "
            f"thinking={thinking}, dcw={'on' if self.dcw_enabled else 'off'}, "
            f"fade={self.fade_in}/{self.fade_out}s, lm_cfg={self.lm_cfg_scale}, "
            f"prompt={prompt[:60]!r}"
        )

        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=lyrics,
            instrumental=is_instrumental,
            duration=float(duration) if duration is not None else None,
            guidance_scale=float(guidance_scale),
            thinking=thinking,
            bpm=bpm,
            seed=-1,
            inference_steps=self.inference_steps,
            # Broadcast-quality output settings
            enable_normalization=True,
            normalization_db=self.normalization_db,
            fade_in_duration=self.fade_in,
            fade_out_duration=self.fade_out,
            # Diffusion schedule — 3.0 required for all ACE-Step v1.5 variants
            shift=self.shift,
            # LM quality — higher cfg_scale = closer prompt adherence
            lm_cfg_scale=self.lm_cfg_scale,
            # DCW — enabled only for turbo; disabled for sft/xl-sft per demo
            dcw_enabled=self.dcw_enabled,
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
            progress=on_progress,
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

