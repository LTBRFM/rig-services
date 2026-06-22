"""
ACE-Step 1.5 music generation engine.

Model: ACE-Step/acestep-v15-xl-sft  (4B DiT, ~18.8 GB bf16)
LM:    ACE-Step/acestep-5Hz-lm-1.7B  (PyTorch backend, no vllm required)

For a 24 GB GeForce card:
  DiT alone  ≈ 18-20 GB   → loads fine without CPU offload
  1.7B LM    ≈  3-4 GB   → together: ~22-23 GB — fits in 24 GB
  (4B LM is also listed as ≥24 GB capable if you prefer the larger LM)

NEVER loaded at the same time as XTTS-v2 (ModelManager enforces this).
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def _download_hf_model(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    if (local_dir / "config.json").exists():
        logger.info(f"Model already present at {local_dir}")
        return
    logger.info(f"Downloading '{repo_id}' → {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    logger.info(f"Download complete: {local_dir}")


class MusicEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config     = config
        self.output_dir = Path(config["output_dir"]) / "music"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        music_cfg       = config["music"]
        self.dit_repo   = music_cfg["dit_repo"]
        self.lm_repo    = music_cfg["lm_repo"]
        self.lm_backend = music_cfg.get("lm_backend", "pt")
        self.ckpt_dir   = Path(music_cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Point ACE-Step at our local checkpoint directory
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(self.ckpt_dir.resolve())

        # ── Download model weights if needed ─────────────────────────────────
        dit_name = self.dit_repo.split("/")[-1]
        lm_name  = self.lm_repo.split("/")[-1]
        dit_local = self.ckpt_dir / dit_name
        lm_local  = self.ckpt_dir / lm_name

        _download_hf_model(self.dit_repo, dit_local)
        _download_hf_model(self.lm_repo,  lm_local)

        # ── Initialise DiT handler ────────────────────────────────────────────
        logger.info(f"Initialising ACE-Step DiT ({dit_name}) on {self.device} ...")
        from acestep.handler import AceStepHandler

        self._dit = AceStepHandler()
        status, ok = self._dit.initialize_service(
            project_root=str(self.ckpt_dir.parent.resolve()),
            config_path=dit_name,
            device=self.device,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step DiT init failed: {status}")
        logger.info(f"DiT ready: {status}")

        # ── Initialise LLM handler ────────────────────────────────────────────
        logger.info(f"Initialising ACE-Step LM ({lm_name}, backend={self.lm_backend}) ...")
        from acestep.llm_inference import LLMHandler

        self._llm = LLMHandler()
        status, ok = self._llm.initialize(
            checkpoint_dir=str(self.ckpt_dir.resolve()),
            lm_model_path=lm_name,
            backend=self.lm_backend,
            device=self.device,
        )
        if not ok:
            logger.warning(f"LLM init warning: {status} — continuing without LM (DiT-only mode)")
            self._llm = None

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
        bpm: int | None = None,
        thinking: bool = True,
    ) -> Path:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        cache_key = hashlib.sha256(
            f"{prompt}|{duration}|{guidance_scale}|{lyrics}|{bpm}|{thinking}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if output_path.exists():
            logger.info(f"Music cache hit: {prompt[:50]!r}")
            return output_path

        logger.info(
            f"ACE-Step generating: duration={duration}s, guidance={guidance_scale}, "
            f"thinking={thinking}, prompt={prompt[:60]!r}"
        )

        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=lyrics,
            instrumental=(lyrics.strip() == "[Instrumental]"),
            duration=float(duration),
            guidance_scale=float(guidance_scale),
            thinking=thinking,
            bpm=bpm,
            seed=-1,
            inference_steps=50,   # SFT model recommended steps
            enable_normalization=True,
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

        # Rename to stable cache-key filename
        generated_path.rename(output_path)
        logger.info(f"Music saved → {output_path}")

        return output_path
