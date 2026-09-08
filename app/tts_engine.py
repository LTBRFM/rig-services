import gc
import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking TTS.
_orig_torch_load = torch.load
def _torch_load_compat(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_compat

warnings.filterwarnings("ignore", message="The attention mask is not set")

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}

PROFILE_DEFAULTS: Dict[str, Any] = {
    "gpt_cond_len":          30,
    "gpt_cond_chunk_len":     6,
    "max_ref_len":            30,
    "sound_norm_refs":        False,
    "temperature":            0.75,
    "top_p":                  0.85,
    "top_k":                  50,
    "repetition_penalty":     10.0,
    "length_penalty":         1.0,
    "speed":                  1.0,
    "enable_text_splitting":  True,
}


def _download_model(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    logger.info(f"Downloading TTS model '{repo_id}' → {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    logger.info("TTS model download complete.")


class TTSEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config     = config
        self.voices_dir = Path(config["voices_dir"])
        self.output_dir = Path(config["output_dir"]) / "tts"
        self.model_dir  = Path(config["tts"]["model_dir"])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.cache_outputs    = config.get("cache_outputs", True)
        self.max_cached_files = config.get("max_cached_files", 100)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not (self.model_dir / "model.pth").exists():
            _download_model(config["tts"]["model_repo"], self.model_dir)

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        xtts_config = XttsConfig()
        xtts_config.load_json(str(self.model_dir / "config.json"))
        self.model = Xtts.init_from_config(xtts_config)
        self.model.load_checkpoint(xtts_config, checkpoint_dir=str(self.model_dir), eval=True)
        self.model.to(self.device)
        self.xtts_config = xtts_config

    def unload(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        gc.collect()

    # ── Voice / profile ───────────────────────────────────────────────────────

    def _find_voice_path(self, voice_name: str) -> Path:
        for ext in SUPPORTED_AUDIO_FORMATS:
            p = self.voices_dir / f"{voice_name}{ext}"
            if p.exists():
                return p
        for f in self.voices_dir.iterdir():
            if f.is_file() and f.stem.lower() == voice_name.lower() \
                    and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                return f
        available = [v["name"] for v in self.list_voices()]
        raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")

    def _load_profile(self, voice_name: str) -> Dict[str, Any]:
        path = self.voices_dir / f"{voice_name}.json"
        overrides = {}
        if path.exists():
            with open(path) as f:
                overrides = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        return {**PROFILE_DEFAULTS, **overrides}

    def list_voices(self) -> List[Dict]:
        voices = []
        for f in sorted(self.voices_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                voices.append({
                    "name":    f.stem,
                    "file":    f.name,
                    "format":  f.suffix.lstrip("."),
                    "profile": self._load_profile(f.stem),
                })
        return voices

    def get_profile_defaults(self) -> Dict[str, Any]:
        return dict(PROFILE_DEFAULTS)

    # ── Output cache ──────────────────────────────────────────────────────────

    def _evict_old_outputs(self):
        files = sorted(self.output_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        while len(files) > self.max_cached_files:
            files.pop(0).unlink(missing_ok=True)

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(self, text: str, voice_name: str, language: str) -> Path:
        voice_path = self._find_voice_path(voice_name)
        profile    = self._load_profile(voice_name)

        # Include the sample's mtime + size so replacing a voice WAV invalidates
        # previously cached outputs for that voice.
        vstat       = voice_path.stat()
        cache_key   = hashlib.sha256(
            f"{text}|{voice_name}|{language}|{json.dumps(profile, sort_keys=True)}"
            f"|{vstat.st_mtime_ns}|{vstat.st_size}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if self.cache_outputs and output_path.exists():
            logger.info(f"Cache hit: voice='{voice_name}' lang='{language}'")
            return output_path

        logger.info(f"TTS synthesising: voice='{voice_name}', lang='{language}', chars={len(text)}")

        outputs = self.model.synthesize(
            text=text,
            config=self.xtts_config,
            speaker_wav=str(voice_path),
            language=language,
            gpt_cond_len=profile["gpt_cond_len"],
            gpt_cond_chunk_len=profile["gpt_cond_chunk_len"],
            max_ref_len=profile["max_ref_len"],
            sound_norm_refs=profile["sound_norm_refs"],
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            top_k=profile["top_k"],
            repetition_penalty=profile["repetition_penalty"],
            length_penalty=profile["length_penalty"],
            speed=profile["speed"],
            enable_text_splitting=profile["enable_text_splitting"],
        )

        import numpy as np
        import soundfile as sf
        wav_np = np.array(outputs["wav"], dtype=np.float32)
        sf.write(str(output_path), wav_np, samplerate=24000)

        if self.cache_outputs:
            self._evict_old_outputs()

        return output_path
