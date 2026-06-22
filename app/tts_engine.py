import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any

import torch

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking TTS checkpoint loading.
_orig_torch_load = torch.load
def _torch_load_compat(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_compat

warnings.filterwarnings("ignore", message="The attention mask is not set")

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}

# Defaults applied when a voice profile doesn't override a value.
# See: TTS/tts/models/xtts.py :: full_inference()
PROFILE_DEFAULTS: Dict[str, Any] = {
    # — Voice cloning / reference audio —
    "gpt_cond_len":       30,    # seconds of reference audio used for cloning (more = better clone, slower)
    "gpt_cond_chunk_len": 6,     # chunk size for conditioning; must be <= gpt_cond_len
    "max_ref_len":        30,    # max reference audio length in seconds
    "sound_norm_refs":    False, # normalise volume of reference audio before cloning

    # — Generation quality —
    "temperature":        0.75,  # randomness/expressiveness (0.0 = robotic, 1.0 = very expressive)
    "top_p":              0.85,  # nucleus sampling threshold (lower = safer/less creative)
    "top_k":              50,    # top-k candidates per step (lower = more predictable)
    "repetition_penalty": 10.0,  # penalise repeated tokens (higher = less repetition)
    "length_penalty":     1.0,   # prefer shorter (<1) or longer (>1) outputs

    # — Delivery —
    "speed":              1.0,   # speech rate multiplier (0.5 = half speed, 2.0 = double)
    "enable_text_splitting": True, # split long texts into sentences for better quality
}


def _download_model(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    logger.info(f"Downloading model '{repo_id}' → {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    logger.info("Download complete.")


def _load_xtts(model_dir: Path, device: str):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
    model.to(device)
    return model, config


class TTSEngine:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.voices_dir   = Path(self.config["voices_dir"])
        self.output_dir   = Path(self.config["output_dir"])
        self.model_dir    = Path(self.config["model_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.cache_outputs    = self.config.get("cache_outputs", True)
        self.max_cached_files = self.config.get("max_cached_files", 100)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not (self.model_dir / "model.pth").exists():
            _download_model(self.config["model_repo"], self.model_dir)

        logger.info(f"Loading XTTS-v2 from {self.model_dir} on {self.device} ...")
        self.model, self.xtts_config = _load_xtts(self.model_dir, self.device)
        logger.info("XTTS-v2 ready.")

    # ── Voice / profile management ────────────────────────────────────────────

    def _find_voice_path(self, voice_name: str) -> Path:
        for ext in SUPPORTED_AUDIO_FORMATS:
            candidate = self.voices_dir / f"{voice_name}{ext}"
            if candidate.exists():
                return candidate
        for f in self.voices_dir.iterdir():
            if f.is_file() and f.stem.lower() == voice_name.lower() \
                    and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                return f
        available = [v["name"] for v in self.list_voices()]
        raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")

    def _load_profile(self, voice_name: str) -> Dict[str, Any]:
        """Load per-voice profile JSON if it exists, merged with defaults."""
        profile_path = self.voices_dir / f"{voice_name}.json"
        overrides: Dict[str, Any] = {}
        if profile_path.exists():
            with open(profile_path) as f:
                overrides = json.load(f)
        return {**PROFILE_DEFAULTS, **overrides}

    def list_voices(self) -> List[Dict]:
        voices = []
        for f in sorted(self.voices_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                profile = self._load_profile(f.stem)
                voices.append({
                    "name":    f.stem,
                    "file":    f.name,
                    "format":  f.suffix.lstrip("."),
                    "profile": profile,
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

        cache_key   = hashlib.sha256(
            f"{text}|{voice_name}|{language}|{json.dumps(profile, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        output_path = self.output_dir / f"{cache_key}.wav"

        if self.cache_outputs and output_path.exists():
            logger.info(f"Cache hit: voice='{voice_name}' lang='{language}'")
            return output_path

        logger.info(
            f"Synthesising: voice='{voice_name}', lang='{language}', "
            f"chars={len(text)}, speed={profile['speed']}, temp={profile['temperature']}"
        )

        outputs = self.model.synthesize(
            text=text,
            config=self.xtts_config,
            speaker_wav=str(voice_path),
            language=language,
            # reference audio params
            gpt_cond_len=profile["gpt_cond_len"],
            gpt_cond_chunk_len=profile["gpt_cond_chunk_len"],
            max_ref_len=profile["max_ref_len"],
            sound_norm_refs=profile["sound_norm_refs"],
            # generation params
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            top_k=profile["top_k"],
            repetition_penalty=profile["repetition_penalty"],
            length_penalty=profile["length_penalty"],
            speed=profile["speed"],
            enable_text_splitting=profile["enable_text_splitting"],
        )

        import torchaudio
        torchaudio.save(
            str(output_path),
            torch.tensor(outputs["wav"]).unsqueeze(0),
            sample_rate=24000,
        )

        if self.cache_outputs:
            self._evict_old_outputs()

        return output_path
