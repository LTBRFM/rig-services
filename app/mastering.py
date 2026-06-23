"""
Audio mastering pipeline.

Chain (when matchering is enabled):
  1. Pedalboard DSP  — dynamics: highpass → compression → high-shelf EQ → limiter
  2. Matchering      — tonal colour: spectral/dynamic matching to reference track
  3. LUFS normalise  — final loudness + true-peak ceiling

When matchering is disabled only steps 1 + 3 run.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def apply_mastering(
    raw_path: Path,
    mastered_path: Path,
    settings: Dict,
    reference_path: Optional[Path] = None,
) -> Path:
    """
    Master *raw_path* → *mastered_path* using settings from the preset.

    Returns *mastered_path* on success.  On any error the raw file is copied
    as-is so the caller always gets a valid output file.
    """
    if not settings.get("enabled", True):
        logger.info("Mastering disabled — copying raw file as result")
        shutil.copy2(raw_path, mastered_path)
        return mastered_path

    try:
        ref_ok = (
            settings.get("matchering_enabled", False)
            and reference_path is not None
            and reference_path.exists()
        )

        # Log input levels
        _log_audio_levels(raw_path, "Input (raw)")

        # Step 1: pedalboard DSP chain (always)
        logger.info(
            f"Mastering step 1/{'3' if ref_ok else '2'}: pedalboard DSP  "
            f"[highpass={settings.get('low_cut_hz', 30)}Hz  "
            f"comp={settings.get('compression_ratio', 3.0)}:{1}@{settings.get('compression_threshold_db', -20)}dB  "
            f"shelf=+{settings.get('high_shelf_gain_db', 1.5)}dB@{settings.get('high_shelf_hz', 8000)}Hz  "
            f"limiter={settings.get('limiter_ceiling_db', -1.0)}dBFS]"
        )
        _run_pedalboard(raw_path, mastered_path, settings)
        _log_audio_levels(mastered_path, "After pedalboard")

        # Step 2: matchering on top for tonal colour (when reference available)
        if ref_ok:
            logger.info(f"Mastering step 2/3: matchering against {reference_path.name}")
            _run_matchering(mastered_path, mastered_path, reference_path)
            _log_audio_levels(mastered_path, "After matchering")

        # Step 3: final LUFS + true-peak pass
        target_lufs = settings.get("target_lufs", -14.0)
        ceiling_db  = settings.get("limiter_ceiling_db", -1.0)
        logger.info(
            f"Mastering step {'3/3' if ref_ok else '2/2'}: LUFS normalise  "
            f"[target={target_lufs} LUFS  ceiling={ceiling_db} dBFS]"
        )
        _normalise_lufs_inplace(mastered_path, target_lufs=target_lufs, ceiling_db=ceiling_db)
        _log_audio_levels(mastered_path, "Final output")

        logger.info(f"Mastering complete → {mastered_path.name}")

    except Exception:
        logger.exception("Mastering failed — falling back to raw copy")
        shutil.copy2(raw_path, mastered_path)

    return mastered_path


def _log_audio_levels(path: Path, label: str) -> None:
    """Measure and log LUFS + peak for a WAV file."""
    try:
        import pyloudnorm as pyln  # type: ignore
        audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
        meter     = pyln.Meter(sr)
        loudness  = meter.integrated_loudness(audio)
        peak_db   = 20 * np.log10(max(np.max(np.abs(audio)), 1e-9))
        lufs_str  = f"{loudness:.1f} LUFS" if np.isfinite(loudness) else "LUFS=∞"
        logger.debug(f"  {label}: {lufs_str}  peak={peak_db:.1f} dBFS  sr={sr}Hz  dur={len(audio)/sr:.1f}s")
    except Exception as exc:
        logger.debug(f"  {label}: level measurement failed — {exc}")


# ── Pedalboard chain ──────────────────────────────────────────────────────────

def _run_pedalboard(raw_path: Path, out_path: Path, s: Dict) -> None:
    from pedalboard import Pedalboard, HighpassFilter, HighShelfFilter, Compressor, Limiter  # type: ignore

    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=float(s.get("low_cut_hz", 30.0))),
        Compressor(
            threshold_db=float(s.get("compression_threshold_db", -20.0)),
            ratio=float(s.get("compression_ratio", 3.0)),
            attack_ms=float(s.get("compression_attack_ms", 10.0)),
            release_ms=float(s.get("compression_release_ms", 100.0)),
        ),
        HighShelfFilter(
            cutoff_frequency_hz=float(s.get("high_shelf_hz", 8000.0)),
            gain_db=float(s.get("high_shelf_gain_db", 1.5)),
        ),
        Limiter(threshold_db=float(s.get("limiter_ceiling_db", -1.0))),
    ])

    audio, sr = sf.read(str(raw_path), dtype="float32", always_2d=True)
    # pedalboard: (channels, samples)
    processed = chain(audio.T, sr)
    # soundfile: (samples, channels)
    sf.write(str(out_path), processed.T, sr, subtype="PCM_24")


# ── Matchering ────────────────────────────────────────────────────────────────

def _run_matchering(raw_path: Path, out_path: Path, reference_path: Path) -> None:
    import matchering as mg  # type: ignore
    mg.process(
        target=str(raw_path),
        reference=str(reference_path),
        results=[mg.pcm24(str(out_path))],
    )


# ── LUFS normalisation ────────────────────────────────────────────────────────

def _normalise_lufs_inplace(path: Path, target_lufs: float, ceiling_db: float) -> None:
    try:
        import pyloudnorm as pyln  # type: ignore
    except ImportError:
        logger.warning("pyloudnorm not installed — skipping LUFS normalisation")
        return

    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        logger.warning(f"LUFS measurement returned {loudness} — skipping normalisation")
        return

    audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # True-peak ceiling
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(audio))
    if peak > ceiling_lin:
        audio *= ceiling_lin / peak

    sf.write(str(path), audio, sr, subtype="PCM_24")
