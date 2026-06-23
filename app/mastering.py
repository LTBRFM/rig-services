"""
Audio mastering pipeline.

Two modes:
  1. Pedalboard DSP chain  — always available; no reference needed.
  2. Matchering            — reference-based spectral/dynamic matching;
                             activates when matchering_enabled=True AND a
                             reference WAV exists at reference_path.

After either mode a final LUFS normalisation + true-peak ceiling is applied.
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
        shutil.copy2(raw_path, mastered_path)
        return mastered_path

    try:
        ref_ok = (
            settings.get("matchering_enabled", False)
            and reference_path is not None
            and reference_path.exists()
        )

        if ref_ok:
            logger.info(f"Mastering: matchering against {reference_path.name}")
            _run_matchering(raw_path, mastered_path, reference_path)
        else:
            logger.info("Mastering: pedalboard DSP chain")
            _run_pedalboard(raw_path, mastered_path, settings)

        # Final LUFS + true-peak pass on the mastered file
        _normalise_lufs_inplace(
            mastered_path,
            target_lufs=settings.get("target_lufs", -14.0),
            ceiling_db=settings.get("limiter_ceiling_db", -1.0),
        )
        logger.info(f"Mastering complete → {mastered_path.name}")

    except Exception:
        logger.exception("Mastering failed — falling back to raw copy")
        shutil.copy2(raw_path, mastered_path)

    return mastered_path


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
