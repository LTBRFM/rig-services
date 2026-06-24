"""
Global music preset manager.

A single preset stores the default prompt and all generation parameters.
POST /music only needs `lyrics` when a preset is configured — all other
fields fall back to the preset.

Persisted to preset.json at the project root; PATCH merges deeply so
callers only need to send the fields they want to change.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_PRESET: Dict[str, Any] = {
    "prompt":         "",
    "guidance_scale": 7.0,
    "bpm":            None,
    "thinking":       True,
    "duration":       None,
    "mastering": {
        "enabled":                  True,
        "target_lufs":              -12.0,    # broadcast feel, matches commercial AI generators
        "low_cut_hz":               30.0,
        "compression_ratio":        4.0,      # tighter dynamics = more perceived loudness
        "compression_threshold_db": -18.0,
        "compression_attack_ms":    10.0,
        "compression_release_ms":   100.0,
        "high_shelf_gain_db":       2.0,      # brighter top end
        "high_shelf_hz":            8000.0,
        "limiter_ceiling_db":       -0.5,     # harder ceiling for more loudness
        "matchering_enabled":       False,
    },
}


class PresetManager:
    """Thread-safe preset manager backed by a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return _deep_copy(self._data)

    def patch(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge *updates* into the current preset and persist."""
        with self._lock:
            self._data = _deep_merge(self._data, updates)
            self._save()
            return _deep_copy(self._data)

    def mastering_settings(self) -> Dict[str, Any]:
        with self._lock:
            return _deep_copy(self._data.get("mastering", {}))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    stored = json.load(f)
                return _deep_merge(_deep_copy(DEFAULT_PRESET), stored)
            except Exception:
                pass
        return _deep_copy(DEFAULT_PRESET)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict, updates: Dict) -> Dict:
    result = dict(base)
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_copy(d: Dict) -> Dict:
    """Simple deep-copy for JSON-compatible dicts."""
    return json.loads(json.dumps(d))
