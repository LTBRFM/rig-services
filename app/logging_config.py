"""
Logging configuration.

Features:
- Daily rotation at midnight, 31-day retention (1 month)
- Gzip compression of rotated files
- request_id context variable injected into every log record
- DEBUG to file, INFO to console
- Noisy third-party loggers silenced
"""
from __future__ import annotations

import gzip
import logging
import logging.handlers
import os
import shutil
from contextvars import ContextVar
from pathlib import Path

# ── Request ID propagation ────────────────────────────────────────────────────

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# ── Constants ─────────────────────────────────────────────────────────────────

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "tts-api.log"

_FMT = (
    "%(asctime)s  %(levelname)-8s  [%(request_id)-8s]  "
    "%(name)-30s — %(message)s"
)
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# Third-party loggers that are too chatty at INFO
_QUIET_LOGGERS = [
    "uvicorn.access",   # replaced by our middleware
    "httpx",
    "httpcore",
    "multipart",
    "numba",
    "matplotlib",
    "filelock",
    "PIL",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "huggingface_hub.file_download",
    "huggingface_hub.repocard",
]


# ── Filters ───────────────────────────────────────────────────────────────────

class _RequestIdFilter(logging.Filter):
    """Attach the current request_id context variable to every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


# ── Gzip-rotating handler ─────────────────────────────────────────────────────

class _GzipTimedRotatingHandler(logging.handlers.TimedRotatingFileHandler):
    """
    TimedRotatingFileHandler that compresses each rotated file with gzip.
    Rotates daily at midnight; keeps 31 backups (≈ 1 month).
    """

    def namer(self, default_name: str) -> str:
        return default_name + ".gz"

    def rotator(self, source: str, dest: str) -> None:
        with open(source, "rb") as f_in, gzip.open(dest, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(source)


# ── Public setup ──────────────────────────────────────────────────────────────

def setup(log_dir: Path = LOG_DIR) -> Path:
    """
    Configure the root logger.  Call once at server startup before any imports
    that create module-level loggers.

    Returns the path to the active (current) log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "tts-api.log"

    fmt_obj   = logging.Formatter(_FMT, datefmt=_DATE_FMT)
    id_filter = _RequestIdFilter()

    # ── File handler — DEBUG+, daily rotation, 31 days, gzip ─────────────────
    fh = _GzipTimedRotatingHandler(
        filename=str(log_file),
        when="midnight",
        backupCount=31,
        encoding="utf-8",
        utc=False,
        delay=False,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_obj)
    fh.addFilter(id_filter)

    # ── Console handler — INFO+ ───────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_obj)
    ch.addFilter(id_filter)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.handlers.clear()          # remove any handlers basicConfig already added
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    # ── Quieten chatty third-party libraries ──────────────────────────────────
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn.error stays at INFO (startup/shutdown messages are useful)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    return log_file
