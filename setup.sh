#!/usr/bin/env bash
set -e

echo "=== TTS API Setup ==="

OS="$(uname -s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Find Python 3.10+ ──────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        MINOR="$("$candidate" -c 'import sys; print(sys.version_info.minor)')"
        MAJOR="$("$candidate" -c 'import sys; print(sys.version_info.major)')"
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
        echo "→ Python 3.10+ not found. Installing python@3.11 via Homebrew..."
        brew install python@3.11
        PYTHON="$(brew --prefix)/bin/python3.11"
    elif [ "$OS" = "Linux" ]; then
        echo "→ Python 3.10+ not found."
        if command -v apt-get &>/dev/null; then
            echo "→ Installing python3.11 via apt..."
            sudo apt-get update -qq && sudo apt-get install -y python3.11 python3.11-venv
            PYTHON="python3.11"
        elif command -v dnf &>/dev/null; then
            echo "→ Installing python3.11 via dnf..."
            sudo dnf install -y python3.11
            PYTHON="python3.11"
        else
            echo "ERROR: Cannot install Python 3.10+ automatically. Please install it manually."
            exit 1
        fi
    else
        echo "ERROR: Python 3.10+ is required. Please install it and re-run this script."
        exit 1
    fi
fi

echo "→ Using: $PYTHON ($($PYTHON --version))"

# ── 2. Recreate venv if it uses wrong Python version ─────────────────────────
if [ -d ".venv" ]; then
    VENV_PY="$(.venv/bin/python --version 2>&1 | awk '{print $2}')"
    MINOR=$(echo "$VENV_PY" | cut -d. -f2)
    if [ "$MINOR" -lt 10 ]; then
        echo "→ Existing .venv uses Python $VENV_PY — recreating..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    echo "→ Creating virtualenv..."
    if "$PYTHON" -m venv .venv 2>/dev/null; then
        :
    else
        # Debian/Ubuntu without python3-venv: ensurepip is missing and we may
        # not have sudo.  Create the venv without pip and bootstrap it manually.
        echo "→ ensurepip unavailable — bootstrapping pip via get-pip.py (no sudo needed)..."
        rm -rf .venv
        "$PYTHON" -m venv --without-pip .venv
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o .venv/get-pip.py
        .venv/bin/python .venv/get-pip.py --quiet
        rm -f .venv/get-pip.py
    fi
fi

source .venv/bin/activate
pip install --upgrade pip --quiet

# ── 3. Install PyTorch with the right CUDA/CPU variant ───────────────────────
echo "→ Detecting hardware for PyTorch install..."

GPU_VRAM_MB=0
if [ "$OS" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
    # Detect CUDA version from driver
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "→ NVIDIA GPU detected (driver reports CUDA $CUDA_VER, ${GPU_VRAM_MB} MiB VRAM)"

    # Pick best matching PyTorch CUDA wheel.  Drivers are backward compatible,
    # so a CUDA 13.x driver runs the cu124 wheel fine.
    if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; }; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        # CUDA 11.x
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi

    # Pin to 2.5.1 — torchaudio 2.6 switched to torchcodec which breaks TTS
    echo "→ Installing PyTorch 2.5.1 with CUDA support ($TORCH_INDEX)..."
    pip install "torch==2.5.1" "torchaudio==2.5.1" --index-url "$TORCH_INDEX" --quiet
else
    echo "→ No NVIDIA GPU detected (or not Linux). Installing CPU-only PyTorch 2.5.1..."
    pip install "torch==2.5.1" "torchaudio==2.5.1" --quiet
fi

# ── 4. Install remaining dependencies ────────────────────────────────────────
echo "→ Installing TTS and API dependencies..."
pip install -r requirements.txt --quiet

# ── 5. Install ACE-Step 1.5 (optional — music generation) ────────────────────
# ACE-Step xl-sft needs ~14 GB VRAM, turbo ~7 GB.  It also drags in a large
# dependency set that can upgrade transformers past what coqui-tts supports.
# Opt in explicitly:   INSTALL_MUSIC=1 ./setup.sh
if [ "${INSTALL_MUSIC:-0}" = "1" ]; then
    if [ "$GPU_VRAM_MB" -gt 0 ] && [ "$GPU_VRAM_MB" -lt 10000 ]; then
        echo "→ WARNING: GPU has only ${GPU_VRAM_MB} MiB VRAM — ACE-Step xl-sft needs ~14 GB."
        echo "           Consider config.json music.checkpoint_variant = acestep-v15-turbo."
    fi
    echo "→ Installing ACE-Step 1.5 music generation..."
    pip install "git+https://github.com/ace-step/ACE-Step-1.5.git" --quiet
else
    echo "→ Skipping ACE-Step (music).  Re-run with INSTALL_MUSIC=1 to enable POST /music."
fi

# ── 6. Download XTTS-v2 weights now (otherwise fetched lazily on first job) ──
echo "→ Ensuring XTTS-v2 weights are present..."
python - <<'PYEOF'
import json, pathlib
from huggingface_hub import snapshot_download
cfg = json.load(open("config.json"))
d = pathlib.Path(cfg["tts"]["model_dir"])
if not (d / "model.pth").exists():
    snapshot_download(repo_id=cfg["tts"]["model_repo"], local_dir=str(d))
print(f"  XTTS-v2 ready in {d}")
PYEOF

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
python - <<'PYEOF'
import torch, TTS
print(f"  coqui-tts {TTS.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  {p.total_memory/1e9:.1f} GB")
PYEOF

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Drop voice sample WAV files into voices/"
echo "  2. Run:  ./start.sh          (or install the systemd unit — see README)"
echo "  3. Open: http://<this-host>:$(python -c 'import json;print(json.load(open("config.json")).get("port",8000))')/docs"
