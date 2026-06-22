#!/usr/bin/env bash
set -e

echo "=== TTS API Setup ==="

OS="$(uname -s)"

# ── 1. Find Python 3.10+ ──────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
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
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet

# ── 3. Install PyTorch with the right CUDA/CPU variant ───────────────────────
echo "→ Detecting hardware for PyTorch install..."

if [ "$OS" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
    # Detect CUDA version from driver
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo "→ NVIDIA GPU detected (driver reports CUDA $CUDA_VER)"

    # Pick best matching PyTorch CUDA wheel
    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
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

# ── 5. Install ACE-Step 1.5 ──────────────────────────────────────────────────
echo "→ Installing ACE-Step 1.5 music generation..."
pip install "git+https://github.com/ace-step/ACE-Step-1.5.git" --quiet

# ── 6. Patch TTS for transformers 4.46+ / 4.50+ compatibility ─────────────────
# ACE-Step upgrades transformers, which:
#   - 4.46+: removes BeamSearchScorer from public namespace
#   - 4.50+: PreTrainedModel no longer inherits GenerationMixin
echo "→ Patching TTS for transformers 4.46+/4.50+ compatibility..."
python - <<'PYEOF'
import pathlib, TTS

tts_root = pathlib.Path(TTS.__file__).parent

# Patch 1: stream_generator.py — BeamSearchScorer removed from public API in 4.46+
sg = tts_root / "tts/layers/xtts/stream_generator.py"
if sg.exists():
    src = sg.read_text()
    old = """from transformers import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
    DisjunctiveConstraint,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    PhrasalConstraint,
    PreTrainedModel,
    StoppingCriteriaList,
)"""
    new = """from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)"""
    if old in src:
        sg.write_text(src.replace(old, new))
        print(f"  Patched: {sg}")
    else:
        print(f"  Already patched: {sg}")

# Patch 2: gpt_inference.py — GenerationMixin not inherited in 4.50+
gi = tts_root / "tts/layers/xtts/gpt_inference.py"
if gi.exists():
    src = gi.read_text()
    if "from transformers.generation import GenerationMixin" not in src:
        src = src.replace(
            "from transformers import GPT2PreTrainedModel",
            "from transformers import GPT2PreTrainedModel\nfrom transformers.generation import GenerationMixin"
        ).replace(
            "class GPT2InferenceModel(GPT2PreTrainedModel):",
            "class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):"
        )
        gi.write_text(src)
        print(f"  Patched: {gi}")
    else:
        print(f"  Already patched: {gi}")
PYEOF


echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Drop voice sample WAV files into voices/"
echo "  2. Run:  ./start.sh"
echo "  3. Open: http://localhost:8000/docs"
echo ""
echo "Note: First run will download model weights:"
echo "  - XTTS-v2:                ~2 GB  (TTS)"
echo "  - ACE-Step XL SFT (DiT): ~19 GB (Music)"
echo "  - ACE-Step LM 1.7B:       ~3 GB  (Music)"
