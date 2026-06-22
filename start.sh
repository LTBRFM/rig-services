#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source .venv/bin/activate

echo "Starting TTS API..."
echo "  Docs:   http://localhost:8000/docs"
echo "  Voices: http://localhost:8000/voices"
echo ""

python -m app.main
