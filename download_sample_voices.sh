#!/usr/bin/env bash
# Downloads the official XTTS-v2 sample voices from HuggingFace.
# These are the same reference clips the model was tested with — guaranteed to work well.

set -e
cd "$(dirname "$0")"

BASE="https://huggingface.co/coqui/XTTS-v2/resolve/main/samples"

# local_name:remote_filename
VOICES="
en_female:en_sample.wav
de_male:de_sample.wav
es_male:es_sample.wav
fr_female:fr_sample.wav
pt_male:pt_sample.wav
tr_male:tr_sample.wav
zh_female:zh-cn-sample.wav
ja_female:ja-sample.wav
"

echo "=== Downloading XTTS-v2 sample voices ==="
echo "  Destination: voices/"
echo ""

for ENTRY in $VOICES; do
    NAME="${ENTRY%%:*}"
    REMOTE="${ENTRY##*:}"
    DEST="voices/${NAME}.wav"
    if [ -f "$DEST" ]; then
        echo "  ✓ $NAME (already exists, skipping)"
    else
        echo "  ↓ $NAME ..."
        curl -fL --progress-bar "$BASE/$REMOTE" -o "$DEST"
    fi
done

echo ""
echo "✓ Done. Available voices:"
for f in voices/*.wav; do
    echo "    $(basename "$f" .wav)"
done
echo ""
echo "Example request:"
echo '  curl -X POST http://localhost:8000/tts \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"text": "Hello world!", "voice": "en_female", "language": "en"}'"'"' \'
echo '    --output speech.wav'
