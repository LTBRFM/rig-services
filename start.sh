#!/usr/bin/env bash
# Persistent server launcher with auto-restart.
# Usage:  ./start.sh [port]   (default: "port" from config.json, else 8000)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

CONFIG_PORT="$(python3 -c 'import json; print(json.load(open("config.json")).get("port", 8000))')"
PORT="${1:-$CONFIG_PORT}"
HOST="$(python3 -c 'import json; print(json.load(open("config.json")).get("host", "0.0.0.0"))')"
LOG="$SCRIPT_DIR/server.log"
PID_FILE="$SCRIPT_DIR/server.pid"

# Allow up to 2 hours per generation — CPU needs ~72 min for 175s/50-step;
# 24 GB GPU needs ~5 min.  Override with ACESTEP_GENERATION_TIMEOUT env var.
export ACESTEP_GENERATION_TIMEOUT="${ACESTEP_GENERATION_TIMEOUT:-7200}"

echo "[start.sh] Starting TTS-API server on $HOST:$PORT  (log: $LOG)"
echo "[start.sh] ACESTEP_GENERATION_TIMEOUT=${ACESTEP_GENERATION_TIMEOUT}s"

# Forward SIGTERM/SIGINT (e.g. from systemd) to the child so shutdown is clean.
trap 'kill "$SERVER_PID" 2>/dev/null; exit 0' TERM INT

while true; do
    python3 -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        >> "$LOG" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    echo "[start.sh] Server PID $SERVER_PID — waiting..."
    wait $SERVER_PID && EXIT=0 || EXIT=$?
    echo "[start.sh] Server exited with code $EXIT — restarting in 3s..."
    sleep 3
done
