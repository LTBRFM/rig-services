#!/usr/bin/env bash
# Persistent server launcher with auto-restart.
# Usage:  ./start.sh [port]   (default port from config.json or 8001)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8001}"
LOG="$SCRIPT_DIR/server.log"
PID_FILE="$SCRIPT_DIR/server.pid"

cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "[start.sh] Starting TTS-API server on port $PORT  (log: $LOG)"

while true; do
    python3 -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info \
        >> "$LOG" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    echo "[start.sh] Server PID $SERVER_PID — waiting..."
    wait $SERVER_PID
    EXIT=$?
    echo "[start.sh] Server exited with code $EXIT — restarting in 3s..."
    sleep 3
done
