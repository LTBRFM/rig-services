# TTS API

High-quality Text-to-Speech REST API powered by **XTTS-v2** — the best open-source voice-cloning model.

## Features

- 🎙️ **Voice cloning** — drop any WAV file into `voices/` and use it as a speaker
- 🌍 **17 languages** — `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-cn`, `hu`, `ko`, `ja`, `hi`
- 💾 **Output caching** — identical requests are served instantly from disk
- 🖥️ **GPU + CPU** — automatically uses CUDA if available, falls back to CPU
- 📄 **Auto-generated docs** — FastAPI Swagger UI at `/docs`

---

## Setup

```bash
cd tts-api
./setup.sh        # creates .venv and installs dependencies (~4–6 GB first run)
```

> **First run** downloads the XTTS-v2 model weights (~2 GB) from HuggingFace automatically.

---

## Adding Voices

Drop audio files into the `voices/` directory:

```
voices/
├── alice.wav          ← 5–30 sec clean recording works best
├── bob.wav
└── narrator.wav
```

**Tips for best quality:**
- Use **WAV** format, 22 kHz or higher
- 10–30 seconds of clean speech (no background noise)
- Single speaker only
- Any language is fine — just match the `language` parameter at request time

---

## Start the Server

```bash
./start.sh
```

The API listens on all interfaces (`0.0.0.0:8000`) — accessible on your local network.

---

## API Reference

### `GET /voices`
List all voice samples found in `voices/`.

```bash
curl http://localhost:8000/voices
```
```json
{
  "voices": [
    {"name": "alice", "file": "alice.wav", "format": "wav"}
  ]
}
```

---

### `POST /tts`
Synthesise speech. Returns a WAV audio file.

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "alice", "language": "en"}' \
  --output speech.wav
```

**Request body:**

| Field      | Type   | Required | Default          | Description                        |
|------------|--------|----------|------------------|------------------------------------|
| `text`     | string | ✅       | —                | Text to synthesise                 |
| `voice`    | string | ✅       | —                | Voice name (filename without ext)  |
| `language` | string | ❌       | `en` (from config) | Language code                    |

---

### `GET /languages`
List all supported language codes.

---

### `GET /health`
Health check — returns `{"status": "ok"}`.

---

## Configuration (`config.json`)

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "voices_dir": "voices",
  "output_dir": "output",
  "default_language": "en",
  "cache_outputs": true,
  "max_cached_files": 100
}
```

| Key                | Description                                              |
|--------------------|----------------------------------------------------------|
| `host`             | Bind address (`0.0.0.0` = all interfaces)               |
| `port`             | TCP port                                                 |
| `model`            | TTS model identifier (do not change for XTTS-v2)        |
| `voices_dir`       | Directory containing voice sample files                  |
| `output_dir`       | Directory where generated WAVs are cached               |
| `default_language` | Language used when `language` is omitted in request     |
| `cache_outputs`    | Whether to cache outputs to disk (avoids re-synthesis)  |
| `max_cached_files` | Max number of cached WAVs before oldest are deleted     |

---

## Hardware Requirements

| Mode | RAM  | VRAM | Speed (typical sentence) |
|------|------|------|--------------------------|
| CPU  | 8 GB | —    | ~30–120 s                |
| GPU  | 8 GB | 4 GB | ~3–10 s                  |
