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
cd rig-services
./setup.sh                    # TTS only: creates .venv, installs deps, downloads XTTS-v2 (~2 GB)
INSTALL_MUSIC=1 ./setup.sh    # additionally install ACE-Step music generation (needs ≥10 GB VRAM)
```

`setup.sh` needs no root: if `python3-venv`/ensurepip is missing it bootstraps
pip from `get-pip.py` inside the venv.

> **Python 3.12 note:** the original `TTS` package refuses Python ≥3.12, so the
> maintained fork [`coqui-tts`](https://pypi.org/project/coqui-tts/) is used.
> It provides the same `TTS.*` import namespace and needs no transformers patches.

### Linux rig (Ubuntu 24.04, GTX 1660 Ti 6 GB) — verified

| Component     | Version                   |
|---------------|---------------------------|
| Python        | 3.12.3 (system)           |
| torch         | 2.5.1+cu124 (driver 595 / CUDA 13.2 is backward compatible) |
| coqui-tts     | 0.27.5                    |
| transformers  | 4.57.x (pinned `<5`)      |
| XTTS-v2 VRAM  | ~2.0 GB peak              |
| Speed         | ~2.5 s for a 5 s sentence; ~20 s cold model load on first job |

Music generation (ACE-Step) is **not installed** on this rig: `xl-sft` needs
~14 GB VRAM and the 1660 Ti has 6 GB.  `POST /music` returns **503** with an
explanatory message.  TTS is unaffected.

---

## Adding Voices

Either upload through the API (no restart needed, works from any machine):

```bash
curl -X POST http://192.168.50.61:8000/voices \
  -F "file=@narrator.wav" -F "name=narrator" \
  -F 'profile={"temperature": 0.65, "speed": 1.0}'     # profile is optional
```

or drop audio files straight into the `voices/` directory:

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

### Option A — foreground with auto-restart

```bash
./start.sh            # port from config.json (8000); ./start.sh 8001 to override
```

### Option B — systemd user service (recommended on the Linux rig)

Runs at boot without a login (linger enabled), restarts on crash, logs to journald.

```bash
cp tts-api.service ~/.config/systemd/user/     # edit WorkingDirectory/ExecStart if the repo moved
systemctl --user daemon-reload
systemctl --user enable --now tts-api.service
loginctl enable-linger "$USER"                  # keep it running after logout / start at boot

systemctl --user status tts-api.service         # state
journalctl --user -u tts-api.service -f         # live log
systemctl --user restart tts-api.service        # after code changes
```

The API listens on all interfaces (`0.0.0.0:8000`) — accessible on your local network.
On the rig that is **http://192.168.50.61:8000** (Swagger UI at `/docs`).
Application logs also go to `logs/tts-api.log` and are readable remotely via `GET /logs`.

### From another machine

```bash
API=http://192.168.50.61:8000
JOB=$(curl -s -X POST $API/tts -H 'Content-Type: application/json' \
  -d '{"text":"Good morning, you are listening to the breakfast show.","voice":"en_female","language":"en"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
until curl -s $API/jobs/$JOB | grep -q '"status":"done"'; do sleep 1; done
curl -s $API/result/$JOB -o speech.wav
```

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
    {"name": "alice", "file": "alice.wav", "format": "wav", "profile": {"temperature": 0.75, "...": "..."}}
  ]
}
```

---

### `POST /voices`
Upload a speaker sample and register it as a voice (`multipart/form-data`).
Returns **201** with the new voice entry; the voice is usable in `POST /tts` immediately.

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | `.wav`, `.mp3`, `.flac` or `.ogg`, max 50 MB — decoded on upload to make sure it is valid audio |
| `name` | no | Voice identifier (`[A-Za-z0-9_-]`, max 64 chars). Defaults to the file name without extension |
| `overwrite` | no | `true` to replace an existing voice; otherwise a duplicate name returns **409** |
| `profile` | no | JSON object of XTTS overrides, stored as `voices/<name>.json` (see `GET /profile/defaults`) |

```bash
curl -X POST http://localhost:8000/voices -F "file=@narrator.wav" -F "name=narrator"
```
```json
{
  "name": "narrator", "file": "narrator.wav", "format": "wav", "profile": {"...": "..."},
  "status": "ok", "replaced": false, "size_kb": 812, "duration_s": 18.9, "sample_rate": 22050
}
```

Replacing a sample automatically invalidates cached TTS outputs for that voice.

---

### `DELETE /voices/{name}`
Remove a voice sample and its `<name>.json` profile. Returns **404** if the voice does not exist.

```bash
curl -X DELETE http://localhost:8000/voices/narrator
```

---

### `POST /tts`
Submit a synthesis job. Returns `{"job_id": ..., "status": "pending", "queue_position": N}`
immediately; poll `GET /jobs/{job_id}` until `status == "done"`, then download the
WAV from `GET /result/{job_id}`.  Identical requests are served from the on-disk cache.

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "alice", "language": "en"}'
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
| GPU  | 8 GB | 4 GB | ~2–10 s (GTX 1660 Ti: ~2.5 s) |

Music generation (ACE-Step, optional) additionally needs ~14 GB VRAM for `xl-sft`
or ~7 GB for `turbo`.
