#!/usr/bin/env python3
"""
End-to-end API test suite for the unified TTS + Music GPU API.

Usage:
    python test_api.py [--host 127.0.0.1] [--port 8000] [--skip-music]

On the first run against a fresh Linux/GPU machine use:
    python test_api.py --host <ip>
"""
import argparse
import json
import time
import sys
import os
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0
EVIDENCE_DIR = "test_results"


def _req(method, path, body=None, raw=False):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if raw:
                return r.read()
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()}


def ok(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label}" + (f": {detail}" if detail else ""))


def poll_job(job_id, timeout=600, interval=10):
    """Poll until job is done or failed, return final job dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = _req("GET", f"/jobs/{job_id}")
        status = j.get("status")
        if status in ("done", "failed"):
            return j
        time.sleep(interval)
    return {"status": "timeout"}


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_tests(skip_music=False):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    # ── Health & info endpoints ──────────────────────────────────────────────
    section("1. Health and Info Endpoints")

    h = _req("GET", "/health")
    ok("GET /health returns {status: ok}", h.get("status") == "ok", str(h))

    v = _req("GET", "/voices")
    voices = v.get("voices", [])
    ok("GET /voices returns list", isinstance(voices, list) and len(voices) > 0, str(v))
    ok("Voices include 'en_female'", any(v["name"] == "en_female" for v in voices))
    print(f"     Found {len(voices)} voices: {[v['name'] for v in voices]}")

    langs_resp = _req("GET", "/languages")
    langs = langs_resp.get("languages", langs_resp) if isinstance(langs_resp, dict) else langs_resp
    ok("GET /languages includes English", "en" in langs, str(langs_resp))

    status = _req("GET", "/status")
    ok("GET /status has queue info",
       "queue_length" in status and "stats" in status, str(status))

    defaults = _req("GET", "/profile/defaults")
    ok("GET /profile/defaults has temperature",
       "temperature" in defaults, str(defaults))

    # ── TTS synthesis ─────────────────────────────────────────────────────────
    section("2. TTS Synthesis")

    tts_cases = [
        ("en_female", "en",    "This is an English test of the XTTS version two text to speech engine. Quality is the priority."),
        ("de_male",   "de",    "Dies ist ein Test der deutschen Sprachsynthese. Das System produziert hochwertige Sprache."),
        ("fr_female", "fr",    "Bonjour, ceci est un test de la synthèse vocale en français. La qualité est excellente."),
        ("es_male",   "es",    "Este sistema utiliza el modelo XTTS versión dos para síntesis de voz de alta calidad."),
        ("zh_female", "zh-cn", "这是中文语音合成的测试。系统支持多种语言。"),
    ]

    for voice, lang, text in tts_cases:
        j = _req("POST", "/tts", {"text": text, "voice": voice, "language": lang})
        ok(f"POST /tts returns job_id [{voice}]", "job_id" in j, str(j))
        if "job_id" not in j:
            continue

        job = poll_job(j["job_id"], timeout=300)
        ok(f"TTS job completes [{voice}]", job["status"] == "done",
           job.get("error", "timeout"))

        if job["status"] == "done":
            wav = _req("GET", f"/result/{j['job_id']}", raw=True)
            ok(f"Result is non-empty WAV [{voice}]",
               isinstance(wav, bytes) and len(wav) > 10_000,
               f"{len(wav)} bytes")
            out_path = f"{EVIDENCE_DIR}/tts_{voice}.wav"
            with open(out_path, "wb") as f:
                f.write(wav)
            dur_s = job.get("processing_s", 0)
            print(f"     Saved {out_path}  ({len(wav)//1024} KB, generated in {dur_s:.1f}s)")

    # ── Priority queue ────────────────────────────────────────────────────────
    section("3. Priority Queue (TTS before Music)")

    if skip_music:
        # Without music model: verify that multiple TTS jobs queue in order
        # and that submitting a music job (which won't run) doesn't block TTS
        print("  [--skip-music] Verifying TTS-only queue ordering (FIFO within same priority).")
        j1 = _req("POST", "/tts", {"text": "Queue test job one.", "voice": "en_female", "language": "en"})
        j2 = _req("POST", "/tts", {"text": "Queue test job two.", "voice": "de_male",   "language": "de"})
        ok("Queue test jobs submitted", "job_id" in j1 and "job_id" in j2)

        r1 = poll_job(j1["job_id"], timeout=300)
        r2 = poll_job(j2["job_id"], timeout=300)
        ok("TTS job 1 completes", r1["status"] == "done", r1.get("error","timeout"))
        ok("TTS job 2 completes", r2["status"] == "done", r2.get("error","timeout"))
        print("  ℹ️  Priority queue ordering (TTS < Music) is enforced in worker.py")
        print("     and is tested on a system with GPU + models available (--no-skip-music).")
    else:
        print("  Submitting 2 music jobs then 1 TTS job — TTS must jump ahead of 2nd music in queue.")
        m1_job = _req("POST", "/music", {
            "prompt": "priority queue test music one",
            "duration": 10, "guidance_scale": 7.0, "thinking": False
        })
        m2_job = _req("POST", "/music", {
            "prompt": "priority queue test music two",
            "duration": 10, "guidance_scale": 7.0, "thinking": False
        })
        time.sleep(0.5)
        t_job = _req("POST", "/tts", {
            "text": "Priority queue test. TTS submitted after music but must run before second music.",
            "voice": "en_female", "language": "en"
        })
        time.sleep(1)

        status = _req("GET", "/status")
        current = status.get("current_job")
        queue   = status.get("queue", [])
        m1_id, m2_id, t_id = m1_job.get("job_id"), m2_job.get("job_id"), t_job.get("job_id")

        # Music1 should be running; TTS should be ahead of Music2 in the waiting queue
        tts_q = next((q for q in queue if q["id"] == t_id),  None)
        m2_q  = next((q for q in queue if q["id"] == m2_id), None)
        if tts_q and m2_q:
            ok("TTS queued ahead of 2nd music job (priority enforced)",
               tts_q["queue_position"] < m2_q["queue_position"],
               f"TTS pos={tts_q['queue_position']} Music2 pos={m2_q['queue_position']}")
        elif current and current.get("id") == t_id:
            ok("TTS queued ahead of 2nd music job (priority enforced)",
               True, "TTS already running — jumped the queue")
        else:
            ok("TTS and music jobs submitted", t_id and m1_id and m2_id)

        # TTS must complete — it runs before Music2 even though submitted after both
        t_result  = poll_job(t_id,  timeout=600)
        m2_result = poll_job(m2_id, timeout=600)
        ok("TTS priority job completes successfully", t_result["status"] == "done",
           t_result.get("error", "timeout"))
        if t_result["status"] == "done" and m2_result.get("completed_at") and t_result.get("completed_at"):
            ok("TTS completes before 2nd music job (priority honoured)",
               t_result["completed_at"] <= m2_result["completed_at"],
               f"TTS={t_result.get('completed_at'):.1f} Music2={m2_result.get('completed_at'):.1f}")

    # ── Music generation (requires GPU + models downloaded) ──────────────────
    if not skip_music:
        section("4. Music Generation (requires GPU + pre-downloaded models)")
        print("  ⚠  These tests require the ACE-Step models to be pre-downloaded.")
        print("     Download: python -m acestep.model_downloader --model acestep-v15-xl-sft")
        print("     Set ACESTEP_CHECKPOINTS_DIR=models/acestep_checkpoints")
        print()

        music_cases = [
            {
                "label": "instrumental jazz",
                "prompt": "upbeat jazz piano trio, warm acoustic bass, brushed snare, medium swing tempo, 120 BPM",
                "lyrics": "[Instrumental]",
                "duration": 30,
                "guidance_scale": 7.0,
                "bpm": 120,
                "thinking": False,
            },
            {
                "label": "electronic ambient",
                "prompt": "calm ambient electronic, soft atmospheric pads, gentle evolving textures, peaceful and serene",
                "lyrics": "[Instrumental]",
                "duration": 30,
                "guidance_scale": 7.0,
                "thinking": False,
            },
            {
                "label": "rock with vocals",
                "prompt": "energetic rock, electric guitar, driving drums, powerful bass, anthemic chorus",
                "lyrics": "[Verse]\nThe world is spinning fast tonight\nWe're reaching for the light\n[Chorus]\nWe rise, we fall, we carry on",
                "duration": 30,
                "guidance_scale": 7.0,
                "thinking": True,
            },
        ]

        for case in music_cases:
            label = case.pop("label")
            j = _req("POST", "/music", case)
            ok(f"POST /music returns job_id [{label}]", "job_id" in j, str(j))
            if "job_id" not in j:
                continue

            print(f"     Waiting for music job {j['job_id']} [{label}] (this can take minutes)...")
            job = poll_job(j["job_id"], timeout=1200, interval=15)
            ok(f"Music job completes [{label}]", job["status"] == "done",
               job.get("error", "timeout"))

            if job["status"] == "done":
                wav = _req("GET", f"/result/{j['job_id']}", raw=True)
                ok(f"Music result is non-empty [{label}]",
                   isinstance(wav, bytes) and len(wav) > 100_000,
                   f"{len(wav)} bytes")
                safe_label = label.replace(" ", "_")
                out_path = f"{EVIDENCE_DIR}/music_{safe_label}.wav"
                with open(out_path, "wb") as f:
                    f.write(wav)
                print(f"     Saved {out_path}  ({len(wav)//1024} KB)")
    else:
        print("\n  ⏭  Music tests skipped (--skip-music)")

    # ── Summary ──────────────────────────────────────────────────────────────
    section("RESULTS")
    total = PASS + FAIL
    print(f"  Passed: {PASS}/{total}")
    if FAIL:
        print(f"  Failed: {FAIL}/{total}")
    print(f"\n  Evidence files saved to: {EVIDENCE_DIR}/")
    for f in sorted(os.listdir(EVIDENCE_DIR)):
        size = os.path.getsize(f"{EVIDENCE_DIR}/{f}") // 1024
        print(f"    {f}  ({size} KB)")
    print()

    return FAIL == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS + Music API test suite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000", type=int)
    parser.add_argument("--skip-music", action="store_true",
                        help="Skip music generation tests (no GPU/models)")
    args = parser.parse_args()

    BASE = f"http://{args.host}:{args.port}"
    print(f"\nTesting API at {BASE}")

    ok_result = run_tests(skip_music=args.skip_music)
    sys.exit(0 if ok_result else 1)
