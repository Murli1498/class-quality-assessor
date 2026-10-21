# 8x8 Class Quality Assessor

Upload (or link) an online class recording → get **Poor / Average / Good / Excellent** plus timestamped,
evidence-based reasoning.

## How it works

```
recording ─┬─ ffmpeg → 16 kHz audio ─┬─ energy VAD: speech ratio, dead air, SNR, clipping   (audio.py)
           │                         └─ faster-whisper (LOCAL) → timestamped transcript      (transcribe.py)
           └─ ffmpeg @ 0.5 fps ───────── screen activity, scene changes, faces, sharpness     (video.py)
                         │
                         ▼
        evidence packet (transcript + signals [+ optional screenshots])
                         │
                         ▼
   LLM judge scores 7 rubric dimensions 1-5, each with [mm:ss] evidence          (rubric.py, llm.py)
                         │
                         ▼
   weighted score → label via calibrated cut points (deterministic, auditable)    (scoring.py)
```

Design choices worth knowing:

* **The label is computed in code, not taken from the LLM's verdict.** The LLM scores dimensions with evidence;
  `scoring.py` applies weights and thresholds. Thresholds can be fitted on the labelled samples (`calibrate.py`).
* **Objective signals are context, not verdicts.** Loud ≠ engaging, silence can be healthy wait time, no faces may be a
  slide view. The prompt spells out these traps, and the judge is told the transcript is machine-generated (children's
  voices are often missed) so it doesn't blame the teacher for ASR errors.
* **Privacy:** audio never leaves the host (Whisper runs locally). Only the *text transcript* plus numeric signals go
  to the LLM. Screenshots are **off by default** (children may be identifiable) - enable only with an approved
  provider. Temp files are deleted after each run; nothing is stored. `recordings/`, `*.mp4` etc. are git-ignored.
* **Optional platform transcript (.vtt):** if the recording comes with a meeting-platform transcript, upload it for real
  teacher/student speaker labels (talk share, student turns). Without it, Whisper is used and speakers are inferred.
* **Graceful fallback:** with no API key, a crude metrics-only estimate is returned and flagged low-confidence.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # ffmpeg is bundled via imageio-ffmpeg, nothing to install
export LLM_API_KEY=...                     # any OpenAI-compatible provider (or put it in .env)
# optional: export LLM_BASE_URL=... LLM_MODEL=...
streamlit run app.py
```

Gemini works through its OpenAI-compatible endpoint:
`export GEMINI_API_KEY=...` (defaults to `gemini-3.8-flash`), or set `LLM_BASE_URL`/`LLM_MODEL` explicitly.
Use only a provider the organiser permits for this data.

Optional: `APP_PASSWORD=...` puts a password gate in front of the app (recommended for a hosted link, since the
recordings are confidential and the API key costs money).

## Calibrate on the labelled samples

```bash
# layout: recordings/{Poor,Average,Good,Excellent}/*.mp4   (or filename prefix: poor_01.mp4)
python calibrate.py recordings --runs 2          # prints scores, confusion matrix, per-dimension correlation
python calibrate.py recordings --runs 2 --write  # fit cut points -> calibration.json (used by the app)
```

Signals are cached in `.cache/`, so only the cheap LLM step is repeated when you tune the rubric/prompt in
`assessor/rubric.py`. A handful of recordings gives a sanity check and a threshold fit, **not** a validated model.

## Deploy

**Hugging Face Spaces (Docker SDK):** create a Space with SDK = Docker, set it **private** (or password-gated), add
`app_port: 7860` to the Space README front-matter, push this repo (the `Dockerfile` is included), then add secrets
`GEMINI_API_KEY` (or `LLM_API_KEY`), `LLM_MODEL`, optional `LLM_FALLBACK_MODELS` (comma separated) and `APP_PASSWORD`.

**Streamlit Community Cloud:** connect the GitHub repo, main file `app.py`, add the same values under *Secrets*.

Notes: a free Gemini key allows only ~20 requests/day/model - enable billing for anything shared. Every LLM call has a
hard wall-clock deadline (`LLM_DEADLINE`, default 120 s), retries transient 5xx errors, and falls through
`LLM_FALLBACK_MODELS` on quota errors or hangs. For long recordings on small hosts use the sidebar's
"first N minutes" option or the `tiny`/`base` Whisper model.

## Layout

```
app.py                Streamlit UI
calibrate.py          fit thresholds / evaluate on labelled recordings
assessor/
  rubric.py           dimensions, weights, judge prompt   <- main tuning surface
  scoring.py          weighted score -> label, calibration, no-LLM fallback
  llm.py              OpenAI-compatible judge client
  pipeline.py         orchestration
  media.py audio.py video.py transcribe.py
```

## Limitations

* Whisper `base` misses quiet/overlapping child speech and struggles with code-switched (e.g. Hindi-English) speech, so
  participation can be under-counted; there is no speaker diarisation (teacher vs student is inferred by the LLM).
* Visual analysis is shallow by default (motion/face statistics); the LLM sees screenshots only if enabled.
* LLM scoring has run-to-run variance; `calibrate.py --runs N` averages it, the app uses a single pass.
* Very long recordings: only head+tail of the transcript are sent if it exceeds ~45k characters.
