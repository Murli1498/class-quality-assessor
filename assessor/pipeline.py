"""End-to-end pipeline: recording -> signals (audio, video, transcript) -> LLM rubric -> label + explanation."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from . import audio as audio_mod
from . import llm, media, scoring, transcribe as tr, video as video_mod
from .rubric import DIMENSIONS


@dataclass
class Options:
    whisper_model: str = "base"
    max_minutes: Optional[float] = None   # analyse only the first N minutes (speed on small hosts)
    use_llm: bool = True
    send_frames: bool = False              # off by default: screenshots may show identifiable children
    frame_every_s: float = 2.0
    transcript_text: Optional[str] = None  # optional platform .vtt content; skips Whisper when given


Progress = Callable[[float, str], None]


def _load_cache(cache_file: str) -> Tuple[Dict, List[Tuple[float, bytes]]]:
    with open(cache_file) as f:
        sig = json.load(f)
    kf: List[Tuple[float, bytes]] = []
    d = cache_file + ".frames"
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d), key=lambda x: float(x[:-4])):
            with open(os.path.join(d, fn), "rb") as fh:
                kf.append((float(fn[:-4]), fh.read()))
    return sig, kf


def extract_signals(path: str, opts: Options, progress: Optional[Progress] = None,
                    cache_file: Optional[str] = None) -> Tuple[Dict, List[Tuple[float, bytes]]]:
    """Run the (slow, deterministic) signal extraction. Optionally cached to disk (used by calibrate.py)."""
    p = progress or (lambda f, m: None)
    if cache_file and os.path.exists(cache_file):
        return _load_cache(cache_file)

    max_s = opts.max_minutes * 60 if opts.max_minutes else None
    p(0.02, "Reading recording...")
    full_duration = media.probe_duration(path)
    duration = min(full_duration, max_s) if max_s else full_duration
    is_video = media.has_video_stream(path)

    workdir = media.new_workdir()
    try:
        p(0.05, "Extracting audio...")
        wav = media.extract_audio(path, os.path.join(workdir, "a.wav"), max_s)
        p(0.10, "Measuring audio signals...")
        a = audio_mod.analyze_audio(wav)

        keyframes: List[Tuple[float, bytes]] = []
        v: Dict = {"error": "no video stream (audio-only recording)"}
        if is_video:
            p(0.15, "Sampling video frames...")
            v, keyframes = video_mod.analyze_video(path, opts.frame_every_s, max_s)

        if opts.transcript_text:
            from .vtt import parse_vtt
            segs = [s for s in parse_vtt(opts.transcript_text) if s["start"] < duration]
            t = {"language": "from transcript file", "language_prob": 1.0, "segments": segs}
        else:
            p(0.30, f"Transcribing speech (Whisper '{opts.whisper_model}', local)...")
            t = tr.transcribe(wav, opts.whisper_model, duration, lambda f: p(0.30 + 0.60 * f, "Transcribing speech..."))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # recordings are confidential: leave nothing behind

    signals = {
        "duration_s": round(duration, 1),
        "full_duration_s": round(full_duration, 1),
        "truncated": duration < full_duration,
        "has_video": is_video,
        "audio": a,
        "video": v,
        "language": t["language"],
        "language_prob": t["language_prob"],
        "transcript_stats": tr.transcript_stats(t["segments"], duration),
        "segments": t["segments"],
    }
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(signals, f)
        d = cache_file + ".frames"
        os.makedirs(d, exist_ok=True)
        for ts, jpg in keyframes:
            with open(os.path.join(d, f"{ts}.jpg"), "wb") as fh:
                fh.write(jpg)
    return signals, keyframes


def build_context(sig: Dict) -> str:
    """The evidence packet the judge sees."""
    a = {k: v for k, v in sig["audio"].items() if k not in ("speech_ratio_per_minute", "silence_spans")}
    v = {k: v_ for k, v_ in sig["video"].items() if k != "activity_per_minute"}
    spans = ", ".join(f"{tr.fmt_ts(s['start'])}-{tr.fmt_ts(s['end'])}" for s in sig["audio"].get("silence_spans", [])[:15])
    parts = [
        f"RECORDING: {sig['duration_s']/60:.1f} min analysed" + (f" (of {sig['full_duration_s']/60:.1f} min)" if sig["truncated"] else ""),
        f"Detected language: {sig['language']} (p={sig['language_prob']})",
        "AUDIO SIGNALS: " + json.dumps(a),
        "Silences of 8s+ at: " + (spans or "none"),
        "VIDEO SIGNALS: " + json.dumps(v),
        "TRANSCRIPT STATS: " + json.dumps(sig["transcript_stats"]),
        "TRANSCRIPT (machine-generated, timestamps = segment start):\n" + tr.render_transcript(sig["segments"]),
    ]
    return "\n\n".join(parts)


def assess(sig: Dict, keyframes: Optional[List[Tuple[float, bytes]]], opts: Options,
           progress: Optional[Progress] = None) -> Dict:
    p = progress or (lambda f, m: None)
    word_count = sig["transcript_stats"]["word_count"]
    result: Dict = {"signals": sig, "mode": "llm", "warnings": []}

    if word_count < 30:
        result["warnings"].append("Almost no speech was transcribed - the audio may be silent, very noisy, or not in a "
                                  "supported language. Treat the result with caution.")

    judged = None
    if opts.use_llm and llm.llm_available():
        p(0.92, f"Assessing class quality with {llm.model_name()}...")
        try:
            judged = llm.judge(build_context(sig), keyframes if opts.send_frames else None)
        except Exception as e:
            result["warnings"].append(f"LLM assessment failed ({e}); fell back to metrics-only estimate.")
    elif opts.use_llm:
        result["warnings"].append("No LLM API key configured; using a crude metrics-only estimate.")

    if judged is None:
        result["mode"] = "metrics_only"
        ds = scoring.heuristic_dimension_scores(sig["audio"], sig["video"], sig["transcript_stats"])
        judged = {
            "summary": "Metrics-only estimate (no LLM judge): based on speech activity, questions per 10 min, pauses and "
                       "audio quality. This cannot assess teaching content or how children responded.",
            "class_overview": "", "strengths": [], "improvements": [], "critical_issues": [],
            "confidence": "low", "confidence_reason": "No LLM judge was used.",
            "dimensions": {k: {"score": round(v), "evidence": [], "rationale": "heuristic from signals"}
                           for k, v in ds.items()},
        }

    dim_scores = {k: judged["dimensions"][k]["score"] for k in DIMENSIONS}
    result["assessment"] = judged
    result["verdict"] = scoring.final_label(dim_scores, judged.get("critical_issues", []))
    result["verdict"]["cuts"] = scoring.load_cuts()
    p(1.0, "Done")
    return result


def analyze_recording(path: str, opts: Options, progress: Optional[Progress] = None,
                      cache_file: Optional[str] = None) -> Tuple[Dict, List[Tuple[float, bytes]]]:
    sig, kf = extract_signals(path, opts, progress, cache_file)
    return assess(sig, kf, opts, progress), kf
