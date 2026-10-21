"""Audio signals from the raw waveform: how much of the class is actually speech, dead air, audio quality.

Deliberately simple (energy-based VAD with an adaptive noise floor). These are *context* signals for the
LLM judge and for the technical-quality dimension - they are not treated as quality verdicts on their own
(a loud class is not a good class; a quiet minute can be a child thinking).
"""
from __future__ import annotations

import wave
from typing import Dict, List

import numpy as np

FRAME_MS = 30


def _load(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _runs(mask: np.ndarray):
    """Yield (start_idx, end_idx, value) for runs in a boolean array."""
    if len(mask) == 0:
        return
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(mask)]
    for s, e in zip(starts, ends):
        yield int(s), int(e), bool(mask[s])


def analyze_audio(wav_path: str, sr: int = 16000) -> Dict:
    x = _load(wav_path)
    hop = int(sr * FRAME_MS / 1000)
    n = len(x) // hop
    if n < 10:
        return {"error": "audio too short"}
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)

    db = np.maximum(db, -80.0)  # digital silence would otherwise give absurd floors / SNR
    floor = float(np.percentile(db, 10))
    top = float(np.percentile(db, 95))
    # Speech = clearly above the noise floor. Guard for near-constant (silent / pure noise) recordings.
    thr = floor + max(8.0, 0.35 * (top - floor))
    voiced = db > thr
    # smooth: drop blips < 90ms and bridge gaps < 300ms
    voiced = _merge(voiced, max_gap=10, min_len=3)

    sec = FRAME_MS / 1000.0
    total = n * sec
    speech_ratio = float(voiced.mean())

    pauses: List[float] = []
    segments = 0
    for s, e, v in _runs(voiced):
        if v:
            segments += 1
        else:
            pauses.append((e - s) * sec)
    long_silences = [p for p in pauses if p >= 8.0]
    # place long silences on the timeline
    silence_spans = []
    for s, e, v in _runs(voiced):
        if not v and (e - s) * sec >= 8.0:
            silence_spans.append({"start": round(s * sec, 1), "end": round(e * sec, 1)})

    # per-minute speech ratio timeline
    per_min = int(60 / sec)
    timeline = [round(float(voiced[i:i + per_min].mean()), 3) for i in range(0, n, per_min)]

    speech_db = db[voiced]
    dyn = float(np.percentile(speech_db, 90) - np.percentile(speech_db, 10)) if speech_db.size > 20 else 0.0
    clip = float(np.mean(np.abs(x) > 0.98))

    return {
        "duration_s": round(total, 1),
        "speech_ratio": round(speech_ratio, 3),
        "noise_floor_db": round(floor, 1),
        "loudness_dynamic_range_db": round(dyn, 1),
        "clipping_fraction": round(clip, 5),
        "speech_segments_per_min": round(segments / (total / 60), 1),
        "median_pause_s": round(float(np.median(pauses)), 2) if pauses else 0.0,
        "long_silences_8s_plus": len(long_silences),
        "longest_silence_s": round(max(pauses), 1) if pauses else 0.0,
        "silence_spans": silence_spans[:30],
        "speech_ratio_per_minute": timeline,
    }


def _merge(mask: np.ndarray, max_gap: int, min_len: int) -> np.ndarray:
    m = mask.copy()
    for s, e, v in list(_runs(m)):
        if not v and (e - s) <= max_gap and s > 0 and e < len(m):
            m[s:e] = True
    for s, e, v in list(_runs(m)):
        if v and (e - s) < min_len:
            m[s:e] = False
    return m
