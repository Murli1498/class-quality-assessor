"""Local speech-to-text (faster-whisper). Runs on the host: the audio itself never leaves the machine."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional


@lru_cache(maxsize=2)
def _model(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(wav_path: str, model_size: str = "base", duration: Optional[float] = None,
               progress: Optional[Callable[[float], None]] = None) -> Dict:
    model = _model(model_size)
    segs, info = model.transcribe(wav_path, vad_filter=True, beam_size=1, language=None,
                                  condition_on_previous_text=False)
    out: List[Dict] = []
    for s in segs:
        out.append({"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()})
        if progress and duration:
            progress(min(1.0, s.end / duration))
    return {"language": info.language, "language_prob": round(info.language_probability, 2), "segments": out}


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def transcript_stats(segments: List[Dict], duration_s: float) -> Dict:
    text = " ".join(s["text"] for s in segments)
    words = re.findall(r"\w+", text)
    speech_s = sum(s["end"] - s["start"] for s in segments)
    questions = text.count("?")
    # gaps between consecutive segments = candidate "wait time" / turn changes
    gaps = [b["start"] - a["end"] for a, b in zip(segments, segments[1:]) if b["start"] - a["end"] > 0]
    stats = {
        "word_count": len(words),
        "words_per_min_overall": round(len(words) / max(duration_s / 60, 1e-6), 1),
        "words_per_min_when_speaking": round(len(words) / max(speech_s / 60, 1e-6), 1),
        "question_marks": questions,
        "questions_per_10min": round(questions / max(duration_s / 600, 1e-6), 1),
        "gaps_over_3s": int(sum(g > 3 for g in gaps)),
        "segments": len(segments),
    }
    # speaker-labelled transcripts (platform VTT): the speaker with the most words is taken to be the teacher
    spk_words: Dict[str, int] = {}
    for s in segments:
        if s.get("speaker"):
            spk_words[s["speaker"]] = spk_words.get(s["speaker"], 0) + len(re.findall(r"\w+", s["text"]))
    if len(spk_words) >= 2:
        teacher = max(spk_words, key=spk_words.get)
        total = sum(spk_words.values())
        others = {k: v for k, v in spk_words.items() if k != teacher}
        stats.update({
            "speaker_labels": True,
            "teacher_word_share": round(spk_words[teacher] / total, 2),
            "distinct_student_speakers": len(others),
            "student_speaking_turns": sum(1 for s in segments if s.get("speaker") and s["speaker"] != teacher),
            "students_with_5plus_words_total": sum(1 for v in others.values() if v >= 5),
        })
    return stats


def render_transcript(segments: List[Dict], max_chars: int = 45000) -> str:
    # speaker names are replaced by roles so personal names don't travel to the LLM
    roles: Dict[str, str] = {}
    words: Dict[str, int] = {}
    for s in segments:
        if s.get("speaker"):
            words[s["speaker"]] = words.get(s["speaker"], 0) + len(s["text"].split())
    if words:
        for i, name in enumerate(sorted(words, key=words.get, reverse=True)):
            roles[name] = "TEACHER" if i == 0 else f"STUDENT{i}"

    def who(s):
        return f"{roles.get(s['speaker'], '?')}: " if s.get("speaker") else ""

    lines = [f"[{fmt_ts(s['start'])}] {who(s)}{s['text']}" for s in segments]
    txt = "\n".join(lines)
    if len(txt) <= max_chars:
        return txt
    # keep head and tail, drop the middle - say so explicitly so the judge doesn't over-trust it
    half = max_chars // 2
    return txt[:half] + "\n[... middle of transcript omitted for length ...]\n" + txt[-half:]
