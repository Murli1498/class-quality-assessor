"""Turns per-dimension scores into a Poor/Average/Good/Excellent label.

The label is computed deterministically from the weighted rubric score, NOT taken from the LLM's free-text verdict.
Reason: thresholds can be fitted on labelled recordings (see calibrate.py) and the result is auditable.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .rubric import DIMENSIONS, LABELS

CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration.json")
# score (1-5) cut points between Poor|Average, Average|Good, Good|Excellent. Overridden by calibration.json.
DEFAULT_CUTS = [2.5, 3.3, 4.1]


def load_cuts() -> List[float]:
    try:
        with open(CAL_PATH) as f:
            cuts = json.load(f)["cuts"]
        if len(cuts) == 3 and cuts == sorted(cuts):
            return [float(c) for c in cuts]
    except Exception:
        pass
    return DEFAULT_CUTS


def weighted_score(dim_scores: Dict[str, float]) -> float:
    tot = sum(DIMENSIONS[k]["weight"] for k in DIMENSIONS)
    return sum(DIMENSIONS[k]["weight"] * float(dim_scores.get(k, 3)) for k in DIMENSIONS) / tot


def label_from_score(score: float, cuts: Optional[List[float]] = None) -> str:
    cuts = cuts or load_cuts()
    return LABELS[sum(score >= c for c in cuts)]


def final_label(dim_scores: Dict[str, float], critical_issues: List[str]) -> Dict:
    score = weighted_score(dim_scores)
    label = label_from_score(score)
    capped = False
    if critical_issues and LABELS.index(label) > LABELS.index("Average"):
        label, capped = "Average", True
    return {"score": round(score, 2), "label": label, "capped_by_critical_issue": capped}


def heuristic_dimension_scores(audio: Dict, video: Dict, tstats: Dict) -> Dict[str, float]:
    """Crude no-LLM fallback so the app still returns something when no API key is configured.
    Built from the signals that separated the sample classes best (speech activity, dead air, words/min, screen
    activity, student turns when speaker labels exist). Flagged as low confidence in the UI."""
    def clamp(v): return max(1.0, min(5.0, v))
    sr = audio.get("speech_ratio", 0.5)
    wpm = tstats.get("words_per_min_overall", 0)
    dur_min = max(audio.get("duration_s", 600) / 60, 1)
    sil10 = audio.get("long_silences_8s_plus", 0) / dur_min * 10
    act = (video or {}).get("screen_activity_mean", 5)
    turns_pm = tstats.get("student_speaking_turns", 0) / dur_min if tstats.get("speaker_labels") else None
    engage = clamp(1 + turns_pm / 2.5) if turns_pm is not None else clamp(1.5 + tstats.get("questions_per_10min", 0) / 15)
    return {
        "individual_engagement": engage,
        "rapport_opening_closing": 3.0,
        "instruction_clarity": clamp(1.5 + wpm / 40),
        "feedback_followup": 3.0,
        "energy_warmth": clamp(1.5 + wpm / 50 + sr),
        "presentation_setup": clamp(2.5 + act / 10),
        "pacing_management": clamp(5 - sil10 / 3 - (1.0 if sr < 0.4 else 0)),
    }
