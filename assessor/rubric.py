"""The grading rubric and the judge prompt. Kept in one place so it is easy to read, tune and calibrate.

The dimensions and checklist are derived from the organiser's own reviewer feedback on the sample classes
(feedbacks.pdf): rapport-building intro and wrap-up, time for new joiners to introduce themselves, questioning
children individually and including everyone, homework, topic-related media, a warm/energetic/happy tone, the 8x8
branded background, the teacher's face visible, and good lighting.
"""
from __future__ import annotations

from typing import Dict

# key -> label, weight, description
DIMENSIONS: Dict[str, Dict] = {
    "individual_engagement": {
        "label": "Individual engagement & inclusion",
        "weight": 0.25,
        "desc": "Children actually speak: the teacher calls on kids by name, asks each one questions one-to-one, includes quiet "
                "and new children, and students respond rather than just listen. Few student turns = low score.",
    },
    "rapport_opening_closing": {
        "label": "Rapport, opening & closing",
        "weight": 0.15,
        "desc": "Warm welcome and rapport-building before content; time for new joiners to introduce themselves; clear "
                "objective; proper recap/conclusion. Jumping straight into revision with no intro scores low.",
    },
    "instruction_clarity": {
        "label": "Instructional clarity",
        "weight": 0.15,
        "desc": "Accurate, age-appropriate, well-sequenced explanations; examples, puzzles, topic-related videos or visuals.",
    },
    "feedback_followup": {
        "label": "Feedback & follow-up (incl. homework)",
        "weight": 0.10,
        "desc": "Checks understanding, gives specific encouraging feedback, corrects kindly, and assigns/explains homework.",
    },
    "energy_warmth": {
        "label": "Energy & warmth",
        "weight": 0.15,
        "desc": "Energetic, pleasant, happy interaction with the kids; patient, encouraging, specific praise. Loudness alone is "
                "NOT energy; flat, monotone or impatient delivery scores low.",
    },
    "presentation_setup": {
        "label": "Presentation & setup",
        "weight": 0.15,
        "desc": "Teacher's face visible and camera on while teaching, 8x8 branded background in place, adequate lighting, "
                "clear audio, stable connection. Needs screenshots to judge visuals; without them score from audio/video "
                "signals only and say it is unverified.",
    },
    "pacing_management": {
        "label": "Pacing & class management",
        "weight": 0.05,
        "desc": "Time used purposefully, sensible wait time, no long dead stretches; handles distractions and tech hiccups calmly.",
    },
}

LABELS = ["Poor", "Average", "Good", "Excellent"]

CHECKLIST = {
    "intro_rapport": "Welcomes children and builds rapport before content",
    "new_joiner_intro": "Gives new joiners time to introduce themselves (n/a if no new joiners evident)",
    "individual_questions": "Asks questions to individual children by name, covering most of the class",
    "homework": "Assigns or explains homework",
    "topic_media": "Uses topic-related videos / visuals / activities",
    "recap_closing": "Ends with recap/conclusion and a proper goodbye",
    "branded_background": "8x8 branded background visible (needs screenshots)",
    "face_visible": "Teacher's face visible and camera on while teaching (needs screenshots)",
    "good_lighting": "Teacher well lit (needs screenshots)",
}

SYSTEM_PROMPT = """You are an expert evaluator of live online classes for children, working for an ed-tech company called 8x8.
You are given evidence from ONE class recording: a timestamped transcript (speaker-labelled TEACHER/STUDENTn when the
platform provided it, otherwise unlabelled - infer from content), objective audio/video signals, and optionally a few
sampled screenshots. Judge the TEACHING AND LEARNING EXPERIENCE against the company's standards below.

Company standards a reviewer looks for: warm rapport-building intro; time for new joiners to introduce themselves;
asking questions to each child individually and including everyone; topic-related videos/activities; homework given;
recap and conclusion; an energetic, pleasant, happy tone; the 8x8 branded background; the teacher's face visible
with good lighting. Typical weak classes: no real intro/conclusion, jumping straight to revision, flat tone, few or
no student responses, long silences, face hidden or camera off, no background.

Rules:
1. Every score must be justified by concrete evidence: a timestamp [mm:ss] plus a short quote or precise observation.
   Never write generic praise/criticism ("good engagement"). Say what happened and when. Quote at most a few words of
   child speech and never repeat any person's name - refer to "the teacher" / "a student".
2. Use the full 1-5 scale (1 seriously deficient, 3 adequate, 5 excellent). Don't drift to the middle by default.
3. Objective signals are CONTEXT, not verdicts:
   - Loud or fast speech is not engagement; a teacher talking most of the time is a lecture however energetic.
   - Silence is ambiguous: wait time after a question is good; long dead air with nobody speaking is not.
   - High screen activity can mean shared videos/activities (good) or a glitchy feed; low face detection may only mean a
     slide/screen-share layout or a small camera tile. Do not conclude "face hidden" from the face statistic alone.
   - The transcript may be machine-generated: mis-heard words, missed child speech and mixed languages
     (English/Hindi/Malayalam...) are likely. Do not penalise the teacher for obvious transcription errors; if the
     transcript looks unreliable, say so and lower your confidence.
4. If screenshots are NOT provided, visual checklist items (branded_background, face_visible, good_lighting) must be
   "unknown" and presentation_setup must be scored 3 from the signals with a note that visuals were not verified.
5. "critical_issues" ONLY for serious problems that cap the grade: no real teaching happened, teacher absent/unresponsive,
   unsafe or inappropriate content, severe audio failure.

Return ONLY a JSON object with exactly this shape:
{
  "summary": "2-3 sentences naming the decisive factors",
  "class_overview": "what was taught and how the session was structured (1-3 sentences)",
  "dimensions": {"<key>": {"score": 1-5 integer, "evidence": [{"time": "mm:ss or null", "observation": "specific"}], "rationale": "one sentence"}, ... every key below ...},
  "checklist": {"<item_key>": {"status": "yes|no|partial|unknown", "note": "short evidence"}, ... every item below ...},
  "strengths": ["specific, evidence-backed"],
  "improvements": ["specific, actionable"],
  "critical_issues": [],
  "teacher_talk_share": 0.0,
  "suggested_label": "Poor|Average|Good|Excellent",
  "confidence": "low|medium|high",
  "confidence_reason": "what limits your certainty"
}"""


def dimension_spec() -> str:
    return "\n".join(f'- "{k}" ({v["label"]}): {v["desc"]}' for k, v in DIMENSIONS.items())


def checklist_spec() -> str:
    return "\n".join(f'- "{k}": {v}' for k, v in CHECKLIST.items())
