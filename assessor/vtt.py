"""Parse WebVTT transcripts exported by meeting platforms (e.g. 'Speaker Name: text' cues) into segments.

Optional input: when a recording ships with a platform transcript we can use real speaker labels
(teacher vs students) instead of guessing from Whisper text.
"""
from __future__ import annotations

import re
from typing import Dict, List

_TS = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*-->\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def _sec(h, m, s) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + float(s)


def parse_vtt(text: str) -> List[Dict]:
    segs: List[Dict] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").lstrip("\ufeff"))
    for b in blocks:
        lines = [l for l in b.split("\n") if l.strip()]
        for i, l in enumerate(lines):
            m = _TS.search(l)
            if not m:
                continue
            start, end = _sec(*m.groups()[:3]), _sec(*m.groups()[3:])
            body = " ".join(x.strip() for x in lines[i + 1:])
            body = re.sub(r"<[^>]+>", "", body)
            spk = None
            sm = re.match(r"^([^:]{1,60}):\s+(.*)$", body)
            if sm:
                spk, body = sm.group(1).strip(), sm.group(2).strip()
            if body:
                segs.append({"start": round(start, 1), "end": round(end, 1), "text": body, "speaker": spk})
            break
    return segs
