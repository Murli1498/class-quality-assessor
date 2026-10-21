"""LLM judge. Uses any OpenAI-compatible endpoint (OpenAI, Gemini's OpenAI-compat API, Azure, local vLLM/Ollama...).

Configure with env vars / Streamlit secrets:
  LLM_API_KEY   (or OPENAI_API_KEY / GEMINI_API_KEY)
  LLM_BASE_URL  (optional, e.g. https://generativelanguage.googleapis.com/v1beta/openai/)
  LLM_MODEL     (default gpt-4o-mini)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from .rubric import CHECKLIST, DIMENSIONS, SYSTEM_PROMPT, checklist_spec, dimension_spec


def _cfg() -> Tuple[Optional[str], Optional[str], str]:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    base = os.getenv("LLM_BASE_URL")
    if not base and not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY") and os.getenv("GEMINI_API_KEY"):
        base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = os.getenv("LLM_MODEL") or ("gemini-3.8-flash" if base and "generativelanguage" in base else "gpt-4o-mini")
    return key, base, model


def llm_available() -> bool:
    return _cfg()[0] is not None


def model_name() -> str:
    return _cfg()[2]


def _extract_json(txt: str) -> Dict:
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _validate(d: Dict) -> Dict:
    dims = d.get("dimensions") or {}
    for k in DIMENSIONS:
        e = dims.get(k) or {}
        try:
            s = int(round(float(e.get("score", 3))))
        except (TypeError, ValueError):
            s = 3
        e["score"] = max(1, min(5, s))
        e.setdefault("evidence", [])
        e.setdefault("rationale", "")
        dims[k] = e
    d["dimensions"] = dims
    cl = d.get("checklist") or {}
    d["checklist"] = {k: {"status": str((cl.get(k) or {}).get("status", "unknown")).lower(),
                          "note": (cl.get(k) or {}).get("note", "")} for k in CHECKLIST}
    d["critical_issues"] = [c for c in (d.get("critical_issues") or []) if c]
    d.setdefault("strengths", [])
    d.setdefault("improvements", [])
    return d


class _Deadline(Exception):
    pass


def _with_deadline(fn, seconds: float):
    """httpx timeouts are per-read, so a server that trickles keepalive bytes can hang a request forever.
    Enforce a hard wall-clock limit instead (the abandoned thread is a daemon and dies with the process)."""
    import threading
    box = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa
            box["e"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise _Deadline(f"no response within {int(seconds)}s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def judge(context_text: str, keyframes: Optional[List[Tuple[float, bytes]]] = None) -> Dict:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

    key, base, model = _cfg()
    kw = dict(api_key=key, timeout=float(os.getenv('LLM_TIMEOUT', '150')), max_retries=0)
    client = OpenAI(base_url=base, **kw) if base else OpenAI(**kw)

    user_text = (
        "Score these dimensions (use these exact keys):\n" + dimension_spec()
        + "\n\nChecklist items (use these exact keys):\n" + checklist_spec()
        + ("\n\nScreenshots ARE provided below." if keyframes else "\n\nNo screenshots are provided.")
        + "\n\n" + context_text
    )
    content: List[Dict] = [{"type": "text", "text": user_text}]
    for ts, jpg in (keyframes or []):
        content.append({"type": "text", "text": f"Screenshot at {int(ts)//60:02d}:{int(ts)%60:02d}"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "detail": "low"}})

    # Try the primary model, then LLM_FALLBACK_MODELS (comma separated). Transient errors (5xx, timeouts) are retried
    # with backoff; quota errors (429) skip straight to the next model so we don't burn requests.
    DEADLINE = float(os.getenv("LLM_DEADLINE", "120"))
    models = [model] + [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    last_err = None
    for m in models:
        for attempt in range(2):
            kwargs = dict(model=m, temperature=0.2,
                          messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}])
            try:
                try:
                    r = _with_deadline(lambda: client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs), DEADLINE)
                except (APIStatusError, APITimeoutError, APIConnectionError, _Deadline):
                    raise
                except Exception:  # some compat endpoints reject response_format
                    r = _with_deadline(lambda: client.chat.completions.create(**kwargs), DEADLINE)
                out = _validate(_extract_json(r.choices[0].message.content))
                out["_model"] = m
                return out
            except APIStatusError as e:
                last_err = e
                if e.status_code == 429 or e.status_code in (400, 401, 403, 404):
                    break                                   # quota / bad model / bad key: next model
                time.sleep(8 * (attempt + 1))               # 5xx: back off and retry
            except _Deadline as e:
                last_err = e
                break                                       # model is hanging: try the next one
            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                time.sleep(8 * (attempt + 1))
            except Exception as e:                          # unparseable JSON: one more try
                last_err = e
    raise RuntimeError(f"LLM assessment failed on all models: {str(last_err)[:300]}")
