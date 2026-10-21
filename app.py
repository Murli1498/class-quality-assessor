"""Streamlit UI: upload (or link) a class recording -> Poor / Average / Good / Excellent + evidence."""
from __future__ import annotations

import json
import os
import shutil

import streamlit as st

# Streamlit Cloud secrets -> env vars (so the same code runs locally with .env / shell exports)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from assessor import llm, media, pipeline
from assessor.rubric import CHECKLIST, DIMENSIONS, LABELS
from assessor.transcribe import fmt_ts

st.set_page_config(page_title="8x8 Class Quality Assessor", page_icon="🎓", layout="wide")

COLORS = {"Poor": "#d64545", "Average": "#e0a030", "Good": "#3d9a5a", "Excellent": "#1f6fd1"}

# ---------------------------------------------------------------- optional access gate
pw = os.getenv("APP_PASSWORD")
if pw:
    if st.session_state.get("authed") is not True:
        entered = st.text_input("Access password", type="password")
        if entered and entered == pw:
            st.session_state["authed"] = True
            st.rerun()
        st.stop()

st.title("🎓 8x8 Class Quality Assessor")
st.caption("Upload a class recording. The app transcribes it, measures audio/visual signals, and has an AI judge "
           "score it against a teaching rubric - with timestamped evidence.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    whisper = st.selectbox("Speech-to-text model (runs locally)", ["tiny", "base", "small"], index=1,
                           help="Larger = more accurate (esp. for children / accents) but slower on CPU.")
    max_min = st.number_input("Analyse only first N minutes (0 = all)", 0, 180, 0,
                              help="Handy on small hosts to keep processing time down.")
    send_frames = st.checkbox("Send a few screenshots to the AI judge", value=False,
                              help="Lets the judge see the screen/activity. Off by default because frames may show "
                                   "identifiable children - only enable if your LLM provider is approved for this.")
    use_llm = st.checkbox("Use LLM judge", value=True, help="Untick for a crude metrics-only estimate.")
    st.divider()
    if llm.llm_available():
        st.success(f"LLM judge: `{llm.model_name()}`")
    else:
        st.warning("No LLM API key set - results will be metrics-only. See README.")
    st.caption("Recordings are processed in a temp folder and deleted afterwards. Audio never leaves this server; "
               "only the text transcript (and optionally screenshots) is sent to the LLM.")

# ---------------------------------------------------------------- input
tab_up, tab_link = st.tabs(["Upload a recording", "Paste a link"])
with tab_up:
    up = st.file_uploader("Class recording (mp4, mov, mkv, webm, mp3, m4a, wav)",
                          type=["mp4", "mov", "mkv", "webm", "avi", "mp3", "m4a", "wav", "aac", "ogg"])
with tab_link:
    url = st.text_input("Direct video URL or a Google Drive link (shared as 'anyone with the link')")
vtt_file = st.file_uploader("Optional: platform transcript (.vtt) - gives speaker labels and skips speech-to-text",
                            type=["vtt"])

go = st.button("Assess class", type="primary", disabled=not (up or url))


def run(path: str):
    opts = pipeline.Options(whisper_model=whisper, max_minutes=max_min or None, use_llm=use_llm, send_frames=send_frames,
                            transcript_text=vtt_file.getvalue().decode("utf-8-sig", "ignore") if vtt_file else None)
    bar = st.progress(0.0, text="Starting...")
    res, _ = pipeline.analyze_recording(path, opts, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    return res


if go:
    work = media.new_workdir()
    try:
        if up:
            path = os.path.join(work, "upload" + os.path.splitext(up.name)[1])
            with open(path, "wb") as f:
                shutil.copyfileobj(up, f)
        else:
            with st.spinner("Downloading recording..."):
                path = media.download_recording(url, work)
        st.session_state["result"] = run(path)
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Could not process this recording: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

# ---------------------------------------------------------------- results
res = st.session_state.get("result")
if res:
    a, v, sig = res["assessment"], res["verdict"], res["signals"]

    for w in res["warnings"]:
        st.warning(w)

    c1, c2 = st.columns([1, 2])
    with c1:
        col = COLORS[v["label"]]
        st.markdown(
            f"<div style='border-radius:14px;padding:22px;background:{col};color:white;text-align:center'>"
            f"<div style='font-size:14px;opacity:.9'>Overall quality</div>"
            f"<div style='font-size:44px;font-weight:700'>{v['label']}</div>"
            f"<div style='font-size:16px'>rubric score {v['score']} / 5</div></div>", unsafe_allow_html=True)
        st.caption(f"Confidence: **{a.get('confidence', 'n/a')}** - {a.get('confidence_reason', '')}")
        cuts = v["cuts"]
        st.caption(f"Bands: Poor < {cuts[0]} ≤ Average < {cuts[1]} ≤ Good < {cuts[2]} ≤ Excellent")
        if v["capped_by_critical_issue"]:
            st.error("Grade capped at Average because of a critical issue.")
        if res["mode"] == "metrics_only":
            st.info("Metrics-only estimate: low reliability.")
    with c2:
        st.subheader("Why")
        st.write(a.get("summary", ""))
        if a.get("class_overview"):
            st.caption("**What happened:** " + a["class_overview"])
        for ci in a.get("critical_issues", []):
            st.error(f"Critical: {ci}")

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Strengths")
        for s in a.get("strengths", []) or ["-"]:
            st.markdown(f"- {s}")
    with s2:
        st.subheader("What would improve it")
        for s in a.get("improvements", []) or ["-"]:
            st.markdown(f"- {s}")

    cl = a.get("checklist")
    if cl:
        st.subheader("8x8 standards checklist")
        icon = {"yes": "✅", "partial": "🟡", "no": "❌", "unknown": "➖"}
        for k, label in CHECKLIST.items():
            e = cl.get(k, {})
            st.markdown(f"{icon.get(e.get('status'), '➖')} **{label}**" + (f" - {e['note']}" if e.get("note") else ""))

    st.subheader("Rubric breakdown")
    for k, meta in DIMENSIONS.items():
        d = a["dimensions"][k]
        with st.expander(f"{meta['label']}  -  {d['score']}/5   (weight {int(meta['weight'] * 100)}%)", expanded=d["score"] <= 2):
            st.progress(d["score"] / 5)
            if d.get("rationale"):
                st.write(d["rationale"])
            for ev in d.get("evidence", []):
                t = ev.get("time")
                st.markdown(f"- `{t}` {ev.get('observation', '')}" if t and t != "null" else f"- {ev.get('observation', '')}")

    st.subheader("Objective signals")
    audio, vid, ts = sig["audio"], sig["video"], sig["transcript_stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{sig['duration_s'] / 60:.1f} min")
    m2.metric("Speech activity", f"{audio.get('speech_ratio', 0) * 100:.0f}%")
    m3.metric("Questions / 10 min", ts["questions_per_10min"])
    m4.metric("Silences ≥ 8 s", audio.get("long_silences_8s_plus", 0))
    if ts.get("speaker_labels"):
        n1, n2, n3 = st.columns(3)
        n1.metric("Teacher share of words", f"{ts['teacher_word_share'] * 100:.0f}%")
        n2.metric("Distinct students heard", ts["distinct_student_speakers"])
        n3.metric("Student turns / min", f"{ts['student_speaking_turns'] / max(sig['duration_s'] / 60, 1):.1f}")
    if a.get("teacher_talk_share") not in (None, "null"):
        try:
            st.metric("Estimated teacher talk share", f"{float(a['teacher_talk_share']) * 100:.0f}%")
        except (TypeError, ValueError):
            pass

    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Speech activity per minute (dips = silence / dead air)")
        st.bar_chart(audio.get("speech_ratio_per_minute", []))
    with ch2:
        if "error" not in vid:
            st.caption("Screen activity per minute (change between frames)")
            st.bar_chart(vid.get("activity_per_minute", []))
        else:
            st.caption(f"Video: {vid['error']}")
    with st.expander("All raw signals"):
        st.json({"audio": {k: x for k, x in audio.items() if k != "speech_ratio_per_minute"},
                 "video": {k: x for k, x in vid.items() if k != "activity_per_minute"}, "transcript": ts})
    with st.expander(f"Transcript ({sig['language']})"):
        st.text("\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in sig["segments"]))

    export = {k: res[k] for k in ("verdict", "assessment", "mode")}
    st.download_button("Download assessment (JSON)", json.dumps(export, indent=2), "assessment.json", "application/json")
