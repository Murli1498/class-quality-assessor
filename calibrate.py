"""Calibrate the label thresholds on the labelled sample recordings.

Usage:
  python calibrate.py PATH_TO_RECORDINGS [--whisper base] [--frames] [--write]

Expected layout (either works):
  recordings/Poor/*.mp4  recordings/Average/*.mp4  recordings/Good/*.mp4  recordings/Excellent/*.mp4
  or label as a filename prefix:  Poor_class1.mp4, excellent-03.mp4 ...

What it does:
  1. Extracts signals for every recording (cached in .cache/ so re-runs are cheap).
  2. Runs the LLM rubric judge (--runs times, averaged, to reduce sampling noise).
  3. Prints per-recording rubric scores, a confusion matrix, and which dimensions best separate the classes.
  4. With --write, fits 3 cut points on the weighted rubric score (max accuracy, ties -> widest margin) and saves
     calibration.json, which the app then uses.
With ~10-20 recordings this is a sanity check and a threshold fit, NOT a validated model - treat accordingly.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

from assessor import llm, pipeline, scoring
from assessor.rubric import DIMENSIONS, LABELS

EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".wav"}


# folder names used by the organiser's sample set -> our 4 labels ("demo" folders are unlabelled and skipped)
FOLDER_LABELS = {"excellent": "Excellent", "good": "Good", "average": "Average", "weak": "Poor",
                 "quality issue": "Poor"}  # ASSUMPTION: "known quality issues" classes are graded as Poor


def label_of(path: str):
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for folder, lab in FOLDER_LABELS.items():
        if folder in parts[:-1]:
            return lab
    for lab in LABELS:
        if lab.lower() in parts[:-1] or os.path.basename(path).lower().startswith(lab.lower()):
            return lab
    return None


def find_vtt(video: str):
    import glob, re
    m = re.search(r"B\d+", os.path.basename(video))
    if not m:
        return None
    hits = glob.glob(os.path.join(os.path.dirname(video), f"*{m.group(0)}*.vtt"))
    return hits[0] if hits else None


def fit_cuts(scores, labels):
    y = np.array([LABELS.index(l) for l in labels])
    s = np.array(scores)
    cands = sorted(set(np.round(np.r_[s - 0.05, s + 0.05], 2)))
    best = (-1, 0, None)
    for cuts in itertools.combinations(cands, 3):
        pred = (s[:, None] >= np.array(cuts)[None, :]).sum(1)
        acc = float((pred == y).mean())
        margin = float(min(np.diff(cuts)))
        if (acc, margin) > (best[0], best[1]):
            best = (acc, margin, cuts)
    return list(best[2]), best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--frames", action="store_true", help="send screenshots to the judge (provider must be approved)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--extract-only", action="store_true", help="only run the slow local signal extraction (no LLM)")
    ap.add_argument("--use-vtt", action="store_true", help="use the sibling platform .vtt transcript instead of Whisper")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.splitext(fn)[1].lower() in EXT and label_of(p):
                files.append(p)
    if not files:
        sys.exit("No labelled recordings found - see the docstring for the expected layout.")

    opts = pipeline.Options(whisper_model=args.whisper, send_frames=args.frames)
    rows = []
    for p in sorted(files):
        h = hashlib.md5((p + str(os.path.getsize(p))).encode()).hexdigest()[:10]
        o = pipeline.Options(whisper_model=args.whisper, send_frames=args.frames)
        src = args.whisper
        if args.use_vtt:
            v = find_vtt(p)
            if v:
                o.transcript_text, src = open(v, encoding="utf-8-sig").read(), "vtt"
        cache = os.path.join(".cache", f"{h}_{src}.json")
        print(f"\n== {p}  (true: {label_of(p)})", flush=True)
        sig, kf = pipeline.extract_signals(p, o, progress=lambda f, m: None, cache_file=cache)
        if args.extract_only:
            print("   cached:", cache, flush=True)
            continue
        per_dim = defaultdict(list)
        for i in range(args.runs):
            model_tag = llm.model_name().replace("/", "_")
            jc = f"{cache}.{model_tag}.judge{i}{'f' if args.frames else ''}.json"   # judge outputs cached: re-runs are free
            if os.path.exists(jc):
                assessment = json.load(open(jc))
            else:
                r = pipeline.assess(sig, kf, opts)
                if r["mode"] != "llm":
                    sys.exit("LLM judge failed for %s: %s" % (p, r["warnings"]))
                assessment = r["assessment"]
                json.dump(assessment, open(jc, "w"))
            for k in DIMENSIONS:
                per_dim[k].append(assessment["dimensions"][k]["score"])
        dims = {k: float(np.mean(v)) for k, v in per_dim.items()}
        score = scoring.weighted_score(dims)
        rows.append({"file": p, "true": label_of(p), "score": score, "dims": dims,
                     "critical": bool(assessment.get("critical_issues"))})
        print("   score %.2f  " % score + "  ".join(f"{k[:6]}={v:.1f}" for k, v in dims.items()))

    if args.extract_only:
        return
    scores = [r["score"] for r in rows]
    labels = [r["true"] for r in rows]
    cuts, acc = fit_cuts(scores, labels) if len(set(labels)) > 1 else (scoring.DEFAULT_CUTS, float("nan"))
    print("\n--- score by true label")
    for lab in LABELS:
        s = [r["score"] for r in rows if r["true"] == lab]
        if s:
            print(f"{lab:10s} n={len(s)}  mean={np.mean(s):.2f}  range={min(s):.2f}-{max(s):.2f}")

    def confusion(cs):
        m = np.zeros((4, 4), int)
        for r in rows:
            m[LABELS.index(r["true"]), LABELS.index(scoring.label_from_score(r["score"], cs))] += 1
        return m

    for name, cs in (("default cuts", scoring.DEFAULT_CUTS), ("fitted cuts", cuts)):
        m = confusion(cs)
        print(f"\n--- confusion ({name} {cs}); rows=true, cols=pred {LABELS}")
        print(m, " exact acc = %.2f  within-one = %.2f" % (
            np.trace(m) / m.sum(),
            sum(m[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / m.sum()))

    print("\n--- dimension separation (corr. with true ordinal label)")
    y = np.array([LABELS.index(l) for l in labels])
    for k in DIMENSIONS:
        x = np.array([r["dims"][k] for r in rows])
        c = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
        print(f"{k:24s} r = {c:+.2f}")

    if args.write:
        with open(scoring.CAL_PATH, "w") as f:
            json.dump({"cuts": cuts, "n": len(rows), "fit_accuracy": acc}, f, indent=2)
        print(f"\nWrote {scoring.CAL_PATH}: cuts={cuts} (in-sample accuracy {acc:.2f}, n={len(rows)})")


if __name__ == "__main__":
    main()
