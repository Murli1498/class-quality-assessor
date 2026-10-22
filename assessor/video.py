"""Visual signals: sampled frames -> screen activity, scene changes, face presence, picture quality.

Frames are streamed out of ffmpeg at a low rate (default 1 frame / 2 s) so even long recordings are cheap.
These are weak signals by design. "Lots of motion" can be a lively screen share or a flickering webcam;
"no faces" can be a slide-only view. They give the LLM context, they do not decide the grade.
"""
from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from .media import ffmpeg_exe

W, H = 640, 360


def analyze_video(path: str, every_s: float = 2.0, max_seconds: Optional[float] = None,
                  n_keyframes: int = 8) -> Tuple[Dict, List[Tuple[float, bytes]]]:
    import cv2  # local import: a missing system library then fails this step, not app startup
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-an", "-vf", f"fps=1/{every_s},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd.append("-")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = W * H * 3

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prev = None
    diffs: List[float] = []
    faces: List[int] = []
    bright: List[float] = []
    sharp: List[float] = []
    jpgs: List[bytes] = []
    i = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(small, prev))))
        else:
            diffs.append(0.0)
        prev = small
        bright.append(float(gray.mean()))
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        f = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(28, 28))
        faces.append(int(len(f)))
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpgs.append(enc.tobytes() if ok else b"")
        i += 1
    proc.stdout.close()
    proc.wait()

    if i == 0:
        return {"error": "no video frames decoded"}, []

    d = np.array(diffs)
    fa = np.array(faces)
    metrics = {
        "frames_sampled": i,
        "sample_interval_s": every_s,
        "screen_activity_mean": round(float(d.mean()), 2),
        "screen_activity_p90": round(float(np.percentile(d, 90)), 2),
        "static_fraction": round(float((d < 0.5).mean()), 3),
        "scene_changes": int((d > 12).sum()),
        "scene_changes_per_10min": round(float((d > 12).sum()) / (i * every_s / 600), 1),
        "frames_with_face_fraction": round(float((fa > 0).mean()), 3),
        "avg_faces_when_present": round(float(fa[fa > 0].mean()), 2) if (fa > 0).any() else 0.0,
        "mean_brightness": round(float(np.mean(bright)), 1),
        "mean_sharpness": round(float(np.mean(sharp)), 1),
        "activity_per_minute": [round(float(d[j:j + int(60 / every_s)].mean()), 2)
                                for j in range(0, i, max(1, int(60 / every_s)))],
    }

    # keyframes: evenly spaced + the biggest changes (likely slide / activity switches)
    idx = set(np.linspace(0, i - 1, min(n_keyframes, i)).astype(int).tolist())
    for j in np.argsort(-d)[: max(0, n_keyframes // 3)]:
        idx.add(int(j))
    keyframes = [(round(j * every_s, 1), jpgs[j]) for j in sorted(idx) if jpgs[j]]
    return metrics, keyframes
