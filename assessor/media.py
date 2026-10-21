"""Media helpers: ffmpeg discovery, audio extraction, and fetching a recording from a link."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional

import imageio_ffmpeg


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: str) -> float:
    """Duration in seconds, parsed from `ffmpeg -i` stderr (no ffprobe needed)."""
    p = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        raise RuntimeError("Could not read the media file - is it a valid audio/video recording?")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def has_video_stream(path: str) -> bool:
    p = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True)
    return bool(re.search(r"Stream #.*Video:", p.stderr)) and "Video: mjpeg" not in p.stderr


def extract_audio(path: str, out_wav: str, max_seconds: Optional[float] = None) -> str:
    """16 kHz mono PCM wav - the format both the VAD analysis and Whisper want."""
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd.append(out_wav)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError("Audio extraction failed (does the file contain an audio track?): " + r.stderr[-300:])
    return out_wav


def download_recording(url: str, workdir: str) -> str:
    """Fetch a recording from a link. Google Drive via gdown, everything else via yt-dlp."""
    url = url.strip()
    if "drive.google.com" in url or "docs.google.com" in url:
        import gdown  # lazy: only needed for Drive links

        out = gdown.download(url=url, output=os.path.join(workdir, "recording"), quiet=True, fuzzy=True)
        if not out or not os.path.exists(out):
            raise RuntimeError("Could not download from Google Drive. The file must be shared as 'anyone with the link'.")
        return out
    import yt_dlp

    opts = {"outtmpl": os.path.join(workdir, "recording.%(ext)s"), "quiet": True, "noprogress": True,
            "format": "best[height<=720]/best", "ffmpeg_location": ffmpeg_exe()}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def new_workdir() -> str:
    return tempfile.mkdtemp(prefix="classassess_")
