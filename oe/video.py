"""Video helpers: normalise a phone clip, read frames, write H.264 overlays."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    """System ffmpeg if present, otherwise the static build bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # pragma: no cover - dependency guard
        raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg") from error


@dataclass
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def prepare_video(
    src: Path,
    dst: Path,
    *,
    max_size: int = 1280,
    fps: float | None = None,
    start: float | None = None,
    end: float | None = None,
) -> VideoInfo:
    """Trim, fix the frame rate (so frame N is exactly N/fps seconds) and shrink the clip.

    Phones often record variable frame rate and big files; SAM's frame indices refer to
    the uploaded file, so the timing must be constant for the physics to be right.
    """
    source = probe(src)
    target_fps = fps or (round(source.fps, 3) if source.fps > 1 else 30.0)
    scale = (
        f"scale='if(gte(iw,ih),min({max_size},iw),-2)':'if(gte(iw,ih),-2,min({max_size},ih))'"
    )
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += [
        "-i", str(src),
        "-vf", f"fps={target_fps},{scale}",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    info = probe(dst)
    info.fps = target_fps  # use the exact rate we asked for, not a rounded container value
    return info


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    """Read one frame by decoding forward (exact, and fine for short clips)."""
    for i, frame in enumerate(iter_frames(path)):
        if i == index:
            return frame
    raise IndexError(f"{path} has no frame {index}")


class H264Writer:
    """Pipe BGR frames into ffmpeg to get an MP4 that plays everywhere."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the overlay video")
