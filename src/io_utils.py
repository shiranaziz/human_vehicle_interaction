"""Video input/output helpers: metadata probing, frame iteration, MP4 writing.

Kept deliberately small and dependency-light (only OpenCV + numpy). Everything
downstream (detection, tracking, visualization) reads frames through here so the
frame-index bookkeeping and the ``FRAME_STRIDE`` sub-sampling live in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Static properties of a clip, read once up front.

    ``frame_count``/``fps`` come straight from the container headers, so they can
    occasionally be wrong or zero for exotic encodings; callers should treat them
    as best-effort. ``duration_s`` is derived rather than read to stay consistent
    with ``frame_count``.
    """

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def resolution(self) -> list[int]:
        return [self.width, self.height]

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize in the exact shape the output JSON schema expects."""
        return {
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "resolution": self.resolution,
            "duration_s": round(self.duration_s, 3),
        }


def probe_video(path: str | Path) -> VideoMetadata:
    """Read fps/size/frame-count without decoding the whole clip."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        return VideoMetadata(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, stride: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(original_frame_index, bgr_frame)`` for every ``stride``-th frame.

    The yielded index is the position in the *original* video (0, stride, 2*stride,
    ...), not a compacted counter, so timestamps computed as ``index / fps`` stay
    correct after sub-sampling.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def open_writer(
    path: str | Path, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    """Create an MP4 writer (``mp4v``) for the given output path and frame size.

    ``size`` is ``(width, height)`` to match OpenCV's convention.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {path}")
    return writer
