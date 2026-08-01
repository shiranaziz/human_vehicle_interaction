"""YOLO detection + ByteTrack tracking wrapper.

This is the single entry point that turns a raw MP4 into a stream of
per-frame tracked detections with stable IDs. Downstream steps (tracklet
assembly, interaction heuristics, visualization) all consume the
``TrackedDetection`` objects produced here so the Ultralytics API is touched in
exactly one place.

Design notes:
- We drive frame reading ourselves via :mod:`src.io_utils` and feed frames to
  ``model.track(..., persist=True)`` one at a time. This lets ``FRAME_STRIDE``
  sub-sampling and ByteTrack's temporal state stay consistent: the tracker only
  ever sees the (sub-sampled) sequence we actually process.
- ``config`` is imported first so its cache-dir env vars are set before
  Ultralytics/torch import (they read those at import time).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Local config MUST be imported before ultralytics/torch: it sets cache-dir
# environment variables at import time that those libraries read on import.
from . import config
from .io_utils import iter_frames
from ultralytics import YOLO


@dataclass
class TrackedDetection:
    """One tracked box in one frame.

    ``xyxy`` is absolute pixel coordinates ``(x1, y1, x2, y2)`` in the original
    frame. ``track_id`` is ByteTrack's stable id for the object across frames.
    """

    frame_idx: int
    track_id: int
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass
class FrameResult:
    """All tracked detections for a single processed frame, plus the frame image.

    The image is carried alongside so a single detection pass can feed both the
    tracklet builder (which ignores ``frame``) and the visualizer (which draws on
    it) without decoding or running inference twice.
    """

    frame_idx: int
    frame: np.ndarray
    detections: list[TrackedDetection]


class DetectorTracker:
    """Thin wrapper around ``YOLO(...).track(..., tracker='bytetrack.yaml')``."""

    def __init__(self, model_path: str | Path = config.MODEL) -> None:
        # Determinism must be set before the model builds any RNG state.
        config.set_determinism()

        self.model = YOLO(str(model_path))
        # id -> class name (e.g. {0: 'person', 2: 'car', ...}); used for labels.
        self.names: dict[int, str] = self.model.names

    def class_name(self, cls: int) -> str:
        return self.names.get(cls, str(cls))

    def track(
        self, video_path: str | Path, stride: int = config.FRAME_STRIDE
    ) -> Iterator[FrameResult]:
        """Yield a :class:`FrameResult` for every processed frame of the clip.

        Only classes in ``config.KEEP_CLASSES`` (person + vehicles) are kept, and
        only boxes that received a track id are emitted (untracked one-off
        detections are dropped to keep IDs meaningful downstream).
        """
        for frame_idx, frame in iter_frames(video_path, stride=stride):
            results = self.model.track(
                frame,
                persist=True,
                tracker=config.TRACKER,
                classes=list(config.KEEP_CLASSES),
                conf=config.CONF,
                iou=config.IOU,
                imgsz=config.IMGSZ,
                verbose=False,
            )
            yield FrameResult(
                frame_idx=frame_idx,
                frame=frame,
                detections=self._parse(results[0], frame_idx),
            )

    @staticmethod
    def _parse(result, frame_idx: int) -> list[TrackedDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        detections: list[TrackedDetection] = []
        for i in range(len(ids)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            detections.append(
                TrackedDetection(
                    frame_idx=frame_idx,
                    track_id=int(ids[i]),
                    cls=int(cls[i]),
                    conf=float(conf[i]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections


def collect_detections(
    video_path: str | Path,
    model: DetectorTracker | None = None,
    stride: int = config.FRAME_STRIDE,
) -> list[TrackedDetection]:
    """Run tracking over a clip and return a flat list of all detections.

    Convenience for Step 3 (tracklet assembly), which does not need the frame
    images. Reuses an existing :class:`DetectorTracker` if provided.
    """
    tracker = model or DetectorTracker()
    out: list[TrackedDetection] = []
    for fr in tracker.track(video_path, stride=stride):
        out.extend(fr.detections)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO+ByteTrack on a clip.")
    parser.add_argument("video", type=Path, help="Path to an MP4 clip")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    args = parser.parse_args()

    tracker = DetectorTracker()
    dets = collect_detections(args.video, model=tracker, stride=args.stride)

    ids = {d.track_id for d in dets}
    by_class = Counter(tracker.class_name(d.cls) for d in dets)
    print(f"clip={args.video.name} stride={args.stride}")
    print(f"detections={len(dets)} unique_track_ids={len(ids)}")
    print("per-class detections:", dict(by_class))
