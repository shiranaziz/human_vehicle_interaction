"""Annotated-MP4 visualization of YOLO tracking + Qwen-bound crops.

Every person/vehicle YOLO track box is drawn in red (class, id, confidence).
Boxes that Step 5 actually crops and sends to Qwen (person + vehicle on the
selected describe frames of a kept interaction) are drawn in yellow.

Output fps is scaled by ``1/stride`` so the annotated clip plays back at roughly
the same wall-clock speed as the source despite frame sub-sampling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detect_track import DetectorTracker, collect_detections
from .interactions import Interaction, find_interactions, select_describe_frames
from .io_utils import iter_frames, open_writer, probe_video
from .tracklets import Observation, Tracklet, TrackletCollection, build_tracklets

# BGR: YOLO track boxes red; Qwen-bound person/vehicle crops yellow.
_TRACK_COLOR = (0, 0, 255)
_QWEN_COLOR = (0, 255, 255)
_TEXT_COLOR = (255, 255, 255)


def _draw_label(
    frame: np.ndarray,
    x1: int,
    y1: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled label chip above the box (or below if it would clip)."""
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = y1 - th - baseline
    if top < 0:
        top = y1 + th + baseline
    cv2.rectangle(
        frame, (x1, top - th - baseline), (x1 + tw, top), color, -1
    )
    cv2.putText(
        frame,
        label,
        (x1, top - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_tracklets_on_frame(
    frame: np.ndarray,
    items: list[tuple[Tracklet, Observation]],
    names: dict[int, str],
    color: tuple[int, int, int] = _TRACK_COLOR,
    highlight_ids: set[int] | None = None,
    highlight_labels: dict[int, str] | None = None,
    highlight_color: tuple[int, int, int] = _QWEN_COLOR,
) -> np.ndarray:
    """Draw YOLO track boxes; yellow for person/vehicle ids sent to Qwen."""
    highlight_ids = highlight_ids or set()
    highlight_labels = highlight_labels or {}
    for tracklet, obs in items:
        x1, y1, x2, y2 = (int(v) for v in obs.xyxy)
        is_qwen = tracklet.track_id in highlight_ids
        color_use = highlight_color if is_qwen else color
        thickness = 3 if is_qwen else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_use, thickness)
        kind = names.get(tracklet.cls, str(tracklet.cls))
        label = f"{kind} #{tracklet.track_id} {obs.conf:.2f}"
        if is_qwen:
            tag = highlight_labels.get(tracklet.track_id, "qwen")
            label = f"{label} [{tag}]"
        _draw_label(frame, x1, y1, label, color_use)
    return frame


def _index_by_frame(
    collection: TrackletCollection,
) -> dict[int, list[tuple[Tracklet, Observation]]]:
    """Map frame index -> list of (tracklet, observation) active in that frame."""
    by_frame: dict[int, list[tuple[Tracklet, Observation]]] = defaultdict(list)
    for tracklet in collection.all_tracklets:
        for obs in tracklet.observations:
            by_frame[obs.frame_idx].append((tracklet, obs))
    return by_frame


def _qwen_highlights(
    interactions: list[Interaction],
    collection: TrackletCollection,
) -> tuple[dict[int, set[int]], dict[int, dict[int, str]]]:
    """Build per-frame track-id sets / labels for crops sent to Qwen.

    Only accepted interactions contribute. Highlights match
    :func:`select_describe_frames` (the frames Step 5 actually captions), not
    every geometric near frame.
    """
    ids_by_frame: dict[int, set[int]] = defaultdict(set)
    labels_by_frame: dict[int, dict[int, str]] = defaultdict(dict)
    for inter in interactions:
        if not inter.is_accepted:
            continue
        person = collection.persons.get(inter.person_id)
        vehicle = collection.vehicles.get(inter.vehicle_id)
        if person is None or vehicle is None:
            continue

        # Prefer Qwen action detail when present; else the action label (type).
        detail = ""
        if isinstance(inter.evidence, dict):
            detail = str(inter.evidence.get("action_detail") or "").strip()
        tag = detail if detail else f"action:{inter.type}"
        # Keep overlay labels short enough to read on the video.
        if len(tag) > 28:
            tag = tag[:27] + "…"

        for f in select_describe_frames(inter, person, vehicle):
            ids_by_frame[f].add(inter.person_id)
            ids_by_frame[f].add(inter.vehicle_id)
            labels_by_frame[f][inter.person_id] = tag
            labels_by_frame[f][inter.vehicle_id] = tag
    return ids_by_frame, labels_by_frame


def annotate_clip(
    video_path: str | Path,
    out_path: str | Path | None = None,
    model: DetectorTracker | None = None,
    stride: int = config.FRAME_STRIDE,
    collection: TrackletCollection | None = None,
    interactions: list[Interaction] | None = None,
) -> Path:
    """Write an annotated MP4; return the path.

    Red = all person/vehicle YOLO track boxes. Yellow = person + vehicle boxes
    on the describe frames that Step 5 sends to Qwen.

    If ``collection`` is provided, the video is re-read and those tracklets are
    drawn (no second inference pass). Otherwise detection+tracking runs once,
    tracklets are assembled, then the same clip is drawn from the collection.

    If ``interactions`` is None and a collection is available, Step 4 is run
    automatically so yellow highlights still appear.

    ``out_path`` defaults to ``outputs/<clip>_annotated.mp4``.
    """
    video_path = Path(video_path)
    if out_path is None:
        out_path = config.OUTPUTS_DIR / f"{video_path.stem}_annotated.mp4"
    out_path = Path(out_path)

    meta = probe_video(video_path)
    tracker = model or DetectorTracker()
    out_fps = max(meta.fps / stride, 1.0)

    if collection is None:
        # Single inference pass: collect detections, then assemble tracklets.
        dets = collect_detections(video_path, model=tracker, stride=stride)
        collection = build_tracklets(dets)

    if interactions is None:
        interactions = find_interactions(
            collection, meta.fps, include_passing_by=False
        )

    by_frame = _index_by_frame(collection)
    qwen_ids, qwen_labels = _qwen_highlights(interactions, collection)
    n_qwen_frames = len(qwen_ids)

    writer = open_writer(out_path, out_fps, (meta.width, meta.height))
    try:
        n = 0
        for frame_idx, frame in iter_frames(video_path, stride=stride):
            items = by_frame.get(frame_idx, [])
            writer.write(
                draw_tracklets_on_frame(
                    frame,
                    items,
                    tracker.names,
                    highlight_ids=qwen_ids.get(frame_idx, set()),
                    highlight_labels=qwen_labels.get(frame_idx, {}),
                )
            )
            n += 1
    finally:
        writer.release()

    print(
        f"wrote {out_path} ({n} frames @ {out_fps:.2f} fps; "
        f"{collection.summary()}; qwen_describe_frames={n_qwen_frames})"
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Write an annotated MP4 of YOLO person/vehicle tracks "
            "(red=YOLO, yellow=person+vehicle crops sent to Qwen)."
        )
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 clip")
    parser.add_argument("--out", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    args = parser.parse_args()

    annotate_clip(args.video, out_path=args.out, stride=args.stride)
