"""Assemble per-ID tracklets from flat tracked detections.

Groups :class:`~src.detect_track.TrackedDetection` rows by ByteTrack id into
ordered sequences of ``(frame, bbox, class, conf)``, then splits them into
person and vehicle maps. Step 4's interaction heuristics consume these
tracklets (plus fps from :mod:`src.io_utils`) without touching the detector
again.

Class labels can flicker across frames for a single id; we resolve each
tracklet's class by majority vote (ties broken by lowest class id) so the
person/vehicle split is stable and deterministic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import config
from .detect_track import DetectorTracker, TrackedDetection, collect_detections


@dataclass
class Observation:
    """One detection of a tracklet in a single frame."""

    frame_idx: int
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Top-left ``(x, y, w, h)`` form used by the output JSON schema."""
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class Tracklet:
    """Full lifetime of one tracked object, ordered by frame index."""

    track_id: int
    cls: int
    observations: list[Observation]

    @property
    def is_person(self) -> bool:
        return self.cls == config.PERSON_CLASS

    @property
    def is_vehicle(self) -> bool:
        return self.cls in config.VEHICLE_CLASSES

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_idx

    @property
    def length(self) -> int:
        """Number of sampled frames this id was observed in."""
        return len(self.observations)

    @property
    def mean_conf(self) -> float:
        if not self.observations:
            return 0.0
        return sum(o.conf for o in self.observations) / len(self.observations)

    def bbox_at(self, frame_idx: int) -> Observation | None:
        """Return the observation at ``frame_idx``, or ``None`` if absent."""
        # Linear scan is fine: tracklets are short and Step 4 walks them once.
        for obs in self.observations:
            if obs.frame_idx == frame_idx:
                return obs
            if obs.frame_idx > frame_idx:
                return None
        return None


@dataclass
class TrackletCollection:
    """Person and vehicle tracklets for one clip, keyed by track id."""

    persons: dict[int, Tracklet]
    vehicles: dict[int, Tracklet]

    @property
    def all_tracklets(self) -> Iterator[Tracklet]:
        yield from self.persons.values()
        yield from self.vehicles.values()

    def summary(self) -> str:
        """One-line human-readable counts for CLI / logging."""
        p_len = sum(t.length for t in self.persons.values())
        v_len = sum(t.length for t in self.vehicles.values())
        return (
            f"persons={len(self.persons)} ({p_len} obs) "
            f"vehicles={len(self.vehicles)} ({v_len} obs)"
        )


def _majority_cls(detections: list[TrackedDetection]) -> int:
    """Most common class id; ties broken by lowest class id (deterministic)."""
    counts = Counter(d.cls for d in detections)
    return min(counts.keys(), key=lambda c: (-counts[c], c))


def build_tracklets(
    detections: Iterable[TrackedDetection],
) -> TrackletCollection:
    """Group flat detections into per-ID tracklets, split by person/vehicle.

    Observations within each tracklet are sorted by ``frame_idx``. Track ids
    whose majority class is neither person nor vehicle (should not happen with
    ``KEEP_CLASSES``) are dropped.
    """
    by_id: dict[int, list[TrackedDetection]] = defaultdict(list)
    for det in detections:
        by_id[det.track_id].append(det)

    persons: dict[int, Tracklet] = {}
    vehicles: dict[int, Tracklet] = {}

    for track_id in sorted(by_id):
        group = sorted(by_id[track_id], key=lambda d: d.frame_idx)
        cls = _majority_cls(group)
        tracklet = Tracklet(
            track_id=track_id,
            cls=cls,
            observations=[
                Observation(frame_idx=d.frame_idx, conf=d.conf, xyxy=d.xyxy)
                for d in group
            ],
        )
        if tracklet.is_person:
            persons[track_id] = tracklet
        elif tracklet.is_vehicle:
            vehicles[track_id] = tracklet

    return TrackletCollection(persons=persons, vehicles=vehicles)


def tracklets_from_video(
    video_path: str | Path,
    model: DetectorTracker | None = None,
    stride: int = config.FRAME_STRIDE,
) -> TrackletCollection:
    """Run detection+tracking on a clip and assemble person/vehicle tracklets."""
    dets = collect_detections(video_path, model=model, stride=stride)
    return build_tracklets(dets)


def _format_tracklet(t: Tracklet, names: dict[int, str]) -> str:
    name = names.get(t.cls, str(t.cls))
    return (
        f"  id={t.track_id:<4} {name:<10} "
        f"frames=[{t.start_frame}->{t.end_frame}] "
        f"n={t.length:<4} mean_conf={t.mean_conf:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble person/vehicle tracklets for a clip."
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 clip")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    args = parser.parse_args()

    tracker = DetectorTracker()
    collection = tracklets_from_video(
        args.video, model=tracker, stride=args.stride
    )

    print(f"clip={args.video.name} stride={args.stride}")
    print(collection.summary())
    print("persons:")
    for t in collection.persons.values():
        print(_format_tracklet(t, tracker.names))
    print("vehicles:")
    for t in collection.vehicles.values():
        print(_format_tracklet(t, tracker.names))
