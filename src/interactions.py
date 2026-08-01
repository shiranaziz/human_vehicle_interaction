"""Person–vehicle interaction heuristics (Step 4).

Consumes person/vehicle tracklets from :mod:`src.tracklets` and fps from
:mod:`src.io_utils`. For every person–vehicle pair that co-exist in time, we
compute per-frame proximity metrics (IoU, containment, center gap), mark
"near" frames, require a minimum dwell, then classify the pair as
``enter`` / ``exit`` / ``interacting`` / ``passing_by`` using dwell + relative
motion + track birth/death near the vehicle.

Accepted proposals (sent to the VLM): get-in, get-out, and stricter sustained
contact without a boarding signature (``interacting``, e.g. trunk/load).
Door-side boarding beside a truncated vehicle box (no IoU) is also accepted as
``enter``. Brief or transit proximity stays ``passing_by`` and is not exported.

Thresholds live in :mod:`src.config` and are tuned in Step 7.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Sequence

from . import config
from .detect_track import DetectorTracker
from .io_utils import probe_video
from .tracklets import Tracklet, TrackletCollection, tracklets_from_video

InteractionType = Literal["enter", "exit", "interacting", "passing_by"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _area(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Fraction of ``inner`` area that lies inside ``outer`` (0..1)."""
    inner_area = _area(inner)
    if inner_area <= 0.0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area


def center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Euclidean distance between box centers (pixels)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def horizontal_edge_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Horizontal gap between boxes (0 if they overlap in x)."""
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    return max(0.0, max(ax1 - bx2, bx1 - ax2))


def vertical_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Length of vertical overlap between two boxes (0 if none)."""
    _, ay1, _, ay2 = a
    _, by1, _, by2 = b
    return max(0.0, min(ay2, by2) - max(ay1, by1))


def door_side_near(
    person_xyxy: tuple[float, float, float, float],
    vehicle_xyxy: tuple[float, float, float, float],
    gap_frac: float = config.NEAR_DOOR_GAP_FRAC,
) -> bool:
    """True when the person stands beside the vehicle (door-side), even if boxes
    do not overlap — common when YOLO truncates the vehicle box.

    Requires a positive horizontal gap (strictly beside, not overlapping in x)
    and meaningful vertical overlap so sidewalk walk-bys under the car box are
    not treated as door contact.
    """
    v_ov = vertical_overlap(person_xyxy, vehicle_xyxy)
    if v_ov <= 0.0:
        return False
    person_h = max(1.0, person_xyxy[3] - person_xyxy[1])
    # At least ~25% of the person height must share the vehicle's vertical band.
    if v_ov < 0.25 * person_h:
        return False
    gap = horizontal_edge_gap(person_xyxy, vehicle_xyxy)
    if gap <= 0.0:
        # Overlapping in x belongs to IoU/containment near, not door-side.
        return False
    vw = max(1.0, vehicle_xyxy[2] - vehicle_xyxy[0])
    return gap <= gap_frac * vw


# ---------------------------------------------------------------------------
# Per-frame / per-pair structures
# ---------------------------------------------------------------------------


@dataclass
class FrameMetrics:
    """Proximity metrics for one co-visible person–vehicle frame."""

    frame_idx: int
    iou: float
    containment: float
    center_dist: float
    person_xyxy: tuple[float, float, float, float]
    vehicle_xyxy: tuple[float, float, float, float]

    @property
    def is_near(self) -> bool:
        return (
            self.iou >= config.NEAR_IOU
            or self.containment >= config.NEAR_CONTAINMENT
        )


@dataclass
class Interaction:
    """One classified person–vehicle interaction (or rejected passing-by)."""

    person_id: int
    vehicle_id: int
    vehicle_cls: int
    type: InteractionType
    frame_range: tuple[int, int]
    time_span_s: tuple[float, float]
    confidence: float
    near_frames: list[int] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        """True for proposals kept for VLM / export (not walk-bys)."""
        return self.type in ("enter", "exit", "interacting")

    def summary(self) -> str:
        return (
            f"person={self.person_id} vehicle={self.vehicle_id} "
            f"type={self.type:<12} frames=[{self.frame_range[0]}->{self.frame_range[1]}] "
            f"t=[{self.time_span_s[0]:.2f}->{self.time_span_s[1]:.2f}]s "
            f"conf={self.confidence:.2f} dwell={self.evidence.get('dwell_frames', 0)}"
        )


def select_describe_frames(
    interaction: Interaction,
    person: Tracklet,
    vehicle: Tracklet,
    max_frames: int = config.DESCRIBE_MAX_FRAMES,
) -> list[int]:
    """Pick up to ``max_frames`` near frames: start, peak-containment, end.

    If peak lands on start or end, backfill a mid near-frame between the
    endpoints so Qwen still gets more than two crops. These are the frames
    Step 5 crops and sends to Qwen (and that visualization highlights in yellow).
    """
    near = interaction.near_frames
    if not near:
        return []
    if len(near) <= max_frames:
        return list(near)

    best_f = near[0]
    best_c = -1.0
    for f in near:
        p = person.bbox_at(f)
        v = vehicle.bbox_at(f)
        if p is None or v is None:
            continue
        c = containment(p.xyxy, v.xyxy)
        if c > best_c:
            best_c = c
            best_f = f

    ordered = sorted({near[0], best_f, near[-1]})
    # Peak often equals start/end; fill remaining slots from between them.
    between = [f for f in near[1:-1] if f not in ordered]
    while between and len(ordered) < max_frames:
        pick = between[len(between) // 2]
        ordered = sorted(set(ordered) | {pick})
        between = [f for f in between if f != pick]
    return ordered[:max_frames]


# ---------------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------------


def _pair_metrics(person: Tracklet, vehicle: Tracklet) -> list[FrameMetrics]:
    """Compute metrics on every frame where both tracklets have an observation."""
    v_by_frame = {o.frame_idx: o for o in vehicle.observations}
    out: list[FrameMetrics] = []
    for p_obs in person.observations:
        v_obs = v_by_frame.get(p_obs.frame_idx)
        if v_obs is None:
            continue
        out.append(
            FrameMetrics(
                frame_idx=p_obs.frame_idx,
                iou=iou(p_obs.xyxy, v_obs.xyxy),
                containment=containment(p_obs.xyxy, v_obs.xyxy),
                center_dist=center_distance(p_obs.xyxy, v_obs.xyxy),
                person_xyxy=p_obs.xyxy,
                vehicle_xyxy=v_obs.xyxy,
            )
        )
    return out


def _frac_near_in_window(
    person: Tracklet, near_set: set[int], start_frac: float, from_start: bool
) -> float:
    """Fraction of a prefix/suffix of the person track that is near the vehicle."""
    n = person.length
    if n == 0:
        return 0.0
    k = max(1, int(round(n * start_frac)))
    window = (
        person.observations[:k] if from_start else person.observations[-k:]
    )
    hits = sum(1 for o in window if o.frame_idx in near_set)
    return hits / len(window)


def _containment_trend(near_metrics: list[FrameMetrics]) -> float:
    """Late-minus-early mean containment over the near span (positive = closer)."""
    if len(near_metrics) < 2:
        return 0.0
    third = max(1, len(near_metrics) // 3)
    early = sum(m.containment for m in near_metrics[:third]) / third
    late = sum(m.containment for m in near_metrics[-third:]) / third
    return late - early


def _distance_trend(near_metrics: list[FrameMetrics]) -> float:
    """Late-minus-early mean center distance (positive = moving away)."""
    if len(near_metrics) < 2:
        return 0.0
    third = max(1, len(near_metrics) // 3)
    early = sum(m.center_dist for m in near_metrics[:third]) / third
    late = sum(m.center_dist for m in near_metrics[-third:]) / third
    return late - early


def _door_gap_trend(door_metrics: list[FrameMetrics]) -> float:
    """Late-minus-early mean horizontal edge gap (negative = approaching door)."""
    if len(door_metrics) < 2:
        return 0.0
    third = max(1, len(door_metrics) // 3)
    early = (
        sum(
            horizontal_edge_gap(m.person_xyxy, m.vehicle_xyxy)
            for m in door_metrics[:third]
        )
        / third
    )
    late = (
        sum(
            horizontal_edge_gap(m.person_xyxy, m.vehicle_xyxy)
            for m in door_metrics[-third:]
        )
        / third
    )
    return late - early


def _frac_door_in_window(
    person: Tracklet,
    vehicle: Tracklet,
    start_frac: float,
    from_start: bool,
) -> float:
    """Fraction of a person-track prefix/suffix that is door-side near the vehicle."""
    n = person.length
    if n == 0:
        return 0.0
    k = max(1, int(round(n * start_frac)))
    window = (
        person.observations[:k] if from_start else person.observations[-k:]
    )
    v_by = {o.frame_idx: o for o in vehicle.observations}
    hits = 0
    for o in window:
        v_obs = v_by.get(o.frame_idx)
        if v_obs is not None and door_side_near(o.xyxy, v_obs.xyxy):
            hits += 1
    return hits / len(window)


def _classify_pair(
    person: Tracklet,
    vehicle: Tracklet,
    metrics: list[FrameMetrics],
    fps: float,
) -> Interaction | None:
    """Classify one person–vehicle pair, or return None if they never co-occur."""
    if not metrics:
        return None

    near = [m for m in metrics if m.is_near]
    door = [
        m
        for m in metrics
        if door_side_near(m.person_xyxy, m.vehicle_xyxy)
    ]
    near_idxs = [m.frame_idx for m in near]
    near_set = set(near_idxs)
    dwell = len(near)
    door_dwell = len(door)

    # No overlap-near and no door-side proximity → ignore pair.
    if dwell == 0 and door_dwell == 0:
        return None

    # Prefer overlap-near frames for span/evidence; fall back to door-side.
    span_metrics = near if near else door
    span_idxs = [m.frame_idx for m in span_metrics]
    peak_containment = max(m.containment for m in span_metrics)
    peak_iou = max(m.iou for m in span_metrics)
    start_f, end_f = span_idxs[0], span_idxs[-1]
    t0 = start_f / fps if fps > 0 else 0.0
    t1 = end_f / fps if fps > 0 else 0.0

    # Person track birth/death relative to the vehicle's lifetime.
    vehicle_present_at_person_start = (
        vehicle.start_frame <= person.start_frame <= vehicle.end_frame
    )
    person_ends_during_vehicle = (
        vehicle.start_frame <= person.end_frame <= vehicle.end_frame
    )
    last_person_near = person.end_frame in near_set or (
        person.observations
        and any(
            o.frame_idx in near_set
            for o in person.observations[-max(1, person.length // 5) :]
        )
    )
    first_person_near = person.start_frame in near_set or (
        person.observations
        and any(
            o.frame_idx in near_set
            for o in person.observations[: max(1, person.length // 5)]
        )
    )

    start_near_frac = _frac_near_in_window(
        person, near_set, config.EXIT_START_NEAR_FRAC, from_start=True
    )
    end_near_frac = _frac_near_in_window(
        person, near_set, config.ENTER_END_NEAR_FRAC, from_start=False
    )
    early_end_near = _frac_near_in_window(
        person, near_set, config.EXIT_END_NEAR_FRAC, from_start=False
    )
    late_start_near = _frac_near_in_window(
        person, near_set, config.ENTER_START_NEAR_FRAC, from_start=True
    )

    cont_trend = _containment_trend(near) if near else 0.0
    dist_trend = _distance_trend(near) if near else _distance_trend(door)
    approaching = cont_trend >= config.APPROACH_CONTAINMENT_DELTA or dist_trend < 0
    leaving = cont_trend <= -config.LEAVE_CONTAINMENT_DELTA or dist_trend > 0

    evidence = {
        "peak_containment": round(peak_containment, 3),
        "peak_iou": round(peak_iou, 3),
        "dwell_frames": dwell,
        "door_dwell_frames": door_dwell,
        "track_terminated_near": bool(last_person_near and person_ends_during_vehicle),
        "track_started_near": bool(first_person_near and vehicle_present_at_person_start),
        "containment_trend": round(cont_trend, 3),
        "distance_trend": round(dist_trend, 3),
        "start_near_frac": round(start_near_frac, 3),
        "end_near_frac": round(end_near_frac, 3),
    }

    # --- Door-side enter: person approaches a truncated vehicle box and the
    # track dies at the door (boarding) without ever overlapping enough for
    # standard near. Kept separate so sidewalk walk-bys with real IoU are
    # not inflated into enters.
    door_idxs = [m.frame_idx for m in door]
    gap_trend = _door_gap_trend(door)
    start_door_frac = _frac_door_in_window(
        person, vehicle, config.ENTER_START_NEAR_FRAC, from_start=True
    )
    dies_at_door = bool(door) and door_idxs[-1] >= person.end_frame - 1
    is_door_enter = (
        door_dwell >= config.MIN_DOOR_ENTER_FRAMES
        and person_ends_during_vehicle
        and dies_at_door
        and start_door_frac <= 0.25
        and gap_trend < -5.0
        # Only when overlap-near never fired (truncated vehicle box case).
        and dwell < config.MIN_DWELL_FRAMES
        and peak_containment < config.NEAR_CONTAINMENT
    )
    if is_door_enter:
        evidence["proposal_reason"] = "door_side_boarding_truncated_vehicle"
        evidence["door_gap_trend"] = round(gap_trend, 3)
        evidence["start_door_frac"] = round(start_door_frac, 3)
        evidence["track_terminated_near"] = True
        conf = 0.55 + 0.2 * min(
            1.0, door_dwell / (2 * config.MIN_DOOR_ENTER_FRAMES)
        )
        return Interaction(
            person_id=person.track_id,
            vehicle_id=vehicle.track_id,
            vehicle_cls=vehicle.cls,
            type="enter",
            frame_range=(door_idxs[0], door_idxs[-1]),
            time_span_s=(
                round(door_idxs[0] / fps if fps > 0 else 0.0, 3),
                round(door_idxs[-1] / fps if fps > 0 else 0.0, 3),
            ),
            confidence=round(min(conf, 0.99), 3),
            near_frames=door_idxs,
            evidence=evidence,
        )

    # --- passing-by: too brief, or transit signature -----------------------
    if dwell < config.MIN_DWELL_FRAMES:
        conf = min(0.4, dwell / max(config.MIN_DWELL_FRAMES, 1) * 0.4)
        return Interaction(
            person_id=person.track_id,
            vehicle_id=vehicle.track_id,
            vehicle_cls=vehicle.cls,
            type="passing_by",
            frame_range=(start_f, end_f),
            time_span_s=(round(t0, 3), round(t1, 3)),
            confidence=round(conf, 3),
            near_frames=near_idxs,
            evidence=evidence,
        )

    # --- enter: approaches, nearness at end, track dies near vehicle -------
    is_enter = (
        person_ends_during_vehicle
        and last_person_near
        and end_near_frac >= config.ENTER_MIN_END_NEAR_RATIO
        and late_start_near <= end_near_frac
        and (approaching or end_near_frac > start_near_frac)
    )

    # --- exit: appears near already-present vehicle, then leaves -----------
    is_exit = (
        vehicle_present_at_person_start
        and first_person_near
        and start_near_frac >= 0.5
        and early_end_near <= start_near_frac
        and (leaving or start_near_frac > end_near_frac)
        and not (last_person_near and end_near_frac >= 0.7)
    )

    if is_enter and is_exit:
        # Ambiguous: prefer the stronger end-vs-start concentration.
        if end_near_frac >= start_near_frac:
            is_exit = False
        else:
            is_enter = False

    if is_enter:
        itype: InteractionType = "enter"
        conf = 0.55 + 0.25 * min(1.0, peak_containment) + 0.2 * min(
            1.0, dwell / (2 * config.MIN_DWELL_FRAMES)
        )
    elif is_exit:
        itype = "exit"
        conf = 0.55 + 0.25 * min(1.0, peak_containment) + 0.2 * min(
            1.0, dwell / (2 * config.MIN_DWELL_FRAMES)
        )
    elif (
        # Sustained contact: longer near-span + moderate overlap.
        dwell >= config.INTERACTING_MIN_DWELL_FRAMES
        and (
            peak_containment >= config.INTERACTING_MIN_PEAK_CONTAINMENT
            or peak_iou >= config.INTERACTING_MIN_PEAK_IOU
        )
    ) or (
        # Short but tight contact: grab/unload/take item from vehicle.
        # Track IDs often fragment these actions, so dwell alone is unreliable.
        dwell >= config.MIN_DWELL_FRAMES
        and (
            peak_containment >= config.INTERACTING_STRONG_PEAK_CONTAINMENT
            or peak_iou >= config.INTERACTING_STRONG_PEAK_IOU
        )
    ):
        # No boarding/alighting signature, but enough contact for a VLM look.
        # Qwen must confirm or it is filtered as passing_by.
        itype = "interacting"
        conf = 0.45 + 0.3 * min(1.0, peak_containment) + 0.15 * min(
            1.0, dwell / (2 * config.INTERACTING_MIN_DWELL_FRAMES)
        )
        if dwell >= config.INTERACTING_MIN_DWELL_FRAMES:
            evidence["proposal_reason"] = "sustained_proximity_no_enter_exit"
        else:
            evidence["proposal_reason"] = "strong_short_contact_no_enter_exit"
    else:
        # Near, but too brief / loose for a soft interacting proposal.
        itype = "passing_by"
        conf = 0.35 + 0.2 * min(1.0, dwell / (3 * config.MIN_DWELL_FRAMES))
        evidence["rejected_reason"] = "no_enter_exit_or_interacting_signature"

    return Interaction(
        person_id=person.track_id,
        vehicle_id=vehicle.track_id,
        vehicle_cls=vehicle.cls,
        type=itype,
        frame_range=(start_f, end_f),
        time_span_s=(round(t0, 3), round(t1, 3)),
        confidence=round(min(conf, 0.99), 3),
        near_frames=near_idxs,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Temporal NMS (fragment merge)
# ---------------------------------------------------------------------------


def _dwell_score(interaction: Interaction) -> int:
    """Overlap dwell for ranking; door-side dwell only if there was no IoU near."""
    ev = interaction.evidence or {}
    dwell = int(ev.get("dwell_frames", 0) or 0)
    if dwell > 0:
        return dwell
    return int(ev.get("door_dwell_frames", 0) or 0)


def _merge_rank(interaction: Interaction) -> tuple[int, float, float]:
    """Higher is better: dwell, then confidence, then span duration."""
    t0, t1 = interaction.time_span_s
    return (_dwell_score(interaction), float(interaction.confidence), float(t1 - t0))


def _spans_close(
    a0: float, a1: float, b0: float, b1: float, gap_s: float
) -> bool:
    """True if spans overlap or the gap between them is ≤ ``gap_s``."""
    if a1 < b0:
        return (b0 - a1) <= gap_s
    if b1 < a0:
        return (a0 - b1) <= gap_s
    return True


def _same_event_link(a: Interaction, b: Interaction, gap_s: float) -> bool:
    """True if ``a``/``b`` look like fragments of one physical event.

    Same TYPE, temporally close, and share a person track **or** vehicle track
    (either side can fragment under occlusion).
    """
    if a.type != b.type:
        return False
    if a.person_id != b.person_id and a.vehicle_id != b.vehicle_id:
        return False
    a0, a1 = a.time_span_s
    b0, b1 = b.time_span_s
    return _spans_close(a0, a1, b0, b1, gap_s)


def _connected_clusters(
    items: Sequence[Interaction], gap_s: float
) -> list[list[Interaction]]:
    """Connected components under :func:`_same_event_link`."""
    n = len(items)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _same_event_link(items[i], items[j], gap_s):
                union(i, j)

    buckets: dict[int, list[Interaction]] = defaultdict(list)
    for i, item in enumerate(items):
        buckets[find(i)].append(item)
    return list(buckets.values())


def merge_overlapping_interactions(
    interactions: Sequence[Interaction],
    *,
    gap_s: float | None = None,
) -> list[Interaction]:
    """Merge fragment proposals that overlap or are near in time.

    Track fragmentation often yields multiple person/vehicle IDs for one
    physical event. Two proposals are linked when they share TYPE, are
    temporally close (overlap or gap ≤ ``gap_s``), and share a person **or**
    vehicle id. Each connected component keeps the member with the strongest
    dwell/confidence; siblings go on ``evidence["merged_from"]``.

    Different types (e.g. exit then enter) are never merged.
    """
    if not interactions:
        return []
    gap = float(config.MERGE_GAP_S if gap_s is None else gap_s)

    # Partition by type first so enter/exit/interacting never mix.
    by_type: dict[InteractionType, list[Interaction]] = defaultdict(list)
    for inter in interactions:
        by_type[inter.type].append(inter)

    merged: list[Interaction] = []
    for items in by_type.values():
        for cluster in _connected_clusters(items, gap):
            winner = max(cluster, key=_merge_rank)
            if len(cluster) > 1:
                suppressed = []
                for other in cluster:
                    if other is winner:
                        continue
                    suppressed.append(
                        {
                            "person_id": other.person_id,
                            "vehicle_id": other.vehicle_id,
                            "time_span_s": [
                                round(other.time_span_s[0], 3),
                                round(other.time_span_s[1], 3),
                            ],
                            "confidence": other.confidence,
                            "dwell_frames": _dwell_score(other),
                        }
                    )
                evidence = dict(winner.evidence)
                evidence["merged_from"] = suppressed
                evidence["merge_count"] = len(cluster)
                winner = replace(winner, evidence=evidence)
            merged.append(winner)

    merged.sort(
        key=lambda i: (i.frame_range[0], i.person_id, i.vehicle_id, i.type)
    )
    return merged


class GeometricInteractionFinder:
    """Classify person–vehicle pairs from tracklets using geometry heuristics.

    Owns the Step-4 policy (proximity → dwell → enter/exit/interacting/
    passing_by). Stateless aside from config thresholds.
    """

    def find(
        self,
        collection: TrackletCollection,
        fps: float,
        *,
        include_passing_by: bool = True,
    ) -> list[Interaction]:
        """Classify all person–vehicle pairs in a clip.

        By default includes ``passing_by`` rows (useful for tuning). Pass
        ``include_passing_by=False`` to keep only accepted proposals
        (enter / exit / interacting).
        """
        results: list[Interaction] = []
        for person in collection.persons.values():
            for vehicle in collection.vehicles.values():
                # Cheap temporal overlap gate before per-frame work.
                if person.end_frame < vehicle.start_frame:
                    continue
                if vehicle.end_frame < person.start_frame:
                    continue
                metrics = _pair_metrics(person, vehicle)
                interaction = _classify_pair(person, vehicle, metrics, fps)
                if interaction is None:
                    continue
                if not include_passing_by and not interaction.is_accepted:
                    continue
                results.append(interaction)

        results.sort(
            key=lambda i: (i.frame_range[0], i.person_id, i.vehicle_id, i.type)
        )
        return results


def find_interactions(
    collection: TrackletCollection,
    fps: float,
    *,
    include_passing_by: bool = True,
) -> list[Interaction]:
    """Classify all person–vehicle pairs in a clip (thin wrapper)."""
    return GeometricInteractionFinder().find(
        collection, fps, include_passing_by=include_passing_by
    )


def interactions_from_video(
    video_path: str | Path,
    model: DetectorTracker | None = None,
    stride: int = config.FRAME_STRIDE,
    *,
    include_passing_by: bool = True,
) -> tuple[TrackletCollection, list[Interaction], float]:
    """Run tracklets + Step 4 on a clip. Returns ``(collection, interactions, fps)``."""
    video_path = Path(video_path)
    meta = probe_video(video_path)
    collection = tracklets_from_video(video_path, model=model, stride=stride)
    interactions = find_interactions(
        collection, meta.fps, include_passing_by=include_passing_by
    )
    return collection, interactions, meta.fps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_interaction(i: Interaction, names: dict[int, str]) -> str:
    vname = names.get(i.vehicle_cls, str(i.vehicle_cls))
    return f"  {i.summary()} vehicle_type={vname}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify person–vehicle interactions for a clip (Step 4)."
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 clip")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    parser.add_argument(
        "--no-passing-by",
        action="store_true",
        help="Hide rejected passing-by pairs",
    )
    args = parser.parse_args()

    tracker = DetectorTracker()
    collection, interactions, fps = interactions_from_video(
        args.video,
        model=tracker,
        stride=args.stride,
        include_passing_by=not args.no_passing_by,
    )

    accepted = [i for i in interactions if i.is_accepted]
    rejected = [i for i in interactions if not i.is_accepted]
    print(f"clip={args.video.name} fps={fps:.2f} stride={args.stride}")
    print(collection.summary())
    print(f"interactions: accepted={len(accepted)} passing_by={len(rejected)}")
    print("accepted:")
    for i in accepted:
        print(_format_interaction(i, tracker.names))
    if not args.no_passing_by:
        print("passing_by:")
        for i in rejected:
            print(_format_interaction(i, tracker.names))
