"""Load GT / prediction events and match them by temporal IoU."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


POINT_SPAN_PAD_S = 0.5


@dataclass(frozen=True)
class GtEvent:
    clip_id: str
    interaction_id: int
    type: str
    time_span_s: tuple[float, float]
    notes: str
    vehicle_type: str = ""


@dataclass(frozen=True)
class PredEvent:
    clip_id: str
    interaction_id: int
    type: str
    time_span_s: tuple[float, float]
    connection: str
    action_detail: str
    person_description: str
    vehicle_description: str


@dataclass
class MatchPair:
    gt: GtEvent
    pred: PredEvent
    iou: float
    type_ok: bool
    desc_score: int | None = None
    desc_reason: str = ""


@dataclass
class ClipMatchResult:
    clip_id: str
    pairs: list[MatchPair]
    unmatched_gt: list[GtEvent]
    unmatched_pred: list[PredEvent]


def expand_span(
    span: tuple[float, float], pad: float = POINT_SPAN_PAD_S
) -> tuple[float, float]:
    """Expand zero-length spans so point annotations can still match."""
    t0, t1 = float(span[0]), float(span[1])
    if t1 <= t0:
        return (t0 - pad, t1 + pad)
    return (t0, t1)


def temporal_iou(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Intersection-over-union of two time intervals."""
    a0, a1 = expand_span(a)
    b0, b1 = expand_span(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def load_gt_events(path: Path) -> list[GtEvent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clip_id = str(data.get("clip_id") or path.stem)
    events: list[GtEvent] = []
    for row in data.get("interactions", []):
        span = row["time_span_s"]
        events.append(
            GtEvent(
                clip_id=clip_id,
                interaction_id=int(row["interaction_id"]),
                type=str(row["type"]),
                time_span_s=(float(span[0]), float(span[1])),
                notes=str(row.get("notes") or ""),
                vehicle_type=str(row.get("vehicle_type") or ""),
            )
        )
    return events


def load_pred_events(path: Path) -> list[PredEvent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clip_id = str(data.get("clip_id") or path.stem)
    events: list[PredEvent] = []
    for row in data.get("interactions", []):
        span = row["time_span_s"]
        evidence = row.get("evidence") or {}
        person = row.get("person") or {}
        vehicle = row.get("vehicle") or {}
        events.append(
            PredEvent(
                clip_id=clip_id,
                interaction_id=int(row["interaction_id"]),
                type=str(row["type"]),
                time_span_s=(float(span[0]), float(span[1])),
                connection=str(row.get("connection") or ""),
                action_detail=str(evidence.get("action_detail") or ""),
                person_description=str(person.get("description") or ""),
                vehicle_description=str(vehicle.get("description") or ""),
            )
        )
    return events


def discover_clip_pairs(
    videos_dir: Path, outputs_dir: Path
) -> list[tuple[str, Path, Path]]:
    """Return ``(clip_id, gt_path, pred_path)`` for clips that have both."""
    pairs: list[tuple[str, Path, Path]] = []
    for gt_path in sorted(videos_dir.glob("*.json")):
        pred_path = outputs_dir / gt_path.name
        if not pred_path.is_file():
            continue
        pairs.append((gt_path.stem, gt_path, pred_path))
    return pairs


def greedy_match(
    gts: list[GtEvent],
    preds: list[PredEvent],
    iou_threshold: float = 0.2,
) -> ClipMatchResult:
    """Greedy 1:1 matching by highest temporal IoU above ``iou_threshold``."""
    if not gts and not preds:
        clip_id = ""
        return ClipMatchResult(clip_id, [], [], [])

    clip_id = (gts[0].clip_id if gts else preds[0].clip_id)
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            iou = temporal_iou(gt.time_span_s, pred.time_span_s)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[MatchPair] = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        gt = gts[gi]
        pred = preds[pi]
        pairs.append(
            MatchPair(
                gt=gt,
                pred=pred,
                iou=iou,
                type_ok=(gt.type == pred.type),
            )
        )

    unmatched_gt = [g for i, g in enumerate(gts) if i not in used_gt]
    unmatched_pred = [p for i, p in enumerate(preds) if i not in used_pred]
    return ClipMatchResult(clip_id, pairs, unmatched_gt, unmatched_pred)


def match_all_clips(
    videos_dir: Path,
    outputs_dir: Path,
    iou_threshold: float = 0.2,
) -> list[ClipMatchResult]:
    results: list[ClipMatchResult] = []
    for clip_id, gt_path, pred_path in discover_clip_pairs(
        videos_dir, outputs_dir
    ):
        gts = load_gt_events(gt_path)
        preds = load_pred_events(pred_path)
        result = greedy_match(gts, preds, iou_threshold=iou_threshold)
        # Prefer stem when both sides empty (should not happen)
        if not result.clip_id:
            result.clip_id = clip_id
        results.append(result)
    return results
