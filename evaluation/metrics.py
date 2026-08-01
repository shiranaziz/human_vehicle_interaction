"""Rank-1 / Rank-2 aggregates from matched clip results."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .match import ClipMatchResult, MatchPair

TYPE_LABELS = ("enter", "exit", "interacting")


@dataclass(frozen=True)
class DetectionScores:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TypeScores:
    n_matched: int
    n_correct: int
    accuracy: float
    confusion: dict[str, dict[str, int]]


@dataclass(frozen=True)
class DescriptionScores:
    n_scored: int
    mean_raw: float
    mean_normalized: float
    histogram: dict[str, int]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def detection_scores(
    results: Iterable[ClipMatchResult],
) -> DetectionScores:
    tp = fp = fn = 0
    for clip in results:
        tp += len(clip.pairs)
        fp += len(clip.unmatched_pred)
        fn += len(clip.unmatched_gt)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return DetectionScores(tp, fp, fn, precision, recall, f1)


def detection_scores_for_clip(clip: ClipMatchResult) -> DetectionScores:
    return detection_scores([clip])


def type_scores(pairs: Iterable[MatchPair]) -> TypeScores:
    pair_list = list(pairs)
    n = len(pair_list)
    n_correct = sum(1 for p in pair_list if p.type_ok)
    confusion: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in TYPE_LABELS} for gt in TYPE_LABELS
    }
    for p in pair_list:
        gt_t = p.gt.type if p.gt.type in confusion else None
        pred_t = p.pred.type if p.pred.type in TYPE_LABELS else None
        if gt_t is None:
            confusion.setdefault(p.gt.type, {lab: 0 for lab in TYPE_LABELS})
            gt_t = p.gt.type
            for lab in TYPE_LABELS:
                confusion[gt_t].setdefault(lab, 0)
        if pred_t is None:
            for row in confusion.values():
                row.setdefault(p.pred.type, 0)
            pred_t = p.pred.type
            confusion[gt_t].setdefault(pred_t, 0)
        confusion[gt_t][pred_t] += 1
    return TypeScores(
        n_matched=n,
        n_correct=n_correct,
        accuracy=_safe_div(n_correct, n),
        confusion=confusion,
    )


def description_scores(pairs: Iterable[MatchPair]) -> DescriptionScores:
    scored = [p for p in pairs if p.desc_score is not None]
    hist_counter: Counter[int] = Counter(int(p.desc_score) for p in scored)
    histogram = {str(k): hist_counter.get(k, 0) for k in (0, 1, 2)}
    if not scored:
        return DescriptionScores(0, 0.0, 0.0, histogram)
    mean_raw = sum(int(p.desc_score) for p in scored) / len(scored)
    return DescriptionScores(
        n_scored=len(scored),
        mean_raw=mean_raw,
        mean_normalized=mean_raw / 2.0,
        histogram=histogram,
    )


def all_pairs(results: Iterable[ClipMatchResult]) -> list[MatchPair]:
    out: list[MatchPair] = []
    for clip in results:
        out.extend(clip.pairs)
    return out
