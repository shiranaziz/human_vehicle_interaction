"""Write eval JSON/CSV reports and print a console summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .match import ClipMatchResult, MatchPair
from .metrics import (
    DescriptionScores,
    DetectionScores,
    TypeScores,
    all_pairs,
    description_scores,
    detection_scores,
    detection_scores_for_clip,
    type_scores,
)


def _detection_dict(s: DetectionScores) -> dict[str, Any]:
    return {
        "tp": s.tp,
        "fp": s.fp,
        "fn": s.fn,
        "precision": round(s.precision, 4),
        "recall": round(s.recall, 4),
        "f1": round(s.f1, 4),
    }


def _type_dict(s: TypeScores) -> dict[str, Any]:
    return {
        "n_matched": s.n_matched,
        "n_correct": s.n_correct,
        "accuracy": round(s.accuracy, 4),
        "confusion": s.confusion,
    }


def _desc_dict(s: DescriptionScores) -> dict[str, Any]:
    return {
        "n_scored": s.n_scored,
        "mean_raw": round(s.mean_raw, 4),
        "mean_normalized": round(s.mean_normalized, 4),
        "histogram": s.histogram,
    }


def _pair_row(pair: MatchPair) -> dict[str, Any]:
    return {
        "clip_id": pair.gt.clip_id,
        "gt_interaction_id": pair.gt.interaction_id,
        "pred_interaction_id": pair.pred.interaction_id,
        "gt_type": pair.gt.type,
        "pred_type": pair.pred.type,
        "type_ok": pair.type_ok,
        "iou": round(pair.iou, 4),
        "gt_time_span_s": list(pair.gt.time_span_s),
        "pred_time_span_s": list(pair.pred.time_span_s),
        "gt_notes": pair.gt.notes,
        "pred_connection": pair.pred.connection,
        "pred_action_detail": pair.pred.action_detail,
        "desc_score": pair.desc_score,
        "desc_reason": pair.desc_reason,
    }


def build_report(
    clips: list[ClipMatchResult],
    *,
    iou_threshold: float,
    ran_llm_judge: bool,
) -> dict[str, Any]:
    pairs = all_pairs(clips)
    det = detection_scores(clips)
    typ = type_scores(pairs)
    type_correct = [p for p in pairs if p.type_ok]
    desc_all = description_scores(pairs)
    desc_type_ok = description_scores(type_correct)

    per_clip: list[dict[str, Any]] = []
    for clip in clips:
        clip_pairs = clip.pairs
        per_clip.append(
            {
                "clip_id": clip.clip_id,
                "detection": _detection_dict(detection_scores_for_clip(clip)),
                "type": _type_dict(type_scores(clip_pairs)),
                "description": _desc_dict(description_scores(clip_pairs)),
                "n_unmatched_gt": len(clip.unmatched_gt),
                "n_unmatched_pred": len(clip.unmatched_pred),
                "pairs": [_pair_row(p) for p in clip_pairs],
                "unmatched_gt": [
                    {
                        "interaction_id": g.interaction_id,
                        "type": g.type,
                        "time_span_s": list(g.time_span_s),
                        "notes": g.notes,
                    }
                    for g in clip.unmatched_gt
                ],
                "unmatched_pred": [
                    {
                        "interaction_id": p.interaction_id,
                        "type": p.type,
                        "time_span_s": list(p.time_span_s),
                        "connection": p.connection,
                    }
                    for p in clip.unmatched_pred
                ],
            }
        )

    return {
        "iou_threshold": iou_threshold,
        "ran_llm_judge": ran_llm_judge,
        "summary": {
            "rank1_detection": _detection_dict(det),
            "rank2_type": _type_dict(typ),
            "rank3_description_all_tp": _desc_dict(desc_all),
            "rank3_description_type_correct": _desc_dict(desc_type_ok),
            "n_clips": len(clips),
            "n_matched_pairs": len(pairs),
        },
        "clips": per_clip,
    }


def write_report_json(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_pairs_csv(clips: list[ClipMatchResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "gt_interaction_id",
        "pred_interaction_id",
        "gt_type",
        "pred_type",
        "type_ok",
        "iou",
        "gt_t0",
        "gt_t1",
        "pred_t0",
        "pred_t1",
        "gt_notes",
        "pred_connection",
        "pred_action_detail",
        "desc_score",
        "desc_reason",
        "match_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for clip in clips:
            for pair in clip.pairs:
                writer.writerow(
                    {
                        "clip_id": pair.gt.clip_id,
                        "gt_interaction_id": pair.gt.interaction_id,
                        "pred_interaction_id": pair.pred.interaction_id,
                        "gt_type": pair.gt.type,
                        "pred_type": pair.pred.type,
                        "type_ok": pair.type_ok,
                        "iou": f"{pair.iou:.4f}",
                        "gt_t0": pair.gt.time_span_s[0],
                        "gt_t1": pair.gt.time_span_s[1],
                        "pred_t0": pair.pred.time_span_s[0],
                        "pred_t1": pair.pred.time_span_s[1],
                        "gt_notes": pair.gt.notes,
                        "pred_connection": pair.pred.connection,
                        "pred_action_detail": pair.pred.action_detail,
                        "desc_score": pair.desc_score
                        if pair.desc_score is not None
                        else "",
                        "desc_reason": pair.desc_reason,
                        "match_status": "matched",
                    }
                )
            for gt in clip.unmatched_gt:
                writer.writerow(
                    {
                        "clip_id": gt.clip_id,
                        "gt_interaction_id": gt.interaction_id,
                        "pred_interaction_id": "",
                        "gt_type": gt.type,
                        "pred_type": "",
                        "type_ok": "",
                        "iou": "",
                        "gt_t0": gt.time_span_s[0],
                        "gt_t1": gt.time_span_s[1],
                        "pred_t0": "",
                        "pred_t1": "",
                        "gt_notes": gt.notes,
                        "pred_connection": "",
                        "pred_action_detail": "",
                        "desc_score": "",
                        "desc_reason": "",
                        "match_status": "fn",
                    }
                )
            for pred in clip.unmatched_pred:
                writer.writerow(
                    {
                        "clip_id": pred.clip_id,
                        "gt_interaction_id": "",
                        "pred_interaction_id": pred.interaction_id,
                        "gt_type": "",
                        "pred_type": pred.type,
                        "type_ok": "",
                        "iou": "",
                        "gt_t0": "",
                        "gt_t1": "",
                        "pred_t0": pred.time_span_s[0],
                        "pred_t1": pred.time_span_s[1],
                        "gt_notes": "",
                        "pred_connection": pred.connection,
                        "pred_action_detail": pred.action_detail,
                        "desc_score": "",
                        "desc_reason": "",
                        "match_status": "fp",
                    }
                )
    return path


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    det = summary["rank1_detection"]
    typ = summary["rank2_type"]
    desc = summary["rank3_description_all_tp"]
    desc_ok = summary["rank3_description_type_correct"]
    print("\n=== Evaluation summary ===")
    print(
        f"Rank1 detection: P={det['precision']:.3f} "
        f"R={det['recall']:.3f} F1={det['f1']:.3f} "
        f"(tp={det['tp']} fp={det['fp']} fn={det['fn']})"
    )
    print(
        f"Rank2 type: accuracy={typ['accuracy']:.3f} "
        f"({typ['n_correct']}/{typ['n_matched']})"
    )
    if report.get("ran_llm_judge"):
        print(
            f"Rank3 description (all TP): "
            f"mean={desc['mean_normalized']:.3f} "
            f"(raw={desc['mean_raw']:.3f}/2, n={desc['n_scored']}, "
            f"hist={desc['histogram']})"
        )
        print(
            f"Rank3 description (type-correct): "
            f"mean={desc_ok['mean_normalized']:.3f} "
            f"(raw={desc_ok['mean_raw']:.3f}/2, n={desc_ok['n_scored']})"
        )
    else:
        print("Rank3 description: skipped (LLM judge off)")
    print(f"clips evaluated: {summary['n_clips']}")
