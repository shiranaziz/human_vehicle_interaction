"""Orchestrate three-rank evaluation of pipeline outputs vs GT annotations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src import config
from src.describe import QwenDescriber

from .judge import ensure_describer, judge_pairs
from .match import match_all_clips
from .metrics import all_pairs
from .report import (
    build_report,
    print_summary,
    write_pairs_csv,
    write_report_json,
)


def run_evaluation(
    videos_dir: str | Path,
    description_dir: str | Path,
    *,
    eval_dir: str | Path | None = None,
    describer: QwenDescriber | None = None,
    iou_threshold: float = 0.2,
    run_llm_judge: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Match GT to predictions, score ranks 1–3, write report artifacts.

    Clips without a GT sidecar or without a prediction JSON are skipped.
    Predictions are read from ``description_dir``; reports go to ``eval_dir``
    (defaults to ``config.OUTPUTS_EVAL_DIR``).
    """
    videos_dir = Path(videos_dir)
    description_dir = Path(description_dir)
    eval_dir = (
        Path(eval_dir)
        if eval_dir is not None
        else config.OUTPUTS_EVAL_DIR
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    clips = match_all_clips(
        videos_dir, description_dir, iou_threshold=iou_threshold
    )
    if verbose:
        print(
            f"\nEvaluating {len(clips)} clip(s) with GT+pred "
            f"(iou_threshold={iou_threshold})"
        )

    judged = False
    if run_llm_judge:
        pairs = all_pairs(clips)
        if pairs:
            if verbose:
                print(f"Running LLM judge on {len(pairs)} matched pair(s)...")
            judge_model = ensure_describer(describer)
            judge_pairs(judge_model, pairs, verbose=verbose)
            judged = True
        elif verbose:
            print("No matched pairs — skipping LLM judge.")
    elif verbose:
        print("LLM judge disabled.")

    report = build_report(
        clips, iou_threshold=iou_threshold, ran_llm_judge=judged
    )
    json_path = write_report_json(report, eval_dir / "eval_report.json")
    csv_path = write_pairs_csv(clips, eval_dir / "eval_pairs.csv")

    if verbose:
        print_summary(report)
        print(f"eval report -> {json_path}")
        print(f"eval pairs  -> {csv_path}")

    report["_paths"] = {
        "eval_report_json": str(json_path),
        "eval_pairs_csv": str(csv_path),
    }
    return report
