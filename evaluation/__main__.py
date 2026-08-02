"""Standalone entry: ``python -m evaluation`` against existing outputs."""
from __future__ import annotations

from src import config

from .run import run_evaluation

# Hard-coded knobs — tweak then run ``python -m evaluation``.
IOU_THRESHOLD = 0.2
RUN_LLM_JUDGE = True
VERBOSE = True


def main() -> None:
    run_evaluation(
        videos_dir=config.VIDEOS_DIR,
        description_dir=config.OUTPUTS_DESCRIPTION_DIR,
        eval_dir=config.OUTPUTS_EVAL_DIR,
        describer=None,
        iou_threshold=IOU_THRESHOLD,
        run_llm_judge=RUN_LLM_JUDGE,
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()
