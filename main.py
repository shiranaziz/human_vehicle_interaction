"""Runnable entry point — edit settings below, then press Run.

Orchestration lives in :class:`src.pipeline.InteractionPipeline`. This file
only configures knobs and selects which clips to process. After the pipeline
finishes, optional three-rank evaluation runs against ``Videos/*.json`` GT.
"""
from pathlib import Path

from evaluation import run_evaluation
from src import config
from src.pipeline import InteractionPipeline, PipelineSettings

# ---------------------------------------------------------------------------
# Hard-coded settings — tweak these, then hit Run.
# ---------------------------------------------------------------------------
CLIP = "mKzCQKTHizw_0.mp4"
PROCESS_ALL = True
RUN_EVAL = True

SETTINGS = PipelineSettings(
    stride=config.FRAME_STRIDE,
    run_describe=True,
    write_annotated=True,
    show_passing_by=True,
    show_vlm_filtered=True,
    verbose=True,
)


def resolve_clips() -> list[Path]:
    if PROCESS_ALL:
        clips = sorted(config.VIDEOS_DIR.glob("*.mp4"))
    else:
        clips = [config.VIDEOS_DIR / CLIP]
    if not clips:
        raise SystemExit(f"No clips found under {config.VIDEOS_DIR}")
    return clips


def main() -> None:
    pipeline = InteractionPipeline(SETTINGS)
    pipeline.run(resolve_clips())
    if RUN_EVAL:
        run_evaluation(
            videos_dir=config.VIDEOS_DIR,
            outputs_dir=config.OUTPUTS_DIR,
            describer=pipeline.describer,
            iou_threshold=0.2,
            run_llm_judge=SETTINGS.run_describe,
            verbose=SETTINGS.verbose,
        )


if __name__ == "__main__":
    main()
