"""Central configuration: all tunable knobs, seeds, paths, and determinism.

Import this module BEFORE torch / ultralytics / transformers so the cache-dir
environment variables below take effect (those libraries read them at import
time). The rest of the code imports every threshold from here so there are no
magic numbers scattered across the pipeline.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "Videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# ---------------------------------------------------------------------------
# Redirect third-party caches/config into the project so the pipeline runs
# without needing write access to the user's home directory.
# ---------------------------------------------------------------------------
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
for _sub in ("ultralytics", "matplotlib", "huggingface"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 0  # single master seed fanned out to random / numpy / torch


def set_determinism(seed: int = SEED) -> None:
    """Fan the master seed out to every RNG the pipeline touches.

    Kept import-light: numpy/torch are imported lazily so that merely reading
    config constants elsewhere does not force those heavy imports.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Detection / tracking (Steps 2-3)
# ---------------------------------------------------------------------------
MODEL = "yolo11s.pt"          # swap to "yolo11n.pt" for faster CPU runs
# Project-local tracker config (see src/bytetrack.yaml): lowers ByteTrack's
# new-track/association thresholds so occluded people (e.g. inside a car's
# bounding box) can spawn a track instead of being detected then dropped.
TRACKER = str(Path(__file__).resolve().parent / "bytetrack.yaml")
IMGSZ = 640                   # inference resolution
CONF = 0.15                   # min detection confidence
IOU = 0.5                     # NMS IoU threshold

# COCO class ids we care about
PERSON_CLASS = 0
VEHICLE_CLASSES = (2, 3, 5, 7)  # car, motorcycle, bus, truck
KEEP_CLASSES = (PERSON_CLASS,) + VEHICLE_CLASSES

# Process every Nth frame to keep CPU runtime reasonable
FRAME_STRIDE = 1

# ---------------------------------------------------------------------------
# Interaction heuristics (Step 4) — starting points; tuned in Step 7
# ---------------------------------------------------------------------------
NEAR_IOU = 0.05          # min person-vehicle IoU to count as "near"
NEAR_CONTAINMENT = 0.15  # fraction of person bbox inside vehicle bbox
MIN_DWELL_FRAMES = 5     # sampled frames of proximity before it counts
# Door-side proximity when YOLO's vehicle box is truncated (person stands at a
# door just outside the box, no IoU/containment). Horizontal edge gap must be
# ≤ this fraction of vehicle width, with vertical box overlap.
NEAR_DOOR_GAP_FRAC = 0.45
# Min door-side frames to propose an enter when the person track dies at the door.
MIN_DOOR_ENTER_FRAMES = 3

# Motion / signature checks (enter vs exit vs passing-by)
APPROACH_CONTAINMENT_DELTA = 0.08  # containment rise over near span → approaching
LEAVE_CONTAINMENT_DELTA = 0.08     # containment drop over near span → leaving
# Max fraction of person-track lifetime that may still be "near" at the end
# for an exit (person should leave the vehicle zone).
EXIT_END_NEAR_FRAC = 0.25
# Min fraction of person-track lifetime that must be "near" at the start
# for an exit (person appears already at the vehicle).
EXIT_START_NEAR_FRAC = 0.35
# Same idea for enter: nearness concentrated at the end of the person track.
ENTER_END_NEAR_FRAC = 0.35
ENTER_START_NEAR_FRAC = 0.25
# Min fraction of the enter end-window that must be near (hardcoded 0.5 was too
# strict for long approach tracks that only board in the last seconds).
ENTER_MIN_END_NEAR_RATIO = 0.5

# Soft "interacting" proposals: proximity without an enter/exit signature
# (e.g. trunk/load/unload/taking items). Two tiers keep walk-bys out of the
# VLM queue while still recalling short grab/unload actions:
#   - sustained: longer dwell + moderate overlap
#   - strong:    MIN_DWELL + tight overlap (blanket/bag from car)
# Qwen still filters passing_by.
INTERACTING_MIN_DWELL_FRAMES = 10  # typically ~2× MIN_DWELL_FRAMES
INTERACTING_MIN_PEAK_CONTAINMENT = 0.35
INTERACTING_MIN_PEAK_IOU = 0.08
# Short-dwell tier: requires stronger contact than the sustained tier.
INTERACTING_STRONG_PEAK_CONTAINMENT = 0.55
INTERACTING_STRONG_PEAK_IOU = 0.15

# ---------------------------------------------------------------------------
# VLM descriptions (Step 5)
# ---------------------------------------------------------------------------
VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
VLM_REVISION = "main"  # pin a commit hash later if you need stricter reproducibility
VLM_MAX_NEW_TOKENS = 96  # TYPE + ACTION + CONNECTION lines
# Max near frames used for TYPE/ACTION/CONNECTION votes (start / peak / end).
# Person and vehicle appearance use a single sharpest crop among these.
DESCRIBE_MAX_FRAMES = 3
# Fractional padding around each bbox before cropping.
CROP_PAD = 0.08
# If True, drop heuristic interactions that Qwen labels as passing_by, and
# overwrite the final type with Qwen's enter/exit/interacting label.
# Exception: when geometry already says enter/exit and Qwen flips the other way,
# keep the geometric direction (still-frame VLMs often confuse boarding/alighting).
VLM_FILTER_TYPE = True
# Weighted TYPE aggregation (VLM votes + geometry prior). Enter/exit may be
# visible on a single frame, so we do NOT require a majority of Qwen votes.
VLM_VOTE_WEIGHT = 1.0
VLM_PEAK_VOTE_WEIGHT = 1.5  # peak-containment describe frame
# Geometry prior added to the matching type's score.
GEO_ENTER_EXIT_WEIGHT = 2.5  # strong: track signature already says enter/exit
GEO_INTERACTING_WEIGHT = 1.5  # weaker: proximity-only proposal needs Qwen support
