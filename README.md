# BlackRover — Person–Vehicle Interaction Detection

Detects **enter**, **exit**, and **interacting** (door/trunk/load/unload) events between people and vehicles in short clips. Geometry proposes candidates; a local Qwen2-VL model describes them and filters walk-bys. Runs on **CPU** (no GPU / no external APIs).

## Quick start

```bash
# 1. Create env and install (CPU PyTorch wheels)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Put clips under Videos/  (name: <clip_id>.mp4)
#    Optional GT for eval: Videos/<clip_id>.json

# 3. Edit knobs in main.py, then run
python main.py
```

On the first run (with network), weights download automatically:

- Ultralytics fetches `yolo11s.pt` into the project root
- Hugging Face caches `Qwen/Qwen2-VL-2B-Instruct` under `.cache/huggingface/`

Later runs reuse those files offline. Expect slow CPU inference (YOLO + Qwen on every clip).

### Settings in `main.py`

| Knob | Meaning |
| --- | --- |
| `CLIP` | Single file under `Videos/` when `PROCESS_ALL` is `False` |
| `PROCESS_ALL` | `True` → all `Videos/*.mp4`; `False` → only `CLIP` |
| `RUN_EVAL` | Compare predictions to `Videos/*.json` GT (if present) |
| `SETTINGS.write_annotated` | Write overlay MP4s under `outputs/videos/` |

Thresholds and model names live in `src/config.py`.

## Outputs

| Path | Contents |
| --- | --- |
| `outputs/description/<clip_id>.json` | One clip document (product schema) |
| `outputs/description/interactions.json` | Array of all clip documents from the run |
| `outputs/videos/<clip_id>_annotated.mp4` | Optional visualization |
| `outputs/eval_results/` | Eval reports when `RUN_EVAL=True` |

Only **kept** interactions are written (`enter` / `exit` / `interacting`). Walk-bys (`passing_by`) are dropped.

---

## Output JSON schema

### Clip document

```json
{
  "clip_id": "mKzCQKTHizw_0",
  "video_metadata": {
    "fps": 29.97,
    "frame_count": 300,
    "resolution": [640, 360],
    "duration_s": 10.01
  },
  "interactions": [ /* … */ ]
}
```

| Field | Type | Description |
| --- | --- | --- |
| `clip_id` | string | Stem of the input MP4 (no extension) |
| `video_metadata.fps` | number | Frames per second |
| `video_metadata.frame_count` | int | Total frames |
| `video_metadata.resolution` | `[w, h]` | Width × height in pixels |
| `video_metadata.duration_s` | number | Clip length in seconds |
| `interactions` | array | Kept events in this clip (may be empty) |

`interactions.json` is a JSON **array** of these clip documents.

### Interaction object

```json
{
  "interaction_id": 1,
  "type": "enter",
  "connection": "The person is entering the vehicle, leaning into the cabin.",
  "person": {
    "track_id": 1337,
    "description": "A young woman wearing a light-colored top and khaki pants, with long hair.",
    "representative_bbox": [451.6, 82.4, 118.1, 225.8]
  },
  "vehicle": {
    "track_id": 1340,
    "type": "car",
    "description": "A silver sedan."
  },
  "frame_range": [271, 288],
  "time_span_s": [9.042, 9.61],
  "confidence": 0.99,
  "evidence": { /* geometry + VLM debug */ }
}
```

| Field | Type | Description |
| --- | --- | --- |
| `interaction_id` | int | 1-based index within the clip |
| `type` | string | `enter` \| `exit` \| `interacting` |
| `connection` | string | Short free-text link between person and vehicle (from Qwen) |
| `person.track_id` | int | ByteTrack ID for the person |
| `person.description` | string | Appearance text from Qwen |
| `person.representative_bbox` | `[x, y, w, h]` | Person box on the representative frame (xywh, pixels). Omitted if unavailable |
| `vehicle.track_id` | int | ByteTrack ID for the vehicle |
| `vehicle.type` | string | COCO label: `car`, `motorcycle`, `bus`, or `truck` |
| `vehicle.description` | string | Appearance text from Qwen |
| `frame_range` | `[start, end]` | Inclusive frame indices of the near/dwell span |
| `time_span_s` | `[t0, t1]` | Same span in seconds (`frame / fps`) |
| `confidence` | number | Geometry confidence in `[0, 1]` (higher for clear enter/exit) |
| `evidence` | object | Extra diagnostics (see below) |

### `evidence` (diagnostics)

Useful for debugging; not required to consume the product fields above.

**Geometry (always present on proposals):**

| Key | Meaning |
| --- | --- |
| `peak_containment` | Max fraction of person box inside vehicle box |
| `peak_iou` | Max person–vehicle IoU |
| `dwell_frames` | Count of “near” frames (IoU / containment) |
| `door_dwell_frames` | Door-side proximity frames (truncated vehicle boxes) |
| `track_terminated_near` / `track_started_near` | Person track ends / starts near the vehicle |
| `containment_trend` / `distance_trend` | Late − early motion trends |
| `start_near_frac` / `end_near_frac` | Nearness concentrated at start vs end of the person track |
| `proposal_reason` | Optional tag (e.g. door-side boarding, soft interacting) |
| `merged_from` / `merge_count` | Present when temporal NMS merged ID fragments |

**After Qwen:**

| Key | Meaning |
| --- | --- |
| `heuristic_type` / `vlm_type` | Geometry vs VLM type labels |
| `type_scores` | Weighted TYPE votes (geometry prior + Qwen) |
| `action` / `action_detail` | Coarse type and finer action phrase |
| `qwen_connection` | Raw connection string from Qwen |
| `qwen_votes` | Per-frame TYPE / ACTION / CONNECTION votes |
| `explanation` | Plain-language keep/reject rationale |
| `appearance_person_frame` / `appearance_vehicle_frame` | Frames used for appearance crops |

---

## Layout

```
Videos/                 # input MP4s (+ optional GT JSON)
src/                    # pipeline code
evaluation/             # offline eval against Videos/*.json
outputs/description/    # product JSON
outputs/videos/         # annotated MP4s
outputs/eval_results/   # eval reports
main.py                 # edit settings → run
requirements.txt
```

## Notes

- Clips are processed independently (no cross-clip identity).
- Seed is fixed (`SEED=0` in `src/config.py`) for reproducibility.
- Swap `MODEL` to `yolo11n.pt` in `src/config.py` for a faster (less accurate) detector.
