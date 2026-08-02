# Person–Vehicle Interaction Detection — Summarization

## Approach

We treat interaction detection as a **geometry-first proposal** problem with a **local VLM filter/describer**, rather than end-to-end video classification.

**Pipeline (per clip, independent):**

1. **Detect & track** — YOLO11s (Ultralytics) + ByteTrack on persons and vehicles (COCO: person, car, motorcycle, bus, truck). Custom ByteTrack thresholds keep occluded people (e.g. inside a car box) from being dropped.
2. **Tracklets** — Group detections by ByteTrack ID into per-object trajectories (frame + bbox + conf), majority-vote the class, and split into person vs vehicle maps. (Pairing people with vehicles that overlap in time is the next step.)
3. **Geometric proposals** — For every temporally overlapping person–vehicle pair, compute IoU, containment (fraction of the person box inside the vehicle box), and center distance. Mark “near” frames (IoU or containment above threshold); require minimum **dwell** (count of near frames); classify as `enter` / `exit` / `interacting` / `passing_by` from motion trends and track birth/death near the vehicle (boarding often ends a person track; alighting often starts one). Assign a geometry **confidence** from peak containment + dwell (higher base for enter/exit than soft interacting).
4. **Temporal NMS** — Merge same-type fragments that share a person or vehicle ID and overlap (or gap ≤ 1 s), keeping the strongest dwell/confidence. Cuts duplicate events from ID switches.
5. **Confidence gate** — Drop proposals with geometry conf &lt; 0.8 before the VLM.
6. **Qwen2-VL-2B (local)** — For each proposal we crop: (1) person, (2) vehicle, (3) three joint person+vehicle frames — the first near frame, the peak-containment near frame, and the last near frame. Appearance uses the sharpest person/vehicle crop; TYPE/ACTION/CONNECTION vote on all three joint frames. The geometry TYPE is injected into the prompt as a prior (strong default for enter/exit). Filters `passing_by`.
7. **Export** — Machine-readable JSON (`outputs/description/<clip_id>.json` and `outputs/description/interactions.json`) plus optional annotated MP4s under `outputs/videos/`.

**Note:** The VLM is expensive in runtime, so we filter proposals (and only send a few crops) before calling it. We could send more joint frames to Qwen for more precise typing, but that would cost more runtime. We also chose models that run on **CPU** (YOLO11s + Qwen2-VL-2B) to keep cost low. No external APIs. Seed is fixed (`SEED=0`) for determinism.

## Ambiguities and decisions

- **What counts as an interaction?** Enter, exit, or sustained contact (door/trunk/load/unload/take object). Walk-bys and standing near without contact are rejected. Alternatives: frame-level action recognition; pose-only heuristics.
- **Geometry vs VLM?** Geometry proposes; Qwen describes and may reject walk-bys. Resolve rules: (1) Qwen says `passing_by` → drop; (2) geometry already says enter/exit and Qwen disagrees → keep geometry (still frames confuse boarding/alighting/loading); (3) geometry said soft `interacting` → use Qwen’s type. Door-side enters also need Qwen to say `enter`. Alternatives: VLM-only typing; always override with VLM. 
Note: with more Qwen frames/calls we could rely more on its decision, but that costs a lot of runtime.
- **Person at the door but boxes don’t overlap?** YOLO often draws a car box that is too short, so the person stands beside the car with IoU≈0 and normal “near” never fires. We still propose `enter` if they stand close beside the car (small side gap + same height) and the person track ends there (boarding). Alternatives: enlarge boxes; ignore non-overlapping pairs.
- **Enter and exit both match?** Prefer the side with stronger near-fraction (end vs start). Never emit both for one pair. Alternatives: emit both; defer entirely to VLM.
- **Soft contact without board/alight?** Two tiers of `interacting` (sustained dwell + IoU, or short tight containment and IoU). Containment alone rejected (common curb walk-by FP). Qwen must confirm. Alternatives: export all near pairs; hard-code trunk ROI.
- **Track fragmentation?** Temporal NMS on shared ID + type + time proximity. Alternatives: Re-ID / BoT-SORT; ignore duplicates.

## Output format

Each interaction includes: `clip_id`, `type`, `frame_range`, `time_span_s`, person (`track_id`, description, representative bbox), vehicle (`track_id`, type, description), `connection`, `confidence`, and geometric `evidence`.

## Results (internal GT, 8 clips)

- **Detection** (IoU≥0.2 on time spans) — F1 **0.85** (P 0.92 / R 0.79)
- **Type** (on matched pairs) — Accuracy **0.91** (10/11)
- **Description quality** — LLM judge: Low (notes vs free-form CONNECTION mismatch; not the product goal)

Main misses: short exits/enters under occlusion (`NmlzoaDcOuI_1`, early exit in `iMGR_0AG3a8_2_3`). One type error: exit labeled as trunk `interacting`.

## Assumptions

- Clips are independent; no cross-clip identity.
- “Interaction” is contact-directed use of the vehicle, not mere proximity.
- YOLO + ByteTrack quality is sufficient; we do not require perfect tracks.
- CPU-only local inference is acceptable for a take-home.
- Annotated `Videos/*.json` files are for local evaluation only; the product artifact is `outputs/description/interactions.json`.

## Limitations

- Relies on 2D boxes: no depth, pose, or door/open-state model.
- Occlusion and ID switches still cause missed short exits and split events.
- Small Qwen2-VL-2B on still crops: weak on fine action wording and sometimes invents trunk vs door.
- Thresholds tuned on this clip set; may not transfer without re-tuning.
- Vehicle class can be wrong when boxes cover multiple objects (e.g. car labeled motorcycle).


