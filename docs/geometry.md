# Geometry module (Step 4)

There is no `geometry.py`. **Step 4 “geometry” lives in `src/interactions.py`**, mainly `GeometricInteractionFinder` and `_classify_pair`. Thresholds are in `src/config.py`.

Geometry is a **cheap box-geometry prior**. It proposes interaction types (`enter` / `exit` / `interacting` / `passing_by`) from bounding boxes and track lifetimes. It does **not** look at image pixels. Qwen (Step 5 in `src/describe.py`) reads the crops later.

---

## Pipeline context: `process_clip`

Call stack from the runnable entry point:

```
main.main
→ InteractionPipeline.run
→ process_clips
→ process_clip(video_path)
```

Inside `InteractionPipeline.process_clip` (`src/pipeline.py`):

1. **Probe video** — `probe_video(video_path)` → fps / metadata.
2. **Detect + track → tracklets** — `tracklets_from_video(..., model=self.tracker)` → person/vehicle `TrackletCollection`.
3. **Geometry proposals** — `self.finder.find(collection, meta.fps, ...)` → list of `Interaction`.
4. **Split** — `geometry_accepted` (`enter` / `exit` / `interacting`) vs `geometry_rejected` (`passing_by`).
5. **Optional Qwen** — `describe_interactions(accepted, ...)` if `run_describe=True`.
6. **Per-clip JSON** — `export_clip_json(result)` (uses kept described rows, else geometry accepted).
7. **Optional annotated MP4** — `annotate_clip(..., interactions=result.for_viz)`.
8. **Return** `ClipResult`.

Combined `outputs/interactions.json` is written later by `run()` → `export_json`. Models are loaded once in `__init__` and reused across clips.

---

## Root call stack for geometry

Every geometry step sits under:

```
main.main
→ InteractionPipeline.run
→ process_clips
→ process_clip
→ GeometricInteractionFinder.find
```

Thin wrappers:

- `find_interactions` → `GeometricInteractionFinder().find`
- `interactions_from_video` → tracklets + `find_interactions`
- CLI: `python -m src.interactions <video>`

---

## Step-by-step inside `find()`

### 1. Person × vehicle tracklets

**Input:** `TrackletCollection` from Step 3 (`src/tracklets.py`).

**Functions:** `tracklets_from_video` → `build_tracklets`

Each tracklet is an ordered list of observations `(frame_idx, bbox xyxy, conf)` for one ByteTrack ID, split into persons vs vehicles by majority class.

### 2. For every person–vehicle pair that overlaps in time

**Function:** `GeometricInteractionFinder.find`

```
find
→ for person in collection.persons
→ for vehicle in collection.vehicles
→ if person.end_frame < vehicle.start_frame: continue
→ if vehicle.end_frame < person.start_frame: continue
```

Cheap temporal gate: no shared frame range → skip before any bbox math.

### 3. Per-frame metrics — `_pair_metrics`

**Function:** `_pair_metrics(person, vehicle) → list[FrameMetrics]`

On every frame where **both** tracklets have an observation, build one `FrameMetrics` row:

| Field | Meaning |
| --- | --- |
| `frame_idx` | shared frame index |
| `iou` | intersection / union of the two boxes |
| `containment` | fraction of **person** area inside the **vehicle** box |
| `center_dist` | pixel distance between box centers |
| `person_xyxy` / `vehicle_xyxy` | the two boxes |

If the vehicle has no observation on that person frame → skip.

Example alignment:

```
person frames:  10  11  12  13  14
vehicle frames: 10      12  13      15
co-visible:     10      12  13        → 3 FrameMetrics
```

Stack:

```
find → _pair_metrics → iou / containment / center_distance
```

### 4. Mark near frames — `FrameMetrics.is_near`

A co-visible frame is **near** if either:

- `iou ≥ NEAR_IOU` (default `0.05`), or
- `containment ≥ NEAR_CONTAINMENT` (default `0.15`)

Center distance alone never marks near; it only drives motion trends later.

```
frame:        10   11   12   13   14   15
is_near:       ·    ·    ✓    ✓    ✓    ·
```

### 5. Count dwell

```text
dwell = number of near frames
```

- `dwell == 0` and no door-side → ignore pair (`None`)
- `dwell < MIN_DWELL_FRAMES` (default `5`) → usually `passing_by` (unless door-side enter fires)

### 6. Classify — `_classify_pair`

Uses near frames, trends, track birth/death, and (optionally) door-side signals to assign:

| Type | Meaning | Sent to VLM? |
| --- | --- | --- |
| `enter` | boarding signature | yes |
| `exit` | alighting signature | yes |
| `interacting` | contact without board/alight (trunk/load/etc.) | yes |
| `passing_by` | brief / transit proximity | no (rejected) |

---

## Geometry does not look at pixels

After tracking, each ID is only boxes over time. Geometry answers:

> Were these two IDs close, for how long, and was the person moving into or away from the car?

Flow:

```
boxes over time
  → FrameMetrics (how close each frame)
  → near frames + dwell (was contact long enough?)
  → trends + track birth/death (enter vs exit vs just near)
  → type label for Qwen
```

### Trends

Over the **near** span, compare early third vs late third:

| Signal | How | Meaning |
| --- | --- | --- |
| `containment_trend` | late − early mean containment | `+` → getting more inside vehicle box |
| `distance_trend` | late − early mean center distance | `+` → moving away |
| `approaching` | cont_trend ≥ 0.08 **or** dist_trend &lt; 0 | feeds enter |
| `leaving` | cont_trend ≤ −0.08 **or** dist_trend &gt; 0 | feeds exit |

Enter-like:

```
containment: 0.10 → 0.25 → 0.45   (+trend → approaching)
```

Exit-like:

```
containment: 0.50 → 0.30 → 0.10   (−trend → leaving)
```

Also used:

- `start_near_frac` / `end_near_frac` — fraction of person-track prefix/suffix that is near
- `track_started_near` / `track_terminated_near` — person track birth/death relative to vehicle lifetime (boarding often kills the person track; alighting often births one at the car)

---

## IoU vs containment

Both use the same intersection area; they normalize differently.

**IoU**

```text
IoU = intersection / (person + vehicle − intersection)
```

“How much do the two boxes overlap relative to their **combined** area?”  
Large vehicle boxes make IoU small even when the person clearly overlaps the car.

**Containment (person inside vehicle)**

```text
containment = intersection / person_area
```

“What **fraction of the person** sits inside the vehicle box?”  
Vehicle size barely matters.

Same scene example:

| Quantity | Value |
| --- | --- |
| Person area | 100 |
| Vehicle area | 2000 |
| Intersection | 50 |
| **containment** | 50/100 = **0.50** |
| **IoU** | 50/(100+2000−50) ≈ **0.024** |

Why both are used for `is_near`:

- **Containment** catches boarding/occlusion (person inside a large car box, IoU still low).
- **IoU** catches side-by-side contact where neither box contains the other much.

**center_dist** is different again: no overlap required — only how far centers are. Used for trends, not for marking near.

---

## Classification decision tree (`_classify_pair`)

### Early reject

- Empty metrics → `None`
- No near and no door-side → `None`
- Brief near (`dwell < MIN_DWELL_FRAMES`) → `passing_by` (unless door-side enter already returned)

### Enter (overlap-near boarding)

Typical signature:

- person track ends while vehicle still present
- last part of person track is near
- nearness concentrated at the end (`end_near_frac` high enough)
- approaching, or end-near &gt; start-near

### Exit (alighting)

Typical signature:

- vehicle already present when person track starts
- first part of person track is near
- nearness concentrated at the start
- leaving, or start-near &gt; end-near
- blocked if still heavily near at the end (`end_near_frac ≥ 0.7`)

### Enter ∧ exit both true

Prefer whichever side has higher near-fraction concentration (`end_near_frac` vs `start_near_frac`). Geometry never emits both.

### Interacting (soft contact)

No board/alight signature, but enough contact to send to VLM:

- **Sustained:** dwell ≥ 10 and (peak_containment ≥ 0.35 or peak_iou ≥ 0.08)
- **Strong short:** dwell ≥ 5 and (peak_containment ≥ 0.55 or peak_iou ≥ 0.15)

Needs Qwen confirmation later; else filtered as `passing_by`.

### Everything else near but loose

→ `passing_by`

---

## Door-side proximity (`door_side_near`)

**Not** door detection in the image. No pixels, no “is the door open?”

**Problem it solves:** YOLO often draws a **truncated / too-small vehicle box**. The person stands at a real door just **outside** the box → IoU = 0, containment = 0 → classic near never fires → boarding is missed.

```
          person box          car box (too short)
              ┌──┐            ┌──────────┐
              │  │   ← gap →  │          │
              └──┘            └──────────┘
```

### What `door_side_near` checks (all must pass)

1. **Vertical overlap** — person and vehicle share enough height (~25% of person height). Blocks sidewalk walk-unders.
2. **Strictly beside in x** — positive horizontal gap (no x-overlap). If they already overlap in x, normal IoU/containment handles it.
3. **Small gap** — gap ≤ `NEAR_DOOR_GAP_FRAC` × vehicle width (default 0.45).

```
OK:  [person] gap [====car====]          gap small, same height band
NO:  [person] ............. [====car====]  gap huge
NO:  boxes already overlap in x            use normal near instead
```

### Door-side enter (`is_door_enter`)

Special early path that can label `enter` when:

- enough door-side frames (`door_dwell ≥ MIN_DOOR_ENTER_FRAMES`, default 3)
- person track ends while vehicle still present
- last door-side frame is at the end of the person track (`dies_at_door`)
- person was not door-side at the start (`start_door_frac ≤ 0.25`)
- horizontal gap is shrinking (`gap_trend < -5`) → walking toward the car side
- **and** normal near never really worked (`dwell < MIN_DWELL_FRAMES` and peak containment still low)

Evidence reason: `door_side_boarding_truncated_vehicle`.

Helpers:

| Function | Role |
| --- | --- |
| `horizontal_edge_gap` | gap between boxes in x (0 if overlap) |
| `vertical_overlap` | shared height length |
| `door_side_near` | per-frame door-side predicate |
| `_door_gap_trend` | late − early gap (negative = approaching door) |
| `_frac_door_in_window` | door-side fraction of person-track prefix/suffix |

Config knobs (`src/config.py`):

- `NEAR_DOOR_GAP_FRAC = 0.45`
- `MIN_DOOR_ENTER_FRAMES = 3`

---

## After geometry: frames for Qwen

**Function:** `select_describe_frames`

From `near_frames`, pick up to `DESCRIBE_MAX_FRAMES` (default 3):

1. first near frame  
2. peak-containment near frame  
3. last near frame  

(Temporal order, deduped.)

In `describe.py`:

- geometry TYPE is injected into the prompt as a prior
- weighted votes use `GEO_ENTER_EXIT_WEIGHT` (2.5) and `GEO_INTERACTING_WEIGHT` (1.5)
- if geometry says enter and Qwen says exit (or reverse), **geometry wins**
- soft Qwen `interacting` under geometry enter/exit also keeps the geometric boarding label

---

## Key files

| File | Role |
| --- | --- |
| `src/interactions.py` | geometry Step 4 (`GeometricInteractionFinder`, metrics, classify) |
| `src/config.py` | thresholds (`NEAR_*`, dwell, interacting, door, GEO weights) |
| `src/tracklets.py` | input tracklets |
| `src/pipeline.py` | orchestrates detect → geometry → VLM → export |
| `src/describe.py` | Qwen consumer of accepted geometry |
| `main.py` | settings + clip selection |

---

## Quick reference: functions → stack

| Work | Function | Stack under `find()` |
| --- | --- | --- |
| Iterate pairs + temporal gate | `GeometricInteractionFinder.find` | `find → for person → for vehicle` |
| Per-frame metrics | `_pair_metrics` | `find → _pair_metrics → iou/containment/center_distance` |
| Mark near | `FrameMetrics.is_near` | `find → _classify_pair → m.is_near` |
| Door-side near | `door_side_near` | `find → _classify_pair → door_side_near` |
| Dwell + type | `_classify_pair` | `find → _classify_pair → trends / is_enter / is_exit / interacting` |
| Drop walk-bys (optional) | `Interaction.is_accepted` | `find → filter if not include_passing_by` |
| Describe frame pick | `select_describe_frames` | `describe_interactions → select_describe_frames` |
