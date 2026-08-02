"""VLM descriptions for person, vehicle, and their connection (Step 5).

For each accepted interaction, sample near/intersecting frames, crop the person
and vehicle boxes, then use local **Qwen2-VL-2B-Instruct** (greedy decoding) to
produce:

1. ``person_description`` — from the person crop alone
2. ``vehicle_description`` — from the vehicle crop alone
3. ``connection`` — from one joint crop covering person + vehicle

Person and vehicle appearance use one sharpest crop each (Laplacian variance).
TYPE / ACTION / CONNECTION votes use up to ``DESCRIBE_MAX_FRAMES`` joint crops
(start / peak / end). Votes are aggregated with weights (peak frame heavier)
plus a geometry prior; the geometry TYPE is also injected into the prompt.
Kept types: enter, exit, interacting. Walk-bys are filtered out. When geometry
says enter/exit and Qwen flips the other way, geometry wins.
Qwen must load successfully; no offline fallback.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

# Local config MUST be imported before torch/transformers: it sets cache-dir
# environment variables at import time that those libraries read on import.
from . import config
from .detect_track import DetectorTracker
from .interactions import (
    Interaction,
    InteractionType,
    containment,
    find_interactions,
    merge_overlapping_interactions,
    select_describe_frames,
)
from .io_utils import probe_video
from .tracklets import Tracklet, TrackletCollection, tracklets_from_video

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

Source = Literal["qwen"]

_PERSON_PROMPT = (
    "Describe this person in one short line: clothing, approximate age/gender "
    "if clear, and any notable accessories. No extra commentary."
)
_VEHICLE_PROMPT = (
    "Describe this vehicle in one short line: type, color, and any notable "
    "features. No extra commentary."
)
_GEO_PRIOR_HINTS = {
    "enter": "boarding / getting into the vehicle",
    "exit": "alighting / getting out of the vehicle",
    "interacting": (
        "non-boarding contact (door/trunk/load/unload/taking an object "
        "from the vehicle)"
    ),
    "passing_by": "walking or standing near with no vehicle-directed action",
}


def _connection_type_prompt(heuristic_type: InteractionType) -> str:
    """Build the TYPE/ACTION/CONNECTION prompt with a geometry TYPE prior."""
    hint = _GEO_PRIOR_HINTS.get(heuristic_type, heuristic_type)
    return (
        "First choose TYPE as exactly one of: enter, exit, interacting, passing_by. "
        "This image shows a person and a vehicle in the same scene. "
        "Decide whether this shows a real person-vehicle interaction per this rule: "
        "entering or exiting a vehicle, or sustained contact at a door/trunk "
        "(loading, unloading, opening, taking an object), counts; only walking "
        "past does NOT.\n"
        "Focus on the ACTION the person is doing with the vehicle, not just proximity.\n"
        f"GEOMETRY PRIOR (from tracking; treat as a strong default): "
        f"TYPE={heuristic_type} — {hint}. "
        "Prefer this prior unless the image clearly contradicts it "
        "(e.g. person only walking past with no contact).\n"
        "TYPE meanings:\n"
        "- enter = boarding: person moving into the cabin / body going through an "
        "open door inward (often mostly outside, leaning or stepping in).\n"
        "- exit = alighting: person leaving the cabin / body emerging outward from "
        "an open door (often partly inside or just stepping out).\n"
        "- interacting = real contact/use that is not boarding or alighting "
        "(opening trunk/hood/door, loading or unloading items, taking a blanket/"
        "bag/object from the vehicle, leaning into cabin).\n"
        "- passing_by = walking or standing near with no vehicle-directed action "
        "(includes standing at a door without opening it, loading, or boarding).\n"
        "Do not guess exit just because a door is open; prefer enter when the "
        "person is outside and moving toward/into the door.\n"
        "Reply in EXACTLY three lines:\n"
        "TYPE: <enter|exit|interacting|passing_by>\n"
        "ACTION: <short verb phrase, e.g. entering driver door, exiting rear door, "
        "opening trunk, loading bags, unloading items, taking blanket from car, "
        "walking past, standing near>\n"
        "CONNECTION: <one short line on the human-vehicle relationship>\n"
        "Use TYPE=passing_by when the person is only standing near / at the door "
        "with no clear enter/exit/trunk/object contact."
    )

# Default ACTION phrases when Qwen omits the ACTION line.
_DEFAULT_ACTIONS = {
    "enter": "entering the vehicle",
    "exit": "exiting the vehicle",
    "interacting": "interacting with the vehicle",
    "passing_by": "walking past the vehicle",
}

# Momentary boarding/alighting: one good frame + geometry is enough.
_DECISIVE_TYPES = frozenset({"enter", "exit"})
# Types that count as a real interaction (exported if not filtered as walk-by).
_INTERACTION_TYPES = frozenset({"enter", "exit", "interacting"})


@dataclass
class FrameVlmVote:
    """One Qwen TYPE/ACTION/CONNECTION vote on a near frame (explainability)."""

    frame_idx: int
    vlm_type: InteractionType | None
    action: str
    connection: str


@dataclass
class InteractionDescription:
    """Natural-language fields for one interaction (task-required + action)."""

    person_description: str
    vehicle_description: str
    connection: str
    source: Source
    frames_used: list[int]
    # Final action label (enter/exit/interacting/passing_by) and verb phrase.
    action: str = ""
    action_detail: str = ""
    # Qwen type used for filtering / final label.
    vlm_type: InteractionType | None = None
    heuristic_type: InteractionType | None = None
    kept: bool = True  # False when VLM filter rejects as passing_by
    # Why we kept / rejected this as an interaction (geometry + Qwen).
    explanation: str = ""
    # Per-frame Qwen votes that drove the type/action filter decision.
    vlm_votes: list[FrameVlmVote] = field(default_factory=list)


@dataclass
class DescribedInteraction:
    """Interaction plus Step 5 descriptions (after optional VLM type filter)."""

    interaction: Interaction
    description: InteractionDescription

    @property
    def kept(self) -> bool:
        return self.description.kept

    def summary(self, clip_id: str | None = None) -> str:
        """Human-readable block including task-required time/frame fields."""
        i = self.interaction
        d = self.description
        ht = d.heuristic_type or i.type
        vt = d.vlm_type or "-"
        flag = "kept" if d.kept else "filtered_out"
        f0, f1 = i.frame_range
        t0, t1 = i.time_span_s
        near = i.near_frames
        near_str = (
            f"[{near[0]}..{near[-1]}] n={len(near)}"
            if near
            else "[]"
        )
        head = f"clip_id={clip_id}\n" if clip_id else ""
        vote_lines = ""
        if d.vlm_votes:
            parts = []
            for v in d.vlm_votes:
                t = v.vlm_type or "?"
                act = v.action or "-"
                conn = v.connection or "-"
                parts.append(
                    f"      frame {v.frame_idx}:\n"
                    f"        TYPE: {t}\n"
                    f"        ACTION: {act}\n"
                    f"        CONNECTION: {conn}"
                )
            vote_lines = "\n    qwen_votes:\n" + "\n".join(parts)
        action = d.action or i.type
        action_detail = d.action_detail or d.connection
        return (
            f"{head}"
            f"person={i.person_id} vehicle={i.vehicle_id}\n"
            f"    action: {action}\n"
            f"    source: {d.source}\n"
            f"    filter: {flag}\n"
            f"    heuristic: {ht}\n"
            f"    vlm: {vt}\n"
            f"    action_detail: {action_detail}\n"
            f"    frame_range: [{f0}, {f1}]\n"
            f"    time_span_s: [{t0:.3f}, {t1:.3f}]\n"
            f"    near_frames: {near_str}\n"
            f"    describe_frames: {d.frames_used}\n"
            f"    person: {d.person_description}\n"
            f"    vehicle: {d.vehicle_description}\n"
            f"    connection: {d.connection}\n"
            f"    why: {d.explanation}"
            f"{vote_lines}"
        )


# ---------------------------------------------------------------------------
# Geometry / crops / frame IO
# ---------------------------------------------------------------------------


def _clamp_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float = config.CROP_PAD,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    x1 -= bw * pad
    y1 -= bh * pad
    x2 += bw * pad
    y2 += bh * pad
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def crop_bgr(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    pad: float = config.CROP_PAD,
) -> np.ndarray | None:
    """Return a padded BGR crop, or ``None`` if the box is empty/invalid."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_box(xyxy, w, h, pad=pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def joint_crop_bgr(
    frame: np.ndarray,
    person_xyxy: tuple[float, float, float, float],
    vehicle_xyxy: tuple[float, float, float, float],
    pad: float = config.CROP_PAD,
) -> np.ndarray | None:
    """Padded crop of the union box covering person and vehicle."""
    union = (
        min(person_xyxy[0], vehicle_xyxy[0]),
        min(person_xyxy[1], vehicle_xyxy[1]),
        max(person_xyxy[2], vehicle_xyxy[2]),
        max(person_xyxy[3], vehicle_xyxy[3]),
    )
    return crop_bgr(frame, union, pad=pad)


def _bgr_to_pil(crop: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _blur_score(bgr: np.ndarray) -> float:
    """Higher = sharper (Laplacian variance). Empty/invalid crops score -1."""
    if bgr is None or bgr.size == 0:
        return -1.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sharpest_crop(
    frames: list[int],
    frame_cache: dict[int, np.ndarray],
    crop_for_frame,
) -> tuple[int, np.ndarray] | None:
    """Return ``(frame_idx, crop)`` for the least-blurred usable crop."""
    best_f: int | None = None
    best_crop: np.ndarray | None = None
    best_score = -1.0
    for f in frames:
        frame = frame_cache.get(f)
        if frame is None:
            continue
        crop = crop_for_frame(frame, f)
        if crop is None:
            continue
        score = _blur_score(crop)
        if score > best_score:
            best_f, best_crop, best_score = f, crop, score
    if best_f is None or best_crop is None:
        return None
    return best_f, best_crop


def load_frames(
    video_path: str | Path, frame_idxs: list[int]
) -> dict[int, np.ndarray]:
    """Decode only the requested frame indices (sorted seek through the file)."""
    needed = sorted(set(frame_idxs))
    if not needed:
        return {}

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    out: dict[int, np.ndarray] = {}
    try:
        # Sequential read is more reliable than CAP_PROP_POS_FRAMES on many MP4s.
        target = set(needed)
        idx = 0
        while target:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in target:
                out[idx] = frame
                target.remove(idx)
            idx += 1
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# Qwen2-VL wrapper
# ---------------------------------------------------------------------------


class QwenDescriber:
    """Qwen2-VL-2B-Instruct captioner (greedy / deterministic)."""

    def __init__(self) -> None:
        config.set_determinism()
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            config.VLM_MODEL,
            revision=config.VLM_REVISION,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL,
            revision=config.VLM_REVISION,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        # Model card defaults include sample-only knobs; clear them so greedy
        # decoding does not warn about unused temperature/top_p/top_k.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def _generate(self, messages: list[dict]) -> str:
        assert self.model is not None and self.processor is not None
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        trimmed = [
            o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)
        ]
        text_out = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return " ".join(text_out.strip().split())

    def describe_person(self, crop: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": _PERSON_PROMPT},
                ],
            }
        ]
        return self._generate(messages)

    def describe_vehicle(self, crop: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": _VEHICLE_PROMPT},
                ],
            }
        ]
        return self._generate(messages)

    def describe_connection_and_type(
        self,
        joint_crop: Image.Image,
        *,
        heuristic_type: InteractionType = "interacting",
    ) -> tuple[InteractionType | None, str, str]:
        """Return ``(vlm_type, action_phrase, connection_line)`` from one scene crop.

        ``heuristic_type`` is the Step-4 geometry prior injected into the prompt
        so Qwen starts from the track-based hypothesis.
        """
        prompt = _connection_type_prompt(heuristic_type)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": joint_crop},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return _parse_type_action_connection(self._generate(messages))


# ---------------------------------------------------------------------------
# Aggregation + parsing + per-interaction describe
# ---------------------------------------------------------------------------


_TYPE_RE = re.compile(
    r"TYPE\s*:\s*(enter|exit|interacting|passing_by)", re.IGNORECASE
)
# Stop ACTION at CONNECTION / newline so one-line replies do not swallow fields.
_ACTION_RE = re.compile(
    r"ACTION\s*:\s*(.+?)(?=\s*CONNECTION\s*:|\n|$)", re.IGNORECASE | re.DOTALL
)
_CONN_RE = re.compile(r"CONNECTION\s*:\s*(.+)", re.IGNORECASE)


# Contact verbs that make "standing at door" a real interaction, not a walk-by.
_CONTACT_ACTION_MARKERS = (
    "opening trunk",
    "opening hood",
    "opening door",
    "loading",
    "unloading",
    "taking out",
    "taking a",
    "removing",
    "putting in",
    "blanket",
    "leaning into",
    "boarding",
    "entering",
    "exiting",
    "getting in",
    "getting out",
)
_WEAK_STANDING_MARKERS = (
    "standing at",
    "standing near",
    "standing by",
    "at the door",
    "waiting",
    "observing",
)


def _has_contact_action(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _CONTACT_ACTION_MARKERS)


def _is_weak_standing_text(text: str) -> bool:
    """True for door-loitering phrases with no load/board/trunk contact verb."""
    low = text.lower()
    if _has_contact_action(low):
        return False
    return any(p in low for p in _WEAK_STANDING_MARKERS)


def _infer_type_from_text(text: str) -> InteractionType | None:
    """Best-effort TYPE from free-text ACTION/CONNECTION when TYPE line is missing."""
    low = text.lower()
    if any(
        p in low
        for p in (
            "walking past",
            "walk past",
            "passing by",
            "passing_by",
            "no interaction",
            "without any",
            "talking on phone",
            "on their phone",
            "on the phone",
        )
    ):
        return "passing_by"
    # Standing at/near a door with no contact verb is a walk-by, not interacting.
    if _is_weak_standing_text(low):
        return "passing_by"
    if any(p in low for p in ("exiting", "exits", "getting out", "leaves the")):
        return "exit"
    if any(p in low for p in ("entering", "enters", "getting in", "getting into")):
        return "enter"
    if any(
        p in low
        for p in (
            "opening trunk",
            "opening hood",
            "loading",
            "unloading",
            "taking out",
            "taking a",
            "removing",
            "putting in",
            "blanket",
            "leaning into",
            "interacting",
        )
    ):
        return "interacting"
    return None


def _parse_type_action_connection(
    text: str,
) -> tuple[InteractionType | None, str, str]:
    """Parse the strict TYPE / ACTION / CONNECTION reply (best-effort)."""
    vlm_type: InteractionType | None = None
    m_type = _TYPE_RE.search(text)
    if m_type:
        vlm_type = m_type.group(1).lower()  # type: ignore[assignment]

    m_action = _ACTION_RE.search(text)
    action = m_action.group(1).strip() if m_action else ""
    # If ACTION still contains a nested CONNECTION label, split it.
    if action and "CONNECTION:" in action.upper():
        cut = re.split(r"CONNECTION\s*:", action, maxsplit=1, flags=re.IGNORECASE)
        action = cut[0].strip()

    m_conn = _CONN_RE.search(text)
    if m_conn:
        connection = m_conn.group(1).strip()
        # Drop a duplicated leading "ACTION: ..." if the model echoed it.
        connection = re.sub(
            r"^ACTION\s*:\s*", "", connection, count=1, flags=re.IGNORECASE
        ).strip()
    else:
        connection = " ".join(text.strip().split())

    if vlm_type is None:
        vlm_type = _infer_type_from_text(f"{action} {connection}")

    # Qwen sometimes labels TYPE=interacting while ACTION/CONNECTION clearly
    # describe a walk-by — trust the free text for filtering.
    if vlm_type in _INTERACTION_TYPES:
        walk_by = _infer_type_from_text(f"{action} {connection}")
        if walk_by == "passing_by":
            vlm_type = "passing_by"

    if not action:
        # Fall back to connection text or a type-based default.
        if connection and not m_conn:
            action = connection
        elif vlm_type is not None:
            action = _DEFAULT_ACTIONS.get(vlm_type, connection)
        else:
            action = connection
    return vlm_type, action, connection


def _aggregate(texts: list[str], preferred: str | None = None) -> str:
    """Majority vote over non-empty strings; ties broken by ``preferred`` then first."""
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return preferred.strip() if preferred else ""
    counts = Counter(cleaned)
    best_count = max(counts.values())
    winners = [t for t, c in counts.items() if c == best_count]
    if preferred and preferred in winners:
        return preferred
    return winners[0]


def _vote_weight(frame_idx: int, peak_idx: int | None) -> float:
    if peak_idx is not None and frame_idx == peak_idx:
        return float(config.VLM_PEAK_VOTE_WEIGHT)
    return float(config.VLM_VOTE_WEIGHT)


def _aggregate_type(
    votes: list[FrameVlmVote],
    *,
    peak_idx: int | None,
    heuristic_type: InteractionType,
) -> tuple[InteractionType | None, dict[str, float]]:
    """Weighted VLM votes + geometry prior → proposed TYPE.

    Returns ``(vlm_type, scores)`` where ``scores`` is kept for explainability
    (Qwen votes + geometry prior). Enter↔exit direction conflicts are resolved
    later in ``_apply_vlm_type`` (geometry wins); this function must not invent
    an opposite enter/exit label from the geometry prior alone.

    - Each typed Qwen vote adds ``VLM_VOTE_WEIGHT`` (peak frame:
      ``VLM_PEAK_VOTE_WEIGHT``).
    - Geometry ``enter``/``exit`` adds ``GEO_ENTER_EXIT_WEIGHT``;
      geometry ``interacting`` adds weaker ``GEO_INTERACTING_WEIGHT``.
    - Kept types: ``enter`` / ``exit`` / ``interacting``. Geometry may tip a
      pass-vs-interact tie **only when Qwen cast at least one interaction
      vote**. Unanimous Qwen ``passing_by`` always rejects — the prior must
      not invent an interaction from track shape alone.
    """
    qwen_scores: dict[str, float] = defaultdict(float)
    for v in votes:
        if v.vlm_type is None:
            continue
        qwen_scores[v.vlm_type] += _vote_weight(v.frame_idx, peak_idx)

    scores: dict[str, float] = defaultdict(float, qwen_scores)
    if heuristic_type in _DECISIVE_TYPES:
        scores[heuristic_type] += float(config.GEO_ENTER_EXIT_WEIGHT)
    elif heuristic_type == "interacting":
        scores["interacting"] += float(config.GEO_INTERACTING_WEIGHT)

    # Round for stable logging / JSON.
    scores = {k: round(v, 3) for k, v in scores.items() if v > 0}
    if not scores:
        return None, {}

    qwen_interact = sum(qwen_scores.get(t, 0.0) for t in _INTERACTION_TYPES)
    # No Qwen interaction support → walk-by (geometry prior cannot invent one).
    if qwen_interact <= 0.0:
        if qwen_scores.get("passing_by", 0.0) > 0.0:
            return "passing_by", scores
        return (
            heuristic_type if heuristic_type in _INTERACTION_TYPES else "passing_by",
            scores,
        )

    pass_score = scores.get("passing_by", 0.0)
    interact_score = sum(scores.get(t, 0.0) for t in _INTERACTION_TYPES)

    if pass_score > interact_score:
        return "passing_by", scores
    if pass_score == interact_score:
        # Qwen has some interaction mass; geometry tips keep on a true tie.
        if heuristic_type in _INTERACTION_TYPES:
            return heuristic_type, scores
        return "passing_by", scores

    # Among interaction types: prefer Qwen votes. Do not let geometry priors
    # create a phantom opposite enter/exit (mismatched ACTION text).
    label_scores = {
        t: qwen_scores[t]
        for t in ("enter", "exit", "interacting")
        if qwen_scores.get(t, 0.0) > 0
    }
    if not label_scores:
        return (
            heuristic_type if heuristic_type in _INTERACTION_TYPES else "passing_by",
            scores,
        )
    best = max(label_scores.values())
    winners = [t for t, s in label_scores.items() if s == best]
    if len(winners) == 1:
        return winners[0], scores
    if heuristic_type in winners:
        return heuristic_type, scores
    # Prefer specific boarding labels over soft interacting on a Qwen tie.
    for preferred in ("enter", "exit", "interacting"):
        if preferred in winners:
            return preferred, scores
    return winners[0], scores


def _texts_for_type(
    votes: list[FrameVlmVote],
    vlm_type: InteractionType | None,
    peak_idx: int | None,
) -> tuple[list[str], list[str], str | None, str | None]:
    """ACTION/CONNECTION texts from frames that voted ``vlm_type``.

    Returns ``(actions, connections, preferred_action, preferred_connection)``
    where preferred comes from the peak frame when it matches, else first match.
    Falls back to empty lists when nothing matches (caller uses all-frame texts).
    """
    if vlm_type is None:
        return [], [], None, None
    matching = [v for v in votes if v.vlm_type == vlm_type]
    if not matching:
        return [], [], None, None
    actions = [v.action for v in matching]
    connections = [v.connection for v in matching]
    preferred_action = preferred_connection = None
    for v in matching:
        if v.frame_idx == peak_idx:
            preferred_action, preferred_connection = v.action, v.connection
            break
    if preferred_action is None:
        preferred_action, preferred_connection = matching[0].action, matching[0].connection
    return actions, connections, preferred_action, preferred_connection


def _build_explanation(
    *,
    kept: bool,
    heuristic_type: InteractionType,
    vlm_type: InteractionType | None,
    action_detail: str,
    connection: str,
    votes: list[FrameVlmVote],
    evidence: dict,
    final_type: InteractionType | None = None,
) -> str:
    """Plain-language reason we kept/rejected this action as an interaction."""
    dwell = evidence.get("dwell_frames", "?")
    peak_c = evidence.get("peak_containment", "?")
    geo = (
        f"Geometry proposed type={heuristic_type} "
        f"(dwell_frames={dwell}, peak_containment={peak_c})."
    )

    if vlm_type is None:
        return (
            f"Geometry proposed action={heuristic_type} "
            f"(dwell_frames={dwell}, peak_containment={peak_c}). "
            f"No usable Qwen votes (missing crops or empty replies); "
            f"kept using the geometric action."
        )

    counts = Counter(v.vlm_type for v in votes if v.vlm_type is not None)
    vote_str = ", ".join(f"{t}×{n}" for t, n in sorted(counts.items())) or "none"
    score_map = evidence.get("type_scores") or {}
    score_str = (
        ", ".join(f"{t}={s}" for t, s in sorted(score_map.items()))
        if score_map
        else "n/a"
    )
    action_said = action_detail.strip() or connection.strip() or "(no action text)"
    how = (
        f"weighted TYPE={vlm_type} (votes {vote_str}; scores {score_str}; "
        f"geometry prior={heuristic_type})"
    )

    if not kept:
        return (
            f"Filtered out (not an interaction) because weighted scores favored "
            f"passing_by ({how}) and Qwen described the action as: "
            f"\"{action_said}\". {geo}"
        )

    # Geometry track lifetime wins when Qwen flips enter ↔ exit on a still crop.
    if evidence.get("vlm_enter_exit_overridden"):
        return (
            f"Kept as action={final_type or heuristic_type} because geometry "
            f"classified {heuristic_type} (dwell_frames={dwell}, "
            f"peak_containment={peak_c}), while aggregation gave {how} with "
            f"\"{action_said}\". Enter/exit direction uses the geometric "
            f"track signature when it conflicts with Qwen."
        )

    return (
        f"Kept as action={vlm_type} because {how} and described the action as: "
        f"\"{action_said}\". {geo} The action label comes from Qwen + geometry."
    )


def _text_matches_type(text: str, itype: InteractionType) -> bool:
    """True if free text clearly agrees with enter/exit ``itype``."""
    low = text.lower()
    if itype == "enter":
        return any(
            p in low for p in ("enter", "entering", "getting in", "boarding")
        )
    if itype == "exit":
        return any(
            p in low for p in ("exit", "exiting", "getting out", "alighting")
        )
    return True


def _apply_vlm_type(
    interaction: Interaction,
    vlm_type: InteractionType | None,
    *,
    filter_type: bool,
) -> tuple[Interaction, bool]:
    """Overwrite type with VLM label; return ``(interaction, kept)``.

    When ``filter_type`` is on and VLM says ``passing_by``, the pair is rejected
    (kept=False) but the Interaction object is still returned for logging.

    When geometry already has a decisive enter/exit and Qwen reports the
    opposite enter/exit, keep the pair but prefer the geometric direction —
    single-frame VLM crops often confuse boarding with alighting.

    When geometry says enter/exit and Qwen softens to ``interacting``, keep the
    geometric boarding/alighting label (still-frame Qwen often says "loading"
    / "opening trunk" for real enters/exits).

    Door-side geometry enters are only kept when Qwen also says ``enter`` —
    truncated-box boarding is rare, and door-side + "opening trunk" is a
    common FP.
    """
    heuristic = interaction.type
    evidence = dict(interaction.evidence)
    evidence["heuristic_type"] = heuristic
    if vlm_type is not None:
        evidence["vlm_type"] = vlm_type

    if vlm_type is None:
        return (
            replace(interaction, evidence=evidence),
            True,
        )

    # Door-side path: require Qwen enter confirmation.
    if (
        filter_type
        and evidence.get("proposal_reason")
        == "door_side_boarding_truncated_vehicle"
        and vlm_type != "enter"
    ):
        evidence["rejected_reason"] = "door_side_without_qwen_enter"
        vlm_type = "passing_by"

    kept = not (filter_type and vlm_type == "passing_by")
    # Even when filtered out, surface the VLM label on the object for logs.
    if not kept:
        final_type: InteractionType = "passing_by"
    elif (
        heuristic in _DECISIVE_TYPES
        and vlm_type in _DECISIVE_TYPES
        and heuristic != vlm_type
    ):
        final_type = heuristic
        evidence["vlm_enter_exit_overridden"] = True
    elif heuristic in _DECISIVE_TYPES and vlm_type == "interacting":
        # Track lifetime > still-frame "loading/trunk" paraphrase.
        final_type = heuristic
        evidence["vlm_interacting_overridden"] = True
    elif vlm_type in _INTERACTION_TYPES:
        final_type = vlm_type
    else:
        final_type = heuristic

    return (
        replace(interaction, type=final_type, evidence=evidence),
        kept,
    )


def describe_interaction(
    interaction: Interaction,
    collection: TrackletCollection,
    video_path: str | Path,
    describer: QwenDescriber,
    frame_cache: dict[int, np.ndarray] | None = None,
    *,
    filter_type: bool = config.VLM_FILTER_TYPE,
) -> DescribedInteraction:
    """Describe one interaction and optionally filter type via Qwen.

    Person / vehicle appearance: one Qwen call each on the sharpest crop among
    the describe-frame candidates. Connection / TYPE: one vote per describe
    frame (up to ``DESCRIBE_MAX_FRAMES``).
    """
    heuristic_type = interaction.type
    person = collection.persons[interaction.person_id]
    vehicle = collection.vehicles[interaction.vehicle_id]
    frames = select_describe_frames(interaction, person, vehicle)
    if frame_cache is None:
        frame_cache = load_frames(video_path, frames)

    def _person_crop(frame: np.ndarray, f: int) -> np.ndarray | None:
        obs = person.bbox_at(f)
        return None if obs is None else crop_bgr(frame, obs.xyxy)

    def _vehicle_crop(frame: np.ndarray, f: int) -> np.ndarray | None:
        obs = vehicle.bbox_at(f)
        return None if obs is None else crop_bgr(frame, obs.xyxy)

    # --- appearance: single sharpest crop each ----------------------------
    person_description = ""
    vehicle_description = ""
    person_frame: int | None = None
    vehicle_frame: int | None = None
    sharp_person = _sharpest_crop(frames, frame_cache, _person_crop)
    if sharp_person is not None:
        person_frame, p_crop = sharp_person
        person_description = describer.describe_person(_bgr_to_pil(p_crop))
    sharp_vehicle = _sharpest_crop(frames, frame_cache, _vehicle_crop)
    if sharp_vehicle is not None:
        vehicle_frame, v_crop = sharp_vehicle
        vehicle_description = describer.describe_vehicle(_bgr_to_pil(v_crop))

    # --- connection / TYPE: multi-frame votes -----------------------------
    connection_texts: list[str] = []
    action_texts: list[str] = []
    votes: list[FrameVlmVote] = []
    peak_connection = peak_action = None
    used: list[int] = []

    # Peak-containment frame among candidates (for vote weighting).
    peak_idx: int | None = None
    best_c = -1.0
    for f in frames:
        p_obs = person.bbox_at(f)
        v_obs = vehicle.bbox_at(f)
        if p_obs is None or v_obs is None:
            continue
        c = containment(p_obs.xyxy, v_obs.xyxy)
        if c > best_c:
            best_c = c
            peak_idx = f

    for f in frames:
        frame = frame_cache.get(f)
        if frame is None:
            continue
        p_obs = person.bbox_at(f)
        v_obs = vehicle.bbox_at(f)
        if p_obs is None or v_obs is None:
            continue
        j_crop = joint_crop_bgr(frame, p_obs.xyxy, v_obs.xyxy)
        if j_crop is None:
            continue

        used.append(f)
        vlm_t, a_txt, c_txt = describer.describe_connection_and_type(
            _bgr_to_pil(j_crop),
            heuristic_type=heuristic_type,
        )
        connection_texts.append(c_txt)
        action_texts.append(a_txt)
        votes.append(
            FrameVlmVote(
                frame_idx=f, vlm_type=vlm_t, action=a_txt, connection=c_txt
            )
        )
        if f == peak_idx:
            peak_connection, peak_action = c_txt, a_txt

    frames_used = sorted(
        {f for f in (person_frame, vehicle_frame, *used) if f is not None}
    )

    if votes:
        vlm_type, type_scores = _aggregate_type(
            votes, peak_idx=peak_idx, heuristic_type=heuristic_type
        )
        # Prefer ACTION/CONNECTION from frames that voted for the chosen type
        # so a single exit/enter frame is not drowned out by "walking past".
        type_actions, type_conns, pref_action, pref_conn = _texts_for_type(
            votes, vlm_type, peak_idx
        )
        if type_actions:
            action_detail = _aggregate(type_actions, pref_action)
            connection = _aggregate(type_conns, pref_conn)
        else:
            connection = _aggregate(connection_texts, peak_connection)
            action_detail = _aggregate(action_texts, peak_action)
        # Door-loitering / phone: Qwen often says TYPE=interacting with a
        # non-contact ACTION while geometry prior keeps the score above
        # passing_by. Trust the free-text ACTION/CONNECTION for filtering.
        if filter_type and vlm_type in _INTERACTION_TYPES:
            text_type = _infer_type_from_text(f"{action_detail} {connection}")
            if text_type == "passing_by":
                vlm_type = "passing_by"
                type_scores = {
                    **type_scores,
                    "passing_by": round(
                        float(type_scores.get("passing_by", 0.0)) + 0.01, 3
                    ),
                }
        # Stash scores before apply so explanations can read them.
        interaction = replace(
            interaction,
            evidence={
                **interaction.evidence,
                "type_scores": type_scores,
                "appearance_person_frame": person_frame,
                "appearance_vehicle_frame": vehicle_frame,
            },
        )
        updated, kept = _apply_vlm_type(
            interaction, vlm_type, filter_type=filter_type
        )
        action_label = updated.type
        # Final enter/exit label must not keep opposite Qwen phrases (e.g. exit
        # + "boarding"). Same cleanup as the geometry-override path.
        if action_label in _DECISIVE_TYPES:
            if not _text_matches_type(action_detail, action_label):
                action_detail = _DEFAULT_ACTIONS.get(action_label, action_detail)
            if not _text_matches_type(connection, action_label):
                connection = _DEFAULT_ACTIONS.get(action_label, connection)
        explanation = _build_explanation(
            kept=kept,
            heuristic_type=heuristic_type,
            vlm_type=vlm_type,
            action_detail=action_detail,
            connection=connection,
            votes=votes,
            evidence=updated.evidence,
            final_type=action_label,
        )
        # Persist explainability on the interaction for later JSON (Step 6).
        updated = replace(
            updated,
            evidence={
                **updated.evidence,
                "explanation": explanation,
                "action": action_label,
                "action_detail": action_detail,
                "qwen_connection": connection,
                "qwen_votes": [
                    {
                        "frame": v.frame_idx,
                        "type": v.vlm_type,
                        "action": v.action,
                        "connection": v.connection,
                    }
                    for v in votes
                ],
            },
        )
        desc = InteractionDescription(
            person_description=person_description,
            vehicle_description=vehicle_description,
            connection=connection,
            source="qwen",
            frames_used=frames_used,
            action=action_label,
            action_detail=action_detail,
            vlm_type=vlm_type,
            heuristic_type=heuristic_type,
            kept=kept,
            explanation=explanation,
            vlm_votes=votes,
        )
        return DescribedInteraction(interaction=updated, description=desc)

    # No usable connection crops: keep the geometric proposal.
    action_detail = _DEFAULT_ACTIONS.get(heuristic_type, heuristic_type)
    explanation = _build_explanation(
        kept=True,
        heuristic_type=heuristic_type,
        vlm_type=None,
        action_detail=action_detail,
        connection="",
        votes=[],
        evidence=interaction.evidence,
    )
    desc = InteractionDescription(
        person_description=person_description,
        vehicle_description=vehicle_description,
        connection="",
        source="qwen",
        frames_used=frames_used,
        action=heuristic_type,
        action_detail=action_detail,
        vlm_type=None,
        heuristic_type=heuristic_type,
        kept=True,
        explanation=explanation,
    )
    return DescribedInteraction(interaction=interaction, description=desc)


def merge_described_interactions(
    described: list[DescribedInteraction],
    *,
    gap_s: float | None = None,
) -> list[DescribedInteraction]:
    """Temporal NMS on kept rows; Qwen-filtered rows pass through unchanged.

    Re-runs :func:`merge_overlapping_interactions` on post-VLM types so fragments
    that only align after type rewriting are still collapsed.
    """
    kept = [d for d in described if d.kept]
    not_kept = [d for d in described if not d.kept]
    if len(kept) <= 1:
        return list(described)

    by_key = {
        (
            d.interaction.person_id,
            d.interaction.vehicle_id,
            d.interaction.type,
            d.interaction.frame_range,
        ): d
        for d in kept
    }
    merged_inter = merge_overlapping_interactions(
        [d.interaction for d in kept], gap_s=gap_s
    )
    out = list(not_kept)
    for inter in merged_inter:
        key = (
            inter.person_id,
            inter.vehicle_id,
            inter.type,
            inter.frame_range,
        )
        base = by_key[key]
        if inter.evidence is not base.interaction.evidence:
            out.append(replace(base, interaction=inter))
        else:
            out.append(base)
    out.sort(
        key=lambda d: (
            d.interaction.frame_range[0],
            d.interaction.person_id,
            d.interaction.vehicle_id,
            d.interaction.type,
        )
    )
    return out


def describe_interactions(
    interactions: list[Interaction],
    collection: TrackletCollection,
    video_path: str | Path,
    describer: QwenDescriber | None = None,
    *,
    filter_type: bool = config.VLM_FILTER_TYPE,
    include_filtered: bool = False,
) -> list[DescribedInteraction]:
    """Describe accepted interactions; optionally drop VLM ``passing_by``.

    Geometry (Step 4) proposes candidates; Qwen confirms type. Pairs Qwen labels
    ``passing_by`` are filtered out unless ``include_filtered=True`` (for logs).
    """
    accepted = [i for i in interactions if i.is_accepted]
    if not accepted:
        return []

    if describer is None:
        describer = QwenDescriber()

    all_frames: list[int] = []
    for inter in accepted:
        person = collection.persons[inter.person_id]
        vehicle = collection.vehicles[inter.vehicle_id]
        all_frames.extend(select_describe_frames(inter, person, vehicle))
    frame_cache = load_frames(video_path, all_frames)

    described = [
        describe_interaction(
            inter,
            collection,
            video_path,
            describer=describer,
            frame_cache=frame_cache,
            filter_type=filter_type,
        )
        for inter in accepted
    ]
    described = merge_described_interactions(described)
    if include_filtered:
        return described
    return [d for d in described if d.kept]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Describe person, vehicle, and connection for accepted interactions "
            "(Step 5)."
        )
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 clip")
    parser.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    args = parser.parse_args()

    tracker = DetectorTracker()
    meta = probe_video(args.video)
    collection = tracklets_from_video(
        args.video, model=tracker, stride=args.stride
    )
    interactions = find_interactions(
        collection, meta.fps, include_passing_by=False
    )
    n_geo = len(interactions)
    interactions = [
        i
        for i in interactions
        if i.confidence >= config.MIN_INTERACTION_CONFIDENCE
    ]
    print(
        f"clip={args.video.name} geometry_accepted={n_geo} "
        f"after_conf_gate={len(interactions)} "
        f"(conf≥{config.MIN_INTERACTION_CONFIDENCE:.2f})"
    )
    described = describe_interactions(
        interactions, collection, args.video, include_filtered=True
    )
    kept = [d for d in described if d.kept]
    filtered = [d for d in described if not d.kept]
    clip_id = args.video.stem
    print(f"after Qwen filter: kept={len(kept)} filtered_out={len(filtered)}")
    print("kept:")
    for d in kept:
        print(d.summary(clip_id=clip_id))
    if filtered:
        print("filtered_out:")
        for d in filtered:
            print(d.summary(clip_id=clip_id))
