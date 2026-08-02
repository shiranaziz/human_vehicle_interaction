"""Step 6 — serialize kept interactions to the product JSON schema.

Writes one JSON file per clip (``outputs/description/<clip_id>.json``) and
optionally a combined ``outputs/description/interactions.json`` array. Only
kept (non-passing_by) interactions are included.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import config
from .describe import DescribedInteraction
from .interactions import Interaction
from .io_utils import VideoMetadata
from .tracklets import TrackletCollection


def _round_bbox(xywh: tuple[float, float, float, float]) -> list[float]:
    return [round(v, 1) for v in xywh]


def _representative_frame(
    interaction: Interaction, frames_used: list[int]
) -> int | None:
    """Peak describe frame (middle of ``frames_used``), else first near frame."""
    if frames_used:
        return frames_used[len(frames_used) // 2]
    if interaction.near_frames:
        return interaction.near_frames[0]
    return None


def _representative_bbox(
    collection: TrackletCollection,
    interaction: Interaction,
    frames_used: list[int],
) -> list[float] | None:
    frame_idx = _representative_frame(interaction, frames_used)
    if frame_idx is None:
        return None
    person = collection.persons.get(interaction.person_id)
    if person is None:
        return None
    obs = person.bbox_at(frame_idx)
    if obs is None:
        return None
    return _round_bbox(obs.xywh)


def _json_safe(value: Any) -> Any:
    """Recursively round floats; leave other JSON types alone."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parts(
    item: DescribedInteraction | Interaction,
) -> tuple[Interaction, str, str, str, list[int]]:
    if isinstance(item, DescribedInteraction):
        d = item.description
        return (
            item.interaction,
            d.person_description,
            d.vehicle_description,
            d.connection,
            list(d.frames_used),
        )
    return item, "", "", "", []


def build_interaction_record(
    interaction_id: int,
    item: DescribedInteraction | Interaction,
    collection: TrackletCollection,
    class_name_fn: Callable[[int], str],
) -> dict[str, Any]:
    """One interaction object in the product schema."""
    inter, person_desc, vehicle_desc, connection, frames_used = _parts(item)
    f0, f1 = inter.frame_range
    t0, t1 = inter.time_span_s
    person_block: dict[str, Any] = {
        "track_id": inter.person_id,
        "description": person_desc,
    }
    bbox = _representative_bbox(collection, inter, frames_used)
    if bbox is not None:
        person_block["representative_bbox"] = bbox

    return {
        "interaction_id": interaction_id,
        "type": inter.type,
        "connection": connection,
        "person": person_block,
        "vehicle": {
            "track_id": inter.vehicle_id,
            "type": class_name_fn(inter.vehicle_cls),
            "description": vehicle_desc,
        },
        "frame_range": [f0, f1],
        "time_span_s": [round(t0, 3), round(t1, 3)],
        "confidence": round(inter.confidence, 3),
        "evidence": _json_safe(dict(inter.evidence)),
    }


def build_clip_record(
    clip_id: str,
    meta: VideoMetadata,
    kept: Sequence[DescribedInteraction | Interaction],
    collection: TrackletCollection,
    class_name_fn: Callable[[int], str],
) -> dict[str, Any]:
    """One clip document: metadata + kept interactions."""
    interactions = [
        build_interaction_record(i + 1, item, collection, class_name_fn)
        for i, item in enumerate(kept)
    ]
    return {
        "clip_id": clip_id,
        "video_metadata": meta.to_dict(),
        "interactions": interactions,
    }


def clip_json_path(clip_id: str) -> Path:
    """Default path for a single-clip JSON artifact."""
    return config.OUTPUTS_DESCRIPTION_DIR / f"{clip_id}.json"


class InteractionExporter:
    """Serialize kept interactions to the product JSON schema."""

    def build_clip_record(
        self,
        clip_id: str,
        meta: VideoMetadata,
        kept: Sequence[DescribedInteraction | Interaction],
        collection: TrackletCollection,
        class_name_fn: Callable[[int], str],
    ) -> dict[str, Any]:
        return build_clip_record(
            clip_id, meta, kept, collection, class_name_fn
        )

    def write_clip(
        self,
        record: dict[str, Any],
        path: str | Path | None = None,
    ) -> Path:
        """Write one clip document to ``outputs/description/<clip_id>.json``."""
        out = Path(path) if path is not None else clip_json_path(
            str(record.get("clip_id", "clip"))
        )
        return self._dump(record, out)

    def write(
        self,
        records: Sequence[dict[str, Any]],
        path: str | Path | None = None,
    ) -> Path:
        """Write the clip-record array to ``outputs/description/interactions.json``."""
        out = (
            Path(path)
            if path is not None
            else config.OUTPUTS_DESCRIPTION_DIR / "interactions.json"
        )
        return self._dump(list(records), out)

    @staticmethod
    def _dump(payload: Any, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return path
