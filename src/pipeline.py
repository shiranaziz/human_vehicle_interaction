"""End-to-end person–vehicle interaction pipeline (OOP facade).

Owns heavy models once and runs Steps 3–6 per clip:

1. Detect + track → tracklets
2. Geometric interaction proposals (enter / exit / interacting)
3. Optional Qwen describe + type filter (drops passing_by)
4. Per-clip JSON export (right after Qwen) + optional annotated MP4

``main.py`` should only configure settings and call :class:`InteractionPipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import config
from .describe import DescribedInteraction, QwenDescriber, describe_interactions
from .detect_track import DetectorTracker
from .interactions import GeometricInteractionFinder, Interaction
from .io_utils import VideoMetadata, probe_video
from .output import InteractionExporter
from .tracklets import TrackletCollection, tracklets_from_video
from .visualize import annotate_clip


@dataclass
class PipelineSettings:
    """Run-time knobs (edit from ``main.py`` or construct in tests)."""

    stride: int = config.FRAME_STRIDE
    run_describe: bool = True
    write_annotated: bool = True
    show_passing_by: bool = True
    show_vlm_filtered: bool = True
    verbose: bool = True


@dataclass
class ClipResult:
    """Everything produced for one clip (domain objects + optional artifacts)."""

    clip_id: str
    video_path: Path
    meta: VideoMetadata
    collection: TrackletCollection
    geometry_accepted: list[Interaction]
    geometry_rejected: list[Interaction]
    described: list[DescribedInteraction] = field(default_factory=list)
    json_path: Path | None = None
    annotated_path: Path | None = None

    @property
    def kept(self) -> list[DescribedInteraction]:
        return [d for d in self.described if d.kept]

    @property
    def filtered(self) -> list[DescribedInteraction]:
        return [d for d in self.described if not d.kept]

    @property
    def for_json(self) -> Sequence[DescribedInteraction | Interaction]:
        """Rows that go into the product JSON (kept described, else geometry)."""
        if self.described:
            return self.kept
        return self.geometry_accepted

    @property
    def for_viz(self) -> list[Interaction]:
        if self.described:
            return [d.interaction for d in self.kept]
        return self.geometry_accepted


class InteractionPipeline:
    """Orchestrates detect → geometry → VLM → export for one or many clips.

    Heavy models (:class:`DetectorTracker`, :class:`QwenDescriber`) are loaded
    once in ``__init__`` and reused across clips.
    """

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or PipelineSettings()
        self.tracker = DetectorTracker()
        self.finder = GeometricInteractionFinder()
        self.exporter = InteractionExporter()
        self.describer: QwenDescriber | None = (
            QwenDescriber() if self.settings.run_describe else None
        )

    def process_clip(self, video_path: str | Path) -> ClipResult:
        """Run the full pipeline on a single MP4."""
        video_path = Path(video_path)
        s = self.settings
        meta = probe_video(video_path)
        collection = tracklets_from_video(
            video_path, model=self.tracker, stride=s.stride
        )
        interactions = self.finder.find(
            collection, meta.fps, include_passing_by=s.show_passing_by
        )
        accepted = [i for i in interactions if i.is_accepted]
        rejected = [i for i in interactions if not i.is_accepted]

        result = ClipResult(
            clip_id=video_path.stem,
            video_path=video_path,
            meta=meta,
            collection=collection,
            geometry_accepted=accepted,
            geometry_rejected=rejected,
        )

        if s.verbose:
            self._print_geometry(result)

        if self.describer is not None:
            result.described = describe_interactions(
                accepted,
                collection,
                video_path,
                describer=self.describer,
                include_filtered=True,
            )
            if s.verbose:
                self._print_described(result)

        # Persist this clip as soon as Qwen (or geometry-only) is done,
        # before annotation / the next video.
        result.json_path = self.export_clip_json(result)

        if s.write_annotated:
            result.annotated_path = annotate_clip(
                video_path,
                model=self.tracker,
                stride=s.stride,
                collection=collection,
                interactions=result.for_viz,
            )
            if s.verbose:
                print(f"annotated video -> {result.annotated_path}")

        return result

    def process_clips(
        self, video_paths: Sequence[str | Path]
    ) -> list[ClipResult]:
        """Process many clips; models stay loaded between clips."""
        return [self.process_clip(p) for p in video_paths]

    def export_clip_json(
        self,
        result: ClipResult,
        path: str | Path | None = None,
    ) -> Path:
        """Write ``outputs/<clip_id>.json`` for one finished clip."""
        record = self.exporter.build_clip_record(
            result.clip_id,
            result.meta,
            result.for_json,
            result.collection,
            self.tracker.class_name,
        )
        out = self.exporter.write_clip(record, path)
        if self.settings.verbose:
            print(f"interactions JSON -> {out}")
        return out

    def export_json(
        self,
        results: Sequence[ClipResult],
        path: str | Path | None = None,
    ) -> Path:
        """Write combined ``outputs/interactions.json`` from clip results."""
        records = [
            self.exporter.build_clip_record(
                r.clip_id,
                r.meta,
                r.for_json,
                r.collection,
                self.tracker.class_name,
            )
            for r in results
        ]
        out = self.exporter.write(records, path)
        if self.settings.verbose:
            print(f"\ncombined interactions JSON -> {out}")
        return out

    def run(
        self, video_paths: Sequence[str | Path]
    ) -> tuple[list[ClipResult], Path]:
        """Process all clips (per-clip JSON written after each) + combined JSON."""
        results = self.process_clips(video_paths)
        json_path = self.export_json(results)
        return results, json_path

    # ------------------------------------------------------------------
    # Logging helpers (keep ``main`` free of print formatting)
    # ------------------------------------------------------------------

    def _print_geometry(self, result: ClipResult) -> None:
        s = self.settings
        print(f"\n=== {result.video_path.name} ===")
        print(result.collection.summary())
        print(
            f"geometry: accepted={len(result.geometry_accepted)} "
            f"passing_by={len(result.geometry_rejected)} "
            f"(fps={result.meta.fps:.2f})"
        )
        print("geometry accepted:")
        for i in result.geometry_accepted:
            vname = self.tracker.class_name(i.vehicle_cls)
            print(f"  {i.summary()} vehicle_type={vname}")
        if s.show_passing_by:
            print("geometry passing_by:")
            for i in result.geometry_rejected:
                vname = self.tracker.class_name(i.vehicle_cls)
                print(f"  {i.summary()} vehicle_type={vname}")

    def _print_described(self, result: ClipResult) -> None:
        kept, filtered = result.kept, result.filtered
        print(
            f"after Qwen filter: kept={len(kept)} "
            f"filtered_out={len(filtered)}"
        )
        print("descriptions (kept):")
        for d in kept:
            print(d.summary(clip_id=result.clip_id))
        if self.settings.show_vlm_filtered and filtered:
            print("filtered out by Qwen (passing_by):")
            for d in filtered:
                print(d.summary(clip_id=result.clip_id))
