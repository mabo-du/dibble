"""_annotations.py — 3D mesh annotation data model.

exports: Annotation
         AnnotationSet
used_by: lithicope annotation panel
rules:   Pure dataclasses with JSON serialization. No GUI imports.
         Coordinates are (x, y, z) floats matching mesh vertex space.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Annotation:
    """A single annotation attached to a 3D mesh point.

    Attributes:
        point: (x, y, z) coordinates on the mesh surface.
        title: Short label for the annotation.
        description: Multi-line descriptive notes.
        category: Type classification — e.g. "scar", "ridge",
            "notch", "cortex", "flake", "breakage", "other".
        measurement_mm: Optional numeric measurement at this point.
        confidence: Estimated reliability (0 = uncertain, 1 = certain).
        author: Name or identifier of the annotator.
        timestamp: ISO 8601 datetime string of creation/last edit.
        attached_photos: List of file paths to associated images.
        sub_annotations: Child annotations nested under this one.
    """
    point: tuple[float, float, float]
    title: str
    description: str = ""
    category: str = ""
    measurement_mm: float = 0.0
    confidence: float = 1.0
    author: str = ""
    timestamp: str = ""
    attached_photos: list[str] = field(default_factory=list)
    sub_annotations: list["Annotation"] = field(default_factory=list)


@dataclass
class AnnotationSet:
    """A collection of annotations for a single artefact mesh.

    Attributes:
        format_version: Schema version for forward compatibility.
        artefact_label: Human-readable artefact identifier.
        mesh_path: Relative or absolute path to the associated mesh file.
        mesh_checksum: SHA-256 hex digest of the mesh file for validation.
        author: Name of the person who created this set.
        created: ISO 8601 datetime of initial creation.
        annotations: All top-level annotations for this artefact.
    """
    format_version: int = 1
    artefact_label: str = ""
    mesh_path: str = ""
    mesh_checksum: str = ""
    author: str = ""
    created: str = ""
    annotations: list[Annotation] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize this annotation set to a JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> AnnotationSet:
        """Deserialize a JSON string into an AnnotationSet."""
        raw = json.loads(data)
        # Reconstruct nested Annotation objects
        raw["annotations"] = [
            cls._annotation_from_dict(a) for a in raw.get("annotations", [])
        ]
        return cls(**raw)

    @staticmethod
    def _annotation_from_dict(raw: dict) -> Annotation:
        """Recursively build an Annotation from a dict, handling sub-annotations."""
        subs = raw.pop("sub_annotations", [])
        ann = Annotation(**raw)
        ann.sub_annotations = [
            AnnotationSet._annotation_from_dict(s) for s in subs
        ]
        return ann

    @staticmethod
    def _point_key(point: tuple[float, float, float]) -> tuple[float, float, float]:
        """Round a 3D point to 3 decimal places for stable merge matching."""
        return (round(point[0], 3), round(point[1], 3), round(point[2], 3))

    def merge(self, other: AnnotationSet) -> tuple[AnnotationSet, list[str]]:
        """Merge another AnnotationSet into this one.

        Annotations at the same 3D position (rounded to 3 dp) are merged.
        Unique positions are appended. Conflicts (same position, different
        data) keep both entries with an author suffix and a warning.

        Args:
            other: The incoming annotation set to merge.

        Returns:
            A tuple of (merged AnnotationSet, list of warning strings).
        """
        warnings: list[str] = []
        merged = AnnotationSet(
            format_version=self.format_version,
            artefact_label=self.artefact_label or other.artefact_label,
            mesh_path=self.mesh_path or other.mesh_path,
            mesh_checksum=self.mesh_checksum or other.mesh_checksum,
            author=f"{self.author}+{other.author}" if self.author and other.author
                   else self.author or other.author,
            created=self.created or other.created,
        )

        # Index existing annotations by position key
        existing: dict[tuple[float, float, float], Annotation] = {}
        for ann in self.annotations:
            existing[self._point_key(ann.point)] = ann

        # Add all current annotations
        merged.annotations = list(self.annotations)

        for ann in other.annotations:
            key = self._point_key(ann.point)
            if key in existing:
                existing_ann = existing[key]
                # Conflict detection: same point, different title/desc
                if (existing_ann.title != ann.title or
                        existing_ann.description != ann.description):
                    suffix = f" ({ann.author})" if ann.author else " (imported)"
                    merged_ann = Annotation(
                        point=ann.point,
                        title=ann.title + suffix,
                        description=ann.description,
                        category=ann.category or existing_ann.category,
                        measurement_mm=ann.measurement_mm or existing_ann.measurement_mm,
                        confidence=ann.confidence or existing_ann.confidence,
                        author=ann.author or existing_ann.author,
                        timestamp=max(ann.timestamp, existing_ann.timestamp)
                        if ann.timestamp and existing_ann.timestamp
                        else ann.timestamp or existing_ann.timestamp,
                        attached_photos=list(set(existing_ann.attached_photos + ann.attached_photos)),
                    )
                    # Replace in-place
                    for i, e in enumerate(merged.annotations):
                        if self._point_key(e.point) == key:
                            merged.annotations[i] = merged_ann
                            break
                    warnings.append(
                        f"Merged annotation at ({ann.point[0]:.3f}, {ann.point[1]:.3f}, "
                        f"{ann.point[2]:.3f}): conflicting data resolved with author suffix"
                    )
                else:
                    # Same content — prefer newer timestamp
                    if ann.timestamp and (not existing_ann.timestamp or
                                          ann.timestamp > existing_ann.timestamp):
                        for i, e in enumerate(merged.annotations):
                            if self._point_key(e.point) == key:
                                e.timestamp = ann.timestamp
                                e.author = ann.author or e.author
                                break
            else:
                merged.annotations.append(ann)

        return merged, warnings

    def compute_checksum(self, mesh_path: Path) -> str:
        """Compute SHA-256 hex digest of a mesh file."""
        sha = hashlib.sha256()
        with open(mesh_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return f"sha256:{sha.hexdigest()}"
