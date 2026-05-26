"""_landmarks.py — 3D landmark placement and scheme management.

exports: LandmarkScheme, LANDMARK_SCHEMES, place_landmark, landmark_list_to_morphoj
used_by: GUI landmark placement tool, MorphoJ export
rules:   Landmarks are stored in oriented coordinate space.
         Schemes define expected landmark names for different tool types.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from lithicore._models import Landmark


@dataclass
class LandmarkScheme:
    """A named set of landmark definitions for a specific tool type."""
    name: str
    landmark_names: List[str]
    description: str = ""


# Landmark schemes based on published lithic GMM research
# (13-point flake scheme, 16-point biface/handaxe scheme)
LANDMARK_SCHEMES = {
    "flake_13": LandmarkScheme(
        name="Standard Flake (13-point)",
        description="13-point scheme for flakes: platform, margins, termination",
        landmark_names=[
            "Point of percussion",
            "Left platform extreme",
            "Right platform extreme",
            "Left proximal margin",
            "Right proximal margin",
            "Left medial margin",
            "Right medial margin",
            "Left distal margin",
            "Right distal margin",
            "Distal termination centre",
            "Maximum width left",
            "Maximum width right",
            "Platform centre",
        ],
    ),
    "biface_16": LandmarkScheme(
        name="Standard Biface/Handaxe (16-point)",
        description="16-point scheme for bifaces: base, margins, tip",
        landmark_names=[
            "Base centre",
            "Base left extreme",
            "Base right extreme",
            "Left proximal margin",
            "Right proximal margin",
            "Left lower margin",
            "Right lower margin",
            "Left mid margin",
            "Right mid margin",
            "Left upper margin",
            "Right upper margin",
            "Left distal margin",
            "Right distal margin",
            "Tip",
            "Maximum width left",
            "Maximum width right",
        ],
    ),
    "custom": LandmarkScheme(
        name="Custom",
        description="User-defined landmarks (free placement)",
        landmark_names=["Landmark"],
    ),
}


class LandmarkManager:
    """Manages landmark placement for a single artefact."""

    def __init__(self, scheme: Optional[LandmarkScheme] = None) -> None:
        self.scheme = scheme or LANDMARK_SCHEMES["custom"]
        self.landmarks: List[Landmark] = []
        self._next_index = 0

    def place_landmark(self, position: np.ndarray) -> Landmark:
        """Place a landmark at the given 3D position.

        Uses the next available name from the scheme if possible,
        otherwise falls back to a generic name.
        """
        if self._next_index < len(self.scheme.landmark_names):
            name = self.scheme.landmark_names[self._next_index]
        else:
            name = f"Landmark {self._next_index + 1}"

        lm = Landmark(
            name=name,
            x=round(float(position[0]), 3),
            y=round(float(position[1]), 3),
            z=round(float(position[2]), 3),
        )
        self.landmarks.append(lm)
        self._next_index += 1
        return lm

    def remove_last(self) -> Optional[Landmark]:
        """Remove the most recently placed landmark."""
        if self.landmarks:
            self._next_index -= 1
            return self.landmarks.pop()
        return None

    def remove_by_name(self, name: str) -> bool:
        """Remove a landmark by name."""
        for i, lm in enumerate(self.landmarks):
            if lm.name == name:
                self.landmarks.pop(i)
                # Don't decrement _next_index — scheme order would be wrong
                return True
        return False

    def clear(self) -> None:
        """Remove all landmarks."""
        self.landmarks.clear()
        self._next_index = 0

    def to_morphoj(self, label: str = "artefact") -> str:
        """Export landmarks in MorphoJ-compatible format."""
        lines = [f"# MorphoJ landmark export — {label}"]
        lines.append(f"# Scheme: {self.scheme.name}")
        lines.append(f"# Landmarks: {len(self.landmarks)}")
        lines.append("")
        lines.append("LRMM 3D")
        lines.append(label)
        lines.append("1")
        lines.append(str(len(self.landmarks)))
        for i, lm in enumerate(self.landmarks):
            lines.append(f"{i + 1} {lm.x:.3f} {lm.y:.3f} {lm.z:.3f}")
        return "\n".join(lines)

    @property
    def remaining(self) -> int:
        """Number of landmarks left to place in the current scheme."""
        return max(0, len(self.scheme.landmark_names) - self._next_index)

    @property
    def is_complete(self) -> bool:
        """Whether all scheme landmarks have been placed."""
        return self._next_index >= len(self.scheme.landmark_names)
