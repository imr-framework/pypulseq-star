"""Logical read/phase/slice to physical x/y/z encoding transforms.

The orientation model is deliberately small and explicit:

    rotation[row][column]

where rows are physical scanner axes ``x, y, z`` and columns are logical image
axes ``read, phase, slice``.  In other words, each column is a physical
direction cosine vector for one logical imaging axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


LOGICAL_AXES = ("read", "phase", "slice")
PHYSICAL_AXES = ("x", "y", "z")
ORIENTATION_PRESETS = ("axial", "coronal", "sagittal")

_ORIENTATION_ALIASES = {
    "ax": "axial",
    "axi": "axial",
    "axial": "axial",
    "transverse": "axial",
    "tra": "axial",
    "cor": "coronal",
    "coronal": "coronal",
    "sag": "sagittal",
    "sagittal": "sagittal",
}


def normalize_orientation(value: Any) -> str:
    """Normalize an orientation preset name.

    Accepted aliases include ``"tra"``/``"transverse"`` for axial,
    ``"cor"`` for coronal, and ``"sag"`` for sagittal.
    """

    key = str(value or "axial").strip().lower()
    key = _ORIENTATION_ALIASES.get(key, key)
    if key not in ORIENTATION_PRESETS:
        raise ValueError(
            f"Unknown orientation {value!r}. "
            f"Expected one of {', '.join(ORIENTATION_PRESETS)} or a 3x3 matrix."
        )
    return key


def orientation_choices() -> tuple[str, ...]:
    """Return the supported named orientation presets."""

    return ORIENTATION_PRESETS


def normalize_logical_axis(value: Any, *, channel: str | None = None) -> str:
    """Normalize a logical read/phase/slice axis.

    If no explicit logical axis is provided, the PyPulseq-style physical channel
    is used as a compatibility fallback: x->read, y->phase, z->slice.
    """

    if value is None:
        if channel is None:
            raise ValueError("A logical axis or channel must be supplied.")
        value = {"x": "read", "y": "phase", "z": "slice"}.get(str(channel).lower())

    key = str(value or "").strip().lower()
    aliases = {
        "ro": "read",
        "readout": "read",
        "kx": "read",
        "pe": "phase",
        "phase_encode": "phase",
        "ky": "phase",
        "ss": "slice",
        "slice_select": "slice",
        "kz": "slice",
    }
    key = aliases.get(key, key)
    if key not in LOGICAL_AXES:
        raise ValueError(
            f"Logical axis must be one of {LOGICAL_AXES}; got {value!r}."
        )
    return key


@dataclass(frozen=True, slots=True)
class EncodingFrame:
    """Right-handed logical-to-physical encoding frame.

    Columns of ``rotation`` are physical x/y/z direction cosines of the logical
    read, phase, and slice axes, respectively. ``position`` is the physical
    isocenter-relative image origin in metres.
    """

    rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "axial"

    def __post_init__(self) -> None:
        r = np.asarray(self.rotation, dtype=float)
        p = np.asarray(self.position, dtype=float)

        if r.shape != (3, 3):
            raise ValueError("Encoding rotation must be 3 x 3.")
        if p.shape != (3,):
            raise ValueError("Encoding position must have three values.")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(p)):
            raise ValueError("Encoding frame values must be finite.")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
            raise ValueError("Encoding rotation must be orthonormal.")
        if not np.isclose(np.linalg.det(r), 1.0, atol=1e-6):
            raise ValueError(
                "Encoding rotation must be right-handed with determinant +1."
            )

        object.__setattr__(
            self,
            "rotation",
            tuple(tuple(float(v) for v in row) for row in r),
        )
        object.__setattr__(
            self,
            "position",
            tuple(float(v) for v in p),
        )
        object.__setattr__(self, "name", str(self.name or "encoding"))

    @classmethod
    def identity(cls) -> "EncodingFrame":
        return cls()

    @classmethod
    def axial(cls, *, position=(0.0, 0.0, 0.0)) -> "EncodingFrame":
        # read=x, phase=y, slice=z
        return cls(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            position=tuple(position),
            name="axial",
        )

    @classmethod
    def coronal(cls, *, position=(0.0, 0.0, 0.0)) -> "EncodingFrame":
        # read=x, phase=z, slice=-y; right-handed
        return cls(
            rotation=((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            position=tuple(position),
            name="coronal",
        )

    @classmethod
    def sagittal(cls, *, position=(0.0, 0.0, 0.0)) -> "EncodingFrame":
        # read=y, phase=z, slice=x; right-handed
        return cls(
            rotation=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            position=tuple(position),
            name="sagittal",
        )

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        position=None,
        name=None,
    ) -> "EncodingFrame":
        if isinstance(value, cls):
            if position is None and name is None:
                return value
            return cls(
                value.rotation,
                tuple(position or value.position),
                name or value.name,
            )

        if isinstance(value, Mapping):
            return cls(
                tuple(tuple(float(v) for v in row) for row in value.get("rotation")),
                tuple(position or value.get("position", (0.0, 0.0, 0.0))),
                name or value.get("name", "encoding"),
            )

        if isinstance(value, str):
            key = normalize_orientation(value)
            factory = {
                "axial": cls.axial,
                "coronal": cls.coronal,
                "sagittal": cls.sagittal,
            }[key]
            return factory(position=position or (0.0, 0.0, 0.0))

        # Otherwise treat as a 3x3 oblique matrix.
        return cls(
            tuple(tuple(float(v) for v in row) for row in value),
            tuple(position or (0.0, 0.0, 0.0)),
            name or "oblique",
        )

    def direction(self, logical_axis: str) -> tuple[float, float, float]:
        """Return physical x/y/z direction cosine vector for a logical axis."""

        axis = normalize_logical_axis(logical_axis)
        column = LOGICAL_AXES.index(axis)
        return tuple(self.rotation[row][column] for row in range(3))

    @property
    def read_dir(self) -> tuple[float, float, float]:
        return self.direction("read")

    @property
    def phase_dir(self) -> tuple[float, float, float]:
        return self.direction("phase")

    @property
    def slice_dir(self) -> tuple[float, float, float]:
        return self.direction("slice")

    @property
    def transform4x4(self) -> tuple[tuple[float, float, float, float], ...]:
        rows = [
            tuple(self.rotation[i]) + (self.position[i],)
            for i in range(3)
        ]
        return tuple(rows + [(0.0, 0.0, 0.0, 1.0)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rotation": [list(row) for row in self.rotation],
            "position": list(self.position),
            "transform4x4": [list(row) for row in self.transform4x4],
            "read_dir": list(self.read_dir),
            "phase_dir": list(self.phase_dir),
            "slice_dir": list(self.slice_dir),
            "convention": "columns_are_read_phase_slice_in_physical_xyz",
        }


__all__ = [
    "EncodingFrame",
    "LOGICAL_AXES",
    "PHYSICAL_AXES",
    "ORIENTATION_PRESETS",
    "normalize_orientation",
    "normalize_logical_axis",
    "orientation_choices",
]
