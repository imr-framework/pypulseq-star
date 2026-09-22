"""Spatial encoding and voxel geometry.

This package contains sequence-independent spatial geometry abstractions.

``EncodingFrame``
    Defines the relationship between logical MRI axes
    (read / phase / slice) and physical scanner axes.

``Voxel``
    Defines a rectangular 3D region by its size and signed center position
    relative to scanner isocenter in the logical encoding frame.
"""

from .encoding import (
    LOGICAL_AXES,
    ORIENTATION_PRESETS,
    PHYSICAL_AXES,
    EncodingFrame,
    normalize_logical_axis,
    normalize_orientation,
    orientation_choices,
)

from .voxel import (
    DEFAULT_MAX_ABS_POSITION_M,
    LogicalAxis,
    Voxel,
)


__all__ = [
    # -------------------------------------------------------------------------
    # Encoding frame
    # -------------------------------------------------------------------------
    "EncodingFrame",
    "LOGICAL_AXES",
    "PHYSICAL_AXES",
    "ORIENTATION_PRESETS",
    "normalize_orientation",
    "normalize_logical_axis",
    "orientation_choices",

    # -------------------------------------------------------------------------
    # Voxel geometry
    # -------------------------------------------------------------------------
    "Voxel",
    "LogicalAxis",
    "DEFAULT_MAX_ABS_POSITION_M",
]