"""Spatial encoding geometry."""
from .encoding import (
    LOGICAL_AXES,
    ORIENTATION_PRESETS,
    PHYSICAL_AXES,
    EncodingFrame,
    normalize_logical_axis,
    normalize_orientation,
    orientation_choices,
)

__all__ = [
    "EncodingFrame",
    "LOGICAL_AXES",
    "PHYSICAL_AXES",
    "ORIENTATION_PRESETS",
    "normalize_orientation",
    "normalize_logical_axis",
    "orientation_choices",
]
