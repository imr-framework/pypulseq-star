"""Spatial encoding geometry."""
from .encoding import (
    EncodingFrame,
    LOGICAL_AXES,
    PHYSICAL_AXES,
    ORIENTATION_PRESETS,
    normalize_orientation,
    normalize_logical_axis,
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
