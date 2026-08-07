"""Unit conversion helpers used by event constructors and validators."""

from __future__ import annotations


def seconds(value: float) -> float:
    """Return seconds from a scalar already expressed in seconds."""

    return float(value)


def milliseconds(value: float) -> float:
    """Convert milliseconds to seconds."""

    return float(value) * 1e-3


def microseconds(value: float) -> float:
    """Convert microseconds to seconds."""

    return float(value) * 1e-6
