"""Constant-amplitude block waveform shape."""

from __future__ import annotations

from dataclasses import dataclass

from .shape import SeqStarShape


@dataclass(slots=True)
class SeqStarBlockShape(SeqStarShape):
    """Rectangular waveform described by duration and amplitude."""

    amplitude: float = 1.0
    kind: str = "block"
