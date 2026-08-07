"""Base waveform shape."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SeqStarShape:
    """A waveform geometry independent of timeline placement."""

    name: str
    duration: float
    samples: list[float] = field(default_factory=list)
    kind: str = "arbitrary"
