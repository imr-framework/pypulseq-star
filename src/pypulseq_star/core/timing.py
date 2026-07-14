"""Timing primitives shared by nodes, events, blocks, and sequences."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SeqStarTiming:
    """Absolute or relative timing in seconds."""

    tstart: float | None = None
    duration: float | None = None

    @property
    def tend(self) -> float | None:
        """Return end time when start and duration are both known."""

        if self.tstart is None or self.duration is None:
            return None
        return self.tstart + self.duration
