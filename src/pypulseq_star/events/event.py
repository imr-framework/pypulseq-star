"""Base event class."""

from __future__ import annotations

from dataclasses import dataclass

from pypulseq_star.core import SeqStarNode


@dataclass(slots=True)
class SeqStarEvent(SeqStarNode):
    """A physical sequence event with a role and optional waveform shape."""

    event_type: str = "event"
