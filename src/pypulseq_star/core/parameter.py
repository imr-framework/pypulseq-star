"""Typed parameters attached to SeqStar nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SeqStarParameter:
    """A named value with optional units and source provenance."""

    name: str
    value: Any
    unit: str | None = None
    source: str | None = None
