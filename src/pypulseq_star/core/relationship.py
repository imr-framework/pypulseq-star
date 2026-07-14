"""Relationships between SeqStar nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SeqStarRelationship:
    """A semantic edge such as contains, follows, balances, or samples."""

    source: str
    target: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
