"""Base node for the SeqStar hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .relationship import SeqStarRelationship
from .timing import SeqStarTiming


@dataclass(slots=True)
class SeqStarNode:
    """Common tree node with timing, metadata, and semantic relationships."""

    name: str
    role: str | None = None
    path: str | None = None
    timing: SeqStarTiming = field(default_factory=SeqStarTiming)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: list[SeqStarRelationship] = field(default_factory=list)
    children: list["SeqStarNode"] = field(default_factory=list)
    enabled: bool = True

    def add_child(self, child: "SeqStarNode", relationship: str = "contains") -> "SeqStarNode":
        """Attach a child and record the semantic edge."""

        self.children.append(child)
        self.relationships.append(
            SeqStarRelationship(source=self.name, target=child.name, kind=relationship)
        )
        return child
