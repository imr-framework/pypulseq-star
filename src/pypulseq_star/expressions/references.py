"""Generic deferred references for later Sequence/event integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .base import Expression
from .context import EvaluationContext


@dataclass(frozen=True, slots=True, repr=False)
class ReferenceExpression(Expression):
    """Reference to a value owned by a sequence, event, block, or node.

    The expression core does not know how event anchors or block ranges are
    calculated. Sequence/resolution code will provide that behavior through
    ``EvaluationContext.resolver``.
    """

    kind: str
    identity: str
    property_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reference_key(self) -> str:
        components = [self.kind, self.identity]
        if self.property_name:
            components.append(self.property_name)
        return ":".join(components)

    def _evaluate(self, context: EvaluationContext) -> Any:
        return context.reference_value(
            kind=self.kind,
            identity=self.identity,
            property_name=self.property_name,
            metadata=self.metadata,
            reference_key=self.reference_key,
        )

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset({self.reference_key})

    def to_canonical(self) -> str:
        if self.property_name:
            return f"{self.identity}.{self.property_name}"
        return self.identity


@dataclass(frozen=True, slots=True, repr=False)
class EventAnchorRef(ReferenceExpression):
    """Symbolic reference to an event anchor."""

    def __init__(
        self,
        event_identity: str,
        anchor: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "kind", "event_anchor")
        object.__setattr__(self, "identity", event_identity)
        object.__setattr__(self, "property_name", anchor)
        object.__setattr__(self, "metadata", dict(metadata or {}))


@dataclass(frozen=True, slots=True, repr=False)
class EventPropertyRef(ReferenceExpression):
    """Symbolic reference to an event property."""

    def __init__(
        self,
        event_identity: str,
        property_name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "kind", "event_property")
        object.__setattr__(self, "identity", event_identity)
        object.__setattr__(self, "property_name", property_name)
        object.__setattr__(self, "metadata", dict(metadata or {}))


@dataclass(frozen=True, slots=True, repr=False)
class BlockRangeDurationRef(ReferenceExpression):
    """Symbolic duration of an inclusive named block range."""

    def __init__(
        self,
        start_block_name: str,
        end_block_name: str,
    ) -> None:
        identity = f"{start_block_name}->{end_block_name}"
        object.__setattr__(self, "kind", "block_range_duration")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "property_name", "duration")
        object.__setattr__(
            self,
            "metadata",
            {
                "start_block_name": start_block_name,
                "end_block_name": end_block_name,
                "inclusive": True,
            },
        )


@dataclass(frozen=True, slots=True, repr=False)
class AnchorIntervalDurationRef(ReferenceExpression):
    """Symbolic duration between two event anchors."""

    def __init__(
        self,
        start: EventAnchorRef,
        end: EventAnchorRef,
    ) -> None:
        identity = f"{start.to_canonical()}->{end.to_canonical()}"
        object.__setattr__(self, "kind", "anchor_interval_duration")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "property_name", "duration")
        object.__setattr__(
            self,
            "metadata",
            {
                "start": start,
                "end": end,
            },
        )

    @property
    def dependencies(self) -> frozenset[str]:
        start = self.metadata["start"]
        end = self.metadata["end"]
        return start.dependencies | end.dependencies
