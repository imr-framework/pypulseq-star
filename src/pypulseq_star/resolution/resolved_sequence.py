"""Immutable-style numeric realization wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .resolved_protocol import ResolvedProtocol


@dataclass(slots=True)
class ResolvedSequence:
    """Numeric realization of one symbolic sequence specification."""

    source: Any
    sequence: Any
    protocol: ResolvedProtocol
    diagnostics: tuple[Any, ...] = ()

    def event(self, name: str) -> Any:
        return self.sequence.event(name)

    def block(self, name: str) -> Any:
        return self.sequence.block(name)

    def node(self, name: str) -> Any:
        record = self.sequence.node(name)
        duration = self._node_duration(name)
        if isinstance(record, dict):
            return SimpleNamespace(**record, duration=duration)
        return SimpleNamespace(name=name, duration=duration, record=record)

    def _node_duration(self, name: str) -> float:
        blocks = []
        for block in self.sequence.timeline.blocks:
            node = _metadata_value(block, "seqstar_node")
            repeat_parent = _metadata_value(block, "seqstar_repeat_parent")
            if node == name or str(node).startswith(f"{name}.") or repeat_parent == name:
                blocks.append(block)
        return self.sequence.calc_timeline_duration(blocks)

    def check_timing(self):
        return self.sequence.check_timing()

    def resolve_expression_reference(
        self,
        kind: str,
        identity: str,
        property_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.sequence.resolve_expression_reference(
            kind,
            identity,
            property_name=property_name,
            metadata=metadata,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sequence, name)


def _metadata_value(obj: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, dict) and key in parameters:
        return parameters[key]
    return getattr(obj, key, default)
