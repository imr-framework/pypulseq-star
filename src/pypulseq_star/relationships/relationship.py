"""Relationship records for SeqStar objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .expression import SeqStarExpression
from .validation import SeqStarValidationResult


@dataclass(slots=True)
class SeqStarRelationship:
    """A developer-defined relationship between protocol parameters/events."""

    name: str
    relation_type: str
    description: str
    target: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    protocol_parameters: dict[str, Any] = field(default_factory=dict)
    derived_parameters: list[str] = field(default_factory=list)
    expression: SeqStarExpression | None = None
    resolved: dict[str, Any] = field(default_factory=dict)
    validation: list[SeqStarValidationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relation_type": self.relation_type,
            "description": self.description,
            "developer_defined": bool(self.metadata.get("developer_defined", True)),
            "target": dict(self.target),
            "reference": dict(self.reference),
            "protocol_parameters": dict(self.protocol_parameters),
            "derived_parameters": list(self.derived_parameters),
            "expression": self.expression.to_dict() if self.expression is not None else None,
            "resolved": dict(self.resolved),
            "validation": [item.to_dict() for item in self.validation],
            "metadata": _json_safe(dict(self.metadata)),
        }

    def update_resolution(
        self,
        *,
        resolved: Mapping[str, Any],
        validation: list[SeqStarValidationResult],
    ) -> None:
        self.resolved = dict(resolved)
        self.validation = list(validation)
        if self.expression is not None:
            # Store the most important scalar, when available, for quick debug.
            if "target_value" in resolved:
                self.expression.resolved_value = resolved["target_value"]
            elif "adc_delay" in resolved:
                self.expression.resolved_value = resolved["adc_delay"]
            elif "tr_fill_delay" in resolved:
                self.expression.resolved_value = resolved["tr_fill_delay"]


def attach_relationship(seq: Any, relationship: SeqStarRelationship) -> None:
    """Attach a relationship object to a sequence.

    This avoids requiring an immediate Sequence class refactor. If the sequence
    already exposes ``relationships`` we use it; otherwise we fall back to
    ``seq.metadata['relationships']`` or ``seq._seqstar_relationships``.
    """

    if hasattr(seq, "relationships"):
        relationships = getattr(seq, "relationships")
        if relationships is None:
            relationships = []
            try:
                setattr(seq, "relationships", relationships)
            except AttributeError:
                relationships = None

        if isinstance(relationships, list):
            relationships.append(relationship)
            return

        if hasattr(relationships, "append"):
            relationships.append(relationship)
            return

    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("relationships", []).append(relationship)
        return

    try:
        setattr(seq, "metadata", {"relationships": [relationship]})
        return
    except AttributeError:
        pass

    existing = getattr(seq, "_seqstar_relationships", None)
    if not isinstance(existing, list):
        existing = []
        try:
            setattr(seq, "_seqstar_relationships", existing)
        except AttributeError as exc:
            raise AttributeError(
                "Could not attach relationships to sequence. Add a mutable "
                "relationships list or metadata dict to the Sequence class."
            ) from exc

    existing.append(relationship)


def get_relationships(seq: Any) -> list[SeqStarRelationship]:
    """Return relationships attached to a sequence."""

    out: list[SeqStarRelationship] = []

    if hasattr(seq, "relationships"):
        relationships = getattr(seq, "relationships")
        if isinstance(relationships, list):
            out.extend(item for item in relationships if isinstance(item, SeqStarRelationship))

    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, dict):
        relationships = metadata.get("relationships", [])
        if isinstance(relationships, list):
            out.extend(item for item in relationships if isinstance(item, SeqStarRelationship))

    relationships = getattr(seq, "_seqstar_relationships", None)
    if isinstance(relationships, list):
        out.extend(item for item in relationships if isinstance(item, SeqStarRelationship))

    # Preserve order but remove duplicates by object identity.
    unique: list[SeqStarRelationship] = []
    seen: set[int] = set()
    for item in out:
        item_id = id(item)
        if item_id not in seen:
            unique.append(item)
            seen.add(item_id)

    return unique


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation of debug metadata."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    # Event/object fallback: keep useful identity without serializing the object.
    name = getattr(value, "name", None)
    return {
        "object_type": value.__class__.__name__,
        "name": str(name) if name is not None else None,
        "object_id": id(value),
    }
