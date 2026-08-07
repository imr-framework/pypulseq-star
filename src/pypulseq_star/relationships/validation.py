"""Relationship validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class SeqStarValidationResult:
    """One validation result associated with a relationship."""

    name: str
    passed: bool
    expression: str | None = None
    value: Any | None = None
    minimum: Any | None = None
    maximum: Any | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "expression": self.expression,
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


def validation_result(
    name: str,
    passed: bool,
    *,
    expression: str | None = None,
    value: Any | None = None,
    minimum: Any | None = None,
    maximum: Any | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeqStarValidationResult:
    """Create a validation result."""

    return SeqStarValidationResult(
        name=name,
        passed=passed,
        expression=expression,
        value=value,
        minimum=minimum,
        maximum=maximum,
        message=message,
        metadata=dict(metadata or {}),
    )
