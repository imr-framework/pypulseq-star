"""Structured diagnostics for expression evaluation and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExpressionDiagnostic:
    """Backend-neutral description of an expression failure."""

    code: str
    message: str
    property_path: str | None = None
    expression: str | None = None
    dependencies: tuple[str, ...] = ()
    input_values: dict[str, Any] = field(default_factory=dict)
    evaluated_value: Any = None
    constraint: str | None = None
    stage: str | None = None
    unit: str | None = None
    cause_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic dictionary."""

        return {
            "code": self.code,
            "message": self.message,
            "property_path": self.property_path,
            "expression": self.expression,
            "dependencies": list(self.dependencies),
            "input_values": dict(self.input_values),
            "evaluated_value": self.evaluated_value,
            "constraint": self.constraint,
            "stage": self.stage,
            "unit": self.unit,
            "cause_type": self.cause_type,
        }

    def format_multiline(self) -> str:
        """Return a readable diagnostic for Python and command-line output."""

        lines = [self.message]
        fields = (
            ("Code", self.code),
            ("Property", self.property_path),
            ("Expression", self.expression),
            ("Dependencies", ", ".join(self.dependencies) if self.dependencies else None),
            ("Evaluated value", self.evaluated_value),
            ("Constraint", self.constraint),
            ("Stage", self.stage),
            ("Unit", self.unit),
            ("Cause", self.cause_type),
        )
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")
        if self.input_values:
            lines.append(f"Inputs: {self.input_values}")
        return "\n".join(lines)
