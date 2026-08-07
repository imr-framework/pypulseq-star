"""Structured feasibility diagnostics for pypulseq_star.

This module is intentionally sequence-agnostic.  It reports whether the
currently evaluated relationship graph and event objects satisfy generic
constraints such as non-negative timing, raster alignment, ADC consistency, and
system limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class FeasibilityDiagnostic:
    """One structured feasibility message."""

    code: str
    message: str
    severity: str = "error"
    path: str | None = None
    parameter: str | None = None
    current_value: Any | None = None
    limit_value: Any | None = None
    suggested_value: Any | None = None
    limiting_constraint: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "suggested_value": self.suggested_value,
            "limiting_constraint": self.limiting_constraint,
            "message": self.message,
            "details": dict(self.details or {}),
        }


@dataclass(slots=True)
class FeasibilityReport:
    """Collection of feasibility diagnostics for one evaluation."""

    diagnostics: list[FeasibilityDiagnostic] = field(default_factory=list)
    derived_values: dict[str, Any] = field(default_factory=dict)
    protocol_values: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, diagnostic: FeasibilityDiagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def extend(self, diagnostics: Iterable[FeasibilityDiagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    @property
    def errors(self) -> list[FeasibilityDiagnostic]:
        return [d for d in self.diagnostics if str(d.severity).lower() == "error"]

    @property
    def warnings(self) -> list[FeasibilityDiagnostic]:
        return [d for d in self.diagnostics if str(d.severity).lower() == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "derived_values": dict(self.derived_values),
            "protocol_values": dict(self.protocol_values),
            "context": dict(self.context),
        }

    def summary_text(self, max_items: int = 12) -> str:
        if not self.diagnostics:
            return "Feasibility: OK; no diagnostics."
        lines = [
            f"Feasibility: {self.status.upper()} "
            f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        ]
        for diagnostic in self.diagnostics[:max_items]:
            path = f" at {diagnostic.path}" if diagnostic.path else ""
            lines.append(
                f"- [{diagnostic.severity.upper()}] {diagnostic.code}{path}: "
                f"{diagnostic.message}"
            )
        remaining = len(self.diagnostics) - max_items
        if remaining > 0:
            lines.append(f"... and {remaining} more diagnostic(s).")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.ok
