"""Sequence-agnostic feasibility evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Iterable
from typing import Any

from pypulseq_star.constraints import Constraint, EvaluationState, default_constraints
from pypulseq_star.constraints.base import protocol_values as collect_protocol_values
from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic, FeasibilityReport


def evaluate_feasibility(
    sequence: Any,
    *,
    protocol_overrides: Mapping[str, Any] | None = None,
    constraints: Iterable[Constraint] | None = None,
    resolve: bool = True,
) -> FeasibilityReport:
    """Evaluate the current or proposed protocol against generic constraints.

    Parameters
    ----------
    sequence
        A SeqStar sequence or resolved sequence-like object.
    protocol_overrides
        Optional candidate protocol values.  When ``resolve`` is true and the
        sequence exposes ``resolve()``, the candidate is evaluated on a numeric
        realization without mutating the symbolic source.
    constraints
        Optional explicit constraint list.  Defaults to generic timing, raster,
        ADC, gradient, RF, and system constraints.
    resolve
        If true, try ``sequence.resolve(**protocol_overrides)`` before checking.
    """

    overrides = dict(protocol_overrides or {})
    evaluated = sequence
    diagnostics: list[FeasibilityDiagnostic] = []

    if resolve and overrides and hasattr(sequence, "resolve"):
        try:
            realization = sequence.resolve(**overrides)
            evaluated = getattr(realization, "sequence", realization)
        except Exception as exc:
            report = FeasibilityReport(
                protocol_values={**collect_protocol_values(sequence), **overrides},
                context={"stage": "resolve", "resolve_failed": True},
            )
            report.add(
                FeasibilityDiagnostic(
                    code="PROTOCOL_RESOLUTION_FAILED",
                    severity="error",
                    message="Protocol overrides could not be resolved into a numeric sequence.",
                    details={"exception_type": type(exc).__name__, "exception": str(exc)},
                )
            )
            return report

    values = collect_protocol_values(evaluated)
    values.update(overrides)
    system = getattr(evaluated, "system", getattr(sequence, "system", None))
    state = EvaluationState(
        sequence=evaluated,
        system=system,
        protocol_values=values,
        derived_values={},
    )

    for constraint in list(constraints or default_constraints()):
        try:
            diagnostics.extend(constraint.evaluate(state))
        except Exception as exc:
            diagnostics.append(
                FeasibilityDiagnostic(
                    code="CONSTRAINT_EVALUATION_FAILED",
                    severity="error",
                    limiting_constraint=constraint.name,
                    message=f"Constraint {constraint.name!r} failed during evaluation.",
                    details={"exception_type": type(exc).__name__, "exception": str(exc)},
                )
            )

    report = FeasibilityReport(
        diagnostics=diagnostics,
        derived_values=dict(state.derived_values),
        protocol_values=values,
        context={
            "sequence_name": str(getattr(evaluated, "name", getattr(sequence, "name", "sequence"))),
            "constraint_count": len(list(constraints or default_constraints())),
            "used_overrides": bool(overrides),
        },
    )
    return report


def assert_feasible(sequence: Any, **kwargs: Any) -> FeasibilityReport:
    """Evaluate feasibility and raise ValueError if any error diagnostics exist."""

    report = evaluate_feasibility(sequence, **kwargs)
    if not report.ok:
        raise ValueError(report.summary_text())
    return report
