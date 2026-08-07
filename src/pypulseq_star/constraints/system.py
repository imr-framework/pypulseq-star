"""System-level reusable constraints."""
from __future__ import annotations

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic

from .base import Constraint, EvaluationState


def system_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="positive_system_limits",
            description="System raster and limit fields must be positive or non-negative as appropriate.",
            evaluator=_positive_system_limits,
        )
    ]


def _positive_system_limits(state: EvaluationState):
    system = state.system
    positive = ("rf_raster_time", "grad_raster_time", "block_duration_raster", "adc_raster_time", "gamma")
    nonnegative = ("max_grad", "max_slew", "rf_dead_time", "rf_ringdown_time", "adc_dead_time", "max_rf")
    for name in positive:
        value = getattr(system, name, None)
        if value is None:
            continue
        if float(value) <= 0:
            yield FeasibilityDiagnostic(
                code="SYSTEM_FIELD_INVALID",
                path=f"root.sys.{name}",
                current_value=value,
                message=f"System field {name} must be positive.",
            )
    for name in nonnegative:
        value = getattr(system, name, None)
        if value is None:
            continue
        if float(value) < 0:
            yield FeasibilityDiagnostic(
                code="SYSTEM_FIELD_INVALID",
                path=f"root.sys.{name}",
                current_value=value,
                message=f"System field {name} must be non-negative.",
            )
