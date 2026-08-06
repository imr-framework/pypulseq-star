"""Sequence-agnostic constraints for pypulseq_star feasibility."""
from __future__ import annotations

from .base import Constraint, EvaluationState
from .system import system_constraints
from .timing import timing_constraints
from .adc import adc_constraints
from .gradients import gradient_constraints
from .rf import rf_constraints


def default_constraints() -> list[Constraint]:
    """Return the default generic constraint set.

    These constraints are event/system/relationship-family checks, not
    sequence-specific GRE/EPI/RARE rules.
    """

    constraints: list[Constraint] = []
    constraints.extend(system_constraints())
    constraints.extend(timing_constraints())
    constraints.extend(adc_constraints())
    constraints.extend(gradient_constraints())
    constraints.extend(rf_constraints())
    return constraints


__all__ = [
    "Constraint",
    "EvaluationState",
    "default_constraints",
    "system_constraints",
    "timing_constraints",
    "adc_constraints",
    "gradient_constraints",
    "rf_constraints",
]
