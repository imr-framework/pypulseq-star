"""Generic RF constraints."""
from __future__ import annotations

from typing import Iterable

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic

from .base import (
    Constraint,
    EvaluationState,
    event_path,
    get_float,
    is_rf,
    iter_block_events,
    iter_blocks,
)


def rf_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="rf_positive_duration",
            description="RF events must have positive duration.",
            evaluator=_rf_positive_duration,
        ),
        Constraint(
            name="rf_system_limit",
            description="RF amplitude must stay within system.max_rf when inferable.",
            evaluator=_rf_system_limit,
        ),
    ]


def _rf_positive_duration(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_rf(event):
                continue
            duration = get_float(event, "duration", "active_duration", default=None)
            if duration is None:
                shape = getattr(event, "shape", None)
                if shape is not None:
                    duration = get_float(shape, "duration", "active_duration", default=None)
            if duration is None or duration <= 0:
                yield FeasibilityDiagnostic(
                    code="RF_DURATION_INVALID",
                    path=f"{event_path(block_index, name, event)}.duration",
                    current_value=duration,
                    message="RF duration must be positive.",
                )


def _rf_system_limit(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    max_rf = float(getattr(state.system, "max_rf", 0.0) or 0.0)
    if max_rf <= 0:
        return
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_rf(event):
                continue
            amp = get_float(event, "amplitude_t", "b1_amplitude", default=None)
            if amp is None:
                shape = getattr(event, "shape", None)
                if shape is not None:
                    amp = get_float(shape, "amplitude_t", "b1_amplitude", default=None)
            if amp is not None and abs(amp) - max_rf > 1e-12:
                yield FeasibilityDiagnostic(
                    code="RF_AMPLITUDE_LIMIT",
                    path=f"{event_path(block_index, name, event)}.amplitude_t",
                    current_value=amp,
                    limit_value=max_rf,
                    message="RF amplitude exceeds system.max_rf.",
                )
