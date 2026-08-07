"""Generic ADC constraints."""
from __future__ import annotations

from typing import Iterable

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic

from .base import (
    Constraint,
    EvaluationState,
    event_path,
    get_float,
    get_int,
    is_adc,
    iter_block_events,
    iter_blocks,
)


def adc_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="adc_positive_sampling",
            description="ADC sampling count, dwell, and duration must be positive and self-consistent.",
            evaluator=_adc_positive_sampling,
        ),
        Constraint(
            name="adc_dead_time",
            description="ADC delay must respect system ADC dead time when available.",
            evaluator=_adc_dead_time,
        ),
    ]


def _adc_positive_sampling(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_adc(event):
                continue
            path = event_path(block_index, name, event)
            samples = get_int(event, "num_samples", "number_of_samples", default=None)
            dwell = get_float(event, "dwell", "sample_time", default=None)
            duration = get_float(event, "duration", default=None)
            if samples is None or samples <= 0:
                yield FeasibilityDiagnostic(
                    code="ADC_SAMPLES_INVALID",
                    path=f"{path}.num_samples",
                    current_value=samples,
                    message="ADC number of samples must be positive.",
                )
            if dwell is None or dwell <= 0:
                yield FeasibilityDiagnostic(
                    code="ADC_DWELL_INVALID",
                    path=f"{path}.dwell",
                    current_value=dwell,
                    message="ADC dwell/sample time must be positive.",
                )
            if duration is not None and duration < 0:
                yield FeasibilityDiagnostic(
                    code="ADC_DURATION_INVALID",
                    path=f"{path}.duration",
                    current_value=duration,
                    message="ADC duration must be non-negative.",
                )
            if samples is not None and dwell is not None and duration is not None:
                expected = samples * dwell
                if abs(duration - expected) > max(1e-12, 1e-6 * max(abs(expected), 1.0)):
                    yield FeasibilityDiagnostic(
                        code="ADC_DURATION_INCONSISTENT",
                        severity="warning",
                        path=f"{path}.duration",
                        current_value=duration,
                        limit_value=expected,
                        suggested_value=expected,
                        message="ADC duration is inconsistent with num_samples × dwell.",
                    )


def _adc_dead_time(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    dead_time = float(getattr(state.system, "adc_dead_time", 0.0) or 0.0)
    if dead_time <= 0:
        return
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_adc(event):
                continue
            delay = get_float(event, "delay", "tstart", default=0.0) or 0.0
            if delay + 1e-12 < dead_time:
                yield FeasibilityDiagnostic(
                    code="ADC_DEAD_TIME_VIOLATION",
                    path=f"{event_path(block_index, name, event)}.delay",
                    current_value=delay,
                    limit_value=dead_time,
                    suggested_value=dead_time,
                    message="ADC delay is shorter than system.adc_dead_time.",
                )
