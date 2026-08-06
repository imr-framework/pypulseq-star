"""Generic gradient constraints."""
from __future__ import annotations

from typing import Iterable

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic

from .base import (
    Constraint,
    EvaluationState,
    event_path,
    get_float,
    is_gradient,
    iter_block_events,
    iter_blocks,
)


def gradient_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="gradient_system_limits",
            description="Gradient amplitude and slew must stay within system limits.",
            evaluator=_gradient_system_limits,
        ),
        Constraint(
            name="gradient_area_duration_feasible",
            description="Gradient area/duration pairs must be physically feasible under system limits.",
            evaluator=_gradient_area_duration_feasible,
        ),
    ]


def _gradient_system_limits(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    max_grad = float(getattr(state.system, "max_grad", 0.0) or 0.0)
    max_slew = float(getattr(state.system, "max_slew", 0.0) or 0.0)
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_gradient(event):
                continue
            path = event_path(block_index, name, event)
            amplitude = get_float(event, "amplitude", default=None)
            if amplitude is None:
                shape = getattr(event, "shape", None)
                if shape is not None:
                    amplitude = get_float(shape, "amplitude", default=None)
            if amplitude is not None and max_grad > 0 and abs(amplitude) - max_grad > 1e-9:
                yield FeasibilityDiagnostic(
                    code="GRADIENT_AMPLITUDE_LIMIT",
                    path=f"{path}.amplitude",
                    current_value=amplitude,
                    limit_value=max_grad,
                    suggested_value=max_grad if amplitude >= 0 else -max_grad,
                    message="Gradient amplitude exceeds system.max_grad.",
                )

            # Infer slew for trapezoids from amplitude/rise/fall when present.
            for ramp_name in ("rise_time", "fall_time"):
                ramp = get_float(event, ramp_name, default=None)
                if ramp is None:
                    shape = getattr(event, "shape", None)
                    if shape is not None:
                        ramp = get_float(shape, ramp_name, default=None)
                if amplitude is None or ramp is None or ramp <= 0 or max_slew <= 0:
                    continue
                slew = abs(amplitude) / ramp
                if slew - max_slew > max(1e-9, 1e-9 * max_slew):
                    yield FeasibilityDiagnostic(
                        code="GRADIENT_SLEW_LIMIT",
                        path=f"{path}.{ramp_name}",
                        current_value=slew,
                        limit_value=max_slew,
                        message="Gradient slew exceeds system.max_slew.",
                        details={"amplitude": amplitude, "ramp_time": ramp},
                    )


def _gradient_area_duration_feasible(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    max_grad = float(getattr(state.system, "max_grad", 0.0) or 0.0)
    max_slew = float(getattr(state.system, "max_slew", 0.0) or 0.0)
    if max_grad <= 0 or max_slew <= 0:
        return
    for block_index, block in iter_blocks(state.sequence):
        for _event_index, name, event in iter_block_events(block):
            if not is_gradient(event):
                continue
            path = event_path(block_index, name, event)
            area = get_float(event, "area", default=None)
            if area is None:
                shape = getattr(event, "shape", None)
                if shape is not None:
                    area = get_float(shape, "area", default=None)
            duration = get_float(event, "duration", default=None)
            if duration is None:
                shape = getattr(event, "shape", None)
                if shape is not None:
                    duration = get_float(shape, "duration", default=None)
            if area is None or duration is None or duration <= 0:
                continue
            area = abs(float(area))
            # Conservative maximum area in a fixed duration with symmetric ramps:
            # if duration is long enough to reach max_grad, use trapezoid area;
            # otherwise use triangular area limited by max_slew.
            min_ramp_to_max = max_grad / max_slew
            if duration >= 2 * min_ramp_to_max:
                max_area = max_grad * (duration - min_ramp_to_max)
            else:
                max_area = max_slew * (duration / 2.0) ** 2
            if area - max_area > max(1e-9, 1e-6 * max_area):
                yield FeasibilityDiagnostic(
                    code="GRADIENT_AREA_DURATION_INFEASIBLE",
                    path=f"{path}.area",
                    current_value=area,
                    limit_value=max_area,
                    message="Requested gradient area cannot fit in the available duration under system limits.",
                    details={"duration": duration, "max_grad": max_grad, "max_slew": max_slew},
                )
