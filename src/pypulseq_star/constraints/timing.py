"""Generic timing constraints."""
from __future__ import annotations

from typing import Iterable

from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic
from .base import (
    Constraint,
    EvaluationState,
    active_duration,
    approx_raster_aligned,
    block_path,
    event_path,
    finite,
    get_float,
    iter_block_events,
    iter_blocks,
    occupied_duration,
)


def timing_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="finite_nonnegative_timing",
            description="Timing fields must be finite and non-negative.",
            evaluator=_finite_nonnegative_timing,
        ),
        Constraint(
            name="block_extent_contains_events",
            description="A stored block duration must contain all event extents.",
            evaluator=_block_extent_contains_events,
        ),
        Constraint(
            name="timing_raster_alignment",
            description="Event and block timing should align to the appropriate system raster.",
            evaluator=_timing_raster_alignment,
            severity="warning",
        ),
    ]


def _finite_nonnegative_timing(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    for block_index, block in iter_blocks(state.sequence):
        bpath = block_path(block_index, block)
        bdur = get_float(block, "duration", default=None)
        if bdur is not None and (not finite(bdur) or bdur < 0):
            yield FeasibilityDiagnostic(
                code="BLOCK_DURATION_INVALID",
                path=bpath,
                current_value=bdur,
                message="Block duration must be finite and non-negative.",
            )
        for _event_index, name, event in iter_block_events(block):
            path = event_path(block_index, name, event)
            for field in ("delay", "tstart", "duration", "active_duration", "rise_time", "flat_time", "fall_time", "dwell"):
                value = get_float(event, field, default=None)
                if value is None:
                    continue
                if not finite(value) or value < 0:
                    yield FeasibilityDiagnostic(
                        code="TIMING_FIELD_INVALID",
                        path=f"{path}.{field}",
                        current_value=value,
                        message=f"{field} must be finite and non-negative.",
                    )


def _block_extent_contains_events(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    for block_index, block in iter_blocks(state.sequence):
        stored = get_float(block, "duration", default=None)
        if stored is None:
            continue
        event_extent = 0.0
        for _event_index, _name, event in iter_block_events(block):
            event_extent = max(event_extent, occupied_duration(event))
        if stored + 1e-12 < event_extent:
            yield FeasibilityDiagnostic(
                code="BLOCK_TOO_SHORT",
                path=block_path(block_index, block),
                current_value=stored,
                limit_value=event_extent,
                suggested_value=event_extent,
                message="Stored block duration is shorter than the contained event extent.",
                details={"event_extent": event_extent},
            )


def _timing_raster_alignment(state: EvaluationState) -> Iterable[FeasibilityDiagnostic]:
    system = state.system
    block_raster = float(getattr(system, "block_duration_raster", getattr(system, "grad_raster_time", 10e-6)))
    grad_raster = float(getattr(system, "grad_raster_time", 10e-6))
    rf_raster = float(getattr(system, "rf_raster_time", 1e-6))
    adc_raster = float(getattr(system, "adc_raster_time", 100e-9))

    for block_index, block in iter_blocks(state.sequence):
        bdur = get_float(block, "duration", default=None)
        if bdur is not None and bdur >= 0 and not approx_raster_aligned(bdur, block_raster):
            yield FeasibilityDiagnostic(
                code="BLOCK_RASTER_MISMATCH",
                severity="warning",
                path=block_path(block_index, block),
                current_value=bdur,
                limit_value=block_raster,
                message="Block duration is not aligned to block_duration_raster.",
            )
        for _event_index, name, event in iter_block_events(block):
            path = event_path(block_index, name, event)
            cls = event.__class__.__name__.lower()
            typ = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
            if "adc" in cls or typ.startswith("adc"):
                raster = adc_raster
                raster_name = "adc_raster_time"
            elif "rf" in cls or typ == "rf":
                raster = rf_raster
                raster_name = "rf_raster_time"
            else:
                raster = grad_raster
                raster_name = "grad_raster_time"
            for field in ("delay", "duration", "rise_time", "flat_time", "fall_time"):
                value = get_float(event, field, default=None)
                if value is None or value < 0:
                    continue
                if not approx_raster_aligned(value, raster):
                    yield FeasibilityDiagnostic(
                        code="EVENT_RASTER_MISMATCH",
                        severity="warning",
                        path=f"{path}.{field}",
                        current_value=value,
                        limit_value=raster,
                        message=f"{field} is not aligned to {raster_name}.",
                    )
            dwell = get_float(event, "dwell", "sample_time", default=None)
            if dwell is not None and dwell >= 0 and not approx_raster_aligned(dwell, adc_raster):
                yield FeasibilityDiagnostic(
                    code="ADC_DWELL_RASTER_MISMATCH",
                    severity="warning",
                    path=f"{path}.dwell",
                    current_value=dwell,
                    limit_value=adc_raster,
                    message="ADC dwell/sample time is not aligned to adc_raster_time.",
                )
