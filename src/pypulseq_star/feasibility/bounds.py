"""Selected-parameter Siemens-style feasible-range estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from pypulseq_star.constraints.base import protocol_values as collect_protocol_values
from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic
from pypulseq_star.feasibility.evaluator import evaluate_feasibility


@dataclass(frozen=True, slots=True)
class FeasibleInterval:
    start: float | int
    stop: float | int
    step: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "step": self.step}


@dataclass(slots=True)
class ParameterBoundReport:
    parameter: str
    current_value: Any | None
    declared_min: Any | None = None
    declared_max: Any | None = None
    allowed_intervals: list[FeasibleInterval] = field(default_factory=list)
    sampled_values: int = 0
    feasible_values: int = 0
    diagnostics_at_current: list[FeasibilityDiagnostic] = field(default_factory=list)
    limiting_diagnostics: list[FeasibilityDiagnostic] = field(default_factory=list)
    status: str = "unknown"

    @property
    def ok_current(self) -> bool:
        return not [d for d in self.diagnostics_at_current if d.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "current_value": self.current_value,
            "declared_min": self.declared_min,
            "declared_max": self.declared_max,
            "allowed_intervals": [interval.to_dict() for interval in self.allowed_intervals],
            "sampled_values": self.sampled_values,
            "feasible_values": self.feasible_values,
            "status": self.status,
            "ok_current": self.ok_current,
            "diagnostics_at_current": [d.to_dict() for d in self.diagnostics_at_current],
            "limiting_diagnostics": [d.to_dict() for d in self.limiting_diagnostics[:10]],
        }

    def summary_text(self) -> str:
        if self.allowed_intervals:
            intervals = ", ".join(
                f"[{i.start}, {i.stop}]" if i.step is None else f"[{i.start}, {i.stop}] step {i.step}"
                for i in self.allowed_intervals
            )
        else:
            intervals = "<none found>"
        return (
            f"{self.parameter}: current={self.current_value}, status={self.status}, "
            f"allowed={intervals}, sampled={self.sampled_values}, feasible={self.feasible_values}"
        )


def feasible_range(
    sequence: Any,
    parameter: str,
    *,
    values: Iterable[Any] | None = None,
    declared_min: float | int | None = None,
    declared_max: float | int | None = None,
    steps: int = 64,
    integer: bool | None = None,
    resolve: bool = True,
) -> ParameterBoundReport:
    """Estimate allowed values for one protocol parameter.

    All other protocol values are held fixed.  This intentionally uses
    evaluation/search rather than sequence-specific analytical rules.
    """

    base_values = collect_protocol_values(sequence)
    current = base_values.get(parameter)
    if current is None:
        # Try simple alias matching without being clever about sequence names.
        for key in base_values:
            if str(key).lower() == str(parameter).lower():
                parameter = str(key)
                current = base_values.get(key)
                break

    if values is None:
        candidate_values = _candidate_grid(
            current=current,
            declared_min=declared_min,
            declared_max=declared_max,
            steps=steps,
            integer=integer,
        )
    else:
        candidate_values = list(values)

    current_report = evaluate_feasibility(sequence, resolve=resolve)
    feasible: list[Any] = []
    limiting: list[FeasibilityDiagnostic] = []

    for value in candidate_values:
        report = evaluate_feasibility(
            sequence,
            protocol_overrides={parameter: value},
            resolve=resolve,
        )
        if report.ok:
            feasible.append(value)
        elif report.errors and len(limiting) < 25:
            limiting.extend(report.errors[:2])

    intervals = _values_to_intervals(feasible)
    status = "ok" if feasible else "no_feasible_values_found"
    if current_report.errors:
        status = "current_invalid" if feasible else "current_invalid_no_feasible_values_found"

    return ParameterBoundReport(
        parameter=parameter,
        current_value=current,
        declared_min=declared_min,
        declared_max=declared_max,
        allowed_intervals=intervals,
        sampled_values=len(candidate_values),
        feasible_values=len(feasible),
        diagnostics_at_current=current_report.diagnostics,
        limiting_diagnostics=limiting,
        status=status,
    )


def _candidate_grid(
    *,
    current: Any,
    declared_min: float | int | None,
    declared_max: float | int | None,
    steps: int,
    integer: bool | None,
) -> list[Any]:
    try:
        current_value = float(current)
    except Exception:
        current_value = 1.0

    if integer is None:
        integer = isinstance(current, int) or (isinstance(current, float) and current.is_integer())

    if declared_min is None:
        declared_min = 1 if integer else max(0.0, 0.25 * current_value)
    if declared_max is None:
        declared_max = max(float(declared_min) + 1.0, 4.0 * max(current_value, 1.0))

    if integer:
        lo = int(math.floor(float(declared_min)))
        hi = int(math.ceil(float(declared_max)))
        if hi < lo:
            lo, hi = hi, lo
        span = hi - lo + 1
        if span <= max(1, steps):
            return list(range(lo, hi + 1))
        # Uniform integer subsampling plus endpoints/current.
        step = max(1, int(math.ceil(span / max(1, steps))))
        values = set(range(lo, hi + 1, step))
        values.add(lo); values.add(hi)
        try:
            values.add(int(round(float(current))))
        except Exception:
            pass
        return sorted(values)

    lo = float(declared_min)
    hi = float(declared_max)
    if hi < lo:
        lo, hi = hi, lo
    if steps <= 1:
        return [current_value]
    values = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    if lo <= current_value <= hi:
        values.append(current_value)
    return sorted(set(values))


def _values_to_intervals(values: list[Any]) -> list[FeasibleInterval]:
    if not values:
        return []
    # Integer contiguous intervals.
    if all(isinstance(v, int) for v in values):
        values = sorted(values)
        intervals: list[FeasibleInterval] = []
        start = prev = values[0]
        for value in values[1:]:
            if value == prev + 1:
                prev = value
                continue
            intervals.append(FeasibleInterval(start, prev, 1))
            start = prev = value
        intervals.append(FeasibleInterval(start, prev, 1))
        return intervals

    nums = sorted(float(v) for v in values)
    return [FeasibleInterval(nums[0], nums[-1], None)]
