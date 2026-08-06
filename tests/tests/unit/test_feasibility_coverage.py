from __future__ import annotations

from dataclasses import dataclass

import pytest

from pypulseq_star.feasibility.bounds import (
    FeasibleInterval,
    ParameterBoundReport,
    _candidate_grid,
    _values_to_intervals,
)
from pypulseq_star.feasibility.diagnostics import FeasibilityDiagnostic, FeasibilityReport
from pypulseq_star.feasibility.evaluator import assert_feasible, evaluate_feasibility


@pytest.mark.unit
def test_feasibility_diagnostic_and_report_views() -> None:
    warning = FeasibilityDiagnostic(code="WARN", message="warning", severity="warning", path="adc")
    error = FeasibilityDiagnostic(code="ERR", message="error", path="rf", limit_value=1)
    report = FeasibilityReport(protocol_values={"TR": 1.0})
    report.add(warning)
    report.extend([error])

    assert not report.ok
    assert not bool(report)
    assert report.status == "error"
    assert report.errors == [error]
    assert report.warnings == [warning]
    assert report.to_dict()["error_count"] == 1
    assert "ERR at rf" in report.summary_text()
    assert FeasibilityReport().summary_text() == "Feasibility: OK; no diagnostics."


@pytest.mark.unit
def test_candidate_grid_and_interval_helpers_cover_integer_and_float_paths() -> None:
    assert _candidate_grid(current=3, declared_min=1, declared_max=5, steps=10, integer=True) == [1, 2, 3, 4, 5]
    sampled = _candidate_grid(current=50, declared_min=0, declared_max=100, steps=5, integer=True)
    assert sampled[0] == 0 and sampled[-1] == 100 and 50 in sampled

    floats = _candidate_grid(current=2.0, declared_min=1.0, declared_max=3.0, steps=3, integer=False)
    assert floats == pytest.approx([1.0, 2.0, 3.0])
    assert _candidate_grid(current="bad", declared_min=None, declared_max=None, steps=1, integer=False) == [1.0]

    assert _values_to_intervals([]) == []
    assert _values_to_intervals([1, 2, 3, 7, 8]) == [
        FeasibleInterval(1, 3, 1),
        FeasibleInterval(7, 8, 1),
    ]
    assert _values_to_intervals([0.1, 0.5, 0.9]) == [FeasibleInterval(0.1, 0.9, None)]


@pytest.mark.unit
def test_parameter_bound_report_serialization_and_summary() -> None:
    report = ParameterBoundReport(
        parameter="TR",
        current_value=1.0,
        allowed_intervals=[FeasibleInterval(0.5, 2.0, 0.1)],
        sampled_values=10,
        feasible_values=8,
        status="ok",
    )
    assert report.ok_current
    assert report.to_dict()["allowed_intervals"][0]["step"] == pytest.approx(0.1)
    assert "TR: current=1.0" in report.summary_text()


@pytest.mark.unit
def test_generic_feasibility_evaluator_success_failure_and_constraint_exception() -> None:
    @dataclass
    class Constraint:
        name: str = "test"
        def evaluate(self, state):
            state.derived_values["checked"] = True
            return []

    class Sequence:
        name = "dummy"
        system = None
        protocol = type("Protocol", (), {"parameters": {"TR": 1.0}})()

    report = evaluate_feasibility(Sequence(), constraints=[Constraint()], resolve=False)
    assert report.ok
    assert report.derived_values["checked"] is True
    assert assert_feasible(Sequence(), constraints=[Constraint()], resolve=False).ok

    class BrokenConstraint:
        name = "broken"
        def evaluate(self, state):
            raise RuntimeError("boom")

    broken = evaluate_feasibility(Sequence(), constraints=[BrokenConstraint()], resolve=False)
    assert broken.errors[0].code == "CONSTRAINT_EVALUATION_FAILED"
    with pytest.raises(ValueError, match="CONSTRAINT_EVALUATION_FAILED"):
        assert_feasible(Sequence(), constraints=[BrokenConstraint()], resolve=False)


@pytest.mark.unit
def test_feasibility_resolution_failure_is_structured() -> None:
    class Sequence:
        name = "dummy"
        protocol = type("Protocol", (), {"parameters": {"TR": 1.0}})()
        def resolve(self, **overrides):
            raise RuntimeError("cannot resolve")

    report = evaluate_feasibility(Sequence(), protocol_overrides={"TR": -1})
    assert not report.ok
    assert report.errors[0].code == "PROTOCOL_RESOLUTION_FAILED"
    assert report.context["resolve_failed"] is True
