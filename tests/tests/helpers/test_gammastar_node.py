"""Regression tests for the generic node.vary -> gammaSTAR export contract."""

from __future__ import annotations

from pypulseq_star.writers.gammastar_literal_validator import (
    GammaStarLiteralValidator,
)


def _literal(value):
    return {"inputs": {}, "script": f"return {value!r}"}


def _expr(inputs, script):
    return {"inputs": dict(inputs), "script": script}


def test_phase_offset_variation_drives_backend_phase_field() -> None:
    """A semantic phase_offset variation must drive gammaSTAR's .phase field."""

    params = {
        "root.tstart": _literal(0.0),
        "root.ky.counter": _literal(0),
        "root.prot.rf_spoiling_increment": _literal(117.0),
        "root.ky.kernel.excitation.rf.rf.phase": _literal(0.0),
        "root.ky.kernel.excitation.rf.rf.phase_offset": _expr(
            {
                "counter": "root.ky.counter",
                "step": "root.prot.rf_spoiling_increment",
            },
            "return counter * step",
        ),
    }

    report = GammaStarLiteralValidator(
        fail_on_unresolved=False
    ).validate_and_fix(params)

    assert params["root.ky.kernel.excitation.rf.rf.phase"] == params[
        "root.ky.kernel.excitation.rf.rf.phase_offset"
    ]
    assert any(
        issue.fixed and issue.category == "variation_backend_alias"
        for issue in report.issues
    )


def test_variation_export_warning_count_is_reported() -> None:
    """Writer-recorded variation export warnings must fail validation."""

    params = {
        "root.tstart": _literal(0.0),
        "root.info.seqstar_variation_export_warning_count": _literal(1),
        "root.info.seqstar_variation_export_warnings": _literal(
            [{"kind": "variation_target_missing"}]
        ),
    }

    report = GammaStarLiteralValidator(
        fail_on_unresolved=False
    ).validate_and_fix(params)

    assert any(
        (not issue.fixed)
        and issue.category == "variation_export_contract_failed"
        for issue in report.issues
    )
