"""Regression tests for gammaSTAR writer behavior introduced during the PRESS sprint."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from pypulseq_star.make import make_delay
from pypulseq_star.writers import GammaStarWriter
from pypulseq_star.writers.gammastar_generic_document import (
    _events_extent_duration,
)
from pypulseq_star.writers.gammastar_literal_validator import (
    GammaStarLiteralValidator,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_demo_press():
    path = _repo_root() / "examples" / "demo_PRESS.py"
    module_name = "_ppstar_test_demo_press_writer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def press_document():
    demo = _load_demo_press()
    system = demo.define_system()
    protocol = demo.define_protocol()
    sequence = demo.build_sequence(system, protocol)
    resolved = sequence.resolve()
    ok, report = resolved.check_timing()
    assert ok, "PRESS timing failed:\n" + "\n".join(str(item) for item in report)
    document = GammaStarWriter(sequence).to_dict(defaults=resolved)
    return demo, sequence, resolved, document


def _literal_return(parameter: dict) -> object:
    assert parameter.get("inputs") == {}
    script = str(parameter.get("script", "")).strip()
    match = re.fullmatch(r"return\s+(.+)", script, flags=re.DOTALL)
    assert match, f"Not a simple literal parameter: {parameter!r}"
    return ast.literal_eval(match.group(1))


def test_delay_only_event_extent_is_not_double_counted():
    delay = make_delay(0.125)
    assert _events_extent_duration([delay]) == pytest.approx(0.125)


def test_authored_tr_seeds_gamma_protocol_default(press_document):
    _, _, _, document = press_document
    params = document["parameters"]

    # Regression for the synthetic flattened-duration TR overriding authored TR.
    assert _literal_return(params["root.prot.TR"]) == pytest.approx(2.0)

    canonical = params["root.prot.press_repetition_time_s"]
    assert "root.prot.TR" in set(canonical.get("inputs", {}).values())


def test_semantic_loop_exports_repeat_period_execution_span_and_validation(press_document):
    _, _, _, document = press_document
    params = document["parameters"]

    metabolite = "root.metabolite.averages_metabolite"
    water = "root.water.averages_water"

    for path in (metabolite, water):
        assert f"{path}.repeat_period" in params
        assert f"{path}.seqstar_content_duration" in params
        assert f"{path}.seqstar_execution_span" in params
        assert f"{path}.seqstar_timing_valid" in params

    metabolite_sources = set(params[f"{metabolite}.repeat_period"]["inputs"].values())
    water_sources = set(params[f"{water}.repeat_period"]["inputs"].values())
    allowed_tr_sources = {
        "root.prot.TR",
        "root.prot.press_repetition_time_s",
    }
    assert metabolite_sources & allowed_tr_sources
    assert water_sources & allowed_tr_sources


def test_loop_lengths_are_protocol_driven(press_document):
    _, _, _, document = press_document
    params = document["parameters"]

    metabolite_length = params["root.metabolite.averages_metabolite.length"]
    water_length = params["root.water.averages_water.length"]

    assert "root.prot.spectroscopy_metabolite_nsa" in set(
        metabolite_length.get("inputs", {}).values()
    )
    assert "root.prot.water_reference_nsa" in set(
        water_length.get("inputs", {}).values()
    )


def test_water_family_starts_after_metabolite_execution_span(press_document):
    _, _, _, document = press_document
    params = document["parameters"]

    water_tstart = params["root.water.tstart"]
    sources = set(water_tstart.get("inputs", {}).values())

    assert "root.metabolite.tstart" in sources
    assert "root.metabolite.seqstar_execution_span" in sources


def test_symbolic_press_delay_bindings_depend_on_protocol_te(press_document):
    _, _, _, document = press_document
    params = document["parameters"]

    fill_paths = [
        path
        for path in params
        if "kernel_press" in path
        and path.endswith(".duration")
        and any(token in path for token in (
            "rf1_to_rf2_fill",
            "rf2_to_rf3_fill",
            "rf3_to_echo_fill",
        ))
    ]
    assert fill_paths, "No PRESS symbolic fill durations were exported"

    all_sources = set()
    for path in fill_paths:
        all_sources.update(params[path].get("inputs", {}).values())

    assert "root.prot.press_te1_s" in all_sources
    assert "root.prot.press_te2_s" in all_sources


def test_literal_validator_reports_missing_dependency():
    params = {
        "root.prot.foo": {"inputs": {}, "script": "return 1.0"},
        "root.demo.duration": {
            "inputs": {"missing": "root.prot.does_not_exist"},
            "script": "return missing",
        },
    }

    report = GammaStarLiteralValidator(fail_on_unresolved=False).validate_and_fix(params)
    assert any(issue.category == "invalid_dependency_root" for issue in report.issues)


def test_literal_validator_repairs_adc_header_offcenter_alias():
    position_path = "root.readout.header.position"
    offcenter_path = "root.readout.header.offcenter"
    params = {
        position_path: {"inputs": {}, "script": "return 0.0"},
    }

    report = GammaStarLiteralValidator(fail_on_unresolved=False).validate_and_fix(params)

    assert offcenter_path in params
    assert params[offcenter_path]["inputs"] == {"position": position_path}
    assert offcenter_path in report.fixed_paths
