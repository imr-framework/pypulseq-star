"""Sequence-level regression tests for examples/demo_PRESS.py.

These tests intentionally exercise the public demo construction path:
Protocol -> Sequence -> resolve() -> check_timing().
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    # tests/sequence/test_demo_press.py -> repository root
    return Path(__file__).resolve().parents[2]


def _load_demo_press():
    path = _repo_root() / "examples" / "demo_PRESS.py"
    if not path.exists():
        pytest.fail(f"Expected PRESS demo at {path}")

    module_name = "_ppstar_test_demo_press"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_demo_press()


def _build_and_resolve(demo, overrides: dict[str, object] | None = None):
    system = demo.define_system()
    protocol = demo.define_protocol(overrides=overrides)
    sequence = demo.build_sequence(system, protocol)
    resolved = sequence.resolve()
    ok, report = resolved.check_timing()
    assert ok, "PRESS timing failed:\n" + "\n".join(str(item) for item in report)
    return sequence, resolved


def test_default_press_protocol_resolves(demo):
    sequence, resolved = _build_and_resolve(demo)

    assert resolved.protocol.press_te1_s == pytest.approx(32e-3)
    assert resolved.protocol.press_te2_s == pytest.approx(65e-3)
    assert (
        resolved.protocol.press_te1_s + resolved.protocol.press_te2_s
    ) == pytest.approx(97e-3)
    assert resolved.protocol.press_repetition_time_s == pytest.approx(2.0)
    assert resolved.protocol.spectroscopy_metabolite_nsa == 16
    assert resolved.protocol.water_reference_nsa == 2

    # The semantic hierarchy is part of the sequence contract.
    assert sequence.get_node("root.metabolite.averages_metabolite") is not None
    assert sequence.get_node("root.water.averages_water") is not None


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        ({"press_te1_s": 40e-3}, "press_te1_s", 40e-3),
        ({"press_te2_s": 80e-3}, "press_te2_s", 80e-3),
        ({"press_repetition_time_s": 4.0}, "press_repetition_time_s", 4.0),
        ({"spectroscopy_metabolite_nsa": 3}, "spectroscopy_metabolite_nsa", 3),
        ({"water_reference_nsa": 1}, "water_reference_nsa", 1),
        ({"spectroscopy_num_samples": 1024}, "spectroscopy_num_samples", 1024),
        ({"spectroscopy_dwell_s": 250e-6}, "spectroscopy_dwell_s", 250e-6),
        ({"press_voxel_size_read_m": 25e-3}, "press_voxel_size_read_m", 25e-3),
        ({"press_voxel_size_phase_m": 24e-3}, "press_voxel_size_phase_m", 24e-3),
        ({"press_voxel_size_slice_m": 22e-3}, "press_voxel_size_slice_m", 22e-3),
        ({"press_voxel_position_read_m": 5e-3}, "press_voxel_position_read_m", 5e-3),
        ({"press_voxel_position_phase_m": -4e-3}, "press_voxel_position_phase_m", -4e-3),
        ({"press_voxel_position_slice_m": 3e-3}, "press_voxel_position_slice_m", 3e-3),
        ({"press_transmit_frequency_offset_hz": 123.0}, "press_transmit_frequency_offset_hz", 123.0),
    ],
)
def test_protocol_overrides_survive_resolution(demo, overrides, field, expected):
    _, resolved = _build_and_resolve(demo, overrides)
    value = getattr(resolved.protocol, field)

    if isinstance(expected, float):
        assert value == pytest.approx(expected)
    else:
        assert value == expected


def test_te_relationship_remains_te1_plus_te2_after_override(demo):
    _, resolved = _build_and_resolve(
        demo,
        {
            "press_te1_s": 40e-3,
            "press_te2_s": 80e-3,
            "press_repetition_time_s": 4.0,
        },
    )

    assert resolved.protocol.press_te1_s == pytest.approx(40e-3)
    assert resolved.protocol.press_te2_s == pytest.approx(80e-3)
    assert (
        resolved.protocol.press_te1_s + resolved.protocol.press_te2_s
    ) == pytest.approx(120e-3)


def test_metabolite_and_water_loops_share_prescribed_tr_but_have_independent_counts(demo):
    sequence, resolved = _build_and_resolve(
        demo,
        {
            "press_repetition_time_s": 4.0,
            "spectroscopy_metabolite_nsa": 3,
            "water_reference_nsa": 2,
        },
    )

    metabolite = sequence.get_node("root.metabolite.averages_metabolite")
    water = sequence.get_node("root.water.averages_water")

    assert metabolite is not None
    assert water is not None
    assert metabolite["repeat_mode"] == "loop"
    assert water["repeat_mode"] == "loop"

    # The symbolic values may remain ParameterRefs here; resolution is verified
    # through the resolved protocol and gammaSTAR tests.
    assert resolved.protocol.press_repetition_time_s == pytest.approx(4.0)
    assert resolved.protocol.spectroscopy_metabolite_nsa == 3
    assert resolved.protocol.water_reference_nsa == 2


def test_water_suppression_and_water_reference_can_be_disabled(demo):
    sequence, resolved = _build_and_resolve(
        demo,
        {
            "spectroscopy_water_suppression_enabled": False,
            "water_reference_enabled": False,
            "press_repetition_time_s": 4.0,
        },
    )

    assert resolved.protocol.spectroscopy_water_suppression_enabled is False
    assert resolved.protocol.water_reference_enabled is False
    assert sequence.get_node("root.water") is None
