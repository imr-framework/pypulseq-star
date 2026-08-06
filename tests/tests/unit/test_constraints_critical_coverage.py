from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.constraints import default_constraints
from pypulseq_star.constraints.adc import _adc_dead_time, _adc_positive_sampling
from pypulseq_star.constraints.base import (
    Constraint,
    EvaluationState,
    active_duration,
    approx_raster_aligned,
    block_path,
    event_path,
    finite,
    get_float,
    get_int,
    is_adc,
    is_delay,
    is_gradient,
    is_rf,
    iter_block_events,
    iter_blocks,
    occupied_duration,
    protocol_values,
)
from pypulseq_star.constraints.gradients import (
    _gradient_area_duration_feasible,
    _gradient_system_limits,
)
from pypulseq_star.constraints.rf import _rf_positive_duration, _rf_system_limit
from pypulseq_star.constraints.system import _positive_system_limits
from pypulseq_star.constraints.timing import (
    _block_extent_contains_events,
    _finite_nonnegative_timing,
    _timing_raster_alignment,
)


pytestmark = pytest.mark.unit


class Block:
    def __init__(self, events=None, duration=None, metadata=None, children=None):
        self.events = events
        self.duration = duration
        self.metadata = metadata or {}
        self.children = children


def state(events, *, system=None, duration=1e-3):
    block = Block(events=events, duration=duration)
    sequence = SimpleNamespace(
        timeline=SimpleNamespace(blocks=[block]),
        protocol=SimpleNamespace(parameters={"TR": 1.0}),
        parameters={"TE": 0.01, "_private": 1},
    )
    system = system or SimpleNamespace(
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        block_duration_raster=10e-6,
        adc_raster_time=100e-9,
        gamma=42.575e6,
        max_grad=40.0,
        max_slew=150.0,
        rf_dead_time=0.0,
        rf_ringdown_time=0.0,
        adc_dead_time=0.0,
        max_rf=20e-6,
    )
    return EvaluationState(sequence, system, protocol_values(sequence), {})


def test_constraint_base_helpers_cover_containers_and_paths():
    e1 = SimpleNamespace(name="rf", type="rf", duration=1e-3, delay=1e-4)
    e2 = SimpleNamespace(name="adc", type="adc", num_samples=10, dwell=1e-6, delay=2e-4)
    mapping_block = Block(events={"excitation": e1, "readout": e2}, metadata={"seqstar_node": "root.kernel"})
    sequence = SimpleNamespace(timeline=SimpleNamespace(blocks={"a": mapping_block}))

    assert list(iter_blocks(sequence))[0][1] is mapping_block
    assert [name for _, name, _ in iter_block_events(mapping_block)] == ["excitation", "readout"]
    assert block_path(0, mapping_block) == "root.kernel"

    e1.metadata = {"seqstar_event_path": "root.kernel.rf"}
    assert event_path(0, "rf", e1) == "root.kernel.rf"

    assert is_rf(e1) and not is_adc(e1)
    assert is_adc(e2) and not is_gradient(e2)
    grad = SimpleNamespace(type="trap", rise_time=1e-4, flat_time=2e-4, fall_time=1e-4, delay=5e-5)
    delay = SimpleNamespace(type="delay", duration=2e-3)
    assert is_gradient(grad)
    assert is_delay(delay)
    assert active_duration(e2) == pytest.approx(10e-6)
    assert active_duration(grad) == pytest.approx(4e-4)
    assert active_duration(delay) == pytest.approx(2e-3)
    assert occupied_duration(grad) == pytest.approx(4.5e-4)

    obj = SimpleNamespace(value="3.4", parameters={"other": "7"})
    assert get_float(obj, "value") == pytest.approx(3.4)
    assert get_int(obj, "other") == 7
    assert finite("1.0")
    assert not finite("bad")
    assert approx_raster_aligned(30e-6, 10e-6)
    assert approx_raster_aligned(1.0, 0.0)


def test_iter_blocks_and_events_fallback_paths():
    event = SimpleNamespace(name="child", type="delay", duration=1.0)
    block = SimpleNamespace(children=[event])
    assert list(iter_block_events(block))[0][1] == "child"

    sequence = SimpleNamespace(blocks=[block])
    assert list(iter_blocks(sequence))[0][1] is block

    class PulseqLike:
        block_events = [1, 2]
        def get_block(self, index):
            return f"block-{index}"

    assert list(iter_blocks(PulseqLike())) == [(1, "block-1"), (2, "block-2")]


def test_system_constraints_report_positive_and_nonnegative_fields():
    bad = SimpleNamespace(
        rf_raster_time=0,
        grad_raster_time=-1,
        block_duration_raster=0,
        adc_raster_time=1e-7,
        gamma=0,
        max_grad=-1,
        max_slew=-2,
        rf_dead_time=-3,
        rf_ringdown_time=-4,
        adc_dead_time=-5,
        max_rf=-6,
    )
    diagnostics = list(_positive_system_limits(EvaluationState(None, bad, {}, {})))
    assert len(diagnostics) == 10
    assert {d.code for d in diagnostics} == {"SYSTEM_FIELD_INVALID"}


def test_adc_constraints_cover_invalid_sampling_warning_and_dead_time():
    adc = SimpleNamespace(
        name="adc",
        type="adc",
        num_samples=0,
        dwell=-1e-6,
        duration=5e-3,
        delay=0.0,
    )
    s = state([adc], system=SimpleNamespace(adc_dead_time=50e-6))
    codes = {d.code for d in _adc_positive_sampling(s)}
    assert {"ADC_SAMPLES_INVALID", "ADC_DWELL_INVALID"} <= codes

    adc.num_samples = 10
    adc.dwell = 1e-6
    adc.duration = 20e-6
    diagnostics = list(_adc_positive_sampling(s))
    assert diagnostics[0].code == "ADC_DURATION_INCONSISTENT"
    assert diagnostics[0].severity == "warning"

    dead = list(_adc_dead_time(s))
    assert dead[0].code == "ADC_DEAD_TIME_VIOLATION"
    adc.delay = 50e-6
    assert list(_adc_dead_time(s)) == []


def test_rf_constraints_cover_shape_fallback_and_amplitude_limit():
    shape = SimpleNamespace(duration=0.0, amplitude_t=30e-6)
    rf = SimpleNamespace(name="rf", type="rf", shape=shape)
    s = state([rf], system=SimpleNamespace(max_rf=20e-6))
    assert list(_rf_positive_duration(s))[0].code == "RF_DURATION_INVALID"
    assert list(_rf_system_limit(s))[0].code == "RF_AMPLITUDE_LIMIT"

    shape.duration = 1e-3
    shape.amplitude_t = 10e-6
    assert list(_rf_positive_duration(s)) == []
    assert list(_rf_system_limit(s)) == []


def test_gradient_constraints_cover_amplitude_slew_and_area_duration():
    grad = SimpleNamespace(
        name="gx",
        type="trap",
        amplitude=50.0,
        rise_time=0.1,
        fall_time=0.1,
        area=1.0,
        duration=1e-3,
    )
    s = state([grad], system=SimpleNamespace(max_grad=40.0, max_slew=150.0))
    codes = {d.code for d in _gradient_system_limits(s)}
    assert "GRADIENT_AMPLITUDE_LIMIT" in codes
    assert "GRADIENT_SLEW_LIMIT" in codes
    assert list(_gradient_area_duration_feasible(s))[0].code == "GRADIENT_AREA_DURATION_INFEASIBLE"

    grad.amplitude = 1.0
    grad.slew_rate = 1.0
    grad.area = 1e-6
    assert list(_gradient_system_limits(s)) == []
    assert list(_gradient_area_duration_feasible(s)) == []


def test_timing_constraints_cover_invalid_extent_and_rasters():
    adc = SimpleNamespace(
        name="adc",
        type="adc",
        num_samples=10,
        dwell=1.05e-6,
        duration=10.5e-6,
        delay=5.55e-6,
    )
    block = Block(events=[adc], duration=5e-6)
    sequence = SimpleNamespace(timeline=SimpleNamespace(blocks=[block]))
    system = SimpleNamespace(
        block_duration_raster=10e-6,
        grad_raster_time=10e-6,
        rf_raster_time=1e-6,
        adc_raster_time=100e-9,
    )
    s = EvaluationState(sequence, system, {}, {})

    assert list(_block_extent_contains_events(s))[0].code == "BLOCK_TOO_SHORT"
    raster_codes = {d.code for d in _timing_raster_alignment(s)}
    assert "BLOCK_RASTER_MISMATCH" in raster_codes
    assert "EVENT_RASTER_MISMATCH" in raster_codes
    assert "ADC_DWELL_RASTER_MISMATCH" in raster_codes

    adc.delay = float("nan")
    assert list(_finite_nonnegative_timing(s))[0].code == "TIMING_FIELD_INVALID"


def test_default_constraints_are_generic_and_executable():
    names = {constraint.name for constraint in default_constraints()}
    assert {
        "positive_system_limits",
        "adc_positive_sampling",
        "adc_dead_time",
        "rf_positive_duration",
        "rf_system_limit",
        "gradient_system_limits",
        "gradient_area_duration_feasible",
        "finite_nonnegative_timing",
        "block_extent_contains_events",
        "timing_raster_alignment",
    } <= names

    constraint = Constraint("x", "desc", lambda state: [])
    assert constraint.evaluate(EvaluationState(None, None, {}, {})) == []
