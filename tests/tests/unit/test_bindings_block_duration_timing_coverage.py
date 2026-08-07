"""Focused branch coverage for small core timing/container utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.bindings import (
    bind_event_property_to_loop_table,
    bind_gradient_area_to_loop_table,
)
from pypulseq_star.blocks.block import SeqStarBlock
from pypulseq_star.calc_duration import calc_duration
from pypulseq_star.check_timing import (
    _div_check,
    _event_raster,
    _get_event_active_duration,
    _get_event_delay,
    _get_event_dwell,
    _get_required_post_dead_time,
    _get_required_pre_dead_time,
    _is_adc,
    _is_delay,
    _is_gradient,
    _is_rf,
    check_timing,
    format_string,
    indent_string,
)
from pypulseq_star.events.delay import SeqStarDelayEvent

pytestmark = pytest.mark.unit


class _NoMetadata:
    __slots__ = ("parameters", "_seqstar_loop_binding")

    def __init__(self) -> None:
        self.parameters = {}

    @property
    def metadata(self):
        return None

    @metadata.setter
    def metadata(self, value):
        raise AttributeError("metadata is read-only")


def test_bindings_validate_names_and_store_all_supported_locations() -> None:
    event = SimpleNamespace(metadata={}, parameters={})
    returned = bind_event_property_to_loop_table(
        event,
        property_name=" phase_offset ",
        protocol_table=" rf_phase_table ",
        counter=" echo_index ",
        scale=2,
        indices={"shot": "shot_index"},
    )
    assert returned is event
    binding = event.metadata["seqstar_loop_binding"]
    assert binding == {
        "property": "phase_offset",
        "protocol_table": "rf_phase_table",
        "counter": "echo_index",
        "scale": 2.0,
        "indices": {"shot": "shot_index"},
    }
    assert event.metadata["seqstar_loop_bindings"]["phase_offset"] is binding
    assert event.parameters["_seqstar_loop_bindings"]["phase_offset"] is binding

    gradient = SimpleNamespace(metadata=None, parameters={})
    bind_gradient_area_to_loop_table(
        gradient,
        protocol_table="ky_table",
        counter="ky_index",
    )
    assert gradient.metadata["seqstar_loop_binding"]["property"] == "area"

    fallback = _NoMetadata()
    bind_event_property_to_loop_table(
        fallback,
        property_name="frequency_offset",
        protocol_table="frequency_table",
        counter="average_index",
    )
    assert fallback._seqstar_loop_binding["protocol_table"] == "frequency_table"

    for keyword in ("property_name", "protocol_table", "counter"):
        kwargs = {
            "property_name": "area",
            "protocol_table": "table",
            "counter": "index",
        }
        kwargs[keyword] = "   "
        with pytest.raises(ValueError, match=keyword):
            bind_event_property_to_loop_table(SimpleNamespace(), **kwargs)


def test_block_lookup_supports_occurrence_and_provenance_names() -> None:
    block = SeqStarBlock(name="readout")
    concrete = SeqStarDelayEvent(1e-3, name="delay_occurrence")
    concrete.metadata["source_event_name"] = "delay_template"
    block.add_event(concrete)

    assert block.get_event("delay_occurrence") is concrete
    assert block.get_event("delay_template") is concrete
    assert block.children == [concrete]
    assert block.relationships[-1].kind == "contains_event"

    with pytest.raises(KeyError, match="Available events"):
        block.get_event("missing")

    duplicate = SeqStarDelayEvent(2e-3, name="second")
    duplicate.parameters["source_event_name"] = "delay_template"
    block.add_event(duplicate)
    with pytest.raises(ValueError, match="2 events"):
        block.get_event("delay_template")


def test_calc_duration_covers_event_families_and_container_forms() -> None:
    assert calc_duration() == 0.0
    assert calc_duration(None) == 0.0
    assert calc_duration(0.125) == pytest.approx(0.125)

    delay = SimpleNamespace(type="delay", duration=2e-3, delay=2e-3)
    rf = SimpleNamespace(
        type="rf",
        delay=1e-3,
        duration=3e-3,
        ringdown_time=0.5e-3,
        flip_angle=1.0,
    )
    adc = SimpleNamespace(
        type="adc",
        delay=2e-3,
        num_samples=100,
        dwell=10e-6,
        dead_time=0.25e-3,
    )
    trap = SimpleNamespace(
        type="trap",
        delay=0.5e-3,
        rise_time=0.2e-3,
        flat_time=0.6e-3,
        fall_time=0.2e-3,
    )
    arbitrary_grad = SimpleNamespace(
        event_type="gradient",
        delay=0.1e-3,
        shape=SimpleNamespace(duration=1.4e-3),
    )
    metadata_only = SimpleNamespace(
        type="custom",
        metadata={"timing": {"delay": 1e-3, "duration": 2e-3, "post_time": 0.5e-3}},
    )
    occupied = SimpleNamespace(type="rf", occupied_duration=7e-3, flip_angle=1.0)

    assert calc_duration(delay) == pytest.approx(2e-3)
    assert calc_duration(rf) == pytest.approx(4.5e-3)
    assert calc_duration(adc) == pytest.approx(3.25e-3)
    assert calc_duration(trap) == pytest.approx(1.5e-3)
    assert calc_duration(arbitrary_grad) == pytest.approx(1.5e-3)
    # Arbitrary metadata is intentionally not treated as an executable timing contract.
    assert calc_duration(metadata_only) == 0.0
    assert calc_duration(occupied) == pytest.approx(7e-3)

    block = SimpleNamespace(events=[delay, rf], duration=5e-3)
    assert calc_duration(block) == pytest.approx(5e-3)
    assert calc_duration(delay, rf, adc) == pytest.approx(4.5e-3)

    children_container = SimpleNamespace(children=[delay, adc], role="kernel")
    assert calc_duration(children_container) == pytest.approx(3.25e-3)


def _system(**overrides):
    values = dict(
        block_duration_raster=10e-6,
        grad_raster_time=10e-6,
        rf_raster_time=2e-6,
        adc_raster_time=1e-7,
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=50e-6,
        max_rf=1e-5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_check_timing_public_path_reports_raster_dead_time_and_mismatch() -> None:
    rf = SimpleNamespace(
        name="rf",
        type="rf",
        flip_angle=1.0,
        delay=50e-6,
        duration=101e-6,
        ringdown_time=20e-6,
        signal=[2e-5],
    )
    adc = SimpleNamespace(
        name="adc",
        type="adc",
        delay=10e-6,
        num_samples=16,
        dwell=1.05e-6,
        dead_time=50e-6,
    )
    block = SimpleNamespace(events=[rf, adc], duration=200e-6)
    seq = SimpleNamespace(system=_system(), blocks=[block], block_durations=[250e-6])

    ok, report = check_timing(seq)
    assert not ok
    error_types = {item.error_type for item in report}
    assert "BLOCK_DURATION_MISMATCH" in error_types
    assert "RASTER" in error_types
    assert "RF_DEAD_TIME" in error_types or "DELAY_DEAD_TIME" in error_types
    assert "ADC_DEAD_TIME" in error_types or "DELAY_DEAD_TIME" in error_types


def test_check_timing_helper_classification_and_formatting() -> None:
    system = _system()
    rf = SimpleNamespace(type="rf", flip_angle=1.0, delay=1e-3, duration=2e-3)
    adc = SimpleNamespace(type="adc", delay=1e-3, num_samples=10, dwell=2e-6)
    delay = SimpleNamespace(type="delay", duration=1e-3)
    grad = SimpleNamespace(type="trap", rise_time=1e-4, flat_time=2e-4, fall_time=1e-4)

    assert _is_rf(rf) and not _is_adc(rf)
    assert _is_adc(adc) and not _is_rf(adc)
    assert _is_delay(delay)
    assert _is_gradient(grad)
    assert _event_raster(rf, system) == (system.rf_raster_time, "rf_raster_time")
    assert _event_raster(adc, system) == (system.adc_raster_time, "adc_raster_time")
    assert _event_raster(grad, system) == (system.grad_raster_time, "grad_raster_time")
    assert _get_event_delay(adc) == pytest.approx(1e-3)
    assert _get_event_active_duration(adc) == pytest.approx(20e-6)
    assert _get_event_dwell(adc) == pytest.approx(2e-6)
    assert _get_required_pre_dead_time(rf, system) == pytest.approx(system.rf_dead_time)
    assert _get_required_post_dead_time(rf, system) == pytest.approx(system.rf_ringdown_time)
    assert _get_required_pre_dead_time(adc, system) == pytest.approx(system.adc_dead_time)

    errors = []
    _div_check(
        value=11e-6,
        raster=10e-6,
        block=1,
        event="gx",
        field="duration",
        raster_name="grad_raster_time",
        error_report=errors,
    )
    assert errors and errors[0].error_type == "RASTER"
    assert "11.00" in format_string("{value*multiplier:.2f}", value=11e-6, multiplier=1e6)
    assert indent_string("a\nb", 2) == "  a\n  b"
