from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.relationships.context import (
    event_anchor_time,
    event_center_pos,
    event_delay,
    event_duration,
    event_name,
    protocol_float,
    protocol_int,
    protocol_value,
    set_event_delay,
    set_event_property,
    store_derived_parameter,
    timing_value,
    try_adc_duration,
)

pytestmark = pytest.mark.unit


def test_protocol_value_all_resolution_layers_and_aliases():
    class Preferred:
        def get_protocol_parameter(self, key, default=None):
            return {"echo_time": 0.01}.get(key, default)

    assert protocol_value(Preferred(), "TE") == pytest.approx(0.01)
    assert protocol_value(Preferred(), "root.prot.echo_time") == pytest.approx(0.01)
    assert protocol_value(Preferred(), 7) == 7

    assert protocol_value(SimpleNamespace(parameters={"repetition_time": 2.0}), "TR") == 2.0
    assert protocol_value(SimpleNamespace(parameters={"Flip_Angle": 90}), "FA") == 90

    class WithContext:
        def context_dict(self):
            return {"protocol": {"echoTime": 0.02}}

    assert protocol_value(WithContext(), "TE") == 0.02

    protocol = SimpleNamespace(
        get_parameter=lambda key: {"average": 4}[key],
        parameters={"average": 4},
    )
    assert protocol_value(SimpleNamespace(protocol=protocol), "average") == 4

    metadata_seq = SimpleNamespace(
        metadata={"protocol": {"parameters": {"TR": 3.0}}}
    )
    assert protocol_value(metadata_seq, "TR") == 3.0
    assert protocol_value(SimpleNamespace(), "unknown") == "unknown"


def test_protocol_numeric_coercion_and_errors():
    seq = SimpleNamespace(parameters={"x": "3.5", "n": "4", "bad": "abc"})
    assert protocol_float(seq, "x") == pytest.approx(3.5)
    assert protocol_int(seq, "n") == 4
    with pytest.raises(ValueError, match="Available sequence parameter keys"):
        protocol_float(seq, "bad", name="bad value")
    with pytest.raises(ValueError, match="must resolve to an int"):
        protocol_int(seq, "bad")


def test_event_delay_setter_and_duration_fallbacks():
    event = SimpleNamespace(name="adc", delay=1e-3, parameters={}, metadata={})
    assert event_name(event) == "adc"
    assert event_delay(event) == pytest.approx(1e-3)

    set_event_delay(event, 2e-3)
    assert event.delay == pytest.approx(2e-3)
    assert event.parameters["delay"] == pytest.approx(2e-3)
    assert event.metadata["timing"]["delay"] == pytest.approx(2e-3)

    adc = SimpleNamespace(type="adc", num_samples=10, dwell=2e-6, delay=0.0)
    assert try_adc_duration(adc) == pytest.approx(20e-6)
    assert event_duration(adc) == pytest.approx(20e-6)

    grad = SimpleNamespace(rise_time=1e-4, flat_time=2e-4, fall_time=1e-4)
    assert event_duration(grad) == pytest.approx(2e-4)

    trapezoid = SimpleNamespace(
        metadata={"timing": {"rise_time": 1e-4, "flat_time": 2e-4, "fall_time": 1e-4}}
    )
    assert event_duration(trapezoid) == pytest.approx(4e-4)

    shaped = SimpleNamespace(shape=SimpleNamespace(duration=3e-3))
    assert event_duration(shaped) == pytest.approx(3e-3)


def test_center_anchor_property_and_timing_helpers():
    event = SimpleNamespace(
        delay=1.0,
        duration=2.0,
        center_pos=0.25,
        parameters={},
        metadata={},
    )
    assert event_center_pos(event) == pytest.approx(0.25)
    assert event_anchor_time(event, "start") == pytest.approx(1.0)
    assert event_anchor_time(event, "center") == pytest.approx(1.5)
    assert event_anchor_time(event, "end") == pytest.approx(3.0)
    with pytest.raises(ValueError, match="Unsupported event anchor"):
        event_anchor_time(event, "quarter")

    set_event_property(event, "duration", 4.0)
    assert event.duration == pytest.approx(4.0)
    assert event.parameters["duration"] == pytest.approx(4.0)
    assert event.metadata["derived"]["duration"] == pytest.approx(4.0)

    assert timing_value(event, "duration") == pytest.approx(4.0)
    event.metadata["timing"] = {"echo_time": 0.03}
    assert timing_value(event, "echo_time") == pytest.approx(0.03)
    assert timing_value(event, "missing") is None


def test_store_derived_parameter_fallbacks():
    seq = SimpleNamespace(derived_parameters={})
    store_derived_parameter(seq, "TE_realized", 0.02)
    assert seq.derived_parameters["TE_realized"] == pytest.approx(0.02)

    class SlotsOnly:
        __slots__ = ("metadata",)
        def __init__(self):
            self.metadata = {}

    slots = SlotsOnly()
    store_derived_parameter(slots, "TR_realized", 1.0)
    assert slots.metadata["derived_parameters"]["TR_realized"] == pytest.approx(1.0)
