from __future__ import annotations

from types import SimpleNamespace

import pytest

import pypulseq_star.writers.seq_writer as sw


pytestmark = pytest.mark.unit


def test_event_family_detection_and_duration_dispatch():
    delay = SimpleNamespace(type="delay", duration=2e-3, delay=1e-3)
    adc = SimpleNamespace(type="adc", num_samples=10, dwell=2e-6, delay=1e-4)
    grad = SimpleNamespace(type="trap", rise_time=1e-4, flat_time=2e-4, fall_time=1e-4)
    rf = SimpleNamespace(type="rf", duration=1e-3)

    assert sw._is_delay_event(delay)
    assert sw._is_adc_event(adc)
    assert sw._is_gradient_event(grad)
    assert sw._is_rf_event(rf)
    assert sw._get_event_active_duration(delay) == pytest.approx(2e-3)
    assert sw._get_event_active_duration(adc) == pytest.approx(20e-6)
    assert sw._get_event_active_duration(grad) == pytest.approx(4e-4)
    assert sw._get_event_active_duration(rf) == pytest.approx(1e-3)


def test_timing_lookup_parameter_helpers_and_channels():
    obj = SimpleNamespace(
        parameters={"x": 3, "f": "4.5"},
        metadata={"timing": {"duration": 2.0}},
    )
    assert sw._get_timing_or_attr(obj, keys=("duration",)) == 2.0
    assert sw._get_int_from_parameters(obj, keys=("x",), default=0) == 3
    assert sw._get_float_from_parameters(obj, keys=("f",), default=None) == pytest.approx(4.5)
    assert sw._get_from_parameters(obj, keys=("missing",), default=9) == 9

    assert sw._gradient_channel(SimpleNamespace(channel="x")) == "x"
    assert sw._gradient_channel(SimpleNamespace(name="gradient_y")) == "y"
    assert sw._gradient_channel(SimpleNamespace(name="unknown")) is None


def test_raster_snapping_all_modes_and_errors():
    assert sw._snap_time_to_raster(15e-6, 10e-6, name="x", mode="nearest", warn=False) == pytest.approx(20e-6)
    assert sw._snap_time_to_raster(11e-6, 10e-6, name="x", mode="ceil", warn=False) == pytest.approx(20e-6)
    assert sw._snap_time_to_raster(19e-6, 10e-6, name="x", mode="floor", warn=False) == pytest.approx(10e-6)
    assert sw._ceil_time_to_raster_no_warn(11e-6, 10e-6) == pytest.approx(20e-6)
    with pytest.raises(ValueError, match="raster must be positive"):
        sw._snap_time_to_raster(1, 0, name="x")
    with pytest.raises(ValueError, match="non-negative"):
        sw._snap_time_to_raster(-1, 1, name="x")
    with pytest.raises(ValueError, match="Unsupported"):
        sw._snap_time_to_raster(1, 1, name="x", mode="bad")
    with pytest.warns(UserWarning, match="Snapped"):
        sw._snap_time_to_raster(11e-6, 10e-6, name="x")


def test_pulseq_event_duration_fallbacks():
    assert sw._pulseq_event_duration_fallback(None) == 0.0
    assert sw._pulseq_event_duration_fallback(
        SimpleNamespace(delay=1e-3, num_samples=10, dwell=2e-6)
    ) == pytest.approx(1.02e-3)
    assert sw._pulseq_event_duration_fallback(
        SimpleNamespace(delay=1e-3, duration=2e-3)
    ) == pytest.approx(3e-3)
    assert sw._pulseq_event_duration_fallback(SimpleNamespace(duration=2e-3)) == pytest.approx(2e-3)
    assert sw._pulseq_event_duration_fallback(
        SimpleNamespace(delay=1e-3, rise_time=1e-4, flat_time=2e-4, fall_time=1e-4)
    ) == pytest.approx(1.4e-3)
    assert sw._pulseq_event_duration_fallback(SimpleNamespace()) == 0.0


def test_adc_window_helper_paths():
    event = SimpleNamespace(
        delay=1e-3,
        dead_time=50e-6,
        windows=[
            {"delay": 0.0, "num_samples": 4, "dwell": 1e-6},
            {"tstart": 10e-6, "num_samples": 4, "dwell": 1e-6},
        ],
    )
    windows = sw._get_adc_pulseq_windows(event)
    assert len(windows) == 2
    assert sw._has_adc_train_windows(event)
    shifted = sw._adc_window_with_parent_delay(event, windows[0])
    assert shifted["delay"] == pytest.approx(1e-3)
    assert sw._adc_window_duration(windows[0]) == pytest.approx(4e-6)
    assert sw._get_adc_train_dead_time(event, SimpleNamespace(adc_dead_time=0.0)) == pytest.approx(50e-6)


def test_gradient_event_classification():
    assert sw._is_arbitrary_gradient_event(SimpleNamespace(type="grad", waveform=[0, 1]))
    assert sw._is_split_gradient_event(SimpleNamespace(type="split"))
