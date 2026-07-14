"""Unit tests for RF, ADC, gradient, and delay events."""

from __future__ import annotations

import math

import pytest

import pypulseq_star as ppstar
from pypulseq_star.calc_duration import calc_duration

pytestmark = pytest.mark.unit


def test_rf_event_records_semantics(system: ppstar.Opts) -> None:
    rf = ppstar.make_block_pulse(
        flip_angle=math.pi / 2.0,
        duration=300e-6,
        system=system,
        use="excitation",
        name="rf_excitation",
    )

    assert rf.event_type == "rf"
    assert rf.parameters["flip_angle"] == pytest.approx(math.pi / 2.0)
    assert rf.parameters["use"] == "excitation"
    assert rf.relationships[0].kind == "has_shape"


def test_adc_duration_uses_samples_and_dwell(system: ppstar.Opts) -> None:
    adc = ppstar.make_adc(num_samples=128, dwell=10e-6, system=system)

    assert calc_duration(adc) >= 128 * 10e-6


def test_adc_train_contains_requested_windows(system: ppstar.Opts) -> None:
    adc = ppstar.make_adc_train(
        num_samples=32,
        duration=320e-6,
        num_echoes=4,
        first_delay=20e-6,
        echo_spacing=1e-3,
        system=system,
    )

    assert len(adc.windows) == 4
    assert adc.windows[0].num_samples == 32


def test_delay_duration_is_not_double_counted(system: ppstar.Opts) -> None:
    delay = ppstar.make_delay(2e-3, system=system)
    assert calc_duration(delay) == pytest.approx(2e-3)


def test_rf_duration_includes_delay_and_ringdown(system: ppstar.Opts) -> None:
    rf = ppstar.make_block_pulse(
        flip_angle=math.pi / 2.0,
        duration=300e-6,
        system=system,
    )
    expected = system.rf_dead_time + 300e-6 + system.rf_ringdown_time
    assert calc_duration(rf) == pytest.approx(expected)
