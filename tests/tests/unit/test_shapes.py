"""Unit tests for waveform shape construction."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pypulseq_star as ppstar

pytestmark = pytest.mark.unit


def test_block_rf_has_expected_duration_and_samples() -> None:
    rf = ppstar.make_block_pulse(math.pi / 2.0, 300e-6, use="excitation")

    assert rf.shape is not None
    assert rf.shape.duration == pytest.approx(300e-6)
    assert rf.shape.kind == "rf_block"
    assert rf.shape.rf_raster_time > 0


def test_arbitrary_gradient_preserves_waveform(system: ppstar.Opts) -> None:
    waveform = np.array([0.0, 1.0, 0.5, 0.0])
    gradient = ppstar.make_arbitrary_grad(
        channel="x",
        waveform=waveform,
        system=system,
        name="gx_arbitrary",
    )

    stored = np.asarray(getattr(gradient, "waveform", gradient.parameters.get("waveform")))
    np.testing.assert_allclose(stored, waveform)


def test_trapezoid_area_matches_requested_area(system: ppstar.Opts) -> None:
    gradient = ppstar.make_trapezoid(
        channel="y",
        area=4.0,
        duration=2e-3,
        system=system,
        name="gy_area",
    )

    assert float(gradient.area) == pytest.approx(4.0, rel=1e-6)


def test_invalid_gradient_channel_is_rejected(system: ppstar.Opts) -> None:
    with pytest.raises((ValueError, AssertionError)):
        ppstar.make_trapezoid(channel="q", area=1.0, duration=1e-3, system=system)
