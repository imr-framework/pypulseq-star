from __future__ import annotations

import math

import pytest

from pypulseq_star.shapes.rf import (
    SeqStarRFArbitraryShape,
    SeqStarRFBlockShape,
    SeqStarRFGaussShape,
    SeqStarRFShape,
    SeqStarRFSincShape,
    _sinc,
)


pytestmark = pytest.mark.unit
GAMMA = 42.575575e6


def integrated_flip(samples, raster):
    return 2 * math.pi * GAMMA * sum(samples["v"][0]["am"]) * raster


def test_base_rf_shape_and_common_validation():
    with pytest.raises(NotImplementedError):
        SeqStarRFShape(name="base", duration=1e-3).to_gammastar_samples(
            flip_angle=1.0, gamma_hz_per_t=GAMMA
        )

    block = SeqStarRFBlockShape(name="block", duration=1e-3, rf_raster_time=1e-6)
    with pytest.raises(ValueError, match="flip_angle"):
        block.to_gammastar_samples(flip_angle=0, gamma_hz_per_t=GAMMA)
    with pytest.raises(ValueError, match="gamma_hz_per_t"):
        block.to_gammastar_samples(flip_angle=1, gamma_hz_per_t=0)
    with pytest.raises(ValueError, match="not aligned"):
        SeqStarRFBlockShape(name="block", duration=1.5e-6, rf_raster_time=1e-6).to_gammastar_samples(
            flip_angle=1, gamma_hz_per_t=GAMMA
        )
    with pytest.raises(ValueError, match="max_rf must be non-negative"):
        block.to_gammastar_samples(flip_angle=1, gamma_hz_per_t=GAMMA, max_rf=-1)
    with pytest.raises(ValueError, match="exceeds max_rf"):
        block.to_gammastar_samples(flip_angle=math.pi, gamma_hz_per_t=GAMMA, max_rf=1e-9)


def test_block_rf_samples_have_expected_amplitude():
    shape = SeqStarRFBlockShape(name="block", duration=1e-3)
    result = shape.to_gammastar_samples(flip_angle=math.pi / 2, gamma_hz_per_t=GAMMA)
    amp = result["v"][0]["am"][0]
    assert amp == pytest.approx((math.pi / 2) / (2 * math.pi * GAMMA * 1e-3))
    assert result["t"] == [0.0, 1e-3]


@pytest.mark.parametrize("shape_cls", [SeqStarRFSincShape, SeqStarRFGaussShape])
def test_rastered_rf_shapes_normalize_and_preserve_flip(shape_cls):
    shape = shape_cls(name="shape", duration=20e-6, rf_raster_time=1e-6)
    t, envelope = shape.normalized_envelope()
    assert len(t) == len(envelope) == 20
    assert max(abs(x) for x in envelope) == pytest.approx(1.0)

    samples = shape.to_gammastar_samples(
        flip_angle=math.pi / 3,
        gamma_hz_per_t=GAMMA,
    )
    assert integrated_flip(samples, 1e-6) == pytest.approx(math.pi / 3)


@pytest.mark.parametrize("shape_cls", [SeqStarRFSincShape, SeqStarRFGaussShape])
def test_rastered_rf_shape_validation(shape_cls):
    with pytest.raises(ValueError, match="rf_raster_time"):
        shape_cls(name="shape", duration=1e-3).normalized_envelope()
    with pytest.raises(ValueError, match="at least two"):
        shape_cls(name="shape", duration=1e-6, rf_raster_time=1e-6).normalized_envelope()
    with pytest.raises(ValueError, match="time_bw_product"):
        shape_cls(name="shape", duration=10e-6, rf_raster_time=1e-6, time_bw_product=0).normalized_envelope()
    with pytest.raises(ValueError, match="apodization"):
        shape_cls(name="shape", duration=10e-6, rf_raster_time=1e-6, apodization=2).normalized_envelope()
    with pytest.raises(ValueError, match="center_pos"):
        shape_cls(name="shape", duration=10e-6, rf_raster_time=1e-6, center_pos=2).normalized_envelope()


def test_arbitrary_rf_happy_path_and_validation():
    shape = SeqStarRFArbitraryShape(
        name="arb",
        duration=4e-6,
        rf_raster_time=1e-6,
        signal=[1.0, 0.5, 0.5, 1.0],
    )
    t, envelope = shape.normalized_envelope()
    assert t == pytest.approx([0.5e-6, 1.5e-6, 2.5e-6, 3.5e-6])
    assert envelope == [1.0, 0.5, 0.5, 1.0]
    samples = shape.to_gammastar_samples(flip_angle=1.0, gamma_hz_per_t=GAMMA)
    assert integrated_flip(samples, 1e-6) == pytest.approx(1.0)

    invalids = [
        SeqStarRFArbitraryShape(name="arb", duration=1e-6, rf_raster_time=None, signal=[1]),
        SeqStarRFArbitraryShape(name="arb", duration=1e-6, rf_raster_time=0, signal=[1]),
        SeqStarRFArbitraryShape(name="arb", duration=1e-6, rf_raster_time=1e-6, signal=[]),
        SeqStarRFArbitraryShape(name="arb", duration=1e-6, rf_raster_time=1e-6, signal=[0]),
        SeqStarRFArbitraryShape(name="arb", duration=1e-6, rf_raster_time=1e-6, signal=[float("nan")]),
    ]
    for invalid in invalids:
        with pytest.raises((ValueError, TypeError)):
            invalid.normalized_envelope()

    with pytest.raises(NotImplementedError):
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6, rf_raster_time=1e-6, signal=[1 + 2j]
        ).normalized_envelope()
    with pytest.raises(TypeError):
        SeqStarRFArbitraryShape(
            name="arb",
            duration=1e-6, rf_raster_time=1e-6, signal=["x"]
        ).normalized_envelope()
    with pytest.raises(ValueError, match="signal length"):
        SeqStarRFArbitraryShape(
            name="arb",
            duration=2e-6, rf_raster_time=1e-6, signal=[1]
        ).normalized_envelope()
    with pytest.raises(ValueError, match="zero signed area"):
        SeqStarRFArbitraryShape(
            name="arb",
            duration=2e-6, rf_raster_time=1e-6, signal=[1, -1]
        ).to_gammastar_samples(flip_angle=1, gamma_hz_per_t=GAMMA)


def test_sinc_helper():
    assert _sinc(0.0) == pytest.approx(1.0)
    assert _sinc(1.0) == pytest.approx(0.0, abs=1e-12)
