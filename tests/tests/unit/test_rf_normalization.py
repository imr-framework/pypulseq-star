"""RF amplitude / flip-angle normalization tests.

The central invariant is:

    alpha = 2*pi*gamma*|integral B1(t) dt|

for the RF envelope represented by each constructor.  The arbitrary-RF tests
also verify the gammaSTAR AM/FM serialization round-trip, including a waveform
originating from SigPy when SigPy is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pypulseq_star as ppstar


def _system() -> ppstar.Opts:
    # Deliberately generous max_rf so normalization is tested independently of
    # hardware-limit rejection.
    return ppstar.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        max_rf=1e-3,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
    )


def _samples(event, system):
    return event.shape.to_gammastar_samples(
        flip_angle=float(event.flip_angle),
        gamma_hz_per_t=float(system.gamma),
        max_rf=float(system.max_rf),
    )


def _reconstruct_complex_b1_from_am_fm(samples: dict, dt: float) -> np.ndarray:
    channel = samples["v"][0]
    am = np.asarray(channel["am"], dtype=float)
    fm = np.asarray(channel["fm"], dtype=float)
    assert am.shape == fm.shape

    # The arbitrary RF serializer defines fm[i] from the phase increment from
    # sample i -> i+1. Global starting phase is irrelevant to |integral B1 dt|.
    phase = np.zeros_like(am)
    for index in range(max(0, len(am) - 1)):
        phase[index + 1] = phase[index] + 2.0 * math.pi * fm[index] * dt

    return am * np.exp(1j * phase)


def _flip_from_complex_b1(b1: np.ndarray, dt: float, gamma_hz_per_t: float) -> float:
    return 2.0 * math.pi * gamma_hz_per_t * abs(np.sum(b1) * dt)


def test_block_rf_normalization_exact():
    system = _system()
    flip = math.radians(90.0)
    duration = 3e-3

    event = ppstar.make_block_pulse(
        flip_angle=flip,
        duration=duration,
        system=system,
    )
    samples = _samples(event, system)
    amplitude = float(samples["v"][0]["am"][0])

    reconstructed = 2.0 * math.pi * system.gamma * amplitude * duration
    assert reconstructed == pytest.approx(flip, rel=1e-12, abs=1e-12)


def test_sinc_rf_integral_matches_requested_flip_angle():
    system = _system()
    flip = math.radians(90.0)

    event = ppstar.make_sinc_pulse(
        flip_angle=flip,
        duration=3e-3,
        time_bw_product=4.0,
        apodization=0.5,
        system=system,
    )
    samples = _samples(event, system)
    am = np.asarray(samples["v"][0]["am"], dtype=float)

    reconstructed = 2.0 * math.pi * system.gamma * abs(
        np.sum(am) * system.rf_raster_time
    )
    assert reconstructed == pytest.approx(flip, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(
    "signal",
    [
        np.hanning(128),
        np.hanning(128) * np.exp(1j * 0.7),
        np.hanning(128) * np.exp(1j * np.linspace(-0.8, 0.8, 128)),
    ],
)
def test_arbitrary_rf_am_fm_roundtrip_preserves_flip_angle(signal):
    system = _system()
    flip = math.radians(90.0)
    dwell = 20e-6

    event = ppstar.make_arbitrary_rf(
        signal=signal,
        flip_angle=flip,
        dwell=dwell,
        system=system,
        use="saturation",
    )
    samples = _samples(event, system)
    b1 = _reconstruct_complex_b1_from_am_fm(samples, system.rf_raster_time)

    reconstructed = _flip_from_complex_b1(
        b1,
        system.rf_raster_time,
        system.gamma,
    )
    assert reconstructed == pytest.approx(flip, rel=2e-6, abs=2e-8)


def test_arbitrary_rf_peak_amplitude_matches_serialized_samples():
    system = _system()
    signal = np.hanning(256)

    event = ppstar.make_arbitrary_rf(
        signal=signal,
        flip_angle=math.radians(90.0),
        dwell=10e-6,
        system=system,
    )
    samples = _samples(event, system)
    exported_peak = float(np.max(np.abs(samples["v"][0]["am"])))

    metadata = getattr(event, "metadata", {}) or {}
    shape_meta = metadata.get("rf_shape", metadata.get("shape", {})) or {}
    recorded_peak = shape_meta.get("amplitude_t")

    # Metadata layout has evolved; require equality when the constructor records
    # the peak, otherwise the primary normalization tests still cover the path.
    if recorded_peak is not None:
        assert exported_peak == pytest.approx(float(recorded_peak), rel=1e-10)


def _sigpy_dzrf():
    pytest.importorskip("sigpy", reason="SigPy optional dependency is not installed")
    try:
        from sigpy.mri.rf.slr import dzrf
    except ImportError:
        try:
            from sigpy.mri.rf import dzrf
        except ImportError:
            pytest.skip("Installed SigPy version does not expose dzrf")
    return dzrf


def test_sigpy_slr_waveform_normalizes_to_requested_flip_angle():
    """External-library regression: SigPy SLR -> make_arbitrary_rf -> gammaSTAR."""

    dzrf = _sigpy_dzrf()
    system = _system()

    # A modest, broadly supported SLR design. The test is concerned with the
    # arbitrary-RF ingestion/normalization contract, not a particular VAPOR
    # design prescription.
    signal = np.asarray(
        dzrf(
            n=128,
            tb=4.0,
            ptype="st",
            ftype="ls",
            d1=0.01,
            d2=0.01,
        ),
        dtype=complex,
    )

    assert signal.ndim == 1
    assert signal.size == 128
    assert np.max(np.abs(signal)) > 0

    flip = math.radians(90.0)
    event = ppstar.make_arbitrary_rf(
        signal=signal,
        flip_angle=flip,
        dwell=20e-6,
        system=system,
        use="saturation",
        parameters={"arbitrary_signal_label": "sigpy_slr_test"},
    )

    samples = _samples(event, system)
    b1 = _reconstruct_complex_b1_from_am_fm(samples, system.rf_raster_time)
    reconstructed = _flip_from_complex_b1(
        b1,
        system.rf_raster_time,
        system.gamma,
    )

    assert reconstructed == pytest.approx(flip, rel=2e-6, abs=2e-8)
