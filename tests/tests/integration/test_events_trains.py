"""Self-contained regression tests replacing the legacy RF/gradient/ADC demos."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pypulseq_star as ppstar
from pypulseq_star.calc_duration import calc_duration
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def _system() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=120,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=50e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
        block_duration_raster=10e-6,
    )


def _sequence(name: str, system: ppstar.Opts) -> ppstar.Sequence:
    return ppstar.Sequence(system=system, name=name, parameters={"Name": name, "TR": 20e-3})


@pytest.mark.parametrize("pulse_type", ["block", "sinc", "gauss", "arbitrary"])
def test_rf_train_event_families_lower(pulse_type: str) -> None:
    system = _system()
    if pulse_type == "block":
        rf = ppstar.make_block_pulse(math.pi / 2, 300e-6, system=system, use="excitation")
    elif pulse_type == "sinc":
        rf = ppstar.make_sinc_pulse(math.pi / 2, 3e-3, system=system, use="excitation")
    elif pulse_type == "gauss":
        rf = ppstar.make_gauss_pulse(math.pi / 2, 3e-3, system=system, use="excitation")
    else:
        signal = np.hamming(256).tolist()
        rf = ppstar.make_arbitrary_rf(signal=signal, flip_angle=math.pi / 2, duration=3e-3, system=system, use="excitation")
    seq = _sequence(f"rf_{pulse_type}", system)
    seq.add_block(rf, role="excitation", node="kernel.excitation")
    assert calc_duration(rf) > 0
    document = GammaStarWriter(seq).to_dict()
    assert any(value == "RFPulse" for value in document["sequence_elements"].values())


@pytest.mark.parametrize("grad_type", ["trapezoid", "arbitrary", "split"])
def test_gradient_train_event_families_lower(grad_type: str) -> None:
    system = _system()
    seq = _sequence(f"grad_{grad_type}", system)
    if grad_type == "trapezoid":
        events = [ppstar.make_trapezoid("x", area=120.0, duration=3e-3, system=system, role="readout", axis_role="read")]
    elif grad_type == "arbitrary":
        waveform = (0.001 * np.sin(np.linspace(0, 2 * np.pi, 400))).tolist()
        events = [ppstar.make_arbitrary_grad("y", waveform, system=system, role="phase_encode", axis_role="phase")]
    else:
        parent = ppstar.make_trapezoid("z", area=80.0, duration=4e-3, system=system, role="slice_rephase", axis_role="slice")
        events = list(ppstar.split_gradient(parent, system=system, name="split_gradient"))
    for index, event in enumerate(events):
        seq.add_block(event, role="encoding", node=f"kernel.encoding_{index}")
    assert all(calc_duration(event) > 0 for event in events)
    document = GammaStarWriter(seq).to_dict()
    assert any(value == "GradPulse" for value in document["sequence_elements"].values())


@pytest.mark.parametrize("num_echoes,mode", [(1, "single"), (4, "discrete"), (8, "continuous")])
def test_adc_train_repetition_modes_lower(num_echoes: int, mode: str) -> None:
    system = _system()
    adc = ppstar.make_adc_train(
        num_samples=32,
        duration=320e-6,
        num_echoes=num_echoes,
        first_delay=50e-6,
        echo_spacing=1e-3,
        mode=mode,
        system=system,
        role="readout",
    )
    seq = _sequence(f"adc_{mode}", system)
    seq.add_block(adc, role="readout", node="kernel.readout")
    assert len(adc.windows) == num_echoes
    document = GammaStarWriter(seq).to_dict()
    assert any(value == "ADC" for value in document["sequence_elements"].values())


def test_representative_train_cases_write_to_temporary_directory(tmp_path: Path) -> None:
    system = _system()
    seq = _sequence("representative_train", system)
    rf = ppstar.make_block_pulse(math.pi / 2, 300e-6, system=system, use="excitation")
    gx = ppstar.make_trapezoid("x", area=100.0, duration=3e-3, system=system, role="readout", axis_role="read")
    adc = ppstar.make_adc(num_samples=32, dwell=10e-6, delay=50e-6, system=system, role="acquisition")
    seq.add_block(rf, role="excitation", node="kernel.excitation")
    seq.add_block(gx, adc, role="readout", node="kernel.readout")
    seq_path = PulseqWriter(seq).write(tmp_path / "representative.seq")
    json_path = GammaStarWriter(seq).write(tmp_path / "representative.seq.json")
    assert Path(seq_path).is_file()
    assert Path(json_path).is_file()
