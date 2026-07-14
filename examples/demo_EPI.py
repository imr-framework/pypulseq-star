"""Beginner example: create and export a compact single-shot EPI sequence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def define_protocol() -> ppstar.Protocol:
    return ppstar.Protocol(
        name="epi",
        description="Compact single-shot 2D EPI.",
        parameters={
            "Name": "epi",
            "sequence_type": "EPI",
            "fov_read": 220e-3,
            "fov_phase": 220e-3,
            "n_x": 64,
            "n_y": 64,
            "slice_thickness": 3e-3,
            "n_slices": 3,
            "flip_angle": 90.0,
            "rf_duration": 3e-3,
            "apodization": 0.5,
            "time_bw_product": 4,
            "dwell_time": 4e-6,
            "prephase_duration": 8e-4,
            "trajectory": "cartesian_epi",
        },
    )


def sample_trapezoid(amplitude: float, rise_time: float, flat_time: float, fall_time: float, raster: float) -> np.ndarray:
    """Sample a trapezoid at gradient-raster centers."""
    n_rise = int(round(rise_time / raster))
    n_flat = int(round(flat_time / raster))
    n_fall = int(round(fall_time / raster))
    rise = amplitude * (np.arange(n_rise, dtype=float) + 0.5) / n_rise if n_rise else np.empty(0)
    flat = np.full(n_flat, amplitude, dtype=float)
    fall = amplitude * (1.0 - (np.arange(n_fall, dtype=float) + 0.5) / n_fall) if n_fall else np.empty(0)
    return np.concatenate((rise, flat, fall))


def main(
    *,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    seq_filename: str = "epi.seq",
    json_filename: str = "epi.seq.json",
) -> None:
    # ======
    # SETUP
    # ======
    system = define_system_limits()
    protocol = define_protocol()

    fov_read = float(protocol.get_parameter("fov_read"))
    fov_phase = float(protocol.get_parameter("fov_phase"))
    n_x = int(protocol.get_parameter("n_x"))
    n_y = int(protocol.get_parameter("n_y"))
    n_slices = int(protocol.get_parameter("n_slices"))
    slice_thickness = float(protocol.get_parameter("slice_thickness"))
    dwell_time = float(protocol.get_parameter("dwell_time"))
    prephase_duration = float(protocol.get_parameter("prephase_duration"))

    seq = ppstar.Sequence(system=system, name=str(protocol.get_parameter("Name")), parameters=protocol.parameters, debug=debug)
    seq.set_definition("FOV", [fov_read, fov_phase, slice_thickness])
    seq.set_definition("Name", str(protocol.get_parameter("Name")))
    seq.set_node("kernel", role="kernel", repeat_count=n_slices, counter="slice_index", repeat_mode="expanded")
    seq.set_node("kernel.echo_train", role="train", repeat_mode="expanded", metadata={"trajectory": str(protocol.get_parameter("trajectory")), "representation": "synchronized_train"})
    seq.set_node("kernel.echo_train.readout", role="readout_train", repeat_count=n_y, repeat_every="echo_spacing", counter="line_index", repeat_mode="loop")
    seq.set_node("kernel.echo_train.readout.single_readout", role="readout", repeat_mode="loop")

    # =============
    # CREATE EVENTS
    # =============
    rf, gz, gz_reph = ppstar.make_sinc_pulse(
        flip_angle=float(protocol.get_parameter("flip_angle")) * math.pi / 180.0,
        duration=float(protocol.get_parameter("rf_duration")),
        slice_thickness=slice_thickness,
        apodization=float(protocol.get_parameter("apodization")),
        time_bw_product=float(protocol.get_parameter("time_bw_product")),
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
        name="rf_excitation",
        gz_name="gz_slice_select",
        gzr_name="gz_slice_rephase",
    )

    delta_kx = 1.0 / fov_read
    delta_ky = 1.0 / fov_phase
    readout_time = n_x * dwell_time

    readout_lobe = ppstar.make_trapezoid(channel="x", amplitude=(n_x * delta_kx) / readout_time, flat_time=math.ceil(readout_time / seq.grad_raster_time - 1e-12) * seq.grad_raster_time, system=system, name="gx_readout_lobe", role="readout")
    gx_pre = ppstar.make_trapezoid(channel="x", area=-float(readout_lobe.area) / 2.0, duration=prephase_duration, system=system, name="gx_prephase", role="prephase")
    gy_pre = ppstar.make_trapezoid(channel="y", area=-n_y / 2.0 * delta_ky, duration=prephase_duration, system=system, name="gy_prephase", role="prephase")

    blip_duration = math.ceil((2.0 * math.sqrt(delta_ky / float(system.max_slew))) / seq.grad_raster_time - 1e-12) * seq.grad_raster_time
    phase_blip = ppstar.make_trapezoid(channel="y", area=delta_ky, duration=blip_duration, system=system, name="gy_phase_blip_lobe", role="encoding_step")

    gx_lobe = sample_trapezoid(float(readout_lobe.amplitude), float(readout_lobe.rise_time), float(readout_lobe.flat_time), float(readout_lobe.fall_time), seq.grad_raster_time)
    gy_blip = sample_trapezoid(float(phase_blip.amplitude), float(phase_blip.rise_time), float(phase_blip.flat_time), float(phase_blip.fall_time), seq.grad_raster_time)

    # Arbitrary-gradient samples are connected directly by the plotter. Add
    # explicit zero-valued boundary samples so each Gy blip begins and ends at
    # zero instead of being connected from a nonzero endpoint to the next TR.
    gy_blip = np.concatenate(([0.0], gy_blip, [0.0]))

    echo_spacing = (len(gx_lobe) + len(gy_blip)) * seq.grad_raster_time

    gx_segments = []
    gy_segments = []
    for line_index in range(n_y):
        polarity = 1.0 if line_index % 2 == 0 else -1.0
        gx_segments.append(np.concatenate((polarity * gx_lobe, np.zeros(len(gy_blip)))))
        gy_segments.append(np.concatenate((np.zeros(len(gx_lobe)), gy_blip)))

    if gy_blip[0] != 0.0 or gy_blip[-1] != 0.0:
        raise ValueError("The sampled Gy blip must begin and end at zero.")

    gx_train = ppstar.make_arbitrary_grad(channel="x", waveform=np.concatenate(gx_segments), system=system, name="gx_echo_train", role="readout_train", metadata={"segment_duration": echo_spacing, "trajectory": str(protocol.get_parameter("trajectory")), "terminal_blip": True})
    gy_train = ppstar.make_arbitrary_grad(channel="y", waveform=np.concatenate(gy_segments), system=system, name="gy_blip_train", role="encoding_train", metadata={"segment_duration": echo_spacing, "trajectory": str(protocol.get_parameter("trajectory")), "terminal_blip": True})

    raw_adc_delay = float(readout_lobe.rise_time) + float(readout_lobe.flat_time) / 2.0 - (readout_time - dwell_time) / 2.0
    adc_delay = max(system.adc_dead_time, round(raw_adc_delay / seq.grad_raster_time) * seq.grad_raster_time)
    adc_train = ppstar.make_adc_train(
        num_samples=n_x,
        duration=readout_time,
        num_echoes=n_y,
        first_delay=adc_delay,
        echo_spacing=echo_spacing,
        polarity="alternating",
        trajectory=str(protocol.get_parameter("trajectory")),
        mode="echo_train",
        name="adc_echo_train",
        role="acquisition_train",
        dead_time=system.adc_dead_time,
        dead_time_policy="train_level_once",
        pulseq_export_mode="windows",
        system=system,
        parameters={
            "counter": "line_index",
            "synchronized_with_block_gradients": True,
            "segment_duration": echo_spacing,
            "terminal_blip": True,
            "logical_readout_node": "kernel.echo_train.readout",
            "logical_single_readout_node": "kernel.echo_train.readout.single_readout",
        },
    )
    seq.parameters["echo_spacing"] = echo_spacing

    # ==================
    # CONSTRUCT SEQUENCE
    # ==================
    first_slice_blocks = None

    for slice_index in range(n_slices):
        slice_position = slice_thickness * (slice_index - (n_slices - 1) / 2.0)
        rf.freq_offset = float(gz.amplitude) * slice_position

        excitation_block = seq.add_block(rf, gz, role="excitation", node="kernel.excitation", varies=["slice_index"])
        prephase_block = seq.add_block(gx_pre, gy_pre, gz_reph, role="prephase", node="kernel.prephase", varies=["slice_index"])
        echo_train_block = seq.add_block(gx_train, gy_train, adc_train, role="echo_train", node="kernel.echo_train", varies=["slice_index", "line_index"])

        if first_slice_blocks is None:
            first_slice_blocks = [excitation_block, prephase_block, echo_train_block]

    # =====================
    # RESOLVE AND VALIDATE
    # =====================
    ppstar.relationships.resolve(seq)
    ok, error_report = seq.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in error_report:
        print(error)

    first_slice_duration = seq.calc_timeline_duration(first_slice_blocks or [])
    print(f"Echo spacing: {echo_spacing * 1e6:.1f} us")
    print(f"ADC windows: {len(adc_train.windows)}")
    print(f"First-slice duration: {first_slice_duration * 1e3:.3f} ms")

    # ====
    # PLOT
    # ====
    if plot:
        seq.plot(time_range=(0.0, first_slice_duration), title="SeqStar EPI", gradient_scale="mt_per_m", debug=debug)

    # ======
    # EXPORT
    # ======
    output_dir = Path("out/epi")
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_seq:
        seq_path = PulseqWriter(seq).write(output_dir / seq_filename)
        print(f"Wrote Pulseq: {seq_path}")

    if write_json:
        json_path = GammaStarWriter(seq).write(output_dir / json_filename)
        print(f"Wrote gammaSTAR JSON: {json_path}")


if __name__ == "__main__":
    main()
