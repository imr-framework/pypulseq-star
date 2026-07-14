"""Beginner example: create and export a compact loop-native TSE sequence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


def phase_encode_order(n_y: int, echo_train_length: int) -> list[list[float]]:
    """Return a shot-major TSE phase-encode table."""
    n_shots = math.ceil(n_y / echo_train_length)
    steps = np.arange(1, n_shots * echo_train_length + 1, dtype=float) - 0.5 * n_shots * echo_train_length - 1.0
    if echo_train_length % 2 == 0:
        steps = np.roll(steps, int(-np.round(n_shots / 2.0)))
    return steps.reshape((n_shots, echo_train_length), order="F").tolist()


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        max_rf=21.58e-6,
    )


def define_protocol() -> ppstar.Protocol:
    n_y = 128
    echo_train_length = 16

    return ppstar.Protocol(
        name="tse",
        description="Compact loop-native turbo spin echo sequence.",
        parameters={
            "Name": "tse",
            "sequence_type": "TSE",
            "fov": 256e-3,
            "n_x": 128,
            "n_y": n_y,
            "slice_thickness": 5e-3,
            "n_slices": 3,
            "echo_train_length": echo_train_length,
            "echo_spacing": 12e-3,
            "TE_effective": 96e-3,
            "repetition_time": 2.0,
            "excitation_flip_angle": 90.0,
            "refocusing_flip_angle": 180.0,
            "excitation_duration": 2.5e-3,
            "refocusing_duration": 2.5e-3,
            "rf_ex_phase": math.pi / 2.0,
            "rf_ref_phase": 0.0,
            "apodization": 0.5,
            "time_bw_product": 4.0,
            "sampling_time": 6.4e-3,
            "gradient_rise_time": 250e-6,
            "readout_spoiler_factor": 1.0,
            "slice_spoiler_factor": 0.5,
            "n_shots": math.ceil(n_y / echo_train_length),
            "phase_encode_steps": phase_encode_order(n_y, echo_train_length),
        },
    )


def main(
    *,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    seq_filename: str = "tse.seq",
    json_filename: str = "tse.seq.json",
) -> None:
    # ======
    # SETUP
    # ======
    system = define_system_limits()
    protocol = define_protocol()

    fov = float(protocol.get_parameter("fov"))
    n_x = int(protocol.get_parameter("n_x"))
    n_shots = int(protocol.get_parameter("n_shots"))
    echo_train_length = int(protocol.get_parameter("echo_train_length"))
    echo_spacing = float(protocol.get_parameter("echo_spacing"))
    effective_te = float(protocol.get_parameter("TE_effective"))
    repetition_time = float(protocol.get_parameter("repetition_time"))
    slice_thickness = float(protocol.get_parameter("slice_thickness"))
    phase_encode_steps = protocol.get_parameter("phase_encode_steps")

    effective_echo = int(round(effective_te / echo_spacing))
    if not math.isclose(effective_te, effective_echo * echo_spacing, abs_tol=1e-9):
        raise ValueError("TE_effective must be an integer multiple of echo_spacing.")

    seq = ppstar.Sequence(system=system, name=str(protocol.get_parameter("Name")), parameters=protocol.parameters, debug=debug)
    seq.set_definition("FOV", [fov, fov, slice_thickness])
    seq.set_definition("Name", str(protocol.get_parameter("Name")))
    seq.set_node("shot", role="shot", repeat_count="n_shots", repeat_every="repetition_time", counter="shot_index", repeat_mode="loop")
    seq.set_node("shot.kernel.echo_train", role="echo_train", repeat_count="echo_train_length", repeat_every="echo_spacing", counter="echo_index", repeat_mode="loop")

    # =============
    # CREATE EVENTS
    # =============
    excitation_rf, excitation_gz, excitation_rephaser = ppstar.make_sinc_pulse(
        flip_angle=float(protocol.get_parameter("excitation_flip_angle")) * math.pi / 180.0,
        duration=float(protocol.get_parameter("excitation_duration")),
        slice_thickness=slice_thickness,
        apodization=float(protocol.get_parameter("apodization")),
        time_bw_product=float(protocol.get_parameter("time_bw_product")),
        phase_offset=float(protocol.get_parameter("rf_ex_phase")),
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
        name="rf_excitation",
        gz_name="gz_excitation",
        gzr_name="gz_excitation_rephaser",
    )
    refocusing_rf, refocusing_gz, _ = ppstar.make_sinc_pulse(
        flip_angle=float(protocol.get_parameter("refocusing_flip_angle")) * math.pi / 180.0,
        duration=float(protocol.get_parameter("refocusing_duration")),
        slice_thickness=slice_thickness,
        apodization=float(protocol.get_parameter("apodization")),
        time_bw_product=float(protocol.get_parameter("time_bw_product")),
        phase_offset=float(protocol.get_parameter("rf_ref_phase")),
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="refocusing",
        name="rf_refocusing",
        gz_name="gz_refocusing",
    )

    delta_k = 1.0 / fov
    sampling_time = float(protocol.get_parameter("sampling_time"))
    rise_time = float(protocol.get_parameter("gradient_rise_time"))

    readout_gradient = ppstar.make_trapezoid(channel="x", flat_area=n_x * delta_k, flat_time=sampling_time + 2.0 * system.adc_dead_time, rise_time=rise_time, system=system, name="gx_readout", role="readout")
    adc = ppstar.make_adc(num_samples=n_x, duration=sampling_time, delay=max(float(readout_gradient.rise_time) + float(system.adc_dead_time), float(system.adc_dead_time)), system=system, name="adc_readout", role="acquisition")

    refocusing_block_duration = max(seq.calc_duration(refocusing_rf), seq.calc_duration(refocusing_gz))
    refocusing_center = float(getattr(refocusing_rf, "delay", 0.0)) + 0.5 * float(getattr(refocusing_rf, "duration", 0.0))
    refocusing_tail = refocusing_block_duration - refocusing_center
    adc_center = float(getattr(adc, "delay", 0.0)) + 0.5 * float(getattr(adc, "duration", sampling_time))
    crusher_budget = 0.5 * echo_spacing - refocusing_tail - adc_center
    crusher_duration = math.floor(crusher_budget / seq.grad_raster_time + 1e-12) * seq.grad_raster_time - seq.grad_raster_time

    if crusher_duration < 2.0 * rise_time:
        raise ValueError("Echo spacing is too short for the selected RF, ADC, and crusher timing.")

    gx_crusher = ppstar.make_trapezoid(channel="x", area=readout_gradient.area * float(protocol.get_parameter("readout_spoiler_factor")), duration=crusher_duration, rise_time=rise_time, system=system, name="gx_crusher", role="spoiling")
    gz_crusher = ppstar.make_trapezoid(channel="z", area=abs(excitation_gz.area) * float(protocol.get_parameter("slice_spoiler_factor")), duration=crusher_duration, rise_time=rise_time, system=system, name="gz_crusher", role="spoiling")

    representative_area = float(phase_encode_steps[0][0]) * delta_k
    gy_phase = ppstar.make_trapezoid(channel="y", area=representative_area, duration=crusher_duration, rise_time=rise_time, system=system, name="gy_phase_encode", role="phase_encode")
    gy_rewind = ppstar.make_trapezoid(channel="y", area=-representative_area, duration=crusher_duration, rise_time=rise_time, system=system, name="gy_phase_rewind", role="phase_rewind")

    gy_phase_variants = []
    gy_rewind_variants = []
    for shot_row in phase_encode_steps:
        for phase_step in shot_row:
            area = float(phase_step) * delta_k
            gy_phase_variants.append(ppstar.make_trapezoid(channel="y", area=area, duration=crusher_duration, rise_time=rise_time, system=system, name="gy_phase_encode", role="phase_encode"))
            gy_rewind_variants.append(ppstar.make_trapezoid(channel="y", area=-area, duration=crusher_duration, rise_time=rise_time, system=system, name="gy_phase_rewind", role="phase_rewind"))

    # ==================
    # CONSTRUCT SEQUENCE
    # ==================
    excitation_block = seq.add_block(excitation_rf, excitation_gz, role="excitation", node="shot.kernel.excitation")
    excitation_rephase_block = seq.add_block(excitation_rephaser, role="excitation_rephase", node="shot.kernel.excitation_rephase")
    first_fill_block = seq.add_block(ppstar.make_delay(delay=0.0, name="first_refocus_fill"), role="echo_timing", node="shot.kernel.first_refocus_fill")
    refocusing_block = seq.add_block(refocusing_rf, refocusing_gz, role="refocusing", node="shot.kernel.echo_train.refocusing")
    crusher_pre_block = seq.add_block(gx_crusher, gz_crusher, gy_phase, role="crusher_pre", node="shot.kernel.echo_train.crusher_pre")
    pre_readout_block = seq.add_block(ppstar.make_delay(delay=0.0, name="pre_readout_fill"), role="echo_timing", node="shot.kernel.echo_train.pre_readout_fill")
    readout_block = seq.add_block(readout_gradient, adc, role="readout", node="shot.kernel.echo_train.readout")
    crusher_post_block = seq.add_block(gx_crusher, gz_crusher, gy_rewind, role="crusher_post", node="shot.kernel.echo_train.crusher_post")
    echo_fill_block = seq.add_block(ppstar.make_delay(delay=0.0, name="echo_period_fill"), role="echo_period_fill", node="shot.kernel.echo_train.echo_period_fill")

    for occurrence, variants, scale in (
        (crusher_pre_block.get_event("gy_phase_encode"), gy_phase_variants, delta_k),
        (crusher_post_block.get_event("gy_phase_rewind"), gy_rewind_variants, -delta_k),
    ):
        occurrence.metadata.update(
            {
                "seqstar_repetition_variants": variants,
                "seqstar_nested_loop_binding": {
                    "protocol_table": "phase_encode_steps",
                    "counter_paths": ("root.n_shots.counter", "root.n_shots.kernel.echo_train_length.counter"),
                    "scale": scale,
                },
                "gammastar_variant_policy": "vary",
            }
        )

    # ====================
    # DEFINE RELATIONSHIPS
    # ====================
    ppstar.relationships.set_center_after(
        seq=seq,
        target=refocusing_block.get_event("rf_refocusing"),
        reference=excitation_block.get_event("rf_excitation"),
        offset=0.5 * echo_spacing,
        solve_event=first_fill_block.get_event("first_refocus_fill"),
        solve_property="duration",
        name="first_refocusing_center_after_excitation",
    )
    ppstar.relationships.set_readout_after(
        seq=seq,
        adc=readout_block.get_event("adc_readout"),
        readout_gradient=readout_block.get_event("gx_readout"),
        reference=refocusing_block.get_event("rf_refocusing"),
        offset=0.5 * echo_spacing,
        solve_event=pre_readout_block.get_event("pre_readout_fill"),
        solve_property="duration",
        name="readout_center_after_refocusing_center",
    )
    echo_blocks = [refocusing_block, crusher_pre_block, pre_readout_block, readout_block, crusher_post_block, echo_fill_block]
    ppstar.relationships.fill_to_period(
        seq=seq,
        blocks=echo_blocks,
        period="echo_spacing",
        fill_event=echo_fill_block.get_event("echo_period_fill"),
        fill_property="duration",
        name="echo_motif_fits_echo_spacing",
    )

    # =====================
    # RESOLVE AND VALIDATE
    # =====================
    ppstar.relationships.resolve(seq)
    ok, error_report = seq.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in error_report:
        print(error)

    minimum_shot_duration = seq.calc_timeline_duration([excitation_block, excitation_rephase_block, first_fill_block]) + echo_train_length * echo_spacing
    if minimum_shot_duration > repetition_time:
        raise ValueError("The requested repetition_time is shorter than one logical TSE shot.")

    print(f"Effective TE: {effective_te * 1e3:.3f} ms (echo {effective_echo})")
    print(f"Echo spacing: {echo_spacing * 1e3:.3f} ms")
    print(f"Logical hierarchy: {n_shots} shots x {echo_train_length} echoes")

    # ====
    # PLOT
    # ====
    if plot:
        seq.plot(title="SeqStar TSE", gradient_scale="mt_per_m", debug=debug)

    # ======
    # EXPORT
    # ======
    output_dir = Path("out/tse")
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_seq:
        seq_path = PulseqWriter(seq, auto_resolve_relationships=False).write(output_dir / seq_filename)
        print(f"Wrote Pulseq: {seq_path}")

    if write_json:
        json_path = GammaStarWriter(seq).write(output_dir / json_filename)
        print(f"Wrote gammaSTAR JSON: {json_path}")


if __name__ == "__main__":
    main()
