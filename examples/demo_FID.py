"""Beginner example: create and export a relationship-aware FID sequence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=50e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
    )


def define_protocol() -> ppstar.Protocol:
    return ppstar.Protocol(
        name="fid",
        description="Relationship-aware free-induction-decay sequence.",
        parameters={
            "Name": "fid",
            "sequence_type": "FID",
            "flip_angle": 90.0,
            "rf_duration": 300e-6,
            "TE": 20e-3,
            "TR": 1.0,
            "averages": 4,
            "num_samples": 2048,
            "dwell": 20e-6,
            "rf_phase_offset": 0.0,
            "rf_frequency_offset": 0.0,
            "adc_phase_offset": 0.0,
            "adc_frequency_offset": 0.0,
        },
    )


def main(
    *,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    seq_filename: str = "fid.seq",
    json_filename: str = "fid.seq.json",
) -> None:
    # ======
    # SETUP
    # ======
    system = define_system_limits()
    protocol = define_protocol()
    seq = ppstar.Sequence(system=system, name=str(protocol.get_parameter("Name")), parameters=protocol.parameters, debug=debug)

    seq.set_definition("Name", str(protocol.get_parameter("Name")))
    seq.set_node("kernel", role="kernel", repeat_count="average", repeat_every="repetition_time", counter="average_index", repeat_mode="loop")

    # =============
    # CREATE EVENTS
    # =============
    rf = ppstar.make_block_pulse(
        flip_angle=float(protocol.get_parameter("flip_angle_excitation")) * math.pi / 180.0,
        duration=float(protocol.get_parameter("rf_duration")),
        phase_offset=float(protocol.get_parameter("rf_phase_offset")),
        freq_offset=float(protocol.get_parameter("rf_frequency_offset")),
        use="excitation",
        system=system,
        name="rf_excitation",
    )
    adc = ppstar.make_adc(
        num_samples=int(protocol.get_parameter("num_samples")),
        dwell=float(protocol.get_parameter("dwell")),
        delay=system.adc_dead_time,
        phase_offset=float(protocol.get_parameter("adc_phase_offset")),
        freq_offset=float(protocol.get_parameter("adc_frequency_offset")),
        system=system,
        name="fid_adc",
        role="acquisition",
    )
    te_fill = ppstar.make_delay(0.0, system=system, name="te_fill", role="echo_time_fill")

    # ==================
    # CONSTRUCT SEQUENCE
    # ==================
    excitation_block = seq.add_block(rf, role="excitation", node="kernel.excitation")
    te_fill_block = seq.add_block(te_fill, role="echo_time_fill", node="kernel.echo_delay")
    readout_block = seq.add_block(adc, role="readout", node="kernel.readout")

    rf_occurrence = excitation_block.get_event("rf_excitation")
    te_fill_occurrence = te_fill_block.get_event("te_fill")
    adc_occurrence = readout_block.get_event("fid_adc")

    # ====================
    # DEFINE RELATIONSHIPS
    # ====================
    ppstar.relationships.set_anchor_after(
        seq=seq,
        target=adc_occurrence,
        target_anchor="start",
        reference=rf_occurrence,
        reference_anchor="center",
        offset="echo_time",
        solve_event=te_fill_occurrence,
        solve_property="duration",
        name="fid_adc_start_after_rf_center",
    )

    # =====================
    # RESOLVE AND VALIDATE
    # =====================
    ppstar.relationships.resolve(seq)
    ok, error_report = seq.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in error_report:
        print(error)

    requested_tr = float(protocol.get_parameter("repetition_time"))
    rf_center = ppstar.relationships.get_anchor_time(rf_occurrence, anchor="center", seq=seq, frame="global")
    adc_start = ppstar.relationships.get_anchor_time(adc_occurrence, anchor="start", seq=seq, frame="global")
    kernel_duration = seq.calc_timeline_duration([excitation_block, te_fill_block, readout_block])

    print(f"TE: {(adc_start - rf_center) * 1e3:.3f} ms")
    print(f"TR: {requested_tr * 1e3:.3f} ms")
    print(f"Kernel occupancy: {kernel_duration * 1e3:.3f} ms")

    # ====
    # PLOT
    # ====
    if plot:
        seq.plot(time_range=(0.0, min(2.0 * requested_tr, 2.0)), title="SeqStar FID", debug=debug)

    # ======
    # EXPORT
    # ======
    output_dir = Path("out/fid")
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_seq:
        seq_path = PulseqWriter(seq).write(output_dir / seq_filename)
        print(f"Wrote Pulseq: {seq_path}")

    if write_json:
        json_path = GammaStarWriter(seq).write(output_dir / json_filename)
        print(f"Wrote gammaSTAR JSON: {json_path}")


if __name__ == "__main__":
    main()
