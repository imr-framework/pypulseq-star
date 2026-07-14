"""Beginner example: create and export a spoiled GRE sequence."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


def define_system_limits() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def define_protocol() -> ppstar.Protocol:
    return ppstar.Protocol(
        name="gre",
        description="Relationship-aware spoiled gradient echo sequence.",
        parameters={
            "Name": "gre",
            "sequence_type": "GRE",
            "fov": 256e-3,
            "n_x": 64,
            "n_y": 64,
            "slice_thickness": 3e-3,
            "flip_angle": 10.0,
            "rf_duration": 3e-3,
            "readout_duration": 3.2e-3,
            "TE": 5e-3,
            "TR": 12e-3,
            "apodization": 0.42,
            "time_bw_product": 4,
            "rf_spoiling_inc": 117.0,
        },
    )


def main(
    *,
    plot: bool = True,
    write_seq: bool = True,
    write_json: bool = True,
    debug: bool = False,
    seq_filename: str = "gre.seq",
    json_filename: str = "gre.seq.json",
) -> None:
    # ======
    # SETUP
    # ======
    system = define_system_limits()
    protocol = define_protocol()
    fov = float(protocol.get_parameter("fov"))
    n_x = int(protocol.get_parameter("n_x"))
    n_y = int(protocol.get_parameter("n_y"))
    slice_thickness = float(protocol.get_parameter("slice_thickness"))

    seq = ppstar.Sequence(system=system, name=str(protocol.get_parameter("Name")), parameters=protocol.parameters, debug=debug)
    seq.set_definition("FOV", [fov, fov, slice_thickness])
    seq.set_definition("Name", str(protocol.get_parameter("Name")))
    seq.set_node("kernel", role="kernel", repeat_every="repetition_time", repeat_count="n_y", counter="ky_index", repeat_mode="expanded")

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
        gz_name="gz_ssel",
        gzr_name="gz_rephaser",
    )

    delta_k = 1.0 / fov
    gx = ppstar.make_trapezoid(channel="x", flat_area=n_x * delta_k, flat_time=float(protocol.get_parameter("readout_duration")), system=system, name="gx_readout")
    adc = ppstar.make_adc(num_samples=n_x, duration=gx.flat_time, delay=max(float(gx.rise_time), float(system.adc_dead_time)), system=system, name="adc_readout", role="acquisition")
    gx_pre = ppstar.make_trapezoid(channel="x", area=-gx.area / 2.0, duration=1e-3, system=system, name="gx_prephaser")
    gx_spoil = ppstar.make_trapezoid(channel="x", area=2.0 * n_x * delta_k, system=system, name="gx_spoil")
    gy_spoil = ppstar.make_trapezoid(channel="y", area=2.0 / (fov / n_y), duration=seq.calc_duration(gx_spoil), system=system, name="gy_spoil", role="spoiling")
    gz_spoil = ppstar.make_trapezoid(channel="z", area=4.0 / slice_thickness, system=system, name="gz_spoil")
    phase_areas = (np.arange(n_y, dtype=float) - n_y / 2.0) * delta_k

    # ==================
    # CONSTRUCT SEQUENCE
    # ==================
    rf_phase = 0.0
    rf_inc = 0.0
    rf_spoiling_inc = float(protocol.get_parameter("rf_spoiling_inc"))
    kernel_blocks: list[list[Any]] = []

    for ky_index, phase_area in enumerate(phase_areas):
        rf.phase_offset = rf_phase * math.pi / 180.0
        adc.phase_offset = rf_phase * math.pi / 180.0
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        excitation_block = seq.add_block(rf, gz, role="excitation", node="kernel.excitation", varies=["rf_phase"])
        gy_pre = ppstar.make_trapezoid(channel="y", area=float(phase_area), duration=seq.calc_duration(gx_pre), system=system, name=f"gy_pre_{ky_index:03d}", role="phase_encode")
        prephase_block = seq.add_block(gx_pre, gy_pre, gz_reph, role="prephase", node="kernel.prephase", varies=["ky_index"])
        te_fill = ppstar.make_delay(0.0, system=system, name=f"te_fill_{ky_index:03d}", role="te_fill")
        te_block = seq.add_block(te_fill, role="te_fill", node="kernel.te_fill")
        readout_block = seq.add_block(gx, adc, role="readout", node="kernel.readout", varies=["rf_phase"])
        tr_fill = ppstar.make_delay(0.0, system=system, name=f"tr_fill_{ky_index:03d}", role="tr_fill")
        spoiling_block = seq.add_block(tr_fill, gx_spoil, gy_spoil, gz_spoil, role="spoiling", node="kernel.spoiling", varies=["ky_index"])

        line_blocks = [excitation_block, prephase_block, te_block, readout_block, spoiling_block]
        kernel_blocks.append(line_blocks)

        # ====================
        # DEFINE RELATIONSHIPS
        # ====================
        ppstar.relationships.set_readout_after(
            seq=seq,
            adc=readout_block.get_event("adc_readout"),
            readout_gradient=readout_block.get_event("gx_readout"),
            reference=excitation_block.get_event("rf_excitation"),
            offset="echo_time",
            solve_event=te_block.get_event(f"te_fill_{ky_index:03d}"),
            solve_property="duration",
            name=f"gre_line_{ky_index:03d}_readout_after_rf",
        )
        ppstar.relationships.fill_to_period(
            seq=seq,
            blocks=line_blocks,
            period="repetition_time",
            fill_event=spoiling_block.get_event(f"tr_fill_{ky_index:03d}"),
            fill_property="duration",
            name=f"gre_line_{ky_index:03d}_fits_tr",
        )

    # =====================
    # RESOLVE AND VALIDATE
    # =====================
    ppstar.relationships.resolve(seq)
    ok, error_report = seq.check_timing()
    print("Timing check passed successfully" if ok else "Timing check failed")
    for error in error_report:
        print(error)

    first_line = kernel_blocks[0]
    first_rf = first_line[0].get_event("rf_excitation")
    first_adc = first_line[3].get_event("adc_readout")
    realized_te = ppstar.relationships.get_anchor_time(first_adc, "center", seq=seq, frame="global") - ppstar.relationships.get_anchor_time(first_rf, "center", seq=seq, frame="global")

    print(f"TE: {realized_te * 1e3:.3f} ms")
    print(f"TR: {seq.calc_timeline_duration(first_line) * 1e3:.3f} ms")
    print(f"Phase encodes: {n_y}")

    # ====
    # PLOT
    # ====
    if plot:
        seq.plot(one_tr=True, title="SeqStar GRE", gradient_scale="mt_per_m", debug=debug)

    # ======
    # EXPORT
    # ======
    output_dir = Path("out/gre")
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_seq:
        seq_path = PulseqWriter(seq).write(output_dir / seq_filename)
        print(f"Wrote Pulseq: {seq_path}")

    if write_json:
        json_path = GammaStarWriter(seq).write(output_dir / json_filename)
        print(f"Wrote gammaSTAR JSON: {json_path}")


if __name__ == "__main__":
    main()
