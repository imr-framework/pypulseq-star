# 1. Replace the PyPulseq import
from pathlib import Path

import numpy as np

import pypulseq_star as ppstar
from pypulseq_star.migration import MigrationValidator
from pypulseq_star.sequence import vary
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


def main(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    write_json: bool = False,
    validate_migration: bool = False,
    seq_filename: str = "gre_pypulseq_star.seq",
    json_filename: str = "gre_pypulseq_star.seq.json",
    *,
    fov: float | tuple[float, float] = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    flip_angle_deg: float = 10,
    slice_thickness: float = 3e-3,
    tr: float = 12e-3,
    te: float = 5e-3,
):
    """Create a basic gradient echo (GRE) sequence.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print a test report. Default is False.
    write_seq : bool, optional
        Write the sequence to a .seq file. Default is False.
    write_json : bool, optional
        Write the sequence to a gammaSTAR .seq.json file. Default is False.
    validate_migration : bool, optional
        Run the PyPulseq-to-PyPulseq-Star migration validator. Default is False.
    seq_filename : str, optional
        Output filename for the .seq file. Default is "gre_pypulseq_star.seq".
    json_filename : str, optional
        Output filename for the .seq.json file. Default is "gre_pypulseq_star.seq.json".
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    flip_angle_deg : float, optional
        Flip angle in degrees. Default is 10.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    tr : float, optional
        Repetition time in seconds. Default is 12e-3.
    te : float, optional
        Echo time in seconds. Default is 5e-3.

    Returns
    -------
    seq : pypulseq_star.Sequence
        The symbolic GRE sequence object.
    """
    # 2. Add a protocol alongside the existing system definition
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    system = ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    protocol = ppstar.Protocol(
        name="gre",
        parameters={
            "fov_x": fov_x,
            "fov_y": fov_y,
            "n_x": n_x,
            "n_y": n_y,
            "flip_angle_deg": flip_angle_deg,
            "slice_thickness": slice_thickness,
            "tr": tr,
            "te": te,
        },
    )

    p = protocol.symbols
    seq = ppstar.Sequence(system=system, protocol=protocol, name="gre")

    # 3. Migrate event construction
    rf_spoiling_inc = 117

    # Create slice selection pulse and gradient
    rf, gz, _ = ppstar.make_sinc_pulse(
        flip_angle=p.flip_angle_deg * np.pi / 180,
        duration=3e-3,
        slice_thickness=p.slice_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use="excitation",
    )

    # Define other gradients and ADC events
    delta_kx = 1 / p.fov_x
    gx = ppstar.make_trapezoid(
        channel="x", flat_area=p.n_x * delta_kx, flat_time=3.2e-3, system=system
    )
    adc = ppstar.make_adc(
        num_samples=p.n_x, duration=gx.flat_time, delay=gx.rise_time, system=system
    )
    gx_pre = ppstar.make_trapezoid(
        channel="x", area=-gx.area / 2, duration=1e-3, system=system
    )
    gz_reph = ppstar.make_trapezoid(
        channel="z", area=-gz.area / 2, duration=1e-3, system=system
    )

    # Gradient spoiling
    gx_spoil = ppstar.make_trapezoid(
        channel="x", area=2 * p.n_x * delta_kx, system=system
    )
    gz_spoil = ppstar.make_trapezoid(
        channel="z", area=4 / p.slice_thickness, system=system
    )

    # 4. Retain TE and TR as timing relationships
    excitation_tail = seq.duration([rf, gz]) - rf.anchor("center")
    prephase_duration = seq.duration([gx_pre])
    readout_to_echo = adc.anchor("center")

    te_delay = p.te - excitation_tail - prephase_duration - readout_to_echo
    te_fill = ppstar.make_delay(te_delay, system=system)

    tr_delay = (
        p.tr
        - seq.duration([rf, gz])
        - prephase_duration
        - seq.duration([gx])
        - te_delay
    )
    tr_fill = ppstar.make_delay(tr_delay, system=system)

    # 5. Replace the explicit phase-encoding loop with retained variations
    kernel = seq.set_node(
        "kernel",
        factor=p.n_y,
        repeat_every=p.tr,
        counter="ky_index",
        repeat_mode="loop",
    )

    phase_encode_start = -p.n_y / (2 * p.fov_y)
    phase_encode_step = 1 / p.fov_y

    gy_pre = ppstar.make_trapezoid(
        channel="y", area=1.0, duration=seq.duration([gx_pre]), system=system
    )
    gy_reph = ppstar.make_trapezoid(
        channel="y", area=-1.0, duration=seq.duration([gx_pre]), system=system
    )

    kernel.vary(
        vary(gy_pre, "area", strength=phase_encode_start, step=phase_encode_step),
        vary(gy_reph, "area", strength=-phase_encode_start, step=-phase_encode_step),
        vary(
            [rf, adc],
            "phase_offset",
            strength=0.0,
            step=rf_spoiling_inc * np.pi / 180,
            mode="accumulated",
            wrap=2 * np.pi,
        ),
    )

    # Add one representative GRE kernel. The logical node owns these blocks,
    # allowing the plotter and backend writers to materialize all repetitions.
    seq.add_block(rf, gz, node="kernel.excitation")
    seq.add_block(gx_pre, gy_pre, gz_reph, node="kernel.prephase")
    seq.add_block(te_fill, node="kernel.echo_delay")
    seq.add_block(gx, adc, node="kernel.readout")
    seq.add_block(tr_fill, gx_spoil, gy_reph,gz_spoil,node="kernel.spoiling")

    # Preserve the same sequence definitions used by the PyPulseq reference.
    seq.set_definition("FOV", [p.fov_x, p.fov_y, p.slice_thickness])
    seq.set_definition("Name", "gre")

    # 6. Resolve, validate, plot, and export
    resolved = seq.resolve()

    ok, error_report = resolved.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        for error in error_report:
            print(error)

    if test_report:
        print(resolved.test_report())

    if plot:
        seq.plot(realization=resolved, time_range=(0, 4 * resolved.protocol.repetition_time))

    if write_seq:
        PulseqWriter(seq).write(seq_filename, realization=resolved)

    if write_json:
        GammaStarWriter(seq).write(json_filename, defaults=resolved)
    if validate_migration:
        source_path = Path(__file__).resolve().parents[2] / "src/pypulseq_star/migration/reference_scripts/write_gre.py"
        report = MigrationValidator(
            source_path=source_path,
            migrated_sequence=seq,
            migrated_realization=resolved,
        ).validate()
        print(report.to_text())

    return seq


if __name__ == "__main__":
    output_dir = Path("out/gre_pp_to_ppstar")
    seq_filename = output_dir / "gre_pp_to_ppstar.seq"
    json_filename = output_dir / "gre_pp_to_ppstar.seq.json"
    main(plot=True, write_seq=True, write_json=True, validate_migration=True,
         seq_filename=seq_filename, json_filename=json_filename,)