"""Create repeated RF pulse trains as both Pulseq and gammaSTAR JSON.

Beginner-facing example.

To choose the RF pulse family, edit the variable at the bottom of this file:

    pulse_type = "block"      # rectangular RF pulse
    pulse_type = "sinc"       # shaped sinc RF pulse
    pulse_type = "gauss"      # shaped Gaussian RF pulse
    pulse_type = "arbitrary"  # user-provided RF waveform samples
    pulse_type = "all"        # run all RF examples separately

Current gradient policy:
    This demo remains RF-shape focused.

    Slice-select gradients are now supported at the API level through
    make_sinc_pulse(..., slice_thickness=..., return_gz=True), etc., but this
    demo intentionally keeps RF-only examples so we can verify RF waveform
    fidelity independently.

Important design choice:
    The "all" option does NOT combine the different RF pulse families into one
    kernel. Instead, it runs four independent examples and writes four .seq
    files plus four .seq.json files.

    This mirrors the gradient-demo policy and avoids premature sequence-level
    relationship coupling.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter


SUPPORTED_RF_PULSES = {"block", "sinc", "gauss", "arbitrary", "all"}
INDIVIDUAL_RF_PULSES = ("block", "sinc", "gauss", "arbitrary")


def define_system_limits() -> ppstar.Opts:
    """Step 1: Define the system limits.

    This is intentionally kept as a small explicit function so beginners can
    see the scanner/system assumptions before building the sequence.
    """

    system = ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=50e-6,
        rf_raster_time=2e-6,
    )

    return system


def define_protocol_parameters(*, pulse_type: str) -> ppstar.Protocol:
    """Step 2: Define protocol parameters.

    This is intentionally kept as a small explicit function so beginners can
    see which knobs control each RF pulse family.
    """

    pulse_type = pulse_type.lower()

    if pulse_type not in INDIVIDUAL_RF_PULSES:
        raise ValueError(
            f"Unsupported individual pulse_type={pulse_type!r}. "
            f"Use one of: {sorted(INDIVIDUAL_RF_PULSES)}"
        )

    protocol_parameters = {
        "average": 20,
        "TR": 0.05,
        "rf_pulse_type": pulse_type,
        "flip_angle": 90.0,
    }

    if pulse_type == "block":
        protocol_parameters.update(
            {
                "rf_duration": 300e-6,
            }
        )

    elif pulse_type == "sinc":
        protocol_parameters.update(
            {
                "rf_duration": 3.0e-3,
                "time_bw_product": 4.0,
                "apodization": 0.5,
                "center_pos": 0.5,
            }
        )

    elif pulse_type == "gauss":
        protocol_parameters.update(
            {
                "rf_duration": 3.0e-3,
                "time_bw_product": 4.0,
                "apodization": 0.5,
                "center_pos": 0.5,
            }
        )

    elif pulse_type == "arbitrary":
        protocol_parameters.update(
            {
                "rf_duration": 3.0e-3,
                "arbitrary_num_samples": 256,
                "arbitrary_signal_label": "Hamming window",
            }
        )

    else:
        raise AssertionError("Unreachable pulse_type branch.")

    protocol = ppstar.Protocol(
        name=f"{pulse_type.capitalize()} RF train",
        description=f"A simple repeated {pulse_type} RF pulse train",
        parameters=protocol_parameters,
    )

    return protocol


def _arbitrary_rf_signal(*, num_samples: int) -> list[float]:
    """Return a beginner-friendly arbitrary RF envelope.

    The values are dimensionless envelope samples. The RF constructor scales
    them to the requested flip angle.
    """

    if num_samples < 2:
        raise ValueError("num_samples must be at least 2 for the arbitrary RF demo.")

    return [
        0.54 - 0.46 * math.cos(2.0 * math.pi * n / (num_samples - 1))
        for n in range(num_samples)
    ]


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    pulse_type: str,
) -> ppstar.Sequence:
    """Step 3: Build a repeated RF train with an explicit repetition relationship."""

    pulse_type = pulse_type.lower()

    if pulse_type not in INDIVIDUAL_RF_PULSES:
        raise ValueError(
            f"build_sequence only supports individual RF demos. "
            f"Passed pulse_type={pulse_type!r}. "
            f"Use one of: {sorted(INDIVIDUAL_RF_PULSES)}"
        )

    sequence_name = f"{pulse_type.capitalize()} RF train"

    # Step 3-1: Create the sequence object.
    seq = ppstar.Sequence(
        name=sequence_name,
        system=system,
        parameters=protocol.parameters,
    )

    # Step 3-2: Create the RF event.
    if pulse_type == "block":
        rf = ppstar.make_block_pulse(
            flip_angle=math.pi / 2,
            duration=float(protocol.get_parameter("rf_duration")),
            use="excitation",
            system=system,
            parameters=protocol.parameters,
        )

    elif pulse_type == "sinc":
        if not hasattr(ppstar, "make_sinc_pulse"):
            raise NotImplementedError(
                "ppstar.make_sinc_pulse is not available yet. "
                "Add it in make/rf.py and export it from pypulseq_star.__init__.py."
            )

        rf = ppstar.make_sinc_pulse(
            flip_angle=math.pi / 2,
            duration=float(protocol.get_parameter("rf_duration")),
            time_bw_product=float(protocol.get_parameter("time_bw_product")),
            apodization=float(protocol.get_parameter("apodization")),
            center_pos=float(protocol.get_parameter("center_pos")),
            use="excitation",
            system=system,
            parameters=protocol.parameters,
        )

    elif pulse_type == "gauss":
        if not hasattr(ppstar, "make_gauss_pulse"):
            raise NotImplementedError(
                "ppstar.make_gauss_pulse is not available yet. "
                "Add it in make/rf.py and export it from pypulseq_star.__init__.py."
            )

        rf = ppstar.make_gauss_pulse(
            flip_angle=math.pi / 2,
            duration=float(protocol.get_parameter("rf_duration")),
            time_bw_product=float(protocol.get_parameter("time_bw_product")),
            apodization=float(protocol.get_parameter("apodization")),
            center_pos=float(protocol.get_parameter("center_pos")),
            use="excitation",
            system=system,
            parameters=protocol.parameters,
        )

    elif pulse_type == "arbitrary":
        if not hasattr(ppstar, "make_arbitrary_rf"):
            raise NotImplementedError(
                "ppstar.make_arbitrary_rf is not available yet. "
                "Add it in make/rf.py and export it from pypulseq_star.__init__.py."
            )

        num_samples = int(protocol.get_parameter("arbitrary_num_samples"))
        arbitrary_signal = _arbitrary_rf_signal(num_samples=num_samples)

        rf = ppstar.make_arbitrary_rf(
            signal=arbitrary_signal,
            flip_angle=math.pi / 2,
            duration=float(protocol.get_parameter("rf_duration")),
            use="excitation",
            system=system,
            parameters=protocol.parameters,
        )

    else:
        raise AssertionError("Unreachable pulse_type branch.")

    # Step 3-3: Define repetitions and construct the repeated kernel.
    repetitions = int(protocol.get_parameter("average", 20))
    tr = float(protocol.get_parameter("TR", 0.05))

    seq.add_repeating_block(
        rf,
        repetitions=repetitions,
        tr=tr,
        role="kernel",
    )

    return seq


def write_artifacts(
    seq: ppstar.Sequence,
    *,
    out_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Step 6: Write Pulseq and gammaSTAR artifacts."""

    seq_path = PulseqWriter(seq).write(out_dir / f"{stem}.seq")
    json_path = GammaStarWriter(seq).write(out_dir / f"{stem}.seq.json")

    return seq_path, json_path


def validate_gammastar_graph(seq: ppstar.Sequence) -> None:
    """Step 7: Print a quick gammaSTAR hierarchy sanity check."""

    doc = GammaStarWriter(seq).to_dict()
    sequence_elements = doc["sequence_elements"]
    parameters = doc["parameters"]

    print("\n[gammaSTAR] sequence_elements:")
    for path, blueprint in sequence_elements.items():
        print(f"  {path}: {blueprint}")

    print("\n[gammaSTAR] kernel direct children:")
    prefix = "root.average.kernel."
    found_child = False

    for path in sequence_elements:
        if not path.startswith(prefix):
            continue

        suffix = path.removeprefix(prefix)
        if "." not in suffix:
            found_child = True
            print(f"  {path}: {sequence_elements[path]}")

    if not found_child:
        print("  NONE")

    print("\n[gammaSTAR] RF leaves:")
    found_rf = False

    for path, blueprint in sequence_elements.items():
        if blueprint == "RFPulse":
            found_rf = True
            print(f"  {path}")
            samples_path = f"{path}.samples"
            duration_path = f"{path}.duration"
            enabled_path = f"{path}.enabled"
            print(f"    samples:  {'yes' if samples_path in parameters else 'NO'}")
            print(f"    duration: {'yes' if duration_path in parameters else 'NO'}")
            print(f"    enabled:  {'yes' if enabled_path in parameters else 'NO'}")

    if not found_rf:
        print("  NONE")

    print("\n[gammaSTAR] protocol parameters:")
    for path in sorted(parameters):
        if path.startswith("root.prot."):
            print(f"  {path}")


def run_one_case(
    *,
    pulse_type: str,
    plot_sequence: bool,
    validate_graph: bool,
) -> tuple[Path, Path]:
    """Build, check, optionally plot, and write one RF demo case."""

    pulse_type = pulse_type.lower()

    if pulse_type not in INDIVIDUAL_RF_PULSES:
        raise ValueError(
            f"run_one_case supports only {INDIVIDUAL_RF_PULSES}. "
            f"Passed pulse_type={pulse_type!r}."
        )

    stem = f"{pulse_type}_rf_train"
    out_dir = Path(f"out/{stem}")

    print("\n" + "=" * 80)
    print(f"Running RF demo case: {pulse_type}")
    print("=" * 80)

    # Step 1: Define the system limits.
    system = define_system_limits()

    # Step 2: Define protocol parameters.
    protocol = define_protocol_parameters(pulse_type=pulse_type)

    # Step 3: Create the sequence object and events.
    seq = build_sequence(
        protocol=protocol,
        system=system,
        pulse_type=pulse_type,
    )

    # Step 4: Check timing, hierarchy, and relationships.
    ok, error_report = seq.check_timing()

    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        for error in error_report:
            print(error)

    # Step 5: Display the sequence.
    if plot_sequence:
        seq.plot(
            time_range=(
                0,
                2 * float(protocol.get_parameter("TR")),
            )
        )

    # Step 6: Write the sequence files.
    seq_path, json_path = write_artifacts(
        seq,
        out_dir=out_dir,
        stem=stem,
    )

    print(f"[100%] wrote Pulseq: {seq_path}")
    print(f"[100%] wrote gammaSTAR JSON: {json_path}")

    # Step 7: Validate gammaSTAR hierarchy.
    if validate_graph:
        validate_gammastar_graph(seq)

    return seq_path, json_path


def main(
    *,
    pulse_type: str = "gauss",
    plot_sequence: bool = True,
    validate_graph: bool = True,
) -> None:
    """Write RF train artifacts.

    If pulse_type == "all", run the four individual RF demos separately.
    """

    pulse_type = pulse_type.lower()

    if pulse_type not in SUPPORTED_RF_PULSES:
        raise ValueError(
            f"Unsupported pulse_type={pulse_type!r}. "
            f"Supported values are: {sorted(SUPPORTED_RF_PULSES)}"
        )

    if pulse_type == "all":
        written: list[tuple[str, Path, Path]] = []

        for individual_type in INDIVIDUAL_RF_PULSES:
            seq_path, json_path = run_one_case(
                pulse_type=individual_type,
                plot_sequence=plot_sequence,
                validate_graph=validate_graph,
            )
            written.append((individual_type, seq_path, json_path))

        print("\n" + "=" * 80)
        print("Finished all independent RF demo cases")
        print("=" * 80)

        for individual_type, seq_path, json_path in written:
            print(f"{individual_type}:")
            print(f"  Pulseq:         {seq_path}")
            print(f"  gammaSTAR JSON: {json_path}")

        return

    run_one_case(
        pulse_type=pulse_type,
        plot_sequence=plot_sequence,
        validate_graph=validate_graph,
    )


if __name__ == "__main__":
    # Options:
    #   "block"      = rectangular RF pulse
    #   "sinc"       = sinc RF pulse
    #   "gauss"      = Gaussian RF pulse
    #   "arbitrary"  = user-provided RF envelope samples
    #   "all"        = run block, sinc, gauss, and arbitrary as separate demos
    pulse_type = "all"

    plot = True
    validate_graph = True

    main(
        pulse_type=pulse_type,
        plot_sequence=plot,
        validate_graph=validate_graph,
    )