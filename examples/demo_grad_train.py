"""Create gradient waveform examples as both Pulseq and gammaSTAR JSON.

Beginner-facing example.

To choose the gradient family, edit the variable at the bottom of this file:

    grad_type = "trapezoid"  # trapezoidal gradient pulse
    grad_type = "arbitrary"  # user-provided arbitrary waveform samples
    grad_type = "split"      # trapezoid split into ramp-up / flat-top / ramp-down
    grad_type = "all"        # run the three cases separately

Current relationship policy:
    Gradient-only examples are supported here.
    RF/ADC/gradient relationships will be added later through clean sequence
    hierarchy and relationship modules. This demo is focused only on waveform
    fidelity in Pulseq and gammaSTAR export.

Important design choice:
    The "all" option does NOT combine the different gradient families into one
    kernel. Instead, it runs three independent examples and writes three .seq
    files plus three .seq.json files. This avoids premature sequence/relationship
    coupling and avoids forcing unrelated gradient objects into one Pulseq block.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

SUPPORTED_GRADIENT_TYPES = {"trapezoid", "arbitrary", "split", "all"}
INDIVIDUAL_GRADIENT_TYPES = ("trapezoid", "arbitrary", "split")


def define_system_limits() -> ppstar.Opts:
    """Step 1: Define the system limits.

    This is intentionally kept as a small explicit function so beginners can
    see the scanner/system assumptions before building the sequence.
    """

    system = ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=120,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=500e-6,
        adc_dead_time=50e-6,
        grad_raster_time=10e-6,
        rf_raster_time=2e-6,
    )

    return system


def define_protocol_parameters(*, grad_type: str) -> ppstar.Protocol:
    """Step 2: Define protocol parameters.

    This mirrors ``demo_rf_train.py``: keep protocol knobs explicit and grouped
    by gradient family so the demo is easy to edit.
    """

    grad_type = grad_type.lower()

    protocol_parameters = {
        "average": 12,
        "TR": 0.04,
        "gradient_type": grad_type,
    }

    if grad_type == "trapezoid":
        protocol_parameters.update(
            {
                "channel": "x",
                "gradient_area": 320.0,
                "gradient_duration": 3.0e-3,
                "gradient_delay": 0.2e-3,
                "gradient_role": "readout",
            }
        )

    elif grad_type == "arbitrary":
        protocol_parameters.update(
            {
                "channel": "y",
                "gradient_duration": 4.0e-3,
                "gradient_delay": 0.4e-3,
                "arbitrary_num_samples": 400,
                "arbitrary_peak_amplitude": 0.010,
                "arbitrary_signal_label": "smooth bipolar sinusoid",
                "gradient_role": "phase_encode",
            }
        )

    elif grad_type == "split":
        protocol_parameters.update(
            {
                "channel": "z",
                "gradient_area": 180.0,
                "gradient_duration": 4.0e-3,
                "gradient_delay": 0.2e-3,
                "gradient_role": "slice_rephase",
            }
        )

    else:
        raise ValueError(
            f"Unsupported individual grad_type={grad_type!r}. "
            f"Use one of: {sorted(INDIVIDUAL_GRADIENT_TYPES)}"
        )

    protocol = ppstar.Protocol(
        name=f"{grad_type.capitalize()} gradient train",
        description=f"A simple repeated {grad_type} gradient waveform train",
        parameters=protocol_parameters,
    )

    return protocol


def _arbitrary_waveform(
    *,
    num_samples: int,
    peak_amplitude: float,
) -> list[float]:
    """Return a smooth bipolar waveform for arbitrary-gradient testing."""

    if num_samples < 4:
        raise ValueError("num_samples must be at least 4 for the arbitrary gradient demo.")

    values: list[float] = []

    for n in range(num_samples):
        u = n / (num_samples - 1)
        envelope = math.sin(math.pi * u) ** 2
        bipolar = math.sin(2.0 * math.pi * u)
        values.append(float(peak_amplitude * envelope * bipolar))

    return values


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    grad_type: str,
) -> ppstar.Sequence:
    """Step 3: Build a repeated gradient train."""

    grad_type = grad_type.lower()

    if grad_type not in INDIVIDUAL_GRADIENT_TYPES:
        raise ValueError(
            f"build_sequence only supports individual gradient demos. "
            f"Passed grad_type={grad_type!r}. "
            f"Use one of: {sorted(INDIVIDUAL_GRADIENT_TYPES)}"
        )

    sequence_name = f"{grad_type.capitalize()} gradient train"

    # Step 3-1: Create the sequence object.
    seq = ppstar.Sequence(
        name=sequence_name,
        system=system,
        parameters=protocol.parameters,
    )

    # Step 3-2: Create the gradient event(s).
    gradient_events = []

    if grad_type == "trapezoid":
        grad = ppstar.make_trapezoid(
            channel=str(protocol.get_parameter("channel")),
            area=float(protocol.get_parameter("gradient_area")),
            duration=float(protocol.get_parameter("gradient_duration")),
            delay=float(protocol.get_parameter("gradient_delay")),
            system=system,
            name="trap_readout",
            role=str(protocol.get_parameter("gradient_role")),
            axis_role="frequency_encode",
            metadata={"demo_case": "trapezoid"},
        )
        gradient_events.append(grad)

    elif grad_type == "arbitrary":
        if not hasattr(ppstar, "make_arbitrary_grad"):
            raise NotImplementedError(
                "ppstar.make_arbitrary_grad is not available yet. "
                "Add it in make/grad.py and export it from pypulseq_star.__init__.py."
            )

        num_samples = int(protocol.get_parameter("arbitrary_num_samples"))
        waveform = _arbitrary_waveform(
            num_samples=num_samples,
            peak_amplitude=float(protocol.get_parameter("arbitrary_peak_amplitude")),
        )

        grad = ppstar.make_arbitrary_grad(
            channel=str(protocol.get_parameter("channel")),
            waveform=waveform,
            delay=float(protocol.get_parameter("gradient_delay")),
            system=system,
            name="arbitrary_bipolar",
            role=str(protocol.get_parameter("gradient_role")),
            axis_role="phase_encode",
            metadata={"demo_case": "arbitrary"},
        )
        gradient_events.append(grad)

    elif grad_type == "split":
        if not hasattr(ppstar, "split_gradient"):
            raise NotImplementedError(
                "ppstar.split_gradient is not available yet. "
                "Add it in make/grad.py and export it from pypulseq_star.__init__.py."
            )

        parent = ppstar.make_trapezoid(
            channel=str(protocol.get_parameter("channel")),
            area=float(protocol.get_parameter("gradient_area")),
            duration=float(protocol.get_parameter("gradient_duration")),
            delay=float(protocol.get_parameter("gradient_delay")),
            system=system,
            name="split_parent",
            role=str(protocol.get_parameter("gradient_role")),
            axis_role="slice_encode",
            metadata={"demo_case": "split_parent"},
        )

        ramp_up, flat_top, ramp_down = ppstar.split_gradient(
            parent,
            system=system,
            name="split_gradient",
            metadata={"demo_case": "split"},
        )

        # Keep split-gradient pieces in the same SeqStar repeated kernel for
        # semantic inspection. The Pulseq writer may lower same-channel pieces
        # into sequential Pulseq blocks, which is correct for .seq export.
        gradient_events.extend([ramp_up, flat_top, ramp_down])

    else:
        raise AssertionError("Unreachable grad_type branch.")

    # Step 3-3: Define repetitions and construct the repeated kernel.
    repetitions = int(protocol.get_parameter("average", 12))
    tr = float(protocol.get_parameter("TR", 0.04))

    seq.add_repeating_block(
        *gradient_events,
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

    print("\n[gammaSTAR] gradient direct children:")
    prefix = "root.average.kernel."
    found_child = False

    for path in sequence_elements:
        if not path.startswith(prefix):
            continue

        suffix = path.removeprefix(prefix)
        if "." not in suffix and path != "root.average.kernel":
            found_child = True
            print(f"  {path}: {sequence_elements[path]}")

    if not found_child:
        print("  NONE")

    print("\n[gammaSTAR] GradPulse leaves:")
    found_grad = False

    for path, blueprint in sequence_elements.items():
        if blueprint == "GradPulse":
            found_grad = True
            print(f"  {path}")
            samples_path = f"{path}.samples"
            direction_path = f"{path}.direction"
            print(f"    samples:   {'yes' if samples_path in parameters else 'NO'}")
            print(f"    direction: {'yes' if direction_path in parameters else 'NO'}")

    if not found_grad:
        print("  NONE")

    print("\n[gammaSTAR] protocol parameters:")
    for path in sorted(parameters):
        if path.startswith("root.prot."):
            print(f"  {path}")


def run_one_case(
    *,
    grad_type: str,
    plot_sequence: bool,
    validate_graph: bool,
) -> tuple[Path, Path]:
    """Build, check, optionally plot, and write one gradient demo case."""

    grad_type = grad_type.lower()

    if grad_type not in INDIVIDUAL_GRADIENT_TYPES:
        raise ValueError(
            f"run_one_case supports only {INDIVIDUAL_GRADIENT_TYPES}. "
            f"Passed grad_type={grad_type!r}."
        )

    stem = f"{grad_type}_grad_train"
    out_dir = Path(f"out/{stem}")

    print("\n" + "=" * 80)
    print(f"Running gradient demo case: {grad_type}")
    print("=" * 80)

    # Step 1: Define the system limits.
    system = define_system_limits()

    # Step 2: Define protocol parameters.
    protocol = define_protocol_parameters(grad_type=grad_type)

    # Step 3: Create the sequence object and events.
    seq = build_sequence(
        protocol=protocol,
        system=system,
        grad_type=grad_type,
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
                min(2 * float(protocol.get_parameter("TR")), 0.1),
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
    grad_type: str = "trapezoid",
    plot_sequence: bool = True,
    validate_graph: bool = True,
) -> None:
    """Write the gradient train artifacts.

    If grad_type == "all", run the three individual demos separately.
    """

    grad_type = grad_type.lower()

    if grad_type not in SUPPORTED_GRADIENT_TYPES:
        raise ValueError(
            f"Unsupported grad_type={grad_type!r}. "
            f"Supported values are: {sorted(SUPPORTED_GRADIENT_TYPES)}"
        )

    if grad_type == "all":
        written: list[tuple[str, Path, Path]] = []

        for individual_type in INDIVIDUAL_GRADIENT_TYPES:
            seq_path, json_path = run_one_case(
                grad_type=individual_type,
                plot_sequence=plot_sequence,
                validate_graph=validate_graph,
            )
            written.append((individual_type, seq_path, json_path))

        print("\n" + "=" * 80)
        print("Finished all independent gradient demo cases")
        print("=" * 80)

        for individual_type, seq_path, json_path in written:
            print(f"{individual_type}:")
            print(f"  Pulseq:         {seq_path}")
            print(f"  gammaSTAR JSON: {json_path}")

        return

    run_one_case(
        grad_type=grad_type,
        plot_sequence=plot_sequence,
        validate_graph=validate_graph,
    )


if __name__ == "__main__":
    # Options:
    #   "trapezoid" = trapezoidal gradient pulse
    #   "arbitrary" = arbitrary waveform gradient pulse
    #   "split"     = trapezoid split into ramp-up / flat-top / ramp-down
    #   "all"       = run trapezoid, arbitrary, and split as separate demos
    grad_type = "all"

    plot = True
    validate_graph = True

    main(
        grad_type=grad_type,
        plot_sequence=plot,
        validate_graph=validate_graph,
    )