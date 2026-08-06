"""Create ADC train examples as both Pulseq and gammaSTAR JSON.

Beginner/developer-facing ADC example.

This script is intentionally written before ADC code generation so it can guide
the API design for:

    ppstar.make_adc(...)
    ppstar.make_adc_train(...)

Design principle:
    PyPulseq-compatible at the call site.
    Train-native internally.
    gammaSTAR-editable at the graph level.
    Single-echo ADC is treated as the simplest ADC train.

Current gradient policy:
    Gradients are not required here.
    ADC schedules can be built and plotted first.
    Readout gradients will be added later through clean gradient modules.

To choose an ADC use case, edit the variable at the bottom of this file:

    adc_use_case = "single"
    adc_use_case = "multi_echo_gre"
    adc_use_case = "epi"
    adc_use_case = "segmented_epi"
    adc_use_case = "rare"
    adc_use_case = "grase"
    adc_use_case = "radial"
    adc_use_case = "spiral"
    adc_use_case = "navigator"
    adc_use_case = "calibration"
    adc_use_case = "asymmetric_echo"
    adc_use_case = "custom"
    adc_use_case = "all"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pypulseq_star as ppstar
from pypulseq_star.writers import GammaStarWriter, PulseqWriter

SUPPORTED_ADC_USE_CASES = {
    "single",
    "multi_echo_gre",
    "epi",
    "segmented_epi",
    "rare",
    "grase",
    "radial",
    "spiral",
    "spectroscopy",
    "navigator",
    "calibration",
    "asymmetric_echo",
    "custom",
}


def define_system_limits() -> ppstar.Opts:
    """Step 1: Define the system limits.

    Keep this explicit so beginners can see scanner/frontend assumptions before
    constructing the ADC train.
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
        grad_raster_time=10e-6,
    )

    return system


def define_protocol_parameters(*, adc_use_case: str) -> ppstar.Protocol:
    """Step 2: Define protocol parameters for one ADC use case."""

    adc_use_case = adc_use_case.lower()

    if adc_use_case not in SUPPORTED_ADC_USE_CASES:
        raise ValueError(
            f"Unsupported adc_use_case={adc_use_case!r}. "
            f"Supported values are: {sorted(SUPPORTED_ADC_USE_CASES)}"
        )

    base_parameters: dict[str, Any] = {
        "adc_use_case": adc_use_case,
        "TR": 0.100,
        "num_samples": 256,
        "dwell": 4e-6,
        "adc_dead_time_policy": "train_level_once",
        "window_guard_time": 0.0,
    }

    if adc_use_case == "single":
        base_parameters.update(
            {
                "description": "Single ADC window, equivalent to PyPulseq make_adc.",
                "first_delay": 2.0e-3,
                "num_echoes": 1,
                "mode": "single",
            }
        )

    elif adc_use_case == "multi_echo_gre":
        base_parameters.update(
            {
                "description": "Multi-echo GRE or field-map style readout.",
                "echo_times": [4.0e-3, 7.0e-3, 10.0e-3],
                "mode": "discrete",
            }
        )

    elif adc_use_case == "epi":
        base_parameters.update(
            {
                "description": "Continuous EPI-like ADC train with alternating polarity.",
                "num_samples": 128,
                "dwell": 4e-6,
                "num_echoes": 64,
                "first_delay": 2.0e-3,
                "echo_spacing": 640e-6,
                "mode": "continuous",
                "polarity": "alternating",
                "trajectory": "cartesian_epi",
            }
        )

    elif adc_use_case == "segmented_epi":
        base_parameters.update(
            {
                "description": "Segmented or multi-shot EPI ADC train.",
                "num_samples": 128,
                "dwell": 4e-6,
                "num_shots": 4,
                "lines_per_shot": 16,
                "first_delay": 2.0e-3,
                "echo_spacing": 720e-6,
                "shot_spacing": 25.0e-3,
                "mode": "continuous",
                "polarity": "alternating",
                "trajectory": "segmented_epi",
            }
        )

    elif adc_use_case == "rare":
        base_parameters.update(
            {
                "description": "RARE/TSE-like discrete echo train.",
                "num_samples": 256,
                "dwell": 4e-6,
                "num_echoes": 12,
                "first_delay": 6.0e-3,
                "echo_spacing": 8.0e-3,
                "mode": "discrete",
                "trajectory": "rare_tse",
            }
        )

    elif adc_use_case == "grase":
        base_parameters.update(
            {
                "description": "GRASE-like grouped EPI readouts around spin echoes.",
                "num_samples": 128,
                "dwell": 4e-6,
                "num_spin_echoes": 4,
                "gradient_echoes_per_spin_echo": 5,
                "first_delay": 5.0e-3,
                "spin_echo_spacing": 12.0e-3,
                "gradient_echo_spacing": 700e-6,
                "mode": "grouped",
                "polarity": "alternating",
                "trajectory": "grase",
            }
        )

    elif adc_use_case == "radial":
        base_parameters.update(
            {
                "description": "Radial readout ADC windows with spoke metadata.",
                "num_samples": 384,
                "dwell": 3e-6,
                "num_spokes": 16,
                "first_delay": 2.0e-3,
                "spoke_spacing": 3.5e-3,
                "mode": "discrete",
                "trajectory": "radial",
            }
        )

    elif adc_use_case == "spiral":
        base_parameters.update(
            {
                "description": "Single long spiral readout ADC window.",
                "num_samples": 2048,
                "dwell": 2e-6,
                "first_delay": 2.0e-3,
                "mode": "single",
                "trajectory": "spiral",
            }
        )

    elif adc_use_case == "spectroscopy":
        base_parameters.update(
            {
                "description": "SVS/MRS-like spectroscopy ADC readout.",
                "num_samples": 2048,
                "dwell": 500e-6,
                "first_delay": 30.0e-3,
                "mode": "single",
                "trajectory": "spectroscopy",
                "role": "spectroscopy",
                "SV_method": "PRESS",
                "frequency_shift_ppm": 0.0,
                "receiver_phase_rad": 0.0,
                "center_sample": 1024,
                "spectral_bandwidth_hz": 1.0 / 500e-6,
            }
        )


    elif adc_use_case == "navigator":
        base_parameters.update(
            {
                "description": "Imaging ADC windows plus navigator windows.",
                "num_samples": 256,
                "navigator_num_samples": 64,
                "dwell": 4e-6,
                "first_delay": 2.0e-3,
                "mode": "mixed_roles",
                "trajectory": "cartesian_with_navigators",
            }
        )

    elif adc_use_case == "calibration":
        base_parameters.update(
            {
                "description": "EPI/parallel-imaging reference and calibration ADC lines.",
                "num_samples": 128,
                "dwell": 4e-6,
                "num_imaging_lines": 8,
                "num_reference_lines": 4,
                "first_delay": 2.0e-3,
                "echo_spacing": 750e-6,
                "mode": "calibration_reference",
                "trajectory": "epi_reference",
            }
        )

    elif adc_use_case == "asymmetric_echo":
        base_parameters.update(
            {
                "description": "Partial-Fourier/asymmetric echo readout.",
                "num_samples": 256,
                "dwell": 4e-6,
                "first_delay": 2.0e-3,
                "center_sample": 96,
                "mode": "single",
                "trajectory": "cartesian_asymmetric_echo",
            }
        )

    elif adc_use_case == "custom":
        base_parameters.update(
            {
                "description": "Explicit irregular ADC windows for custom sequence design.",
                "mode": "explicit_windows",
                "trajectory": "custom",
            }
        )

    return ppstar.Protocol(
        name=f"ADC train demo: {adc_use_case}",
        description=base_parameters["description"],
        parameters=base_parameters,
    )


def build_sequence(
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    *,
    adc_use_case: str,
) -> ppstar.Sequence:
    """Step 3: Create the sequence object and ADC train event."""

    adc_use_case = adc_use_case.lower()

    seq = ppstar.Sequence(
        name=f"ADC train demo: {adc_use_case}",
        system=system,
        parameters=protocol.parameters,
    )

    adc = make_adc_for_use_case(
        system=system,
        protocol=protocol,
        adc_use_case=adc_use_case,
    )

    add_event_block(
        seq,
        adc,
        role="readout",
    )

    return seq


def make_adc_for_use_case(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
    adc_use_case: str,
):
    """Create one ADC object for a selected use case.

    This function intentionally assumes the future ADC implementation has:

        ppstar.make_adc(...)
        ppstar.make_adc_train(...)

    where make_adc(...) is PyPulseq-compatible and make_adc_train(...) is the
    preferred SeqStar train-native constructor.
    """

    if adc_use_case == "single":
        return make_single_echo_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "multi_echo_gre":
        return make_multi_echo_gre_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "epi":
        return make_epi_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "segmented_epi":
        return make_segmented_epi_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "rare":
        return make_rare_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "grase":
        return make_grase_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "radial":
        return make_radial_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "spiral":
        return make_spiral_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "spectroscopy":
        return make_spectroscopy_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "navigator":
        return make_navigator_adc(
            system=system,
            protocol=protocol,
        )
    if adc_use_case == "calibration":
        return make_calibration_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "asymmetric_echo":
        return make_asymmetric_echo_adc(
            system=system,
            protocol=protocol,
        )

    if adc_use_case == "custom":
        return make_custom_adc(
            system=system,
            protocol=protocol,
        )

    raise ValueError(
        f"Unsupported adc_use_case={adc_use_case!r}. "
        f"Supported values are: {sorted(SUPPORTED_ADC_USE_CASES)}"
    )


def make_single_echo_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 1: single-echo ADC, close to PyPulseq make_adc syntax."""

    if not hasattr(ppstar, "make_adc"):
        raise NotImplementedError(
            "ppstar.make_adc is not available yet. "
            "Add it in make/adc.py and export it from pypulseq_star.__init__.py."
        )

    return ppstar.make_adc(
        num_samples=int(protocol.get_parameter("num_samples")),
        dwell=float(protocol.get_parameter("dwell")),
        delay=float(protocol.get_parameter("first_delay")),
        freq_offset=0.0,
        phase_offset=0.0,
        system=system,
        parameters=protocol.parameters,
    )


def make_multi_echo_gre_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 2: multi-echo GRE / field-map style discrete ADC windows."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    echo_times = list(protocol.get_parameter("echo_times"))

    windows = []

    for echo_index, echo_time in enumerate(echo_times):
        windows.append(
            {
                "label": f"echo_{echo_index + 1:02d}",
                "role": "imaging",
                "delay": float(echo_time),
                "num_samples": num_samples,
                "dwell": dwell,
                "echo_index": echo_index,
                "echo_time": float(echo_time),
                "trajectory": "cartesian",
            }
        )

    return ppstar.make_adc_train(
        windows=windows,
        mode="discrete",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_epi_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 3: continuous EPI-like ADC train with alternating polarity."""

    ensure_adc_train_exists()

    return ppstar.make_adc_train(
        num_samples=int(protocol.get_parameter("num_samples")),
        dwell=float(protocol.get_parameter("dwell")),
        num_echoes=int(protocol.get_parameter("num_echoes")),
        first_delay=float(protocol.get_parameter("first_delay")),
        echo_spacing=float(protocol.get_parameter("echo_spacing")),
        mode=str(protocol.get_parameter("mode")),
        polarity=str(protocol.get_parameter("polarity")),
        trajectory=str(protocol.get_parameter("trajectory")),
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_segmented_epi_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 4: segmented or multi-shot EPI ADC train."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    num_shots = int(protocol.get_parameter("num_shots"))
    lines_per_shot = int(protocol.get_parameter("lines_per_shot"))
    first_delay = float(protocol.get_parameter("first_delay"))
    echo_spacing = float(protocol.get_parameter("echo_spacing"))
    shot_spacing = float(protocol.get_parameter("shot_spacing"))

    windows = []

    for shot_index in range(num_shots):
        shot_start = first_delay + shot_index * shot_spacing

        for line_in_shot in range(lines_per_shot):
            global_line_index = shot_index * lines_per_shot + line_in_shot
            polarity = 1 if line_in_shot % 2 == 0 else -1

            windows.append(
                {
                    "label": f"shot_{shot_index:02d}_line_{line_in_shot:03d}",
                    "role": "imaging",
                    "delay": shot_start + line_in_shot * echo_spacing,
                    "num_samples": num_samples,
                    "dwell": dwell,
                    "shot_index": shot_index,
                    "segment_index": shot_index,
                    "line_index": global_line_index,
                    "line_index_in_shot": line_in_shot,
                    "polarity": polarity,
                    "trajectory": "segmented_epi",
                }
            )

    return ppstar.make_adc_train(
        windows=windows,
        mode="continuous",
        polarity="alternating",
        trajectory="segmented_epi",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_rare_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 5: RARE/TSE-like discrete ADC windows."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    num_echoes = int(protocol.get_parameter("num_echoes"))
    first_delay = float(protocol.get_parameter("first_delay"))
    echo_spacing = float(protocol.get_parameter("echo_spacing"))

    windows = []

    for echo_index in range(num_echoes):
        windows.append(
            {
                "label": f"rare_echo_{echo_index + 1:02d}",
                "role": "imaging",
                "delay": first_delay + echo_index * echo_spacing,
                "num_samples": num_samples,
                "dwell": dwell,
                "echo_index": echo_index,
                "spin_echo_index": echo_index,
                "phase_encode_index": echo_index,
                "trajectory": "rare_tse",
                "placement_hint": "between_refocusing_rf_pulses",
            }
        )

    return ppstar.make_adc_train(
        windows=windows,
        mode="discrete",
        trajectory="rare_tse",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_grase_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 6: GRASE-like grouped spin-echo and gradient-echo ADC windows."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    num_spin_echoes = int(protocol.get_parameter("num_spin_echoes"))
    gradient_echoes_per_spin_echo = int(
        protocol.get_parameter("gradient_echoes_per_spin_echo")
    )
    first_delay = float(protocol.get_parameter("first_delay"))
    spin_echo_spacing = float(protocol.get_parameter("spin_echo_spacing"))
    gradient_echo_spacing = float(protocol.get_parameter("gradient_echo_spacing"))

    windows = []

    for spin_echo_index in range(num_spin_echoes):
        spin_echo_center = first_delay + spin_echo_index * spin_echo_spacing
        mid_gradient_echo = (gradient_echoes_per_spin_echo - 1) / 2.0

        for gradient_echo_index in range(gradient_echoes_per_spin_echo):
            relative_index = gradient_echo_index - mid_gradient_echo
            delay = spin_echo_center + relative_index * gradient_echo_spacing
            polarity = 1 if gradient_echo_index % 2 == 0 else -1

            windows.append(
                {
                    "label": (
                        f"grase_se_{spin_echo_index:02d}_"
                        f"ge_{gradient_echo_index:02d}"
                    ),
                    "role": "imaging",
                    "delay": delay,
                    "num_samples": num_samples,
                    "dwell": dwell,
                    "spin_echo_index": spin_echo_index,
                    "gradient_echo_index": gradient_echo_index,
                    "echo_index": len(windows),
                    "polarity": polarity,
                    "trajectory": "grase",
                }
            )

    return ppstar.make_adc_train(
        windows=windows,
        mode="grouped",
        polarity="alternating",
        trajectory="grase",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_radial_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 7: radial readout ADC windows with spoke metadata."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    num_spokes = int(protocol.get_parameter("num_spokes"))
    first_delay = float(protocol.get_parameter("first_delay"))
    spoke_spacing = float(protocol.get_parameter("spoke_spacing"))

    windows = []

    for spoke_index in range(num_spokes):
        angle_rad = spoke_index * math.pi / num_spokes

        windows.append(
            {
                "label": f"spoke_{spoke_index:03d}",
                "role": "imaging",
                "delay": first_delay + spoke_index * spoke_spacing,
                "num_samples": num_samples,
                "dwell": dwell,
                "spoke_index": spoke_index,
                "projection_angle_rad": angle_rad,
                "projection_angle_deg": math.degrees(angle_rad),
                "trajectory": "radial",
            }
        )

    return ppstar.make_adc_train(
        windows=windows,
        mode="discrete",
        trajectory="radial",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_spiral_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 8: single long spiral ADC window."""

    ensure_adc_train_exists()

    return ppstar.make_adc_train(
        windows=[
            {
                "label": "spiral_readout",
                "role": "imaging",
                "delay": float(protocol.get_parameter("first_delay")),
                "num_samples": int(protocol.get_parameter("num_samples")),
                "dwell": float(protocol.get_parameter("dwell")),
                "trajectory": "spiral",
                "center_sample": 0,
                "readout_type": "long_non_cartesian",
            }
        ],
        mode="single",
        trajectory="spiral",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_spectroscopy_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 9: SVS/MRS-like spectroscopy ADC readout.

    This is intentionally different from imaging ADC examples. There is no
    line index, phase-encode index, or EPI polarity. The ADC window represents
    a long free-induction or spin-echo spectroscopy acquisition with sample
    count and dwell determining spectral bandwidth and acquisition time.
    """

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    first_delay = float(protocol.get_parameter("first_delay"))
    center_sample = int(protocol.get_parameter("center_sample"))

    frequency_shift_ppm = float(protocol.get_parameter("frequency_shift_ppm"))
    receiver_phase_rad = float(protocol.get_parameter("receiver_phase_rad"))
    spectral_bandwidth_hz = 1.0 / dwell

    return ppstar.make_adc_train(
        windows=[
            {
                "label": "spectroscopy_readout",
                "role": "spectroscopy",
                "delay": first_delay,
                "num_samples": num_samples,
                "dwell": dwell,
                "center_sample": center_sample,
                "echo_time": first_delay + center_sample * dwell,
                "trajectory": "spectroscopy",
                "readout_type": "svs_mrs",
                "SV_method": str(protocol.get_parameter("SV_method")),
                "frequency_shift_ppm": frequency_shift_ppm,
                "receiver_phase_rad": receiver_phase_rad,
                "spectral_bandwidth_hz": spectral_bandwidth_hz,
                "acquisition_duration": num_samples * dwell,
            }
        ],
        mode="single",
        trajectory="spectroscopy",
        role="spectroscopy",
        freq_ppm=frequency_shift_ppm,
        phase_offset=receiver_phase_rad,
        system=system,
        parameters=protocol.parameters,
    )


def make_navigator_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 9: imaging windows plus navigator windows."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    imaging_samples = int(protocol.get_parameter("num_samples"))
    navigator_samples = int(protocol.get_parameter("navigator_num_samples"))
    first_delay = float(protocol.get_parameter("first_delay"))

    windows = [
        {
            "label": "navigator_pre",
            "role": "navigator",
            "delay": first_delay,
            "num_samples": navigator_samples,
            "dwell": dwell,
            "navigator_index": 0,
            "trajectory": "navigator",
        },
        {
            "label": "imaging_echo_01",
            "role": "imaging",
            "delay": first_delay + 2.0e-3,
            "num_samples": imaging_samples,
            "dwell": dwell,
            "echo_index": 0,
            "trajectory": "cartesian",
        },
        {
            "label": "navigator_post",
            "role": "navigator",
            "delay": first_delay + 5.0e-3,
            "num_samples": navigator_samples,
            "dwell": dwell,
            "navigator_index": 1,
            "trajectory": "navigator",
        },
    ]

    return ppstar.make_adc_train(
        windows=windows,
        mode="mixed_roles",
        trajectory="cartesian_with_navigators",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_calibration_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 10: calibration/reference ADC lines."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    num_reference_lines = int(protocol.get_parameter("num_reference_lines"))
    num_imaging_lines = int(protocol.get_parameter("num_imaging_lines"))
    first_delay = float(protocol.get_parameter("first_delay"))
    echo_spacing = float(protocol.get_parameter("echo_spacing"))

    windows = []

    for line_index in range(num_reference_lines):
        windows.append(
            {
                "label": f"reference_line_{line_index:03d}",
                "role": "reference",
                "delay": first_delay + line_index * echo_spacing,
                "num_samples": num_samples,
                "dwell": dwell,
                "line_index": line_index,
                "calibration_type": "epi_reference",
                "trajectory": "epi_reference",
            }
        )

    imaging_start_index = len(windows)

    for line_index in range(num_imaging_lines):
        windows.append(
            {
                "label": f"imaging_line_{line_index:03d}",
                "role": "imaging",
                "delay": first_delay + (imaging_start_index + line_index) * echo_spacing,
                "num_samples": num_samples,
                "dwell": dwell,
                "line_index": line_index,
                "trajectory": "epi",
            }
        )

    return ppstar.make_adc_train(
        windows=windows,
        mode="calibration_reference",
        polarity="alternating",
        trajectory="epi_reference",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_asymmetric_echo_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 11: partial-Fourier/asymmetric echo ADC window."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))
    num_samples = int(protocol.get_parameter("num_samples"))
    center_sample = int(protocol.get_parameter("center_sample"))
    first_delay = float(protocol.get_parameter("first_delay"))

    echo_time = first_delay + center_sample * dwell

    return ppstar.make_adc_train(
        windows=[
            {
                "label": "asymmetric_echo",
                "role": "imaging",
                "delay": first_delay,
                "num_samples": num_samples,
                "dwell": dwell,
                "center_sample": center_sample,
                "echo_time": echo_time,
                "sample_time_to_echo_center": center_sample * dwell,
                "partial_fourier": True,
                "trajectory": "cartesian_asymmetric_echo",
            }
        ],
        mode="single",
        trajectory="cartesian_asymmetric_echo",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def make_custom_adc(
    *,
    system: ppstar.Opts,
    protocol: ppstar.Protocol,
):
    """Use case 12: fully explicit irregular ADC windows."""

    ensure_adc_train_exists()

    dwell = float(protocol.get_parameter("dwell"))

    windows = [
        {
            "label": "custom_short_reference",
            "role": "reference",
            "delay": 1.5e-3,
            "num_samples": 64,
            "dwell": dwell,
            "trajectory": "custom",
        },
        {
            "label": "custom_imaging_a",
            "role": "imaging",
            "delay": 4.0e-3,
            "num_samples": 256,
            "dwell": dwell,
            "trajectory": "custom",
            "line_index": 0,
        },
        {
            "label": "custom_navigator",
            "role": "navigator",
            "delay": 7.3e-3,
            "num_samples": 96,
            "dwell": dwell,
            "trajectory": "custom",
        },
        {
            "label": "custom_imaging_b",
            "role": "imaging",
            "delay": 11.0e-3,
            "num_samples": 256,
            "dwell": dwell,
            "trajectory": "custom",
            "line_index": 1,
        },
    ]

    return ppstar.make_adc_train(
        windows=windows,
        mode="explicit_windows",
        trajectory="custom",
        role="readout",
        system=system,
        parameters=protocol.parameters,
    )


def ensure_adc_train_exists() -> None:
    """Raise a clear message until make_adc_train is implemented."""

    if not hasattr(ppstar, "make_adc_train"):
        raise NotImplementedError(
            "ppstar.make_adc_train is not available yet. "
            "Add it in make/adc.py and export it from pypulseq_star.__init__.py."
        )


def add_event_block(
    seq: ppstar.Sequence,
    event,
    *,
    role: str,
) -> None:
    """Add one event block while tolerating small Sequence API differences."""

    try:
        seq.add_block(event, role=role)
    except TypeError:
        seq.add_block(event)


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

    print("\n[gammaSTAR] kernel/readout direct children:")
    candidate_prefixes = (
        "root.average.kernel.",
        "root.kernel.",
        "root.readout.",
    )

    found_child = False

    for prefix in candidate_prefixes:
        for path in sequence_elements:
            if not path.startswith(prefix):
                continue

            suffix = path.removeprefix(prefix)
            if "." not in suffix:
                found_child = True
                print(f"  {path}: {sequence_elements[path]}")

    if not found_child:
        print("  NONE")

    print("\n[gammaSTAR] protocol parameters:")
    for path in sorted(parameters):
        if path.startswith("root.prot."):
            print(f"  {path}")


def run_one_use_case(
    *,
    adc_use_case: str,
    plot_sequence: bool,
    write_sequence_files: bool,
    validate_graph: bool,
) -> None:
    """Run one ADC use case from system/protocol through writers."""

    print(f"\n=== ADC use case: {adc_use_case} ===")

    # Step 1: Define the system limits.
    system = define_system_limits()

    # Step 2: Define protocol parameters.
    protocol = define_protocol_parameters(adc_use_case=adc_use_case)

    # Step 3: Create the sequence object and ADC train.
    seq = build_sequence(
        system=system,
        protocol=protocol,
        adc_use_case=adc_use_case,
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
                float(protocol.get_parameter("TR", 0.100)),
            )
        )

    # Step 6: Write the sequence files.
    if write_sequence_files:
        stem = f"adc_train_{adc_use_case}"
        out_dir = Path(f"out/{stem}")

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


def main(
    *,
    adc_use_case: str = "single",
    plot_sequence: bool = True,
    write_sequence_files: bool = True,
    validate_graph: bool = True,
) -> None:
    """Run one ADC use case, or run all supported ADC use cases."""

    adc_use_case = adc_use_case.lower()

    if adc_use_case == "all":
        for case_name in sorted(SUPPORTED_ADC_USE_CASES):
            run_one_use_case(
                adc_use_case=case_name,
                plot_sequence=plot_sequence,
                write_sequence_files=write_sequence_files,
                validate_graph=validate_graph,
            )
        return

    if adc_use_case not in SUPPORTED_ADC_USE_CASES:
        raise ValueError(
            f"Unsupported adc_use_case={adc_use_case!r}. "
            f"Supported values are: {sorted(SUPPORTED_ADC_USE_CASES)} or 'all'"
        )

    run_one_use_case(
        adc_use_case=adc_use_case,
        plot_sequence=plot_sequence,
        write_sequence_files=write_sequence_files,
        validate_graph=validate_graph,
    )


if __name__ == "__main__":
    # Options:
    #   "single"            = one ADC window, PyPulseq make_adc-like
    #   "multi_echo_gre"    = multi-echo GRE / field mapping
    #   "epi"               = continuous EPI-like ADC train
    #   "segmented_epi"     = segmented or multi-shot EPI
    #   "rare"              = RARE/TSE-like discrete echo train
    #   "grase"             = GRASE-like grouped ADC train
    #   "radial"            = radial spokes
    #   "spiral"            = long spiral readout
    #   "spectroscopy"      = SVS/MRS-like long spectroscopy ADC readout
    #   "navigator"         = imaging + navigator windows
    #   "calibration"       = reference/calibration lines
    #   "asymmetric_echo"   = partial-Fourier/asymmetric echo
    #   "custom"            = explicit irregular windows
    #   "all"               = run all use cases
    adc_use_case = "all"

    plot = True
    write_files = True
    validate_graph = True

    main(
        adc_use_case=adc_use_case,
        plot_sequence=plot,
        write_sequence_files=write_files,
        validate_graph=validate_graph,
    )