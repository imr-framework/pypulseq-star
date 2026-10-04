"""Regression tests for protocol-driven gammaSTAR trapezoid bindings."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPOSITORY_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import demo_GRE  # noqa: E402

from pypulseq_star.writers import GammaStarWriter  # noqa: E402


def _find_named_gradient_parameter(
    parameters: dict,
    event_name: str,
    field: str,
) -> tuple[str, dict]:
    """Find one gradient field without depending on block/event indices."""

    suffix = f".grad.{field}"
    matches = [
        (path, value)
        for path, value in parameters.items()
        if event_name in path and path.endswith(suffix)
    ]
    assert len(matches) == 1, (
        f"Expected one {event_name!r} gradient parameter ending in {suffix!r}; "
        f"found {[path for path, _ in matches]}"
    )
    return matches[0]


def _build_document() -> dict:
    system = demo_GRE.define_system_limits()
    protocol = demo_GRE.define_protocol(
        orientation="axial",
        overrides={"n_y": 128, "fov": 0.256},
    )
    sequence, _, _ = demo_GRE.build_sequence(
        system,
        protocol,
        orientation="axial",
    )
    resolved = sequence.resolve()
    return GammaStarWriter(sequence).to_dict(defaults=resolved)


def test_export_does_not_fail_on_event_property_reference() -> None:
    """Non-protocol event references remain safely resolved as literals."""

    parameters = _build_document()["parameters"]
    _, gx_pre_area = _find_named_gradient_parameter(
        parameters,
        "gx_prephaser",
        "area",
    )
    assert gx_pre_area["inputs"] == {}


def test_protocol_driven_fixed_shape_trapezoids_keep_live_area() -> None:
    """Area stays live while explicitly authored gradient timing stays fixed."""

    parameters = _build_document()["parameters"]

    for event_name in ("gx_spoil", "gy_spoil"):
        area_path, area_parameter = _find_named_gradient_parameter(
            parameters,
            event_name,
            "area",
        )
        assert area_parameter["inputs"], area_path

        input_sources = set(area_parameter["inputs"].values())
        assert "root.prot.fov" in input_sources
        assert "root.prot.phase_encode_step" not in input_sources

        leaf = area_path.removesuffix(".area")
        samples_parameter = parameters[f"{leaf}.samples"]
        assert samples_parameter["inputs"]["area"] == area_path

        normalized_path = samples_parameter["inputs"]["shape"]
        assert normalized_path == f"{leaf}.seqstar_normalized_samples"
        assert parameters[normalized_path]["inputs"] == {}

        # Explicit rise/flat/fall timing is the authored contract.  Varying a
        # protocol-driven area must scale the waveform, not make duration live.
        duration_parameter = parameters[f"{leaf}.duration"]
        assert duration_parameter["inputs"] == {}

        max_amp_parameter = parameters[f"{leaf}.max_abs_amplitude"]
        assert max_amp_parameter["inputs"]["samples"] == f"{leaf}.samples"


def test_fixed_shape_spoiler_timing_is_not_reintroduced_as_dynamic() -> None:
    """Fixed-shape gradients must not acquire an area-dependent block duration."""

    parameters = _build_document()["parameters"]

    for event_name in ("gx_spoil", "gy_spoil"):
        area_path, _ = _find_named_gradient_parameter(
            parameters,
            event_name,
            "area",
        )
        leaf = area_path.removesuffix(".area")
        assert parameters[f"{leaf}.duration"]["inputs"] == {}

    dynamic_block_durations = []
    for path, parameter in parameters.items():
        if not path.endswith(".duration") or not isinstance(parameter, dict):
            continue
        inputs = parameter.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        sources = {str(value) for value in inputs.values()}
        if any("gx_spoil" in source for source in sources) or any(
            "gy_spoil" in source for source in sources
        ):
            dynamic_block_durations.append(path)

    assert dynamic_block_durations == []
