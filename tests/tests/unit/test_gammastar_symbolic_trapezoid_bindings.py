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


def _find_parameter(parameters: dict, suffix: str) -> tuple[str, dict]:
    matches = [
        (path, value)
        for path, value in parameters.items()
        if path.endswith(suffix)
    ]
    assert len(matches) == 1, (
        f"Expected one parameter ending in {suffix!r}; "
        f"found {[path for path, _ in matches]}"
    )
    return matches[0]


def _find_event_parameter(
    parameters: dict,
    *,
    event_name: str,
    field_suffix: str,
) -> tuple[str, dict]:
    """Find an event parameter without depending on generated block/event IDs."""
    token = f".{event_name}_"
    matches = [
        (path, value)
        for path, value in parameters.items()
        if token in path and path.endswith(field_suffix)
    ]
    assert len(matches) == 1, (
        f"Expected one {event_name!r} parameter ending in {field_suffix!r}; "
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
    """gx_pre uses -0.5 * gx.area and must remain exportable."""
    document = _build_document()
    parameters = document["parameters"]

    _, gx_pre_area = _find_event_parameter(
        parameters,
        event_name="gx_prephaser",
        field_suffix=".grad.area",
    )
    # The unsupported EventPropertyRef form should remain a resolved literal,
    # not abort the export.
    assert gx_pre_area["inputs"] == {}


def test_gre_spoilers_retain_fov_relationships() -> None:
    parameters = _build_document()["parameters"]

    for event_suffix in (
        ".gx_spoil_b004_e007.grad.area",
        ".gy_spoil_b004_e008.grad.area",
    ):
        area_path, area_parameter = _find_parameter(
            parameters,
            event_suffix,
        )
        assert area_parameter["inputs"], area_path
        input_sources = set(area_parameter["inputs"].values())
        # The backend-consumed spoiler area must depend directly on the
        # editable FOV control. gammaSTAR does not reliably invalidate the
        # transitive chain fov -> phase_encode_step -> area -> samples.
        assert "root.prot.fov" in input_sources
        assert "root.prot.phase_encode_step" not in input_sources

        leaf = area_path.removesuffix(".area")
        samples_parameter = parameters[f"{leaf}.samples"]
        assert samples_parameter["inputs"]["area"] == area_path
        assert samples_parameter["inputs"]["grad_set"] == (
            "root.gradient_settings"
        )
        assert samples_parameter["inputs"]["gamma"] == "root.sys.gamma"

        duration_parameter = parameters[f"{leaf}.duration"]
        assert duration_parameter["inputs"]["samples"] == (
            f"{leaf}.samples"
        )


def test_gre_spoiler_block_duration_remains_live() -> None:
    parameters = _build_document()["parameters"]

    block_duration = parameters[
        "root.kernel.kernel.seqstar_blocks.kernel_spoiling.duration"
    ]
    input_paths = set(block_duration["inputs"].values())
    assert any(
        path.endswith(".gx_spoil_b004_e007.duration")
        for path in input_paths
    )
    assert any(
        path.endswith(".gy_spoil_b004_e008.duration")
        for path in input_paths
    )
