"""Additional generic gammaSTAR helper and normalization coverage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pypulseq_star.writers.gammastar_generic_document as gd

pytestmark = pytest.mark.unit


def test_protocol_ui_units_scaling_and_canonicalization() -> None:
    values = {
        "fov": 0.22,
        "n_y": 64,
        "echo_time": 5e-3,
        "repetition_time": 1.0,
        "orientation": "sagittal",
        "enabled": True,
        "table": [1, 2, 3],
        "private": None,
    }
    specification = gd._build_protocol_ui_specification(values)
    paths = {entry["path"] for entry in specification}
    assert {"root.prot.fov", "root.prot.n_y", "root.prot.TE", "root.prot.TR", "root.prot.orientation", "root.prot.enabled"} <= paths
    assert "root.prot.private" not in paths
    # UI construction canonicalizes echo_time/repetition_time to TE/TR.
    assert gd._protocol_unit("TE") == "ms"
    assert gd._protocol_unit_scaling("TE") == pytest.approx(1e-3)
    assert gd._protocol_unit("TR") == "ms"
    assert gd._protocol_unit_scaling("TR") == pytest.approx(1e-3)
    # The noncanonical source key is intentionally not a UI-unit key.
    assert gd._protocol_unit("echo_time") is None
    assert gd._protocol_value_type([1, 2]) is None
    canonical = gd._canonical_protocol_values(
        {"repetition_time": 1.0, "echo_time": 0.01}
    )
    assert canonical["repetition_time"] == pytest.approx(1.0)
    assert canonical["TR"] == pytest.approx(1.0)
    assert canonical["echo_time"] == pytest.approx(0.01)
    assert canonical["TE"] == pytest.approx(0.01)
    assert gd._protocol_key("repetition_time") == "TR"
    assert gd._protocol_key("echo_time") == "TE"


def test_gradient_normalization_and_units() -> None:
    event = SimpleNamespace(
        channel="x",
        system=SimpleNamespace(gamma=42.576e6),
        metadata={},
        parameters={},
    )
    normalized = gd._normalize_gradient_data(
        {
            "tt": [0.0, 1e-3, 2e-3],
            "waveform": [0.0, 42.576e3, 0.0],
            "duration": 2e-3,
            "unit": "Hz/m",
        },
        gradient_event=event,
    )
    assert normalized["samples_unit"] == "T/m"
    assert normalized["samples_source_unit"] == "hz/m"
    assert max(normalized["samples"]["v"]) == pytest.approx(1e-3)
    assert gd._gradient_gamma_hz_per_t(event) == pytest.approx(42.576e6)
    assert gd._grad_to_t_per_m(28, "mT/m") == pytest.approx(0.028)
    assert gd._grad_to_t_per_m(1, "T/m") == pytest.approx(1.0)
    assert gd._slew_to_t_per_m_s(100, "T/m/s") == pytest.approx(100.0)
    # Unknown units currently pass through unchanged for backward compatibility.
    assert gd._grad_to_t_per_m(1, "bad") == pytest.approx(1.0)
    assert gd._slew_to_t_per_m_s(1, "bad") == pytest.approx(1.0)
    # Frequency-normalized units require gamma and therefore cannot be converted
    # by these scalar helpers alone.
    with pytest.raises(ValueError, match="requires system.gamma"):
        gd._grad_to_t_per_m(1, "Hz/m")
    with pytest.raises(ValueError, match="requires system.gamma"):
        gd._slew_to_t_per_m_s(1, "Hz/m/s")


def test_adc_normalization_fallback_and_window_helpers() -> None:
    window = SimpleNamespace(
        delay=1e-3,
        num_samples=8,
        dwell=2e-6,
        phase_offset=0.2,
        frequency_offset=100.0,
        label="echo_0",
    )
    event = SimpleNamespace(
        name="adc_train",
        type="adc",
        windows=[window],
        metadata={"trajectory": "cartesian"},
        parameters={},
    )
    assert gd._get_adc_windows_for_export(event) == [window]
    assert gd._adc_event_num_windows(event) == 1
    assert not gd._has_multiple_adc_windows_for_export(event)
    assert gd._adc_window_value(window, "num_samples") == 8

    # The single-event normalizer consumes a scalar ADC event; train windows are
    # handled by the dedicated window-export path tested above.
    single = SimpleNamespace(
        name="adc",
        type="adc",
        delay=1e-3,
        num_samples=16,
        dwell=4e-6,
        phase_offset=0.0,
        frequency_offset=0.0,
        metadata={},
        parameters={},
    )
    fallback = gd._fallback_single_adc_data(single)
    assert fallback["total_num_samples"] == 16
    assert fallback["num_windows"] == 1
    assert fallback["duration"] == pytest.approx(1e-3 + 64e-6)
    assert fallback["windows"][0]["num_samples"] == 16
    assert fallback["windows"][0]["dwell"] == pytest.approx(4e-6)
    data = gd._normalize_adc_data_from_event(single)
    assert data["total_num_samples"] == 16
    assert data["windows"][0]["num_samples"] == 16


def test_block_metadata_node_and_path_helpers() -> None:
    block = SimpleNamespace(
        name="readout_3",
        role="readout",
        metadata={
            "seqstar_node": "kernel.readout",
            "seqstar_parent_node": "kernel",
            "seqstar_local_name": "readout",
            "source_block_index": 3,
        },
        parameters={},
    )
    mapping = {}
    gd._copy_block_export_metadata_to_mapping(block, mapping)
    assert gd._block_node_for_export(block) == "kernel.readout"
    assert gd._block_parent_for_export(block, "kernel.readout") == "kernel"
    assert gd._block_local_name_for_export(block, "kernel.readout") == "readout"
    assert gd._event_source_block_index_for_export(block) == 3
    assert mapping
    assert gd._node_is_within_for_export("kernel.readout.echo", "kernel.readout")
    assert gd._node_depth_for_export("kernel.readout.echo") == 3
    assert gd._safe_path_token("kernel/readout echo") == "kernel_readout_echo"


def test_event_tags_spoiler_detection_variants_and_extent() -> None:
    spoiler = SimpleNamespace(
        name="gx_spoil",
        role="spoiling",
        type="trap",
        channel="x",
        delay=1e-3,
        rise_time=1e-4,
        flat_time=2e-4,
        fall_time=1e-4,
        duration=4e-4,
        metadata={"axis_role": "read"},
        parameters={},
    )
    assert gd._is_spoiler_like_event(spoiler)
    assert "spoiling" in gd._event_text_tags(spoiler)
    assert gd._event_active_duration(spoiler) == pytest.approx(4e-4)
    assert gd._events_extent_duration([spoiler]) == pytest.approx(1.4e-3)
    assert gd._event_loop_counter_path(spoiler) == "root.average.counter"
    assert gd._event_variant_policy(spoiler) == "auto"


def test_motif_and_sequence_element_ordering_helpers() -> None:
    rf = SimpleNamespace(name="rf", type="rf", role="excitation", metadata={})
    adc = SimpleNamespace(name="adc", type="adc", role="acquisition", metadata={})
    gx = SimpleNamespace(name="gx", type="trap", role="readout", channel="x", metadata={})
    block = SimpleNamespace(name="readout", role="readout", events=[rf, gx, adc])
    signature = gd._block_motif_signature(block)
    assert "role:readout" in signature
    assert "rf" in signature and "adc" in signature and "grad" in signature
    assert gd._normalize_signature_token("Read Out / X") == "read_out_x"

    ordered = gd._order_sequence_elements(
        {
            "root.kernel.readout": "AtomicSequence",
            "root": "Sequence",
            "root.kernel": "Loop",
        }
    )
    assert list(ordered)[0] == "root"
    assert gd._hierarchy_sort_key("root.kernel")[0] < gd._hierarchy_sort_key("root.kernel.readout")[0]
