from __future__ import annotations

from types import SimpleNamespace

import pytest

import pypulseq_star.writers.gammastar_generic_document as gd


pytestmark = pytest.mark.unit


def test_orientation_script_and_gradient_axis_helpers():
    script = gd._orientation_lua_script("rotation")
    assert "coronal" in script and "sagittal" in script
    with pytest.raises(ValueError, match="Unsupported orientation"):
        gd._orientation_lua_script("bad")

    assert gd._gradient_logical_axis_for_orientation(SimpleNamespace(logical_axis="read")) == "read"
    assert gd._gradient_logical_axis_for_orientation(
        SimpleNamespace(metadata={"axis_role": "phase"})
    ) == "phase"
    assert gd._gradient_logical_axis_for_orientation(SimpleNamespace(channel="z")) == "slice"
    assert gd._gradient_logical_axis_for_orientation(SimpleNamespace(channel="q")) is None


def test_protocol_source_detection_direct_and_expression_paths():
    live = {"n_y": 64, "fov": 0.22}
    assert gd._direct_protocol_source_name_from_value("root.prot.n_y", live) == "n_y"
    assert gd._direct_protocol_source_name_from_value("n_y", live) == "n_y"
    assert gd._direct_protocol_source_name_from_value("-0.5*n_y/fov", live) is None

    ref = SimpleNamespace(name="fov")
    assert gd._direct_protocol_source_name_from_value(ref, live) == "fov"

    expr = SimpleNamespace(to_canonical=lambda: "n_y / fov")
    assert gd._protocol_source_name_from_value(expr, live) in {"n_y", "fov"}
    assert gd._protocol_source_name_from_value(None, live) is None


def test_coercion_anchor_and_protocol_ui_helpers():
    assert gd._coerce_vector3([1, 2, 3]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="three values"):
        gd._coerce_vector3([1, 2])
    assert gd._coerce_matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="three rows"):
        gd._coerce_matrix3([[1]])

    assert gd._anchor_fraction("start") == 0.0
    assert gd._anchor_fraction("center") == 0.5
    assert gd._anchor_fraction("end") == 1.0
    # Unknown anchors deliberately fall back to the start of the event.
    assert gd._anchor_fraction("bad") == 0.0
    assert gd._anchor_fraction(None) == 0.0

    assert gd._protocol_value_type(True) == "bool"
    assert gd._protocol_value_type(3) == "int"
    assert gd._protocol_value_type(3.5) == "float"
    assert gd._protocol_value_type("x") == "string"
    assert gd._pretty_protocol_name("echo_time") == "Echo Time"


def test_container_names_directions_and_safe_tokens():
    assert gd._adc_container_name(SimpleNamespace(name="adc_train"), 0) == "adc_train"
    assert gd._gradient_container_name(SimpleNamespace(name="gx"), 0) == "gx"
    assert gd._gradient_direction("x") == [1.0, 0.0, 0.0]
    assert gd._gradient_direction("y") == [0.0, 1.0, 0.0]
    assert gd._gradient_direction("z") == [0.0, 0.0, 1.0]
    assert gd._safe_path_token("root.kernel/read out") == "root_kernel_read_out"


def test_event_classification_duration_and_lua_helpers():
    rf = SimpleNamespace(type="rf", duration=1e-3, delay=1e-4)
    adc = SimpleNamespace(type="adc", num_samples=10, dwell=1e-6, duration=10e-6, delay=2e-4)
    grad = SimpleNamespace(
        type="trap",
        rise_time=1e-4,
        flat_time=2e-4,
        fall_time=1e-4,
        duration=4e-4,
    )
    assert gd._is_rf_event(rf)
    assert gd._is_adc_event(adc)
    assert gd._is_gradient_event(grad)
    assert gd._event_delay(rf) == pytest.approx(1e-4)
    assert gd._event_active_duration(adc) == pytest.approx(10e-6)
    assert gd._event_active_duration(grad) == pytest.approx(4e-4)

    assert gd._lua_string("a'b") == '"a\'b"'
    assert gd._safe_input_name("root.prot.echo-time") == "root_prot_echo_time"
    assert gd._to_lua(True) == "true"
    assert gd._to_lua([1, 2]) == "{1, 2}"
    assert gd._literal(3)["script"] == "return 3"
    assert gd._expr(
        inputs={"x": "root.prot.x"},
        script="return x",
    )["inputs"]["x"] == "root.prot.x"
