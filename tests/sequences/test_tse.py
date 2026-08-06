"""TSE reference-sequence contract tests."""
from __future__ import annotations

import numpy as np
import pytest

from .helpers import assert_timing_passes, evaluate, load_demo

pytestmark = [pytest.mark.sequence, pytest.mark.tse]


def _flatten_fortran(table) -> list[int]:
    return np.asarray(table, dtype=int).reshape(-1, order="F").tolist()


@pytest.mark.parametrize("order", ["linear", "centric", "reverse"])
def test_tse_phase_order_covers_every_ky_line_once(repo_root, order) -> None:
    demo = load_demo(repo_root, "TSE")
    table = demo._phase_encode_order(
        n_y=64,
        echo_train_length=8,
        order=order,
        center_echo_index=3,
    )
    acquired = _flatten_fortran(table)
    assert len(acquired) == 64
    assert sorted(acquired) == list(range(-32, 32))


def test_tse_centric_places_kspace_center_at_first_echo(repo_root) -> None:
    demo = load_demo(repo_root, "TSE")
    table = np.asarray(
        demo._phase_encode_order(
            n_y=64,
            echo_train_length=8,
            order="centric",
            center_echo_index=3,
        )
    )
    assert table[0, 0] == 0
    assert table[:, 0].tolist()[:5] == [0, -1, 1, -2, 2]


def test_tse_linear_and_reverse_are_oppositely_ordered(repo_root) -> None:
    demo = load_demo(repo_root, "TSE")
    linear = _flatten_fortran(demo._phase_encode_order(64, 8, order="linear", center_echo_index=3))
    reverse = _flatten_fortran(demo._phase_encode_order(64, 8, order="reverse", center_echo_index=3))
    assert linear != reverse
    assert sorted(linear) == sorted(reverse)


def test_tse_phase_encode_and_rewind_are_equal_and_opposite(repo_root) -> None:
    demo = load_demo(repo_root, "TSE")
    protocol = demo.define_protocol(overrides={"phase_encode_order": "centric"})
    seq, *fills = demo.build_sequence(demo.define_system_limits(), protocol)
    blocks = {block.name: block for block in seq.timeline.blocks}
    phase = blocks["crusher_phase"].get_event("gy_phase_encode")
    rewind = blocks["rewind_spoil"].get_event("gy_phase_rewind")
    phase_variants = phase.metadata["seqstar_repetition_variants"]
    rewind_variants = rewind.metadata["seqstar_repetition_variants"]
    assert len(phase_variants) == len(rewind_variants) == protocol.parameters["n_y"]
    for left, right in zip(phase_variants, rewind_variants, strict=True):
        assert float(left.area) == pytest.approx(-float(right.area))
    assert all(evaluate(fill, seq) >= 0.0 for fill in fills)
    assert_timing_passes(seq)


def test_tse_live_binding_is_direct_and_backend_generic(repo_root) -> None:
    demo = load_demo(repo_root, "TSE")
    protocol = demo.define_protocol(overrides={"phase_encode_order": "centric"})
    seq, *_ = demo.build_sequence(demo.define_system_limits(), protocol)
    blocks = {block.name: block for block in seq.timeline.blocks}
    phase = blocks["crusher_phase"].get_event("gy_phase_encode")
    binding = phase.metadata["seqstar_nested_loop_expression_binding"]
    assert binding["property"] == "area"
    assert binding["loop_inputs"] == {"shot": "outer", "echo": "inner"}
    assert binding["protocol_inputs"]["order"] == "phase_encode_order"
    assert "if key == 'centric' then target = 0 end" in binding["script"]
