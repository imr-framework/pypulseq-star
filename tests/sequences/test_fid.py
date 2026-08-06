"""FID reference-sequence contract tests."""
from __future__ import annotations

import pytest

from .helpers import assert_timing_passes, evaluate, load_demo

pytestmark = [pytest.mark.sequence, pytest.mark.fid]


def test_fid_protocol_exposes_canonical_aliases(repo_root) -> None:
    demo = load_demo(repo_root, "FID")
    protocol = demo.define_protocol()
    assert protocol.parameters["sequence_type"] == "FID"
    assert protocol.aliases["TE"] == "echo_time"
    assert protocol.aliases["TR"] == "repetition_time"
    assert protocol.aliases["averages"] == "average"
    assert protocol.aliases["num_averages"] == "average"


def test_fid_build_has_positive_te_and_tr_fills(repo_root) -> None:
    demo = load_demo(repo_root, "FID")
    system = demo.define_system_limits()
    protocol = demo.define_protocol()
    seq, te_fill, tr_fill = demo.build_sequence(system, protocol)
    assert evaluate(te_fill, seq) >= 0.0
    assert evaluate(tr_fill, seq) >= 0.0
    assert_timing_passes(seq)


def test_fid_average_loop_remains_protocol_driven(repo_root) -> None:
    demo = load_demo(repo_root, "FID")
    protocol = demo.define_protocol()
    seq, _, _ = demo.build_sequence(demo.define_system_limits(), protocol)
    repeat_count = seq.timeline.nodes["kernel"]["repeat_count"]

    # Guard against replacing the protocol reference with a sequence-specific
    # literal. Evaluate the same expression against an edited protocol context.
    dependencies = set(getattr(repeat_count, "dependencies", ()))
    assert "protocol.average" in dependencies
    assert evaluate(repeat_count, {"average": 7}) == pytest.approx(7)


def test_fid_te_edit_changes_symbolic_fill(repo_root) -> None:
    demo = load_demo(repo_root, "FID")
    system = demo.define_system_limits()
    p1 = demo.define_protocol()
    s1, fill1, _ = demo.build_sequence(system, p1)
    p2 = demo.define_protocol()
    p2.parameters["echo_time"] += 5e-3
    s2, fill2, _ = demo.build_sequence(system, p2)
    assert evaluate(fill2, s2) - evaluate(fill1, s1) == pytest.approx(5e-3)
