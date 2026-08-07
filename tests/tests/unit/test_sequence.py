"""Unit tests for blocks, timelines, nodes, and duration handling."""

from __future__ import annotations

import pytest

import pypulseq_star as ppstar

pytestmark = pytest.mark.unit


def test_add_block_records_timeline_and_containment(basic_sequence) -> None:
    adc = ppstar.make_adc(num_samples=128, dwell=10e-6)
    block = basic_sequence.add_block(adc, role="acquisition")

    assert basic_sequence.timeline.blocks == [block]
    assert block.events == [adc]
    assert basic_sequence.relationships[0].kind == "contains_block"
    assert block.relationships[0].kind == "contains_event"


def test_set_node_records_loop_contract(basic_sequence) -> None:
    basic_sequence.set_node(
        "kernel",
        role="kernel",
        repeat_count="averages",
        repeat_every="TR",
        counter="average_index",
        repeat_mode="loop",
    )

    node = basic_sequence.nodes["kernel"]
    assert node["role"] == "kernel"
    assert node["repeat_count"] == "averages"
    assert node["repeat_every"] == "TR"
    assert node["counter"] == "average_index"
    assert node["repeat_mode"] == "loop"


def test_calc_timeline_duration_matches_ordered_blocks(basic_sequence, system) -> None:
    first = basic_sequence.add_block(
        ppstar.make_delay(1e-3, system=system, name="first"),
        role="first",
    )
    second = basic_sequence.add_block(
        ppstar.make_delay(2e-3, system=system, name="second"),
        role="second",
    )

    assert basic_sequence.calc_timeline_duration([first, second]) == pytest.approx(3e-3)


def test_implicit_block_after_relationships_resolve(basic_sequence, system) -> None:
    basic_sequence.add_block(ppstar.make_delay(1e-3, system=system), role="first")
    basic_sequence.add_block(ppstar.make_delay(2e-3, system=system), role="second")
    relationships = ppstar.relationships.resolve(basic_sequence)

    block_after = [rel for rel in relationships if rel.relation_type == "timing.block_after"]
    assert len(block_after) == 1
    assert all(item.passed for item in block_after[0].validation)
