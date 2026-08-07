"""Regression coverage for symbolic logical-node loop lengths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.expressions.parameters import ParameterRef
from pypulseq_star.writers.gammastar_generic_document import (
    GenericGammaStarDocumentBuilder,
)

pytestmark = pytest.mark.unit


def test_kernel_loop_recovers_ny_from_symbolic_repeat_count() -> None:
    """root.kernel.length must remain bound to n_y, not a resolved literal."""

    values = {"n_y": 128}
    protocol = SimpleNamespace(
        parameters=values,
        get_parameter=lambda key, default=None: values.get(key, default),
    )

    repeat_count = ParameterRef("n_y", protocol=protocol, dtype=int)

    symbolic_sequence = SimpleNamespace(
        timeline=SimpleNamespace(
            nodes={
                "kernel": {
                    "name": "kernel",
                    "node": "kernel",
                    "repeat_count": repeat_count,
                    "counter": "ky_index",
                    "repeat_mode": "loop",
                }
            }
        ),
        metadata={},
    )

    builder = object.__new__(GenericGammaStarDocumentBuilder)
    builder.symbolic_sequence = symbolic_sequence
    builder._live_protocol_values = values

    source = builder._loop_length_source_from_symbolic_nodes(
        loop_token="kernel"
    )

    assert source == "n_y"