from __future__ import annotations

import pytest

from pypulseq_star.core.expression import SeqStarExpression, _lua_table

pytestmark = pytest.mark.unit


def test_seqstar_expression_literals_scripts_and_parameters():
    assert SeqStarExpression.literal(True).to_script() == "return true"
    assert SeqStarExpression.literal("abc").to_script() == "return 'abc'"
    assert SeqStarExpression("x", {"x": "root.prot.x"}).to_script() == "return x"
    assert SeqStarExpression.literal(3).to_script() == "return 3"
    assert SeqStarExpression.literal(2.5).to_script().startswith("return 2.5")
    assert SeqStarExpression.literal([1, False, "x"]).to_script() == "return {1, false, 'x'}"
    assert SeqStarExpression.literal({"a": 1}).to_script() == "return {a=1}"

    relation = SeqStarExpression.relation("a + 2*b", a="root.prot.a", b="root.prot.b")
    parameter = relation.to_gammastar_parameter()
    assert parameter["inputs"] == {"a": "root.prot.a", "b": "root.prot.b"}
    assert parameter["script"].startswith("return ")


def test_lua_table_nested_and_fallback_values():
    assert _lua_table({"x": [1, 2], "flag": True}) == "{x={1, 2}, flag=true}"
    assert _lua_table(None) == "None"
