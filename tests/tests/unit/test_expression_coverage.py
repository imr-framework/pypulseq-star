from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.expressions import (
    EvaluationContext,
    EventAnchorRef,
    EventPropertyRef,
    InvalidExpressionError,
    LiteralExpression,
    ParameterNamespace,
    ParameterRef,
    UnknownReferenceError,
    as_expression,
)
from pypulseq_star.expressions.operations import BinaryExpression, UnaryExpression


@pytest.mark.unit
def test_expression_arithmetic_dependencies_and_explanation() -> None:
    te = ParameterRef("echo_time")
    dwell = ParameterRef("dwell")
    expression = -(te * 2 + dwell / 4 - 1) ** 2

    assert expression.eval({"echo_time": 3.0, "dwell": 4.0}) == pytest.approx(-36.0)
    assert expression.dependencies == frozenset({"protocol.echo_time", "protocol.dwell"})
    assert expression.explain()["type"] == "UnaryExpression"
    assert "echo_time" in expression.to_canonical()
    with pytest.raises(TypeError, match="cannot be used as booleans"):
        bool(expression)


@pytest.mark.unit
def test_expression_literal_and_invalid_operator_paths() -> None:
    literal = as_expression(5)
    assert isinstance(literal, LiteralExpression)
    assert as_expression(literal) is literal
    assert literal.eval() == 5
    assert int(literal) == 5
    assert float(literal) == 5.0
    assert LiteralExpression("abc").to_canonical() == "'abc'"

    with pytest.raises(InvalidExpressionError):
        LiteralExpression([])
    with pytest.raises(InvalidExpressionError):
        UnaryExpression("unsupported", literal)
    with pytest.raises(InvalidExpressionError):
        BinaryExpression("unsupported", literal, literal)


@pytest.mark.unit
def test_evaluation_context_reference_mapping_and_resolver() -> None:
    reference = EventPropertyRef("adc", "duration")
    context = EvaluationContext(references={reference.reference_key: 0.002})
    assert reference.eval(context) == pytest.approx(0.002)

    class Resolver:
        def resolve_expression_reference(self, kind, identity, property_name=None, metadata=None):
            assert kind == "event_anchor"
            assert identity == "rf"
            assert property_name == "center"
            return 0.001

    anchor = EventAnchorRef("rf", "center")
    assert anchor.eval(EvaluationContext(resolver=Resolver())) == pytest.approx(0.001)

    with pytest.raises(UnknownReferenceError):
        EventPropertyRef("missing", "duration").eval(EvaluationContext())


@pytest.mark.unit
def test_evaluation_context_adapters_and_namespace_cache() -> None:
    mapping_context = EvaluationContext.from_object({"echo_time": 0.01})
    assert mapping_context.parameter_value("echo_time") == pytest.approx(0.01)
    with pytest.raises(KeyError):
        mapping_context.parameter_value("missing")

    protocol = SimpleNamespace(parameters={"echo_time": 0.02, "average": 3})
    source = SimpleNamespace(protocol=protocol, parameters={"echo_time": 99, "extra": 4})
    context = EvaluationContext.from_object(source)
    assert context.parameters == {"echo_time": 0.02, "average": 3, "extra": 4}

    namespace = ParameterNamespace(protocol, aliases={"TE": "echo_time"})
    assert namespace.TE is namespace["echo_time"]
    assert "TE" in dir(namespace)
    assert namespace.TE.eval(protocol.parameters) == pytest.approx(0.02)
