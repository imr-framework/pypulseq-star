"""Arithmetic expression nodes."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from .base import Expression
from .context import EvaluationContext
from .diagnostics import ExpressionDiagnostic
from .errors import ExpressionEvaluationError, InvalidExpressionError


@dataclass(frozen=True, slots=True, repr=False)
class UnaryExpression(Expression):
    """A unary arithmetic operation."""

    operator_name: str
    operand: Expression

    _OPERATORS: ClassVar[dict[str, tuple[Callable[[Any], Any], str]]] = {
        "neg": (operator.neg, "-"),
    }

    def __post_init__(self) -> None:
        if self.operator_name not in self._OPERATORS:
            raise InvalidExpressionError(
                f"Unsupported unary operator {self.operator_name!r}."
            )

    def _evaluate(self, context: EvaluationContext) -> Any:
        function, symbol = self._OPERATORS[self.operator_name]
        try:
            return function(self.operand.eval(context))
        except ExpressionEvaluationError:
            raise
        except Exception as exc:
            raise ExpressionEvaluationError(
                f"Unary operation {symbol!r} failed.",
                diagnostic=ExpressionDiagnostic(
                    code="UNARY_OPERATION_FAILED",
                    message=f"Unary operation {symbol!r} failed.",
                    expression=self.to_canonical(),
                    dependencies=tuple(sorted(self.dependencies)),
                    stage="evaluation",
                    cause_type=type(exc).__name__,
                ),
                cause=exc,
            ) from exc

    @property
    def dependencies(self) -> frozenset[str]:
        return self.operand.dependencies

    def to_canonical(self) -> str:
        _, symbol = self._OPERATORS[self.operator_name]
        return f"({symbol}{self.operand.to_canonical()})"


@dataclass(frozen=True, slots=True, repr=False)
class BinaryExpression(Expression):
    """A binary arithmetic operation."""

    operator_name: str
    left: Expression
    right: Expression

    _OPERATORS: ClassVar[
        dict[str, tuple[Callable[[Any, Any], Any], str]]
    ] = {
        "add": (operator.add, "+"),
        "sub": (operator.sub, "-"),
        "mul": (operator.mul, "*"),
        "truediv": (operator.truediv, "/"),
        "pow": (operator.pow, "**"),
    }

    def __post_init__(self) -> None:
        if self.operator_name not in self._OPERATORS:
            raise InvalidExpressionError(
                f"Unsupported binary operator {self.operator_name!r}."
            )

    def _evaluate(self, context: EvaluationContext) -> Any:
        function, symbol = self._OPERATORS[self.operator_name]
        try:
            left_value = self.left.eval(context)
            right_value = self.right.eval(context)
            return function(left_value, right_value)
        except ExpressionEvaluationError:
            raise
        except Exception as exc:
            raise ExpressionEvaluationError(
                f"Binary operation {symbol!r} failed.",
                diagnostic=ExpressionDiagnostic(
                    code="BINARY_OPERATION_FAILED",
                    message=f"Binary operation {symbol!r} failed.",
                    expression=self.to_canonical(),
                    dependencies=tuple(sorted(self.dependencies)),
                    stage="evaluation",
                    cause_type=type(exc).__name__,
                ),
                cause=exc,
            ) from exc

    @property
    def dependencies(self) -> frozenset[str]:
        return self.left.dependencies | self.right.dependencies

    def to_canonical(self) -> str:
        _, symbol = self._OPERATORS[self.operator_name]
        return (
            f"({self.left.to_canonical()} "
            f"{symbol} "
            f"{self.right.to_canonical()})"
        )
