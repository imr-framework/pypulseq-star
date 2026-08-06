"""Literal scalar expression nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Expression
from .context import EvaluationContext
from .errors import InvalidExpressionError

_SUPPORTED_LITERAL_TYPES = (int, float, complex, bool, str, type(None))


@dataclass(frozen=True, slots=True, repr=False)
class LiteralExpression(Expression):
    """A constant value in an expression graph."""

    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.value, _SUPPORTED_LITERAL_TYPES):
            raise InvalidExpressionError(
                f"Unsupported expression literal type: {type(self.value).__name__}."
            )

    def _evaluate(self, context: EvaluationContext) -> Any:
        return self.value

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset()

    def to_canonical(self) -> str:
        if isinstance(self.value, str):
            return repr(self.value)
        return str(self.value)
