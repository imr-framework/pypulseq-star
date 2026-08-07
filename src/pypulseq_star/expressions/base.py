"""Base symbolic scalar expression contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .context import EvaluationContext
from .diagnostics import ExpressionDiagnostic
from .errors import ExpressionEvaluationError


class Expression(ABC):
    """Base class for protocol-derived symbolic scalar values.

    Expressions support ordinary Python arithmetic while retaining a dependency
    graph. They are intentionally lightweight and do not expose SymPy to users.
    """

    __array_priority__ = 1000

    @abstractmethod
    def _evaluate(self, context: EvaluationContext) -> Any:
        """Evaluate this expression using an existing context."""

    @property
    @abstractmethod
    def dependencies(self) -> frozenset[str]:
        """Canonical dependency identifiers required by this expression."""

    @abstractmethod
    def to_canonical(self) -> str:
        """Return a stable human-readable expression."""

    def eval(self, context: Any | None = None) -> Any:
        """Evaluate the expression.

        ``context`` may be an ``EvaluationContext``, a plain parameter mapping,
        a Protocol object, a Sequence object, or a resolved realization.
        """

        evaluation_context = (
            EvaluationContext()
            if context is None
            else EvaluationContext.from_object(context)
        )
        evaluation_context.enter(self)
        try:
            return self._evaluate(evaluation_context)
        except ExpressionEvaluationError:
            raise
        except Exception as exc:
            raise ExpressionEvaluationError(
                "Expression evaluation failed.",
                diagnostic=ExpressionDiagnostic(
                    code="EXPRESSION_EVALUATION_FAILED",
                    message="Expression evaluation failed.",
                    expression=self.to_canonical(),
                    dependencies=tuple(sorted(self.dependencies)),
                    stage="evaluation",
                    cause_type=type(exc).__name__,
                ),
                cause=exc,
            ) from exc
        finally:
            evaluation_context.exit(self)

    def explain(self) -> dict[str, Any]:
        """Return an inspectable, serialization-friendly expression summary."""

        return {
            "expression": self.to_canonical(),
            "dependencies": sorted(self.dependencies),
            "type": type(self).__name__,
        }

    def __add__(self, other: Any) -> "Expression":
        from .operations import BinaryExpression
        return BinaryExpression("add", self, as_expression(other))

    def __radd__(self, other: Any) -> "Expression":
        return as_expression(other).__add__(self)

    def __sub__(self, other: Any) -> "Expression":
        from .operations import BinaryExpression
        return BinaryExpression("sub", self, as_expression(other))

    def __rsub__(self, other: Any) -> "Expression":
        return as_expression(other).__sub__(self)

    def __mul__(self, other: Any) -> "Expression":
        from .operations import BinaryExpression
        return BinaryExpression("mul", self, as_expression(other))

    def __rmul__(self, other: Any) -> "Expression":
        return as_expression(other).__mul__(self)

    def __truediv__(self, other: Any) -> "Expression":
        from .operations import BinaryExpression
        return BinaryExpression("truediv", self, as_expression(other))

    def __rtruediv__(self, other: Any) -> "Expression":
        return as_expression(other).__truediv__(self)

    def __pow__(self, other: Any) -> "Expression":
        from .operations import BinaryExpression
        return BinaryExpression("pow", self, as_expression(other))

    def __rpow__(self, other: Any) -> "Expression":
        return as_expression(other).__pow__(self)

    def __neg__(self) -> "Expression":
        from .operations import UnaryExpression
        return UnaryExpression("neg", self)

    def __pos__(self) -> "Expression":
        return self

    def __float__(self) -> float:
        """Prevent accidental early conversion unless the expression is resolvable."""

        return float(self.eval())

    def __int__(self) -> int:
        return int(self.eval())

    def __bool__(self) -> bool:
        raise TypeError(
            "Symbolic expressions cannot be used as booleans before resolution. "
            "Evaluate the expression explicitly with expression.eval(context)."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_canonical()})"


def as_expression(value: Any) -> Expression:
    """Promote a supported literal or preserve an existing expression."""

    if isinstance(value, Expression):
        return value

    from .literals import LiteralExpression

    return LiteralExpression(value)
