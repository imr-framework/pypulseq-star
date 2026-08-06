"""Protocol-derived symbolic scalar expressions.

This package is intentionally independent of SymPy and of sequence-specific
logic. It provides the arithmetic and dependency graph used by protocol
parameters, event specifications, timing fills, and later writer backends.
"""

from .base import Expression, as_expression
from .context import EvaluationContext, ReferenceResolver
from .diagnostics import ExpressionDiagnostic
from .errors import (
    CircularExpressionError,
    ExpressionConstructionError,
    ExpressionError,
    ExpressionEvaluationError,
    ExpressionValidationError,
    HardwareLimitError,
    InvalidExpressionError,
    RangeValidationError,
    RasterValidationError,
    TypeValidationError,
    UnknownReferenceError,
    UnresolvedDependencyError,
)
from .literals import LiteralExpression
from .operations import BinaryExpression, UnaryExpression
from .parameters import ParameterNamespace, ParameterRef
from .references import (
    AnchorIntervalDurationRef,
    BlockRangeDurationRef,
    EventAnchorRef,
    EventPropertyRef,
    ReferenceExpression,
)

__all__ = [
    "AnchorIntervalDurationRef",
    "BinaryExpression",
    "BlockRangeDurationRef",
    "CircularExpressionError",
    "EvaluationContext",
    "EventAnchorRef",
    "EventPropertyRef",
    "Expression",
    "ExpressionError",
    "TypeValidationError",
    "RasterValidationError",
    "RangeValidationError",
    "HardwareLimitError",
    "ExpressionValidationError",
    "ExpressionDiagnostic",
    "ExpressionConstructionError",
    "ExpressionEvaluationError",
    "InvalidExpressionError",
    "LiteralExpression",
    "ParameterNamespace",
    "ParameterRef",
    "ReferenceExpression",
    "ReferenceResolver",
    "UnaryExpression",
    "UnknownReferenceError",
    "UnresolvedDependencyError",
    "as_expression",
]
