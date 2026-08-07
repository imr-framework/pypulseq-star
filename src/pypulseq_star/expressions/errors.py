"""Structured exceptions raised by the PyPulseq-Star expression layer."""

from __future__ import annotations

from typing import Any

from .diagnostics import ExpressionDiagnostic


class ExpressionError(Exception):
    """Base class for expression-related failures."""

    default_code = "EXPRESSION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        diagnostic: ExpressionDiagnostic | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.diagnostic = diagnostic or ExpressionDiagnostic(
            code=self.default_code,
            message=message,
            cause_type=type(cause).__name__ if cause is not None else None,
        )
        self.cause = cause
        super().__init__(self.diagnostic.format_multiline())

    def to_dict(self) -> dict[str, Any]:
        return self.diagnostic.to_dict()


class ExpressionConstructionError(ExpressionError):
    default_code = "EXPRESSION_CONSTRUCTION_ERROR"


class InvalidExpressionError(ExpressionConstructionError):
    default_code = "INVALID_EXPRESSION"


class ExpressionEvaluationError(ExpressionError):
    default_code = "EXPRESSION_EVALUATION_ERROR"


class UnresolvedDependencyError(ExpressionEvaluationError):
    default_code = "UNRESOLVED_DEPENDENCY"

    def __init__(
        self,
        dependencies: set[str] | frozenset[str] | tuple[str, ...],
        *,
        expression: str | None = None,
        property_path: str | None = None,
        stage: str = "evaluation",
    ) -> None:
        ordered = tuple(sorted(str(item) for item in dependencies))
        super().__init__(
            "Cannot evaluate expression because one or more dependencies are unresolved.",
            diagnostic=ExpressionDiagnostic(
                code=self.default_code,
                message="Cannot evaluate expression because one or more dependencies are unresolved.",
                property_path=property_path,
                expression=expression,
                dependencies=ordered,
                stage=stage,
            ),
        )
        self.dependencies = ordered


class CircularExpressionError(ExpressionEvaluationError):
    default_code = "CIRCULAR_EXPRESSION"


class UnknownReferenceError(ExpressionEvaluationError):
    default_code = "UNKNOWN_REFERENCE"


class ExpressionValidationError(ExpressionError):
    default_code = "EXPRESSION_VALIDATION_ERROR"


class RangeValidationError(ExpressionValidationError):
    default_code = "RANGE_VALIDATION_ERROR"


class TypeValidationError(ExpressionValidationError):
    default_code = "TYPE_VALIDATION_ERROR"


class RasterValidationError(ExpressionValidationError):
    default_code = "RASTER_VALIDATION_ERROR"


class HardwareLimitError(ExpressionValidationError):
    default_code = "HARDWARE_LIMIT_ERROR"
