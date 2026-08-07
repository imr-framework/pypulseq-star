"""Evaluation contexts for symbolic scalar expressions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, runtime_checkable
from typing import Protocol as TypingProtocol

from .diagnostics import ExpressionDiagnostic
from .errors import CircularExpressionError, UnknownReferenceError


@runtime_checkable
class ReferenceResolver(TypingProtocol):
    """Protocol implemented by sequence/resolution objects that resolve references."""

    def resolve_expression_reference(
        self,
        kind: str,
        identity: str,
        property_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Resolve a deferred expression reference."""


@dataclass(slots=True)
class EvaluationContext:
    """Values and services used while evaluating an expression.

    Parameters
    ----------
    parameters:
        Canonical protocol parameter values.
    references:
        Optional direct values for deferred references. Keys use the stable
        reference key produced by ``ReferenceExpression.reference_key``.
    resolver:
        Optional object that can resolve event, block, node, anchor, or duration
        references dynamically.
    metadata:
        Free-form diagnostic information.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    resolver: ReferenceResolver | Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _evaluation_stack: list[int] = field(default_factory=list, repr=False)

    @classmethod
    def from_object(cls, source: Any) -> "EvaluationContext":
        """Build a context from a mapping, Protocol, Sequence, or realization.

        This adapter is intentionally permissive so the expression layer can be
        integrated incrementally with the existing repository.
        """

        if isinstance(source, EvaluationContext):
            return source

        if isinstance(source, Mapping):
            return cls(parameters=dict(source))

        parameters: dict[str, Any] = {}

        protocol = getattr(source, "protocol", None)
        protocol_parameters = getattr(protocol, "parameters", None)
        if isinstance(protocol_parameters, Mapping):
            parameters.update(protocol_parameters)

        source_parameters = getattr(source, "parameters", None)
        if isinstance(source_parameters, Mapping):
            for key, value in source_parameters.items():
                parameters.setdefault(str(key), value)

        if not parameters and hasattr(source, "to_dict"):
            try:
                payload = source.to_dict()
                candidate = payload.get("parameters", {})
                if isinstance(candidate, Mapping):
                    parameters.update(candidate)
            except Exception:
                pass

        return cls(parameters=parameters, resolver=source)

    def parameter_value(self, canonical_name: str) -> Any:
        """Return a canonical protocol parameter value."""

        if canonical_name in self.parameters:
            return self.parameters[canonical_name]
        raise KeyError(canonical_name)

    def reference_value(
        self,
        *,
        kind: str,
        identity: str,
        property_name: str | None,
        metadata: Mapping[str, Any] | None,
        reference_key: str,
    ) -> Any:
        """Return a deferred event/block/node reference value."""

        if reference_key in self.references:
            return self.references[reference_key]

        resolver = self.resolver
        if resolver is not None and hasattr(resolver, "resolve_expression_reference"):
            return resolver.resolve_expression_reference(
                kind,
                identity,
                property_name=property_name,
                metadata=metadata,
            )

        raise UnknownReferenceError(
            f"Cannot resolve expression reference {reference_key!r}.",
            diagnostic=ExpressionDiagnostic(
                code="UNKNOWN_REFERENCE",
                message=f"Cannot resolve expression reference {reference_key!r}.",
                expression=reference_key,
                dependencies=(reference_key,),
                stage="reference_resolution",
            ),
        )

    def enter(self, expression: Any) -> None:
        """Register an expression during evaluation for cycle detection."""

        marker = id(expression)
        if marker in self._evaluation_stack:
            raise CircularExpressionError(
                f"Circular expression dependency detected while evaluating {expression!r}.",
                diagnostic=ExpressionDiagnostic(
                    code="CIRCULAR_EXPRESSION",
                    message="Circular expression dependency detected.",
                    expression=getattr(expression, "to_canonical", lambda: repr(expression))(),
                    dependencies=tuple(sorted(getattr(expression, "dependencies", ()))),
                    stage="evaluation",
                ),
            )
        self._evaluation_stack.append(marker)

    def exit(self, expression: Any) -> None:
        """Remove an expression from the active evaluation stack."""

        marker = id(expression)
        if self._evaluation_stack and self._evaluation_stack[-1] == marker:
            self._evaluation_stack.pop()
            return

        if marker in self._evaluation_stack:
            self._evaluation_stack.remove(marker)
