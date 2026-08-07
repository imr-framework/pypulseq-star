"""Protocol parameter references and symbolic namespaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .base import Expression
from .context import EvaluationContext
from .diagnostics import ExpressionDiagnostic
from .errors import TypeValidationError, UnresolvedDependencyError


@dataclass(frozen=True, slots=True, repr=False)
class ParameterRef(Expression):
    """Reference to one canonical protocol parameter."""

    canonical_name: str
    protocol: Any | None = None
    unit: str | None = None
    dtype: type[Any] | None = None
    display_name: str | None = None

    def _evaluate(self, context: EvaluationContext) -> Any:
        try:
            value = context.parameter_value(self.canonical_name)
        except KeyError:
            value = self._default_from_protocol()

        if value is _MISSING:
            raise UnresolvedDependencyError({self.dependency_key})

        if self.dtype is not None and value is not None:
            try:
                return self.dtype(value)
            except Exception as exc:
                raise TypeValidationError(
                    f"Protocol parameter {self.canonical_name!r} could not be converted "
                    f"to {self.dtype.__name__}.",
                    diagnostic=ExpressionDiagnostic(
                        code="PARAMETER_TYPE_CONVERSION_FAILED",
                        message=(
                            f"Protocol parameter {self.canonical_name!r} could not be "
                            f"converted to {self.dtype.__name__}."
                        ),
                        property_path=self.dependency_key,
                        expression=self.to_canonical(),
                        dependencies=(self.dependency_key,),
                        input_values={self.dependency_key: value},
                        stage="parameter_evaluation",
                        cause_type=type(exc).__name__,
                    ),
                    cause=exc,
                ) from exc
        return value

    def _default_from_protocol(self) -> Any:
        protocol = self.protocol
        if protocol is None:
            return _MISSING

        if hasattr(protocol, "get_parameter"):
            sentinel = object()
            try:
                value = protocol.get_parameter(self.canonical_name, sentinel)
                if value is not sentinel:
                    return value
            except TypeError:
                try:
                    value = protocol.get_parameter(self.canonical_name)
                    if value is not None:
                        return value
                except Exception:
                    pass

        parameters = getattr(protocol, "parameters", None)
        if isinstance(parameters, Mapping) and self.canonical_name in parameters:
            return parameters[self.canonical_name]

        return _MISSING

    @property
    def dependency_key(self) -> str:
        return f"protocol.{self.canonical_name}"

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset({self.dependency_key})

    def to_canonical(self) -> str:
        return self.canonical_name


class ParameterNamespace:
    """Attribute-based access to symbolic protocol parameters.

    This class is intended to back ``protocol.symbols`` in the next integration
    step. It accepts aliases but always returns a canonical ``ParameterRef``.
    """

    def __init__(
        self,
        protocol: Any,
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._protocol = protocol
        self._aliases = dict(aliases or {})
        self._cache: dict[str, ParameterRef] = {}

    def canonical_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    def __getattr__(self, name: str) -> ParameterRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, name: str) -> ParameterRef:
        canonical = self.canonical_name(name)
        if canonical not in self._cache:
            self._cache[canonical] = ParameterRef(
                canonical_name=canonical,
                protocol=self._protocol,
                unit=self._parameter_unit(canonical),
                dtype=self._parameter_dtype(canonical),
                display_name=name,
            )
        return self._cache[canonical]

    def _parameter_spec(self, canonical_name: str) -> Any | None:
        protocol_type = type(self._protocol)
        if hasattr(protocol_type, "parameter_spec"):
            try:
                return protocol_type.parameter_spec(canonical_name)
            except Exception:
                return None
        return None

    def _parameter_unit(self, canonical_name: str) -> str | None:
        spec = self._parameter_spec(canonical_name)
        return getattr(spec, "unit", None)

    def _parameter_dtype(self, canonical_name: str) -> type[Any] | None:
        spec = self._parameter_spec(canonical_name)
        value_type = getattr(spec, "value_type", None)
        if value_type == "integer":
            return int
        if value_type == "number":
            return float
        if value_type == "boolean":
            return bool
        if value_type == "string":
            return str
        return None

    def __dir__(self) -> list[str]:
        names = set(self._aliases)
        parameters = getattr(self._protocol, "parameters", None)
        if isinstance(parameters, Mapping):
            names.update(str(key) for key in parameters)
        return sorted(names)


_MISSING = object()
