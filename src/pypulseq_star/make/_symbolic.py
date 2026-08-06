"""Generic symbolic-to-numeric bridge for event constructors.

Constructors need numeric values to build today's event/shape objects, while
Phase 2 also needs to retain the developer's symbolic specification. This
module centralizes that bridge so RF, gradient, ADC, and delay constructors do
not grow sequence-specific symbolic branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pypulseq_star.expressions import EventAnchorRef, Expression


_UNRESOLVED = object()


def is_expression(value: Any) -> bool:
    """Return whether ``value`` is a PyPulseq-Star symbolic expression."""

    return isinstance(value, Expression)


def evaluate_default(
    value: Any,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> Any:
    """Evaluate an expression using its protocol defaults when possible.

    Sequence-dependent expressions may not be evaluable during event
    construction. In that case, ``fallback`` is returned when supplied.
    """

    if not is_expression(value):
        return value

    try:
        return value.eval(context)
    except Exception as exc:
        if fallback is not _UNRESOLVED:
            return fallback
        label = field_name or "value"
        raise ValueError(
            f"Could not evaluate symbolic {label!r} during event construction. "
            "Provide protocol defaults or defer this property through an event "
            "binding that is resolved by Sequence.resolve()."
        ) from exc


def resolve_float(
    value: Any,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> float:
    """Return a numeric float from a literal or symbolic specification."""

    resolved = evaluate_default(
        value,
        context=context,
        fallback=fallback,
        field_name=field_name,
    )
    return float(resolved)


def resolve_int(
    value: Any,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> int:
    """Return a numeric integer from a literal or symbolic specification."""

    resolved = evaluate_default(
        value,
        context=context,
        fallback=fallback,
        field_name=field_name,
    )
    numeric = int(resolved)
    if isinstance(resolved, float) and not resolved.is_integer():
        label = field_name or "value"
        raise ValueError(
            f"Symbolic {label!r} resolved to non-integral value {resolved!r}."
        )
    return numeric


def resolve_string(
    value: Any,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> str:
    """Return a string from a literal or symbolic specification."""

    resolved = evaluate_default(
        value,
        context=context,
        fallback=fallback,
        field_name=field_name,
    )
    return str(resolved)


def resolve_optional_float(
    value: Any | None,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> float | None:
    """Resolve an optional numeric constructor argument."""

    if value is None:
        return None
    return resolve_float(
        value,
        context=context,
        fallback=fallback,
        field_name=field_name,
    )


def resolve_optional_int(
    value: Any | None,
    *,
    context: Any | None = None,
    fallback: Any = _UNRESOLVED,
    field_name: str | None = None,
) -> int | None:
    """Resolve an optional integer constructor argument."""

    if value is None:
        return None
    return resolve_int(
        value,
        context=context,
        fallback=fallback,
        field_name=field_name,
    )


def require_positive(
    value: float | int | None,
    *,
    field_name: str,
    allow_none: bool = False,
) -> float | int | None:
    """Validate positivity only after symbolic resolution."""

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be supplied.")
    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive. Passed: {value!r}."
        )
    return value


def require_nonnegative(
    value: float | int | None,
    *,
    field_name: str,
    allow_none: bool = False,
) -> float | int | None:
    """Validate non-negativity only after symbolic resolution."""

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be supplied.")
    if value < 0:
        raise ValueError(
            f"{field_name} must be non-negative. Passed: {value!r}."
        )
    return value


def symbolic_specs(**values: Any) -> dict[str, Expression]:
    """Return only symbolic constructor arguments."""

    return {
        name: value
        for name, value in values.items()
        if isinstance(value, Expression)
    }


def attach_symbolic_specs(
    event: Any,
    specs: Mapping[str, Expression] | None,
    *,
    placeholders: Mapping[str, Any] | None = None,
) -> Any:
    """Attach symbolic event-property specifications and install anchor support.

    The original expression is kept in both metadata and parameters when those
    stores exist. Numeric event fields remain compatible with current shapes,
    plotters, timing checks, and writers until the dedicated resolver replaces
    them with a final realization.
    """

    specs = dict(specs or {})
    placeholders = dict(placeholders or {})

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("symbolic_properties", {}).update(specs)
        if placeholders:
            metadata.setdefault("symbolic_placeholders", {}).update(placeholders)
        metadata["has_symbolic_properties"] = bool(specs)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters.setdefault("_symbolic_properties", {}).update(specs)
        if placeholders:
            parameters.setdefault("_symbolic_placeholders", {}).update(placeholders)

    _install_anchor_api(type(event))
    return event


def symbolic_property(event: Any, name: str) -> Expression | None:
    """Return an event's symbolic specification for one property."""

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        specs = metadata.get("symbolic_properties")
        if isinstance(specs, Mapping):
            value = specs.get(name)
            if isinstance(value, Expression):
                return value

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        specs = parameters.get("_symbolic_properties")
        if isinstance(specs, Mapping):
            value = specs.get(name)
            if isinstance(value, Expression):
                return value

    return None


def has_unresolved_symbolic_property(event: Any, name: str) -> bool:
    """Return whether a property currently uses a construction placeholder."""

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        placeholders = metadata.get("symbolic_placeholders")
        return isinstance(placeholders, Mapping) and name in placeholders
    return False


def _install_anchor_api(event_type: type[Any]) -> None:
    """Install the common symbolic anchor method on an event class once."""

    if hasattr(event_type, "anchor"):
        return

    def anchor(event: Any, name: str) -> EventAnchorRef:
        anchor_name = str(name).strip().lower()
        if anchor_name not in {"origin", "start", "center", "end"}:
            raise ValueError(
                "Event anchor must be one of: origin, start, center, end. "
                f"Passed: {name!r}."
            )

        identity = (
            getattr(event, "name", None)
            or getattr(event, "path", None)
            or f"{event.__class__.__name__}:{id(event)}"
        )
        return EventAnchorRef(
            str(identity),
            anchor_name,
            metadata={"event": event},
        )

    setattr(event_type, "anchor", anchor)
