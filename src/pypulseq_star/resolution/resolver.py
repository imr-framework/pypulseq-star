"""First-pass generic Phase 2 sequence resolver."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from pypulseq_star.expressions import (
    Expression,
    ExpressionDiagnostic,
    ExpressionEvaluationError,
    RangeValidationError,
)

from .resolved_protocol import ResolvedProtocol
from .resolved_sequence import ResolvedSequence


def resolve_sequence(sequence: Any, *, overrides: Mapping[str, Any] | None = None) -> ResolvedSequence:
    """Resolve symbolic event properties into one numeric realization.

    The symbolic source sequence is never mutated. Resolution proceeds in
    repeated passes so later expressions such as TR fill can depend on earlier
    resolved expressions such as TE fill.
    """

    overrides = dict(overrides or {})
    realized = copy.deepcopy(sequence)

    protocol_values = _resolved_protocol_values(realized, overrides)
    _apply_protocol_values(realized, protocol_values)

    pending = _collect_symbolic_bindings(realized)
    diagnostics: list[ExpressionDiagnostic] = []

    max_passes = max(len(pending) + 2, 4)
    last_errors: dict[str, BaseException] = {}

    for _ in range(max_passes):
        if not pending:
            break

        progress = False
        remaining: list[tuple[Any, str, Expression, str]] = []

        for event, property_name, expression, property_path in pending:
            try:
                value = expression.eval(realized)
                _validate_resolved_value(
                    value,
                    property_name=property_name,
                    property_path=property_path,
                    expression=expression,
                )
                _apply_event_value(event, property_name, value)
                _mark_binding_resolved(event, property_name, value)
                progress = True
                last_errors.pop(property_path, None)
            except ExpressionEvaluationError as exc:
                remaining.append((event, property_name, expression, property_path))
                last_errors[property_path] = exc
            except Exception as exc:
                remaining.append((event, property_name, expression, property_path))
                last_errors[property_path] = exc

        pending = remaining
        if not progress:
            break

    if pending:
        event, property_name, expression, property_path = pending[0]
        cause = last_errors.get(property_path)
        if isinstance(cause, ExpressionEvaluationError):
            raise cause
        raise ExpressionEvaluationError(
            f"Could not resolve symbolic event property {property_path!r}.",
            diagnostic=ExpressionDiagnostic(
                code="EVENT_PROPERTY_RESOLUTION_FAILED",
                message=f"Could not resolve symbolic event property {property_path!r}.",
                property_path=property_path,
                expression=expression.to_canonical(),
                dependencies=tuple(sorted(expression.dependencies)),
                stage="sequence_resolution",
                cause_type=type(cause).__name__ if cause is not None else None,
            ),
            cause=cause,
        ) from cause

    _rebuild_symbolic_constructors(realized)
    _resolve_node_controls(realized)
    _synchronize_protocol(realized, protocol_values)

    protocol_aliases = _resolved_protocol_aliases(realized)

    return ResolvedSequence(
        source=sequence,
        sequence=realized,
        protocol=ResolvedProtocol(
            protocol_values,
            aliases=protocol_aliases,
        ),
        diagnostics=tuple(diagnostics),
    )


def _resolved_protocol_aliases(sequence: Any) -> dict[str, str]:
    """Return the merged protocol alias map for a realization."""

    protocol = getattr(sequence, "protocol", None)
    aliases: dict[str, str] = {}

    alias_map_method = getattr(protocol, "alias_map", None)
    if callable(alias_map_method):
        try:
            aliases.update({str(k): str(v) for k, v in alias_map_method().items()})
        except Exception:
            pass

    built_in_method = getattr(protocol, "built_in_aliases", None)
    if callable(built_in_method):
        try:
            for alias, canonical in built_in_method().items():
                aliases.setdefault(str(alias), str(canonical))
        except Exception:
            pass

    protocol_parameters = getattr(protocol, "parameters", None)
    if isinstance(protocol_parameters, Mapping):
        for canonical in protocol_parameters:
            aliases.setdefault(str(canonical), str(canonical))

    sequence_parameters = getattr(sequence, "parameters", None)
    if isinstance(sequence_parameters, Mapping):
        for canonical in sequence_parameters:
            if not str(canonical).startswith("_"):
                aliases.setdefault(str(canonical), str(canonical))

    return aliases


def _resolved_protocol_values(sequence: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    protocol = getattr(sequence, "protocol", None)
    protocol_parameters = getattr(protocol, "parameters", None)
    if isinstance(protocol_parameters, Mapping):
        values.update(protocol_parameters)

    parameters = getattr(sequence, "parameters", None)
    if isinstance(parameters, Mapping):
        for key, value in parameters.items():
            if not str(key).startswith("_"):
                values.setdefault(str(key), value)

    canonical_name = getattr(protocol, "canonical_name", None)
    for key, value in overrides.items():
        canonical = canonical_name(key) if callable(canonical_name) else key
        values[str(canonical)] = value

    # Protocol values may themselves be expressions.
    unresolved = dict(values)
    for _ in range(max(len(unresolved) + 1, 2)):
        progress = False
        for key, value in list(unresolved.items()):
            if isinstance(value, Expression):
                try:
                    values[key] = value.eval(values)
                    unresolved.pop(key)
                    progress = True
                except ExpressionEvaluationError:
                    continue
            else:
                unresolved.pop(key)
                progress = True
        if not unresolved or not progress:
            break

    return values


def _apply_protocol_values(sequence: Any, values: Mapping[str, Any]) -> None:
    if hasattr(sequence, "parameters") and isinstance(sequence.parameters, dict):
        sequence.parameters.update(values)

    protocol = getattr(sequence, "protocol", None)
    if protocol is not None and hasattr(protocol, "parameters"):
        try:
            protocol.parameters = dict(values)
        except Exception:
            pass


def _collect_symbolic_bindings(sequence: Any) -> list[tuple[Any, str, Expression, str]]:
    bindings: list[tuple[Any, str, Expression, str]] = []

    for block in getattr(sequence.timeline, "blocks", []):
        for event in _iter_events(block):
            specs = _symbolic_specs(event)
            for property_name, expression in specs.items():
                if not isinstance(expression, Expression):
                    continue
                path = (
                    _metadata_value(event, "sequence_event_path")
                    or getattr(event, "path", None)
                    or getattr(event, "name", None)
                    or event.__class__.__name__
                )
                bindings.append(
                    (event, str(property_name), expression, f"{path}.{property_name}")
                )

    # Stable ordering: non-delay event properties first, delay durations last.
    bindings.sort(
        key=lambda item: (
            item[1] in {"duration", "delay"} and _is_delay_event(item[0]),
            item[3],
        )
    )
    return bindings


def _symbolic_specs(event: Any) -> dict[str, Any]:
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        specs = metadata.get("symbolic_properties")
        if isinstance(specs, Mapping):
            return dict(specs)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        specs = parameters.get("_symbolic_properties")
        if isinstance(specs, Mapping):
            return dict(specs)

    return {}


def _apply_event_value(event: Any, property_name: str, value: Any) -> None:
    numeric_value = _coerce_property_value(property_name, value)

    for target_name in _property_targets(property_name):
        try:
            setattr(event, target_name, numeric_value)
        except Exception:
            pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters[property_name] = numeric_value
        if property_name == "duration":
            parameters["shape_duration"] = numeric_value
        if property_name == "num_samples":
            parameters["number_of_samples"] = int(numeric_value)

    timing = getattr(event, "timing", None)
    if timing is not None:
        if property_name == "delay":
            for attr in ("delay", "tstart"):
                try:
                    setattr(timing, attr, float(numeric_value))
                except Exception:
                    pass
        elif property_name == "duration":
            try:
                setattr(timing, "duration", float(numeric_value))
            except Exception:
                pass

    shape = getattr(event, "shape", None)
    if shape is not None:
        if property_name == "duration":
            try:
                setattr(shape, "duration", float(numeric_value))
            except Exception:
                pass

    # Delay events frequently represent duration through their delay field.
    if _is_delay_event(event) and property_name in {"duration", "delay"}:
        for attr in ("duration", "delay"):
            try:
                setattr(event, attr, float(numeric_value))
            except Exception:
                pass
        if timing is not None:
            try:
                setattr(timing, "duration", float(numeric_value))
            except Exception:
                pass


def _property_targets(property_name: str) -> tuple[str, ...]:
    aliases = {
        "freq_offset": ("freq_offset", "frequency_offset"),
        "phase_offset": ("phase_offset",),
        "num_samples": ("num_samples",),
        "flip_angle": ("flip_angle",),
        "dwell": ("dwell",),
        "duration": ("duration",),
        "delay": ("delay",),
    }
    return aliases.get(property_name, (property_name,))


def _coerce_property_value(property_name: str, value: Any) -> Any:
    if property_name in {"num_samples", "num_echoes", "repeat_count"}:
        return int(value)
    if property_name in {
        "duration",
        "delay",
        "dwell",
        "flip_angle",
        "phase_offset",
        "freq_offset",
        "frequency_offset",
        "amplitude",
        "area",
        "rise_time",
        "flat_time",
        "fall_time",
    }:
        return float(value)
    return value


def _validate_resolved_value(
    value: Any,
    *,
    property_name: str,
    property_path: str,
    expression: Expression,
) -> None:
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RangeValidationError(
                f"Resolved value for {property_path!r} is not finite.",
                diagnostic=ExpressionDiagnostic(
                    code="NONFINITE_RESOLVED_VALUE",
                    message=f"Resolved value for {property_path!r} is not finite.",
                    property_path=property_path,
                    expression=expression.to_canonical(),
                    dependencies=tuple(sorted(expression.dependencies)),
                    evaluated_value=value,
                    constraint="finite numeric value",
                    stage="sequence_resolution",
                ),
            )

        if property_name in {
            "duration",
            "delay",
            "dwell",
            "rise_time",
            "flat_time",
            "fall_time",
        } and float(value) < -1e-12:
            raise RangeValidationError(
                f"Resolved timing value for {property_path!r} is negative.",
                diagnostic=ExpressionDiagnostic(
                    code="NEGATIVE_TIMING_VALUE",
                    message=f"Resolved timing value for {property_path!r} is negative.",
                    property_path=property_path,
                    expression=expression.to_canonical(),
                    dependencies=tuple(sorted(expression.dependencies)),
                    evaluated_value=value,
                    constraint="value >= 0",
                    stage="sequence_resolution",
                    unit="s",
                ),
            )


def _mark_binding_resolved(event: Any, property_name: str, value: Any) -> None:
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("resolved_symbolic_properties", {})[property_name] = value
        placeholders = metadata.get("symbolic_placeholders")
        if isinstance(placeholders, dict):
            placeholders.pop(property_name, None)


def _rebuild_symbolic_constructors(sequence: Any) -> None:
    """Rebuild events whose geometry depends on resolved constructor inputs.

    Assigning ``event.duration`` alone is insufficient for trapezoids because
    amplitude, rise time, flat time, fall time, area, shape duration, and event
    timing are coupled. Reconstructing from the retained constructor
    specification keeps those values consistent.
    """

    for block_index, block in enumerate(
        getattr(sequence.timeline, "blocks", [])
    ):
        for event_index, event in enumerate(_iter_events(block)):
            constructor = _symbolic_constructor(event)
            if not constructor:
                continue

            family = str(constructor.get("family", "")).strip().lower()
            if family == "trapezoid":
                _rebuild_trapezoid_event(
                    event,
                    constructor=constructor,
                    sequence=sequence,
                    block=block,
                    block_index=block_index,
                    event_index=event_index,
                )
            elif family == "sinc":
                _rebuild_sinc_rf_event(
                    event,
                    constructor=constructor,
                    sequence=sequence,
                    block=block,
                    block_index=block_index,
                    event_index=event_index,
                )


def _symbolic_constructor(event: Any) -> dict[str, Any] | None:
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        constructor = metadata.get("symbolic_constructor")
        if isinstance(constructor, Mapping):
            return dict(constructor)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        constructor = parameters.get("_symbolic_constructor")
        if isinstance(constructor, Mapping):
            return dict(constructor)

    return None


def _resolve_constructor_value(value: Any, sequence: Any) -> Any:
    if isinstance(value, Expression):
        return value.eval(sequence)
    if isinstance(value, Mapping):
        return {
            key: _resolve_constructor_value(item, sequence)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_resolve_constructor_value(item, sequence) for item in value)
    if isinstance(value, list):
        return [_resolve_constructor_value(item, sequence) for item in value]
    return value


def _rebuild_trapezoid_event(
    event: Any,
    *,
    constructor: Mapping[str, Any],
    sequence: Any,
    block: Any | None = None,
    block_index: int | None = None,
    event_index: int | None = None,
) -> None:
    """Reconstruct one trapezoid from its fully resolved constructor inputs."""

    from pypulseq_star.make.grad import make_trapezoid

    raw_specs = constructor.get("specs", {})
    if not isinstance(raw_specs, Mapping):
        return

    specs = {
        key: _resolve_constructor_value(value, sequence)
        for key, value in raw_specs.items()
    }

    # Event metadata contains sequence provenance and variation bindings that
    # must survive reconstruction. Constructor metadata is supplied separately.
    original_metadata = getattr(event, "metadata", None)
    preserved_metadata = (
        dict(original_metadata)
        if isinstance(original_metadata, Mapping)
        else {}
    )
    constructor_metadata = specs.get("metadata")
    if not isinstance(constructor_metadata, Mapping):
        constructor_metadata = {}

    system = (
        getattr(event, "system", None)
        or getattr(sequence, "system", None)
    )

    try:
        rebuilt = make_trapezoid(
            channel=(
                None
                if specs.get("channel") is None
                else str(specs["channel"])
            ),
            amplitude=specs.get("amplitude"),
            area=specs.get("area"),
            delay=specs.get("delay", 0.0),
            duration=specs.get("duration"),
            fall_time=specs.get("fall_time"),
            flat_area=specs.get("flat_area"),
            flat_time=specs.get("flat_time"),
            max_grad=specs.get("max_grad"),
            max_slew=specs.get("max_slew"),
            rise_time=specs.get("rise_time"),
            system=system,
            name=specs.get("name") or getattr(event, "name", None),
            role=specs.get("role") or getattr(event, "role", None),
            axis_role=(
                specs.get("axis_role")
                or getattr(event, "axis_role", None)
            ),
            encoding_role=(
                specs.get("encoding_role")
                or getattr(event, "encoding_role", None)
            ),
            polarity=(
                specs.get("polarity")
                if specs.get("polarity") is not None
                else getattr(event, "polarity", None)
            ),
            metadata=dict(constructor_metadata),
        )
    except (ValueError, NotImplementedError) as exc:
        raise ValueError(
            _trapezoid_rebuild_error_message(
                event=event,
                block=block,
                block_index=block_index,
                event_index=event_index,
                specs=specs,
                system=system,
                cause=exc,
            )
        ) from exc

    rebuilt_shape = getattr(rebuilt, "shape", None)
    if rebuilt_shape is not None:
        try:
            event.shape = rebuilt_shape
        except Exception:
            pass

    # Copy public gradient properties where setters exist. Most are derived
    # from shape, but this keeps older event implementations synchronized.
    for property_name in (
        "amplitude",
        "area",
        "delay",
        "duration",
        "rise_time",
        "flat_time",
        "fall_time",
        "flat_area",
    ):
        try:
            value = getattr(rebuilt, property_name)
        except Exception:
            continue
        try:
            setattr(event, property_name, value)
        except Exception:
            pass

    rebuilt_parameters = getattr(rebuilt, "parameters", None)
    event_parameters = getattr(event, "parameters", None)
    if isinstance(event_parameters, dict):
        if isinstance(rebuilt_parameters, Mapping):
            for key, value in rebuilt_parameters.items():
                if not str(key).startswith("_symbolic"):
                    event_parameters[key] = value
        event_parameters.setdefault(
            "resolved_symbolic_constructor",
            {},
        ).update(
            {
                "family": "trapezoid",
                "duration": getattr(event, "duration", None),
                "area": getattr(event, "area", None),
                "amplitude": getattr(event, "amplitude", None),
            }
        )

    if isinstance(original_metadata, dict):
        rebuilt_metadata = getattr(rebuilt, "metadata", None)
        if isinstance(rebuilt_metadata, Mapping):
            original_metadata.update(rebuilt_metadata)
        original_metadata.update(preserved_metadata)
        original_metadata.setdefault(
            "resolved_symbolic_constructor",
            {},
        ).update(
            {
                "family": "trapezoid",
                "duration": getattr(event, "duration", None),
                "area": getattr(event, "area", None),
                "amplitude": getattr(event, "amplitude", None),
            }
        )

    timing = getattr(event, "timing", None)
    if timing is not None:
        try:
            timing.tstart = float(getattr(event, "delay", 0.0) or 0.0)
        except Exception:
            pass
        try:
            timing.duration = float(
                getattr(event, "duration", 0.0) or 0.0
            )
        except Exception:
            pass




def _rebuild_sinc_rf_event(
    event: Any,
    *,
    constructor: Mapping[str, Any],
    sequence: Any,
    block: Any | None = None,
    block_index: int | None = None,
    event_index: int | None = None,
) -> None:
    """Reconstruct one sinc RF event from its resolved scientific inputs.

    Sinc RF realization can depend on scanner hardware limits. In particular,
    ``rf_constraint_policy="stretch"`` may lengthen the pulse so that its peak
    B1 does not exceed ``system.max_rf``. Applying a symbolic duration directly
    to an already-realized RF event would undo that adaptation. Rebuilding here
    restores a self-consistent event/shape realization after protocol
    expressions have resolved.
    """

    from pypulseq_star.make.rf import make_sinc_pulse

    raw_specs = constructor.get("specs", {})
    if not isinstance(raw_specs, Mapping):
        return

    # Constructor records may contain a hardware-realized numeric duration,
    # while symbolic_properties retains the authored protocol expression.
    # Overlay symbolic constructor inputs before evaluation so a new
    # realization remains responsive to protocol overrides.
    merged_specs = dict(raw_specs)
    symbolic = _symbolic_specs(event)
    for key in (
        "flip_angle",
        "duration",
        "phase_offset",
        "freq_offset",
        "time_bw_product",
        "apodization",
        "center_pos",
        "delay",
        "slice_thickness",
    ):
        value = symbolic.get(key)
        if isinstance(value, Expression):
            merged_specs[key] = value

    specs = {
        key: _resolve_constructor_value(value, sequence)
        for key, value in merged_specs.items()
    }

    system = (
        getattr(event, "system", None)
        or getattr(sequence, "system", None)
    )

    if system is None:
        raise ValueError(
            "Cannot rebuild symbolic sinc RF event without scanner system limits."
        )

    event_name = (
        specs.get("name")
        or getattr(event, "name", None)
        or _metadata_value(event, "source_event_name", None)
        or "rf"
    )

    try:
        rebuilt = make_sinc_pulse(
            flip_angle=specs.get("flip_angle"),
            duration=specs.get("duration"),
            name=str(event_name),
            use=specs.get("use") or getattr(event, "use", None),
            phase_offset=specs.get("phase_offset"),
            freq_offset=specs.get("freq_offset"),
            time_bw_product=specs.get("time_bw_product"),
            apodization=specs.get("apodization"),
            center_pos=specs.get("center_pos"),
            rf_constraint_policy=str(
                specs.get("rf_constraint_policy", "strict")
            ),
            delay=specs.get("delay"),
            system=system,
            parameters=getattr(sequence, "parameters", None),
            slice_thickness=specs.get("slice_thickness"),
            max_grad=specs.get("max_grad"),
            max_slew=specs.get("max_slew"),
            return_gz=False,
            channel=specs.get("channel"),
            axis_role=(
                specs.get("axis_role")
                or getattr(event, "axis_role", None)
            ),
            gz_name=specs.get("gz_name"),
            gzr_name=specs.get("gzr_name"),
        )
    except (ValueError, NotImplementedError) as exc:
        block_name = (
            getattr(block, "name", None)
            if block is not None
            else None
        ) or "<unnamed block>"

        location_parts = [f"block={block_name!r}"]
        if block_index is not None:
            location_parts.append(f"block_index={block_index}")
        if event_index is not None:
            location_parts.append(f"event_index={event_index}")

        raise ValueError(
            "Failed to rebuild symbolic sinc RF during sequence resolution. "
            f"event={str(event_name)!r}, "
            f"{', '.join(location_parts)}, "
            f"duration={specs.get('duration')!r}, "
            f"flip_angle={specs.get('flip_angle')!r}, "
            f"time_bw_product={specs.get('time_bw_product')!r}, "
            f"rf_constraint_policy={specs.get('rf_constraint_policy', 'strict')!r}, "
            f"system.max_rf={getattr(system, 'max_rf', None)!r}. "
            f"Underlying error: {exc}"
        ) from exc

    # Keep the original symbolic/provenance stores, but replace all realized
    # RF geometry/timing with the freshly hardware-constrained realization.
    original_metadata = getattr(event, "metadata", None)
    preserved_metadata = (
        dict(original_metadata)
        if isinstance(original_metadata, Mapping)
        else {}
    )

    original_parameters = getattr(event, "parameters", None)
    preserved_parameters = (
        dict(original_parameters)
        if isinstance(original_parameters, Mapping)
        else {}
    )

    rebuilt_shape = getattr(rebuilt, "shape", None)
    if rebuilt_shape is not None:
        try:
            event.shape = rebuilt_shape
        except Exception:
            pass

    for property_name in (
        "flip_angle",
        "duration",
        "phase_offset",
        "freq_offset",
        "frequency_offset",
        "use",
        "role",
        "delay",
    ):
        try:
            value = getattr(rebuilt, property_name)
        except Exception:
            continue
        try:
            setattr(event, property_name, value)
        except Exception:
            pass

    rebuilt_parameters = getattr(rebuilt, "parameters", None)
    event_parameters = getattr(event, "parameters", None)
    if isinstance(event_parameters, dict):
        if isinstance(rebuilt_parameters, Mapping):
            for key, value in rebuilt_parameters.items():
                if not str(key).startswith("_symbolic"):
                    event_parameters[key] = value

        # Retain source expressions/constructor intent for provenance.
        for key, value in preserved_parameters.items():
            if str(key).startswith("_symbolic"):
                event_parameters[key] = value

        event_parameters.setdefault(
            "resolved_symbolic_constructor",
            {},
        ).update(
            {
                "family": "sinc",
                "duration": getattr(event, "duration", None),
                "flip_angle": getattr(event, "flip_angle", None),
                "time_bw_product": (
                    getattr(rebuilt_shape, "time_bw_product", None)
                    if rebuilt_shape is not None
                    else specs.get("time_bw_product")
                ),
                "rf_constraint_policy": specs.get(
                    "rf_constraint_policy",
                    "strict",
                ),
                "max_rf": getattr(system, "max_rf", None),
            }
        )

    if isinstance(original_metadata, dict):
        rebuilt_metadata = getattr(rebuilt, "metadata", None)
        if isinstance(rebuilt_metadata, Mapping):
            for key, value in rebuilt_metadata.items():
                if key not in {
                    "symbolic_constructor",
                    "symbolic_properties",
                    "symbolic_placeholders",
                    "resolved_symbolic_properties",
                }:
                    original_metadata[key] = value

        # Preserve source-side symbolic relationships and provenance.
        for key in (
            "symbolic_constructor",
            "symbolic_properties",
            "symbolic_placeholders",
            "resolved_symbolic_properties",
            "sequence_event_path",
            "source_event_name",
        ):
            if key in preserved_metadata:
                original_metadata[key] = preserved_metadata[key]

        original_metadata.setdefault(
            "resolved_symbolic_constructor",
            {},
        ).update(
            {
                "family": "sinc",
                "duration": getattr(event, "duration", None),
                "flip_angle": getattr(event, "flip_angle", None),
                "time_bw_product": (
                    getattr(rebuilt_shape, "time_bw_product", None)
                    if rebuilt_shape is not None
                    else specs.get("time_bw_product")
                ),
                "rf_constraint_policy": specs.get(
                    "rf_constraint_policy",
                    "strict",
                ),
                "max_rf": getattr(system, "max_rf", None),
            }
        )

    timing = getattr(event, "timing", None)
    rebuilt_timing = getattr(rebuilt, "timing", None)
    if timing is not None:
        if rebuilt_timing is not None:
            for attr in ("tstart", "delay", "duration", "tend"):
                try:
                    setattr(timing, attr, getattr(rebuilt_timing, attr))
                except Exception:
                    pass
        else:
            try:
                timing.duration = float(
                    getattr(event, "duration", 0.0) or 0.0
                )
            except Exception:
                pass


def _trapezoid_rebuild_error_message(
    *,
    event: Any,
    block: Any | None,
    block_index: int | None,
    event_index: int | None,
    specs: Mapping[str, Any],
    system: Any,
    cause: BaseException,
) -> str:
    """Return a contextual error for a failed symbolic gradient rebuild."""

    event_name = (
        specs.get("name")
        or getattr(event, "name", None)
        or _metadata_value(event, "source_event_name", None)
        or "<unnamed gradient>"
    )
    block_name = (
        getattr(block, "name", None)
        if block is not None
        else None
    ) or "<unnamed block>"
    role = specs.get("role") or getattr(event, "role", None)
    axis_role = specs.get("axis_role") or getattr(event, "axis_role", None)
    channel = specs.get("channel") or getattr(event, "channel", None)

    location_parts = [f"block={block_name!r}"]
    if block_index is not None:
        location_parts.append(f"block_index={block_index}")
    if event_index is not None:
        location_parts.append(f"event_index={event_index}")

    constructor_keys = (
        "area",
        "flat_area",
        "amplitude",
        "duration",
        "flat_time",
        "rise_time",
        "fall_time",
        "delay",
    )
    resolved_inputs = ", ".join(
        f"{key}={specs.get(key)!r}"
        for key in constructor_keys
        if specs.get(key) is not None
    ) or "<none>"

    limit_parts: list[str] = []
    max_grad = specs.get("max_grad")
    if max_grad is None and system is not None:
        max_grad = getattr(system, "max_grad", None)
    max_slew = specs.get("max_slew")
    if max_slew is None and system is not None:
        max_slew = getattr(system, "max_slew", None)

    if max_grad is not None:
        max_grad_text = f"max_grad={float(max_grad):.9g} Hz/m"
        gamma = getattr(system, "gamma", None) if system is not None else None
        if gamma:
            max_grad_text += (
                f" ({float(max_grad) / float(gamma) * 1e3:.6g} mT/m)"
            )
        limit_parts.append(max_grad_text)
    if max_slew is not None:
        limit_parts.append(f"max_slew={float(max_slew):.9g} Hz/m/s")

    limits = ", ".join(limit_parts) or "<not available>"

    return (
        "Failed to rebuild symbolic trapezoid during sequence resolution. "
        f"event={str(event_name)!r}, "
        f"{', '.join(location_parts)}, "
        f"role={role!r}, axis_role={axis_role!r}, channel={channel!r}. "
        f"Resolved constructor inputs: {resolved_inputs}. "
        f"System limits: {limits}. "
        f"Underlying error: {cause}"
    )

def _resolve_node_controls(sequence: Any) -> None:
    for record in getattr(sequence.timeline, "nodes", {}).values():
        for key in ("repeat_count", "repeat_every"):
            value = record.get(key)
            if isinstance(value, Expression):
                record[key] = value.eval(sequence)
        if record.get("repeat_count") is not None:
            record["repeat_count"] = int(record["repeat_count"])
        if record.get("repeat_every") is not None:
            record["repeat_every"] = float(record["repeat_every"])


def _synchronize_protocol(sequence: Any, values: Mapping[str, Any]) -> None:
    if hasattr(sequence, "parameters") and isinstance(sequence.parameters, dict):
        sequence.parameters.update(values)


def _iter_events(block: Any) -> list[Any]:
    events = getattr(block, "events", None)
    if isinstance(events, Mapping):
        return list(events.values())
    if isinstance(events, (list, tuple)):
        return list(events)
    try:
        return list(events or [])
    except Exception:
        return []


def _metadata_value(obj: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping) and key in parameters:
        return parameters[key]
    return getattr(obj, key, default)


def _is_delay_event(event: Any) -> bool:
    event_type = str(getattr(event, "type", getattr(event, "kind", ""))).lower()
    class_name = event.__class__.__name__.lower()
    role = str(getattr(event, "role", "")).lower()
    return event_type == "delay" or "delay" in class_name or "delay" in role
