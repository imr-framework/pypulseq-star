"""Loop-dependent event-property variation specifications.

The public API is intentionally compact.

Arithmetic progression
-----------------------

    kernel = seq.set_node(
        "kernel",
        factor=p.n_y,
        counter="ky_index",
        repeat_mode="loop",
    )

    kernel.vary(
        vary(
            gy_pre,
            "area",
            strength=-0.5 * p.n_y / p.fov,
            step=1.0 / p.fov,
        ),
        vary(
            [rf, adc],
            "phase_offset",
            strength=0.0,
            step=p.rf_spoiling_increment,
            mode="accumulated",
            wrap=2 * math.pi,
        ),
    )

Explicit table progression
--------------------------

    kernel.vary(
        vary(
            rf,
            "phase_offset",
            values=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
            mode="table",
            wrap=2 * math.pi,
        ),
    )

For table variation, loop index ``i`` selects::

    values[i % len(values)]

This allows reusable schedules such as RF/receiver phase cycles while keeping
the variation mechanism sequence-agnostic.

A variation is declarative. Arithmetic modes preserve symbolic ``strength``,
``step``, and the node's symbolic ``factor`` for writers/resolvers. Table mode
preserves the explicit ordered ``values`` instead.

Realization/writer layers may lower either form into backend-specific loop
relationships or expanded numeric event occurrences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


_ALLOWED_MODES = {
    "linear",
    "accumulated",
    "alternating",
    "table",
}


def _normalize_targets(
    event_or_events: Any,
) -> tuple[Any, ...]:
    """Normalize one or more event targets."""

    if isinstance(
        event_or_events,
        (list, tuple),
    ):
        targets = tuple(event_or_events)
    else:
        targets = (event_or_events,)

    if not targets or any(
        target is None
        for target in targets
    ):
        raise ValueError(
            "Variation requires at least one non-None event."
        )

    return targets


def _normalize_values(
    values: Iterable[Any] | None,
) -> tuple[Any, ...] | None:
    """Normalize an optional explicit variation table."""

    if values is None:
        return None

    if isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "Variation values must be an iterable of values, "
            "not a string or bytes object."
        )

    if isinstance(values, Mapping):
        raise TypeError(
            "Variation values must be an ordered iterable, "
            "not a mapping."
        )

    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "Variation values must be iterable."
        ) from exc

    if not normalized:
        raise ValueError(
            "Table variation requires at least one value."
        )

    return normalized


@dataclass(frozen=True, slots=True)
class SeqStarVariation:
    """One loop-dependent event-property progression.

    Arithmetic modes use ``strength`` and ``step``.

    Table mode uses ``values`` and selects values cyclically according to the
    repeat-node counter.
    """

    events: tuple[Any, ...]
    attribute: str

    # Existing arithmetic variation contract.
    #
    # Defaults are now None only so table variation can omit them. Existing
    # positional and keyword construction remains valid.
    strength: Any | None = None
    step: Any | None = None

    mode: str = "linear"
    wrap: Any | None = None

    # Explicit ordered values for mode="table".
    values: tuple[Any, ...] | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        attribute = str(
            self.attribute
        ).strip()

        mode = str(
            self.mode
        ).strip().lower()

        if not attribute:
            raise ValueError(
                "Variation attribute must not be empty."
            )

        if "." in attribute:
            raise ValueError(
                "Variation attribute must be a property name such as 'area' "
                "or 'phase_offset', not an event path."
            )

        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"Unsupported variation mode {mode!r}. "
                f"Expected one of {sorted(_ALLOWED_MODES)}."
            )

        normalized_values = _normalize_values(
            self.values
        )

        # ---------------------------------------------------------------------
        # Validate the two mutually exclusive variation forms.
        # ---------------------------------------------------------------------

        if mode == "table":
            if normalized_values is None:
                raise ValueError(
                    "Table variation requires values=..."
                )

            if (
                self.strength is not None
                or self.step is not None
            ):
                raise ValueError(
                    "Table variation uses values=... and must not also "
                    "specify strength or step."
                )

        else:
            if normalized_values is not None:
                raise ValueError(
                    "values=... is only valid with mode='table'."
                )

            if (
                self.strength is None
                or self.step is None
            ):
                raise ValueError(
                    f"Variation mode {mode!r} requires strength and step."
                )

        object.__setattr__(
            self,
            "attribute",
            attribute,
        )

        object.__setattr__(
            self,
            "mode",
            mode,
        )

        object.__setattr__(
            self,
            "values",
            normalized_values,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata or {}),
        )

    def to_record(
        self,
        *,
        node: str,
        counter: str,
        factor: Any,
    ) -> dict[str, Any]:
        """Return a writer/resolver-friendly declarative record."""

        return {
            "node": str(node),
            "counter": str(counter),
            "factor": factor,
            "attribute": self.attribute,

            # Existing arithmetic fields remain present for backward
            # compatibility. They are None for table variation.
            "strength": self.strength,
            "step": self.step,

            "mode": self.mode,
            "wrap": self.wrap,

            # New generic table field. Existing consumers can ignore it.
            "values": self.values,

            "metadata": dict(self.metadata),

            "event_names": [
                str(
                    getattr(
                        event,
                        "name",
                        None,
                    )
                    or getattr(
                        event,
                        "path",
                        None,
                    )
                    or event.__class__.__name__
                )
                for event in self.events
            ],

            "event_object_ids": [
                id(event)
                for event in self.events
            ],
        }


def vary(
    event_or_events: Any,
    attribute: str,
    *,
    strength: Any | None = None,
    step: Any | None = None,
    mode: str = "linear",
    wrap: Any | None = None,
    values: Iterable[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeqStarVariation:
    """Create one variation specification for ``node.vary(...)``.

    Arithmetic variation
    --------------------

    Existing usage is unchanged::

        vary(
            gy,
            "area",
            strength=start,
            step=delta,
        )

    Table variation
    ---------------

    Explicit values may be selected cyclically by the node counter::

        vary(
            rf,
            "phase_offset",
            values=(0.0, pi / 2, pi, 3 * pi / 2),
            mode="table",
            wrap=2 * pi,
        )

    Table semantics are::

        value = values[counter % len(values)]
    """

    return SeqStarVariation(
        events=_normalize_targets(
            event_or_events
        ),
        attribute=attribute,
        strength=strength,
        step=step,
        mode=mode,
        wrap=wrap,
        values=_normalize_values(values),
        metadata=dict(
            metadata or {}
        ),
    )


class SeqStarNodeHandle:
    """Public handle returned by ``Sequence.set_node``.

    It behaves like the historical node-record mapping for compatibility,
    while adding the fluent ``vary()`` API.
    """

    __slots__ = (
        "_sequence",
        "_name",
    )

    def __init__(
        self,
        sequence: Any,
        name: str,
    ) -> None:
        self._sequence = sequence
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def record(self) -> dict[str, Any]:
        record = self._sequence.timeline.get_node(
            self._name
        )

        if record is None:
            raise KeyError(
                f"Unknown sequence node {self._name!r}."
            )

        return record

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.record.get(
            key,
            default,
        )

    def __getitem__(
        self,
        key: str,
    ) -> Any:
        return self.record[key]

    def __contains__(
        self,
        key: object,
    ) -> bool:
        return key in self.record

    def __iter__(self):
        return iter(
            self.record
        )

    def __len__(self) -> int:
        return len(
            self.record
        )

    def keys(self):
        return self.record.keys()

    def items(self):
        return self.record.items()

    def values(self):
        return self.record.values()

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return dict(
            self.record
        )

    def vary(
        self,
        *variations: SeqStarVariation,
        event: Any | None = None,
        attribute: str | None = None,
        strength: Any | None = None,
        step: Any | None = None,
        mode: str = "linear",
        wrap: Any | None = None,
        values: Iterable[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SeqStarNodeHandle":
        """Attach one or more property progressions to this repeat node.

        Existing arithmetic convenience form::

            kernel.vary(
                event=gy_pre,
                attribute="area",
                strength=...,
                step=...,
            )

        New table convenience form::

            kernel.vary(
                event=rf,
                attribute="phase_offset",
                values=(
                    0.0,
                    math.pi / 2,
                    math.pi,
                    3 * math.pi / 2,
                ),
                mode="table",
            )

        Multiple-property form::

            kernel.vary(
                vary(
                    gy_pre,
                    "area",
                    strength=...,
                    step=...,
                ),
                vary(
                    [rf, adc],
                    "phase_offset",
                    strength=...,
                    step=...,
                ),
                vary(
                    rf2,
                    "phase_offset",
                    values=(...),
                    mode="table",
                ),
            )
        """

        # ---------------------------------------------------------------------
        # Determine whether the caller is using the inline convenience form.
        #
        # values is included here so positional specifications cannot be mixed
        # accidentally with an inline table declaration.
        # ---------------------------------------------------------------------

        has_inline = (
            any(
                value is not None
                for value in (
                    event,
                    attribute,
                    strength,
                    step,
                    wrap,
                    values,
                )
            )
            or mode != "linear"
            or bool(metadata)
        )

        if variations and has_inline:
            raise ValueError(
                "Use either positional vary(...) specifications or the inline "
                "event/attribute variation form, not both."
            )

        if not variations:
            if (
                event is None
                or attribute is None
            ):
                raise ValueError(
                    "Inline node.vary() requires event and attribute."
                )

            normalized_mode = str(
                mode
            ).strip().lower()

            if normalized_mode == "table":
                if values is None:
                    raise ValueError(
                        "Inline table node.vary() requires values."
                    )

                if (
                    strength is not None
                    or step is not None
                ):
                    raise ValueError(
                        "Inline table node.vary() uses values=... and must "
                        "not also specify strength or step."
                    )

            else:
                if (
                    strength is None
                    or step is None
                ):
                    raise ValueError(
                        "Inline node.vary() requires strength and step "
                        "for non-table variation."
                    )

                if values is not None:
                    raise ValueError(
                        "Inline values=... requires mode='table'."
                    )

            variations = (
                vary(
                    event,
                    attribute,
                    strength=strength,
                    step=step,
                    mode=mode,
                    wrap=wrap,
                    values=values,
                    metadata=metadata,
                ),
            )

        for specification in variations:
            if not isinstance(
                specification,
                SeqStarVariation,
            ):
                raise TypeError(
                    "node.vary() positional arguments must be produced "
                    "by vary(...)."
                )

            self._sequence._register_node_variation(
                self._name,
                specification,
            )

        return self


__all__ = [
    "SeqStarVariation",
    "SeqStarNodeHandle",
    "vary",
]