"""Loop-dependent event-property variation specifications.

The public API is intentionally compact:

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

A variation is declarative. It preserves symbolic ``strength``, ``step``, and
the node's symbolic ``factor`` for gammaSTAR, while realization/writer layers
may lower the same declaration into expanded numeric event occurrences.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


_ALLOWED_MODES = {
    "linear",
    "accumulated",
    "alternating",
}


def _normalize_targets(event_or_events: Any) -> tuple[Any, ...]:
    if isinstance(event_or_events, (list, tuple)):
        targets = tuple(event_or_events)
    else:
        targets = (event_or_events,)

    if not targets or any(target is None for target in targets):
        raise ValueError("Variation requires at least one non-None event.")
    return targets


@dataclass(frozen=True, slots=True)
class SeqStarVariation:
    """One loop-dependent event-property progression."""

    events: tuple[Any, ...]
    attribute: str
    strength: Any
    step: Any
    mode: str = "linear"
    wrap: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        attribute = str(self.attribute).strip()
        mode = str(self.mode).strip().lower()

        if not attribute:
            raise ValueError("Variation attribute must not be empty.")
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

        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

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
            "strength": self.strength,
            "step": self.step,
            "mode": self.mode,
            "wrap": self.wrap,
            "metadata": dict(self.metadata),
            "event_names": [
                str(
                    getattr(event, "name", None)
                    or getattr(event, "path", None)
                    or event.__class__.__name__
                )
                for event in self.events
            ],
            "event_object_ids": [id(event) for event in self.events],
        }


def vary(
    event_or_events: Any,
    attribute: str,
    *,
    strength: Any,
    step: Any,
    mode: str = "linear",
    wrap: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeqStarVariation:
    """Create one variation specification for ``node.vary(...)``."""

    return SeqStarVariation(
        events=_normalize_targets(event_or_events),
        attribute=attribute,
        strength=strength,
        step=step,
        mode=mode,
        wrap=wrap,
        metadata=dict(metadata or {}),
    )


class SeqStarNodeHandle:
    """Public handle returned by ``Sequence.set_node``.

    It behaves like the historical node-record mapping for compatibility, while
    adding the fluent ``vary()`` API.
    """

    __slots__ = ("_sequence", "_name")

    def __init__(self, sequence: Any, name: str) -> None:
        self._sequence = sequence
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def record(self) -> dict[str, Any]:
        record = self._sequence.timeline.get_node(self._name)
        if record is None:
            raise KeyError(f"Unknown sequence node {self._name!r}.")
        return record

    def get(self, key: str, default: Any = None) -> Any:
        return self.record.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.record[key]

    def __contains__(self, key: object) -> bool:
        return key in self.record

    def __iter__(self):
        return iter(self.record)

    def __len__(self) -> int:
        return len(self.record)

    def keys(self):
        return self.record.keys()

    def items(self):
        return self.record.items()

    def values(self):
        return self.record.values()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.record)

    def vary(
        self,
        *variations: SeqStarVariation,
        event: Any | None = None,
        attribute: str | None = None,
        strength: Any | None = None,
        step: Any | None = None,
        mode: str = "linear",
        wrap: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SeqStarNodeHandle":
        """Attach one or more property progressions to this repeat node.

        Single-property convenience form::

            kernel.vary(
                event=gy_pre,
                attribute="area",
                strength=...,
                step=...,
            )

        Multiple-property form::

            kernel.vary(
                vary(gy_pre, "area", strength=..., step=...),
                vary([rf, adc], "phase_offset", strength=..., step=...),
            )
        """

        has_inline = any(
            value is not None
            for value in (event, attribute, strength, step, wrap)
        ) or mode != "linear" or bool(metadata)

        if variations and has_inline:
            raise ValueError(
                "Use either positional vary(...) specifications or the inline "
                "event/attribute/strength/step form, not both."
            )

        if not variations:
            if event is None or attribute is None:
                raise ValueError(
                    "Inline node.vary() requires event and attribute."
                )
            if strength is None or step is None:
                raise ValueError(
                    "Inline node.vary() requires strength and step."
                )
            variations = (
                vary(
                    event,
                    attribute,
                    strength=strength,
                    step=step,
                    mode=mode,
                    wrap=wrap,
                    metadata=metadata,
                ),
            )

        for specification in variations:
            if not isinstance(specification, SeqStarVariation):
                raise TypeError(
                    "node.vary() positional arguments must be produced by vary(...)."
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
