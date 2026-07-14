"""Gradient events for pypulseq_star."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from pypulseq_star.shapes.grad import (
    SeqStarArbitraryGradientShape,
    SeqStarGradientShape,
    SeqStarSplitGradientShape,
    SeqStarTrapezoidGradientShape,
)


@dataclass(slots=True)
class SeqStarGradientEvent:
    """Physical gradient event wrapping a gradient shape.

    The event mirrors PyPulseq's public data fields where practical while
    preserving richer SeqStar metadata and methods.
    """

    shape: SeqStarGradientShape | SeqStarSplitGradientShape
    name: str | None = None
    role: str | None = None
    axis_role: str | None = None
    encoding_role: str | None = None
    polarity: int | None = None
    relationship_tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    system: Any | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            base = getattr(self.shape, "kind", "grad")
            ch = getattr(self.shape, "channel", "x")
            self.name = f"{base}_{ch}"
        if self.role is None:
            self.role = getattr(self.shape, "role", None)

    @property
    def type(self) -> str:
        return getattr(self.shape, "kind", "grad")

    @property
    def channel(self) -> str:
        return getattr(self.shape, "channel")

    @property
    def delay(self) -> float:
        return float(getattr(self.shape, "delay", 0.0))

    @property
    def duration(self) -> float:
        return float(getattr(self.shape, "duration"))

    @property
    def active_duration(self) -> float:
        return float(getattr(self.shape, "active_duration", self.duration - self.delay))

    
    @property
    def first(self) -> float:
        return float(getattr(self.shape, "first"))

    @property
    def last(self) -> float:
        return float(getattr(self.shape, "last"))

    @property
    def area(self) -> float:
        """Return gradient area.

        Event-level parameters may override the wrapped shape area. This allows
        PyPulseq-style mutation such as changing amplitude after construction.
        """

        value = self.parameters.get("area", None)
        if value is not None:
            return float(value)

        return float(getattr(self.shape, "area"))

    @area.setter
    def area(self, value: float) -> None:
        """Set event-level gradient area.

        The wrapped shape remains the geometry object, but the event-level area
        becomes the public value used by writers/adapters.
        """

        value = float(value)
        self.parameters["area"] = value

        # Best-effort synchronization for mutable shape implementations.
        try:
            setattr(self.shape, "area", value)
        except Exception:
            pass

    @property
    def amplitude(self) -> float:
        """Return gradient amplitude.

        Event-level parameters may override the wrapped shape amplitude. This
        supports PyPulseq-style mutation while preserving the enriched shape.
        """

        value = self.parameters.get("amplitude", None)
        if value is not None:
            return float(value)

        value = getattr(self.shape, "amplitude", None)
        if value is not None:
            return float(value)

        flat_time = float(getattr(self, "flat_time", 0.0) or 0.0)
        if flat_time > 0:
            return float(self.area) / flat_time

        duration = float(getattr(self, "duration", 0.0) or 0.0)
        if duration > 0:
            return float(self.area) / duration

        return 0.0

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        """Set gradient amplitude and update area consistently.

        Supports PyPulseq-style mutation, for example:

            gy_pre.amplitude = -gy_pre.amplitude
        """

        value = float(value)

        old_amplitude = float(self.amplitude)
        old_area = float(self.area)

        if old_amplitude != 0.0:
            new_area = old_area * value / old_amplitude
        else:
            flat_time = float(getattr(self, "flat_time", 0.0) or 0.0)
            duration = float(getattr(self, "duration", 0.0) or 0.0)

            if flat_time > 0:
                new_area = value * flat_time
            elif duration > 0:
                new_area = value * duration
            else:
                new_area = old_area

        self.parameters["amplitude"] = value
        self.parameters["area"] = new_area

        # Best-effort synchronization for mutable shape implementations.
        try:
            setattr(self.shape, "amplitude", value)
        except Exception:
            pass

        try:
            setattr(self.shape, "area", new_area)
        except Exception:
            pass

    @property
    def rise_time(self) -> float | None:
        return getattr(self.shape, "rise_time", None)

    @property
    def flat_time(self) -> float | None:
        return getattr(self.shape, "flat_time", None)

    @property
    def fall_time(self) -> float | None:
        return getattr(self.shape, "fall_time", None)

    @property
    def flat_area(self) -> float | None:
        return getattr(self.shape, "flat_area", None)

    @property
    def waveform(self) -> np.ndarray:
        if hasattr(self.shape, "waveform"):
            return self.shape.waveform()
        raise AttributeError("Split-gradient container does not expose one waveform; use .parts instead.")


    @property
    def tt(self) -> np.ndarray:
        if hasattr(self.shape, "time_axis"):
            return self.shape.time_axis()
        raise AttributeError("Split-gradient container does not expose one time axis; use .parts instead.")

    @property
    def parts(self) -> tuple["SeqStarGradientEvent", "SeqStarGradientEvent", "SeqStarGradientEvent"]:
        if not isinstance(self.shape, SeqStarSplitGradientShape):
            raise AttributeError("Only split-gradient events expose .parts.")
        return tuple(
            SeqStarGradientEvent(
                shape=part,
                name=f"{self.name}_{label}",
                role=self.role,
                axis_role=self.axis_role,
                encoding_role=self.encoding_role,
                polarity=self.polarity,
                relationship_tags=list(self.relationship_tags),
                parameters=dict(self.parameters),
                metadata={**self.metadata, "split_part": label},
                system=self.system,
            )
            for part, label in zip(self.shape.parts, ("ramp_up", "flat_top", "ramp_down"), strict=True)
        )  # type: ignore[return-value]

    def validate(self, *, max_grad: float | None = None, max_slew: float | None = None) -> None:
        """Validate gradient event limits."""

        if isinstance(self.shape, SeqStarSplitGradientShape):
            for part in self.shape.parts:
                part.validate(max_grad=max_grad, max_slew=max_slew)
        else:
            self.shape.validate(max_grad=max_grad, max_slew=max_slew)

    def to_simple_namespace(self) -> SimpleNamespace:
        """Return a PyPulseq-style SimpleNamespace without importing PyPulseq."""

        if isinstance(self.shape, SeqStarSplitGradientShape):
            raise ValueError("Split-gradient containers cannot lower to one SimpleNamespace; lower each part instead.")

        ns = SimpleNamespace()
        ns.type = self.type
        ns.channel = self.channel
        ns.delay = self.delay
        ns.area = self.area
        ns.first = self.first
        ns.last = self.last

        if isinstance(self.shape, SeqStarTrapezoidGradientShape):
            ns.type = "trap"
            ns.amplitude = self.shape.amplitude
            ns.rise_time = self.shape.rise_time
            ns.flat_time = self.shape.flat_time
            ns.fall_time = self.shape.fall_time
            ns.flat_area = self.shape.flat_area
        elif isinstance(self.shape, SeqStarArbitraryGradientShape):
            ns.type = "grad"
            ns.waveform = self.shape.waveform()
            ns.tt = self.shape.time_axis()
            ns.shape_dur = self.shape.active_duration
        else:  # pragma: no cover - defensive path
            raise TypeError(f"Unsupported gradient shape: {type(self.shape)!r}")

        return ns

    def to_pulseq(self) -> SimpleNamespace:
        """Return a PyPulseq-compatible gradient event.

        The method uses PyPulseq constructors when PyPulseq is installed, so the
        resulting object should pass standard PyPulseq ``Sequence.add_block``.
        """

        if isinstance(self.shape, SeqStarSplitGradientShape):
            raise ValueError("Split-gradient containers cannot lower to one PyPulseq event; lower each part instead.")

        if isinstance(self.shape, SeqStarTrapezoidGradientShape):
            from pypulseq.make_trapezoid import make_trapezoid

            return make_trapezoid(
                channel=self.shape.channel,
                amplitude=self.shape.amplitude,
                delay=self.shape.delay,
                rise_time=self.shape.rise_time,
                flat_time=self.shape.flat_time,
                fall_time=self.shape.fall_time,
                system=self.system,
            )

        if isinstance(self.shape, SeqStarArbitraryGradientShape):
            from pypulseq.make_arbitrary_grad import make_arbitrary_grad

            return make_arbitrary_grad(
                channel=self.shape.channel,
                waveform=np.asarray(self.shape.samples, dtype=float),
                first=self.shape.first,
                last=self.shape.last,
                delay=self.shape.delay,
                system=self.system,
                oversampling=self.shape.oversampling,
            )

        raise TypeError(f"Unsupported gradient shape: {type(self.shape)!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize gradient event."""

        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "type": self.type,
            "channel": self.channel,
            "delay": self.delay,
            "duration": self.duration,
            "active_duration": self.active_duration,
            "area": None if isinstance(self.shape, SeqStarSplitGradientShape) else self.area,
            "first": None if isinstance(self.shape, SeqStarSplitGradientShape) else self.first,
            "last": None if isinstance(self.shape, SeqStarSplitGradientShape) else self.last,
            "role": self.role,
            "axis_role": self.axis_role,
            "encoding_role": self.encoding_role,
            "polarity": self.polarity,
            "relationship_tags": list(self.relationship_tags),
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
            "shape": self.shape.to_dict(),
        }

    def to_gammastar_waveform(self) -> dict[str, Any]:
        """Return a generic gradient waveform payload for future JSON writers."""

        if isinstance(self.shape, SeqStarSplitGradientShape):
            return {
                "kind": "split",
                "channel": self.channel,
                "parts": [part.to_gammastar_waveform() for part in self.parts],
            }

        return {
            "kind": self.type,
            "channel": self.channel,
            "tstart": self.delay,
            "duration": self.duration,
            "active_duration": self.active_duration,
            "area": self.area,
            "first": self.first,
            "last": self.last,
            "waveform": self.waveform.tolist(),
            "tt": self.tt.tolist(),
            "role": self.role,
            "axis_role": self.axis_role,
            "encoding_role": self.encoding_role,
        }
