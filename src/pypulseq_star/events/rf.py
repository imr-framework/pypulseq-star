"""Radiofrequency event model."""

from __future__ import annotations

from dataclasses import dataclass

from pypulseq_star.core import SeqStarRelationship, SeqStarTiming
from pypulseq_star.shapes import SeqStarRFBlockShape, SeqStarRFShape

from .event import SeqStarEvent


@dataclass(slots=True)
class SeqStarRFEvent(SeqStarEvent):
    """Base RF event enriched with gammaSTAR-facing semantics."""

    flip_angle: float = 0.0
    duration: float = 0.0
    phase_offset: float = 0.0
    freq_offset: float = 0.0
    asymmetry: float = 0.5
    enabled: bool = True
    use: str | None = None
    shape: SeqStarRFShape | None = None
    event_type: str = "rf"

    def __post_init__(self) -> None:
        self.timing = SeqStarTiming(
            tstart=self.timing.tstart,
            duration=self.duration or self.timing.duration,
        )
        self.parameters.setdefault("flip_angle", self.flip_angle)
        self.parameters.setdefault("duration", self.duration or self.timing.duration or 0.0)
        self.parameters.setdefault("phase", self.phase_offset)
        self.parameters.setdefault("frequency", self.freq_offset)
        self.parameters.setdefault("asymmetry", self.asymmetry)
        self.parameters.setdefault("enabled", self.enabled)
        self.parameters.setdefault("type", _gammastar_rf_type(self.use))
        if self.use is not None:
            self.parameters.setdefault("use", self.use)
        if self.shape is not None:
            self.parameters.setdefault("shape_kind", self.shape.kind)
            self.relationships.append(
                SeqStarRelationship(
                    source=self.name,
                    target=self.shape.name,
                    kind="has_shape",
                    metadata={"shape_kind": self.shape.kind},
                )
            )

    @property
    def shape_node_name(self) -> str:
        """Return the stable shape-node name for relationship metadata."""

        return f"{self.name}.shape"

    def gammastar_samples(self, gamma_hz_per_t: float) -> dict[str, object]:
        """Return gammaSTAR RF samples from the event's RF shape."""

        if self.shape is None:
            raise ValueError("RF event requires an RF shape for gammaSTAR export")
        return self.shape.to_gammastar_samples(
            flip_angle=float(self.parameters["flip_angle"]),
            gamma_hz_per_t=gamma_hz_per_t,
        )


@dataclass(slots=True)
class SeqStarRFBlockEvent(SeqStarRFEvent):
    """Rectangular RF block pulse."""

    shape: SeqStarRFShape | None = None
    event_type: str = "rf"

    def __post_init__(self) -> None:
        if self.shape is None:
            self.shape = SeqStarRFBlockShape(
                name=f"{self.name}_shape",
                duration=self.duration,
            )
        SeqStarRFEvent.__post_init__(self)


def _gammastar_rf_type(use: str | None) -> str:
    if use in {None, "excitation", "excite"}:
        return "Excitation"
    if use in {"refocusing", "refocus"}:
        return "Refocusing"
    return str(use)
