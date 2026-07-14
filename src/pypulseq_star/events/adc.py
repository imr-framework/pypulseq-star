"""ADC event classes.

SeqStar treats ADC as train-native:

    SeqStarADCEvent = one semantic ADC train object

A single PyPulseq-style ADC is represented as an ADC train with one window.
For Pulseq .seq export, this enriched event can later be lowered into
individual standard ADC blocks at writer/preprocessing time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pypulseq_star.shapes.adc import SeqStarADCTrainShape, SeqStarADCWindow


@dataclass(slots=True)
class SeqStarADCEvent:
    """ADC train event.

    This class deliberately exposes familiar single-ADC attributes such as
    ``num_samples``, ``dwell``, ``delay``, and ``duration`` so timing/checking
    code can treat a single-window train like a normal ADC event.

    The richer multi-window representation lives in ``shape.windows``.
    """

    name: str = "adc"
    role: str | None = "readout"

    num_samples: int = 0
    dwell: float = 0.0
    duration: float = 0.0
    delay: float = 0.0

    freq_offset: float = 0.0
    phase_offset: float = 0.0
    freq_ppm: float = 0.0
    phase_ppm: float = 0.0
    phase_modulation: list[float] | None = None

    use: str | None = None

    shape: SeqStarADCTrainShape | None = None

    dead_time: float = 0.0
    window_guard_time: float = 0.0
    dead_time_policy: str = "train_level_once"
    pulseq_export_mode: str = "windows"

    mode: str = "single"
    trajectory: str | None = None

    type: str = "adc"
    kind: str = "adc_train"

    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    system: Any | None = None

    @property
    def windows(self) -> list[SeqStarADCWindow]:
        """Return ADC windows."""

        if self.shape is None:
            return []
        return self.shape.windows

    @property
    def num_windows(self) -> int:
        """Number of ADC windows."""

        return len(self.windows)

    @property
    def total_num_samples(self) -> int:
        """Total samples across all ADC windows."""

        if self.shape is None:
            return self.num_samples
        return self.shape.total_num_samples

    @property
    def first_window_delay(self) -> float:
        """First ADC-window start time."""

        if self.shape is None:
            return self.delay
        return self.shape.first_window_delay

    @property
    def last_window_end(self) -> float:
        """Last ADC-window end time."""

        if self.shape is None:
            return self.delay + self.duration
        return self.shape.last_window_end

    def bind_system(self, system: Any) -> None:
        """Attach system context after construction."""

        self.system = system

        if hasattr(system, "to_dict"):
            self.metadata.setdefault("system", system.to_dict())

    def validate(self) -> None:
        """Validate event and shape consistency."""

        if self.num_samples <= 0:
            raise ValueError(f"ADC num_samples must be positive. Passed: {self.num_samples}")

        if self.dwell <= 0:
            raise ValueError(f"ADC dwell must be positive. Passed: {self.dwell}")

        if self.delay < 0:
            raise ValueError(f"ADC delay must be non-negative. Passed: {self.delay}")

        if self.duration < 0:
            raise ValueError(f"ADC duration must be non-negative. Passed: {self.duration}")

        if self.dead_time < 0:
            raise ValueError(f"ADC dead_time must be non-negative. Passed: {self.dead_time}")

        if self.window_guard_time < 0:
            raise ValueError(
                f"ADC window_guard_time must be non-negative. Passed: {self.window_guard_time}"
            )

        if self.phase_modulation is not None:
            if len(self.phase_modulation) != self.num_samples:
                raise ValueError(
                    "phase_modulation length must match representative num_samples. "
                    f"len={len(self.phase_modulation)}, num_samples={self.num_samples}"
                )

        if self.shape is None:
            raise ValueError("SeqStarADCEvent requires a SeqStarADCTrainShape.")

        self.shape.validate()

        if self.duration + 1e-12 < self.shape.last_window_end:
            raise ValueError(
                "ADC event duration does not cover the ADC train shape. "
                f"event.duration={self.duration}, shape.last_window_end={self.shape.last_window_end}"
            )

    def to_gammastar_windows(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Return gammaSTAR-oriented ADC event data."""

        self.validate()

        shape_data = self.shape.to_gammastar_windows(
            include_sample_times=include_sample_times
        )

        return {
            "type": self.type,
            "kind": self.kind,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "trajectory": self.trajectory,
            "delay": self.delay,
            "duration": self.duration,
            "dead_time": self.dead_time,
            "frontend_dead_time": self.dead_time,
            "window_guard_time": self.window_guard_time,
            "dead_time_policy": self.dead_time_policy,
            "pulseq_export_mode": self.pulseq_export_mode,
            "num_samples": self.num_samples,
            "dwell": self.dwell,
            "freq_offset": self.freq_offset,
            "phase_offset": self.phase_offset,
            "freq_ppm": self.freq_ppm,
            "phase_ppm": self.phase_ppm,
            "num_windows": self.num_windows,
            "total_num_samples": self.total_num_samples,
            "first_window_delay": self.first_window_delay,
            "last_window_end": self.last_window_end,
            "shape": shape_data,
            "windows": shape_data["windows"],
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }

    def to_gammastar_samples(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Alias for generic writer code."""

        return self.to_gammastar_windows(include_sample_times=include_sample_times)

    def to_pulseq_windows(self) -> list[dict[str, Any]]:
        """Lower the enriched ADC train to standard Pulseq ADC windows.

        This is the hook the future writer/preprocessor should use before
        calling the feature-limited Pulseq .seq writer.
        """

        self.validate()

        windows = self.shape.to_pulseq_windows()

        for window in windows:
            window.setdefault("metadata", {})
            window["metadata"].setdefault("seqstar_parent_adc", self.name)
            window["metadata"].setdefault("seqstar_pulseq_export_mode", self.pulseq_export_mode)
            window["metadata"].setdefault("seqstar_dead_time_policy", self.dead_time_policy)

        return windows

    def to_dict(self) -> dict[str, Any]:
        """Compact dictionary representation."""

        return self.to_gammastar_windows(include_sample_times=False)