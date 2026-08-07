"""ADC train shapes and gammaSTAR-oriented ADC window serialization.

Design principle
----------------
PyPulseq exposes ADC as a single event. SeqStar keeps that familiar API through
``make_adc(...)`` but internally represents ADC as a train of one or more ADC
windows. A single ADC is simply a train with one window.

This lets EPI, RARE/TSE, GRASE, PROPELLER-GRASE, radial, spiral, navigator,
calibration, and spectroscopy readouts use one robust ADC object instead of
forcing developers to create many frontend ADC objects manually.

Pulseq compatibility policy
---------------------------
The enriched ADC train is not written directly as a nonstandard Pulseq ADC
object. Instead, later in the Pulseq writer/preprocessor, the train can be
lowered into standard one-window Pulseq ADC events/blocks.

The helper method ``to_pulseq_windows(...)`` is provided specifically for that
future lowering step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .shape import SeqStarShape


@dataclass(slots=True)
class SeqStarADCWindow:
    """One ADC acquisition window within an ADC train.

    ``delay`` is the start time of this ADC window relative to the start of the
    block/train event. ``duration`` is normally ``num_samples * dwell``.

    Window-level metadata such as echo index, shot index, polarity, trajectory,
    and role is intentionally stored here so that EPI, RARE/TSE, GRASE, radial,
    navigator, calibration, and spectroscopy readouts can share the same ADC
    object model.
    """

    index: int
    delay: float
    num_samples: int
    dwell: float
    duration: float

    label: str | None = None
    role: str = "imaging"

    freq_offset: float = 0.0
    phase_offset: float = 0.0
    freq_ppm: float = 0.0
    phase_ppm: float = 0.0

    polarity: int = 1

    echo_index: int | None = None
    shot_index: int | None = None
    segment_index: int | None = None
    line_index: int | None = None
    line_index_in_shot: int | None = None
    spin_echo_index: int | None = None
    gradient_echo_index: int | None = None
    spoke_index: int | None = None
    navigator_index: int | None = None

    center_sample: int | None = None
    echo_time: float | None = None
    trajectory: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tstart(self) -> float:
        """gammaSTAR-style alias for ADC window start time."""

        return self.delay

    @property
    def tend(self) -> float:
        """ADC window end time."""

        return self.delay + self.duration

    @property
    def sample_times(self) -> list[float]:
        """Sample-center times relative to the train/block start."""

        return [
            self.delay + (sample_index + 0.5) * self.dwell
            for sample_index in range(self.num_samples)
        ]

    def validate(self, *, adc_raster_time: float | None = None) -> None:
        """Validate window timing and sample metadata."""

        if self.index < 0:
            raise ValueError(f"ADC window index must be non-negative. Passed: {self.index}")

        if self.delay < 0:
            raise ValueError(f"ADC window delay must be non-negative. Passed: {self.delay}")

        if self.num_samples <= 0:
            raise ValueError(f"ADC num_samples must be positive. Passed: {self.num_samples}")

        if self.dwell <= 0:
            raise ValueError(f"ADC dwell must be positive. Passed: {self.dwell}")

        if self.duration <= 0:
            raise ValueError(f"ADC duration must be positive. Passed: {self.duration}")

        expected_duration = self.num_samples * self.dwell

        if not math.isclose(self.duration, expected_duration, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "ADC duration must equal num_samples*dwell. "
                f"duration={self.duration}, expected={expected_duration}"
            )

        if self.polarity not in (-1, 1):
            raise ValueError(f"ADC polarity must be +1 or -1. Passed: {self.polarity}")

        if self.center_sample is not None:
            if not 0 <= self.center_sample < self.num_samples:
                raise ValueError(
                    "center_sample must be inside the ADC sample range. "
                    f"center_sample={self.center_sample}, num_samples={self.num_samples}"
                )

        if adc_raster_time is not None:
            self._validate_raster(value=self.delay, raster=adc_raster_time, name="delay")
            self._validate_raster(value=self.dwell, raster=adc_raster_time, name="dwell")
            self._validate_raster(
                value=self.duration,
                raster=adc_raster_time,
                name="duration",
            )

    @staticmethod
    def _validate_raster(*, value: float, raster: float, name: str) -> None:
        """Validate approximate raster alignment."""

        if raster <= 0:
            raise ValueError(f"ADC raster time must be positive. Passed: {raster}")

        n = value / raster

        if not math.isclose(n, round(n), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"ADC window {name} is not raster-aligned. "
                f"value={value}, adc_raster_time={raster}"
            )

    def to_dict(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Return a gammaSTAR/JSON-friendly representation."""

        data: dict[str, Any] = {
            "index": self.index,
            "label": self.label,
            "role": self.role,
            "delay": self.delay,
            "tstart": self.tstart,
            "duration": self.duration,
            "tend": self.tend,
            "num_samples": self.num_samples,
            "number_of_samples": self.num_samples,
            "dwell": self.dwell,
            "sample_time": self.dwell,
            "freq_offset": self.freq_offset,
            "phase_offset": self.phase_offset,
            "freq_ppm": self.freq_ppm,
            "phase_ppm": self.phase_ppm,
            "polarity": self.polarity,
            "echo_index": self.echo_index,
            "shot_index": self.shot_index,
            "segment_index": self.segment_index,
            "line_index": self.line_index,
            "line_index_in_shot": self.line_index_in_shot,
            "spin_echo_index": self.spin_echo_index,
            "gradient_echo_index": self.gradient_echo_index,
            "spoke_index": self.spoke_index,
            "navigator_index": self.navigator_index,
            "center_sample": self.center_sample,
            "echo_time": self.echo_time,
            "trajectory": self.trajectory,
            "metadata": dict(self.metadata),
        }

        if include_sample_times:
            data["sample_times"] = self.sample_times

        return data

    def to_pulseq_dict(self) -> dict[str, Any]:
        """Return one standard Pulseq-compatible ADC-window description.

        This does not write the .seq file directly. It gives the future Pulseq
        writer/preprocessor exactly what it needs to lower one enriched ADC
        window into one standard Pulseq ADC event/block.
        """

        return {
            "num_samples": self.num_samples,
            "dwell": self.dwell,
            "duration": self.duration,
            "delay": self.delay,
            "freq_offset": self.freq_offset,
            "phase_offset": self.phase_offset,
            "freq_ppm": self.freq_ppm,
            "phase_ppm": self.phase_ppm,
            "label": self.label,
            "role": self.role,
            "metadata": {
                "seqstar_adc_window_index": self.index,
                "trajectory": self.trajectory,
                "polarity": self.polarity,
                "echo_index": self.echo_index,
                "shot_index": self.shot_index,
                "segment_index": self.segment_index,
                "line_index": self.line_index,
                "spin_echo_index": self.spin_echo_index,
                "gradient_echo_index": self.gradient_echo_index,
                "spoke_index": self.spoke_index,
                "navigator_index": self.navigator_index,
                "center_sample": self.center_sample,
                "echo_time": self.echo_time,
                **dict(self.metadata),
            },
        }


@dataclass(slots=True)
class SeqStarADCShape(SeqStarShape):
    """Base class for ADC shapes."""

    kind: str = "adc"
    adc_raster_time: float | None = None

    def to_gammastar_windows(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Return gammaSTAR-oriented ADC data."""

        raise NotImplementedError


@dataclass(slots=True)
class SeqStarADCTrainShape(SeqStarADCShape):
    """Train of ADC windows.

    This is the common shape for single ADC, multi-echo GRE, EPI, RARE/TSE,
    GRASE, PROPELLER-GRASE, radial, spiral, navigator, calibration, and SVS-like
    spectroscopy readouts.
    """

    kind: str = "adc_train"
    windows: list[SeqStarADCWindow] = field(default_factory=list)

    mode: str = "single"
    trajectory: str | None = None

    frontend_dead_time: float = 0.0
    window_guard_time: float = 0.0
    dead_time_policy: str = "train_level_once"
    pulseq_export_mode: str = "windows"

    def __post_init__(self) -> None:
        """Validate immediately after construction."""

        self.validate()

    @property
    def num_windows(self) -> int:
        """Number of ADC windows."""

        return len(self.windows)

    @property
    def total_num_samples(self) -> int:
        """Total acquired samples across all windows."""

        return sum(window.num_samples for window in self.windows)

    @property
    def first_window_delay(self) -> float:
        """Start time of the first ADC window."""

        return min((window.delay for window in self.windows), default=0.0)

    @property
    def last_window_end(self) -> float:
        """End time of the last ADC window."""

        return max((window.tend for window in self.windows), default=0.0)

    def validate(self) -> None:
        """Validate the ADC train."""

        if self.duration < 0:
            raise ValueError(f"ADC train duration must be non-negative. Passed: {self.duration}")

        if self.frontend_dead_time < 0:
            raise ValueError(
                f"frontend_dead_time must be non-negative. Passed: {self.frontend_dead_time}"
            )

        if self.window_guard_time < 0:
            raise ValueError(
                f"window_guard_time must be non-negative. Passed: {self.window_guard_time}"
            )

        if not self.windows:
            raise ValueError("ADC train must contain at least one ADC window.")

        self.windows.sort(key=lambda window: window.delay)

        previous_end: float | None = None

        for index, window in enumerate(self.windows):
            window.index = index
            window.validate(adc_raster_time=self.adc_raster_time)

            if previous_end is not None:
                if window.delay < previous_end - 1e-12:
                    raise ValueError(
                        "ADC windows overlap. "
                        f"window[{index}].delay={window.delay}, previous_end={previous_end}"
                    )

                gap = window.delay - previous_end

                if gap + 1e-12 < self.window_guard_time:
                    raise ValueError(
                        "ADC windows violate window_guard_time. "
                        f"gap={gap}, window_guard_time={self.window_guard_time}"
                    )

            previous_end = window.tend

        if self.duration + 1e-12 < self.last_window_end:
            raise ValueError(
                "ADC train duration does not cover all windows. "
                f"duration={self.duration}, last_window_end={self.last_window_end}"
            )

    def to_gammastar_windows(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Return gammaSTAR-oriented ADC train data."""

        self.validate()

        return {
            "type": "adc",
            "kind": self.kind,
            "name": self.name,
            "mode": self.mode,
            "trajectory": self.trajectory,
            "duration": self.duration,
            "adc_raster_time": self.adc_raster_time,
            "frontend_dead_time": self.frontend_dead_time,
            "window_guard_time": self.window_guard_time,
            "dead_time_policy": self.dead_time_policy,
            "pulseq_export_mode": self.pulseq_export_mode,
            "num_windows": self.num_windows,
            "total_num_samples": self.total_num_samples,
            "first_window_delay": self.first_window_delay,
            "last_window_end": self.last_window_end,
            "windows": [
                window.to_dict(include_sample_times=include_sample_times)
                for window in self.windows
            ],
        }

    def to_gammastar_samples(self, *, include_sample_times: bool = False) -> dict[str, Any]:
        """Alias used by generic writer code that expects sample serialization."""

        return self.to_gammastar_windows(include_sample_times=include_sample_times)

    def to_pulseq_windows(self) -> list[dict[str, Any]]:
        """Lower this ADC train into standard Pulseq-compatible ADC windows.

        The future writer/preprocessor can turn each returned dictionary into
        one normal Pulseq ADC event/block.
        """

        self.validate()
        return [window.to_pulseq_dict() for window in self.windows]

    def can_merge_for_pulseq_continuous(self) -> bool:
        """Return True if windows can be safely merged into one long ADC.

        This is intentionally conservative. The default writer strategy should
        still be window-by-window lowering. Continuous merging is only safe when
        every window is contiguous and has the same dwell/frequency/phase state.
        """

        self.validate()

        if len(self.windows) <= 1:
            return True

        first = self.windows[0]

        for previous, current in zip(self.windows[:-1], self.windows[1:]):
            if not math.isclose(current.delay, previous.tend, rel_tol=0.0, abs_tol=1e-12):
                return False

            if current.num_samples <= 0:
                return False

            if not math.isclose(current.dwell, first.dwell, rel_tol=0.0, abs_tol=1e-12):
                return False

            if not math.isclose(current.freq_offset, first.freq_offset, rel_tol=0.0, abs_tol=1e-12):
                return False

            if not math.isclose(current.phase_offset, first.phase_offset, rel_tol=0.0, abs_tol=1e-12):
                return False

            if not math.isclose(current.freq_ppm, first.freq_ppm, rel_tol=0.0, abs_tol=1e-12):
                return False

            if not math.isclose(current.phase_ppm, first.phase_ppm, rel_tol=0.0, abs_tol=1e-12):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Compact dictionary representation."""

        return self.to_gammastar_windows(include_sample_times=False)