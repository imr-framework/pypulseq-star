"""RF waveform shapes and gammaSTAR sample serialization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .shape import SeqStarShape


@dataclass(slots=True)
class SeqStarRFShape(SeqStarShape):
    """Base class for RF waveform geometry.

    RF events own placement, role, and flip-angle semantics.

    RF shapes own the channel samples that gammaSTAR plots and executes.

    The optional system-derived fields allow shapes to respect scanner limits
    during sample serialization without forcing every shape to receive a full
    Opts object.
    """

    kind: str = "rf"
    rf_raster_time: float | None = None
    max_rf: float | None = None

    def to_gammastar_samples(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
        max_rf: float | None = None,
    ) -> dict[str, object]:
        """Return gammaSTAR RF samples for this shape."""

        raise NotImplementedError

    def _validate_common_rf_inputs(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
    ) -> None:
        """Validate RF inputs common to all RF shapes."""

        if self.duration <= 0:
            raise ValueError("RF duration must be positive")

        if flip_angle <= 0:
            raise ValueError(f"flip_angle must be positive. Passed: {flip_angle}")

        if gamma_hz_per_t <= 0:
            raise ValueError(
                f"gamma_hz_per_t must be positive. Passed: {gamma_hz_per_t}"
            )

        if self.rf_raster_time is not None:
            if self.rf_raster_time <= 0:
                raise ValueError(
                    f"rf_raster_time must be positive. Passed: {self.rf_raster_time}"
                )

            n_raster = self.duration / self.rf_raster_time

            if not math.isclose(n_raster, round(n_raster), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "RF duration is not aligned to rf_raster_time. "
                    f"duration = {self.duration}, "
                    f"rf_raster_time = {self.rf_raster_time}"
                )

    @staticmethod
    def _check_max_rf(
        *,
        amplitude_t: float,
        max_rf: float | None,
    ) -> None:
        """Check RF amplitude against max RF amplitude if provided."""

        if max_rf is None:
            return

        if max_rf < 0:
            raise ValueError(f"max_rf must be non-negative. Passed: {max_rf}")

        if abs(amplitude_t) > max_rf:
            raise ValueError(
                "RF amplitude exceeds max_rf. "
                f"Required B1 = {abs(amplitude_t):.6g} T, "
                f"max_rf = {max_rf:.6g} T."
            )


@dataclass(slots=True)
class SeqStarRFBlockShape(SeqStarRFShape):
    """Constant-amplitude rectangular RF waveform."""

    kind: str = "rf_block"

    def to_gammastar_samples(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
        max_rf: float | None = None,
    ) -> dict[str, object]:
        """Return a two-sample gammaSTAR representation of a block RF pulse.

        The returned amplitude is B1 in tesla.

        For a rectangular RF pulse:

            flip_angle = 2*pi*gamma*B1*duration

        so:

            B1 = flip_angle / (2*pi*gamma*duration)

        where gamma is in Hz/T.
        """

        self._validate_common_rf_inputs(
            flip_angle=flip_angle,
            gamma_hz_per_t=gamma_hz_per_t,
        )

        amplitude_t = flip_angle / (
            2.0 * math.pi * gamma_hz_per_t * self.duration
        )

        effective_max_rf = max_rf if max_rf is not None else self.max_rf

        self._check_max_rf(
            amplitude_t=amplitude_t,
            max_rf=effective_max_rf,
        )

        return {
            "t": [0.0, self.duration],
            "v": [
                {
                    "am": [amplitude_t, amplitude_t],
                    "fm": [0.0, 0.0],
                }
            ],
        }


@dataclass(slots=True)
class SeqStarRFSincShape(SeqStarRFShape):
    """Rastered sinc RF waveform.

    This is the RF-shape counterpart to ``make_sinc_pulse``. It intentionally
    handles only the RF envelope. Slice-select gradients will be added later in
    clean gradient modules.

    The shape is peak-normalized internally, then scaled so that:

        flip_angle = 2*pi*gamma*integral(B1(t) dt)

    where gamma is in Hz/T.
    """

    kind: str = "rf_sinc"
    time_bw_product: float = 4.0
    apodization: float = 0.0
    center_pos: float = 0.5

    def to_gammastar_samples(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
        max_rf: float | None = None,
    ) -> dict[str, object]:
        """Return gammaSTAR RF samples for a sinc RF pulse."""

        self._validate_common_rf_inputs(
            flip_angle=flip_angle,
            gamma_hz_per_t=gamma_hz_per_t,
        )

        self._validate_sinc_inputs()

        if self.rf_raster_time is None:
            raise ValueError(
                "SeqStarRFSincShape requires rf_raster_time for sample generation."
            )

        t_values, envelope = self.normalized_envelope()

        area = sum(envelope) * self.rf_raster_time

        if abs(area) <= 0:
            raise ValueError(
                "Sinc RF envelope has zero area and cannot be scaled to the "
                "requested flip angle."
            )

        peak_scale_t = flip_angle / (2.0 * math.pi * gamma_hz_per_t * area)
        am_values = [peak_scale_t * value for value in envelope]
        fm_values = [0.0 for _ in am_values]

        effective_max_rf = max_rf if max_rf is not None else self.max_rf

        if am_values:
            peak_amplitude_t = max(abs(value) for value in am_values)
            self._check_max_rf(
                amplitude_t=peak_amplitude_t,
                max_rf=effective_max_rf,
            )

        return {
            "t": t_values,
            "v": [
                {
                    "am": am_values,
                    "fm": fm_values,
                }
            ],
        }

    def normalized_envelope(self) -> tuple[list[float], list[float]]:
        """Return rastered, peak-normalized sinc-envelope samples."""

        if self.rf_raster_time is None:
            raise ValueError("rf_raster_time is required for sinc envelope generation.")

        self._validate_sinc_inputs()

        n_samples = int(round(self.duration / self.rf_raster_time))

        if n_samples < 2:
            raise ValueError(
                "Sinc pulse duration must contain at least two RF raster samples. "
                f"duration={self.duration}, rf_raster_time={self.rf_raster_time}"
            )

        actual_duration = n_samples * self.rf_raster_time

        t_values: list[float] = []
        envelope: list[float] = []

        for index in range(n_samples):
            t = (index + 0.5) * self.rf_raster_time
            relative = (t / actual_duration) - self.center_pos

            sinc_arg = self.time_bw_product * relative
            sinc_value = _sinc(sinc_arg)

            # Hamming-style apodization centered at center_pos.
            # apodization = 0 -> pure sinc
            # apodization = 1 -> full Hamming-style window
            window = (1.0 - self.apodization) + self.apodization * (
                0.54 + 0.46 * math.cos(2.0 * math.pi * relative)
            )

            t_values.append(t)
            envelope.append(sinc_value * window)

        max_abs = max(abs(value) for value in envelope)

        if max_abs <= 0:
            raise ValueError("Sinc RF envelope is all zeros.")

        envelope = [value / max_abs for value in envelope]

        return t_values, envelope

    def _validate_sinc_inputs(self) -> None:
        """Validate sinc-shape-specific inputs."""

        if self.time_bw_product <= 0:
            raise ValueError(
                "time_bw_product must be positive. "
                f"Passed: {self.time_bw_product}"
            )

        if not 0.0 <= self.apodization <= 1.0:
            raise ValueError(
                "apodization must be between 0 and 1. "
                f"Passed: {self.apodization}"
            )

        if not 0.0 <= self.center_pos <= 1.0:
            raise ValueError(
                "center_pos must be between 0 and 1. "
                f"Passed: {self.center_pos}"
            )


@dataclass(slots=True)
class SeqStarRFGaussShape(SeqStarRFShape):
    """Rastered Gaussian RF waveform.

    This is the RF-shape counterpart to ``make_gauss_pulse``. It intentionally
    handles only the RF envelope. Slice-select gradients will be added later in
    clean gradient modules.

    The shape is peak-normalized internally, then scaled so that:

        flip_angle = 2*pi*gamma*integral(B1(t) dt)

    where gamma is in Hz/T.

    The Gaussian envelope uses the same user-facing controls as the RF
    constructor:

    ``time_bw_product``
        Controls the width of the Gaussian. Larger values produce a narrower
        pulse.

    ``apodization``
        Controls taper strength. A value of 0 gives a flat envelope; a value
        of 1 gives the strongest Gaussian taper.

    ``center_pos``
        Places the Gaussian center within the pulse duration.
    """

    kind: str = "rf_gauss"
    time_bw_product: float = 4.0
    apodization: float = 0.5
    center_pos: float = 0.5

    def to_gammastar_samples(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
        max_rf: float | None = None,
    ) -> dict[str, object]:
        """Return gammaSTAR RF samples for a Gaussian RF pulse."""

        self._validate_common_rf_inputs(
            flip_angle=flip_angle,
            gamma_hz_per_t=gamma_hz_per_t,
        )

        self._validate_gauss_inputs()

        if self.rf_raster_time is None:
            raise ValueError(
                "SeqStarRFGaussShape requires rf_raster_time for sample generation."
            )

        t_values, envelope = self.normalized_envelope()

        area = sum(envelope) * self.rf_raster_time

        if abs(area) <= 0:
            raise ValueError(
                "Gaussian RF envelope has zero area and cannot be scaled to the "
                "requested flip angle."
            )

        peak_scale_t = flip_angle / (2.0 * math.pi * gamma_hz_per_t * area)
        am_values = [peak_scale_t * value for value in envelope]
        fm_values = [0.0 for _ in am_values]

        effective_max_rf = max_rf if max_rf is not None else self.max_rf

        if am_values:
            peak_amplitude_t = max(abs(value) for value in am_values)
            self._check_max_rf(
                amplitude_t=peak_amplitude_t,
                max_rf=effective_max_rf,
            )

        return {
            "t": t_values,
            "v": [
                {
                    "am": am_values,
                    "fm": fm_values,
                }
            ],
        }

    def normalized_envelope(self) -> tuple[list[float], list[float]]:
        """Return rastered, peak-normalized Gaussian-envelope samples."""

        if self.rf_raster_time is None:
            raise ValueError("rf_raster_time is required for Gaussian envelope generation.")

        self._validate_gauss_inputs()

        n_samples = int(round(self.duration / self.rf_raster_time))

        if n_samples < 2:
            raise ValueError(
                "Gaussian pulse duration must contain at least two RF raster samples. "
                f"duration={self.duration}, rf_raster_time={self.rf_raster_time}"
            )

        actual_duration = n_samples * self.rf_raster_time
        taper_strength = max(self.apodization, 1e-12)

        t_values: list[float] = []
        envelope: list[float] = []

        for index in range(n_samples):
            t = (index + 0.5) * self.rf_raster_time
            relative = (t / actual_duration) - self.center_pos

            gauss_arg = self.time_bw_product * relative
            gauss_value = math.exp(-taper_strength * gauss_arg * gauss_arg)

            t_values.append(t)
            envelope.append(gauss_value)

        max_abs = max(abs(value) for value in envelope)

        if max_abs <= 0:
            raise ValueError("Gaussian RF envelope is all zeros.")

        envelope = [value / max_abs for value in envelope]

        return t_values, envelope

    def _validate_gauss_inputs(self) -> None:
        """Validate Gaussian-shape-specific inputs."""

        if self.time_bw_product <= 0:
            raise ValueError(
                "time_bw_product must be positive. "
                f"Passed: {self.time_bw_product}"
            )

        if not 0.0 <= self.apodization <= 1.0:
            raise ValueError(
                "apodization must be between 0 and 1. "
                f"Passed: {self.apodization}"
            )

        if not 0.0 <= self.center_pos <= 1.0:
            raise ValueError(
                "center_pos must be between 0 and 1. "
                f"Passed: {self.center_pos}"
            )


@dataclass(slots=True)
class SeqStarRFArbitraryShape(SeqStarRFShape):
    """Rastered arbitrary RF waveform.

    This is the RF-shape counterpart to ``make_arbitrary_rf``. It handles
    real-valued RF envelope samples only. Complex RF samples and per-sample
    phase/frequency modulation will be added later.

    The input signal is treated as a dimensionless envelope and scaled so that:

        flip_angle = 2*pi*gamma*integral(B1(t) dt)

    where gamma is in Hz/T.
    """

    kind: str = "rf_arbitrary"
    signal: list[float] = field(default_factory=list)

    def to_gammastar_samples(
        self,
        *,
        flip_angle: float,
        gamma_hz_per_t: float,
        max_rf: float | None = None,
    ) -> dict[str, object]:
        """Return gammaSTAR RF samples for an arbitrary RF pulse.

        Returns
        -------
        dict
            gammaSTAR-compatible RF sample dictionary:

                {
                    "t": [...],
                    "v": [{"am": [...], "fm": [...]}]
                }

        Notes
        -----
        The amplitude samples are B1 in tesla. Frequency modulation is set to
        zero for now.
        """

        self._validate_common_rf_inputs(
            flip_angle=flip_angle,
            gamma_hz_per_t=gamma_hz_per_t,
        )

        self._validate_arbitrary_inputs()

        if self.rf_raster_time is None:
            raise ValueError(
                "SeqStarRFArbitraryShape requires rf_raster_time for sample generation."
            )

        t_values, envelope = self.normalized_envelope()

        area = sum(envelope) * self.rf_raster_time

        if abs(area) <= 0:
            raise ValueError(
                "Arbitrary RF signal has zero signed area and cannot be scaled "
                "to the requested flip angle. Use a signal with nonzero integral."
            )

        scale_t = flip_angle / (2.0 * math.pi * gamma_hz_per_t * area)
        am_values = [scale_t * value for value in envelope]
        fm_values = [0.0 for _ in am_values]

        effective_max_rf = max_rf if max_rf is not None else self.max_rf

        if am_values:
            peak_amplitude_t = max(abs(value) for value in am_values)
            self._check_max_rf(
                amplitude_t=peak_amplitude_t,
                max_rf=effective_max_rf,
            )

        return {
            "t": t_values,
            "v": [
                {
                    "am": am_values,
                    "fm": fm_values,
                }
            ],
        }

    def normalized_envelope(self) -> tuple[list[float], list[float]]:
        """Return rastered arbitrary-envelope samples.

        ``make_arbitrary_rf`` already resamples the incoming waveform to the
        RF raster before constructing this shape. This method validates that
        the signal length and duration are consistent and returns one sample
        per RF raster interval, placed at raster centers.
        """

        if self.rf_raster_time is None:
            raise ValueError(
                "rf_raster_time is required for arbitrary RF envelope generation."
            )

        self._validate_arbitrary_inputs()

        n_expected = int(round(self.duration / self.rf_raster_time))

        if n_expected < 1:
            raise ValueError(
                "Arbitrary RF duration must contain at least one RF raster sample. "
                f"duration={self.duration}, rf_raster_time={self.rf_raster_time}"
            )

        if len(self.signal) != n_expected:
            raise ValueError(
                "Arbitrary RF signal length is not consistent with duration and "
                "rf_raster_time. "
                f"len(signal)={len(self.signal)}, expected={n_expected}, "
                f"duration={self.duration}, rf_raster_time={self.rf_raster_time}"
            )

        t_values = [
            (index + 0.5) * self.rf_raster_time
            for index in range(n_expected)
        ]

        return t_values, list(self.signal)

    def _validate_arbitrary_inputs(self) -> None:
        """Validate arbitrary-shape-specific inputs."""

        if self.rf_raster_time is None:
            raise ValueError("rf_raster_time is required for arbitrary RF shapes.")

        if self.rf_raster_time <= 0:
            raise ValueError(
                f"rf_raster_time must be positive. Passed: {self.rf_raster_time}"
            )

        if not self.signal:
            raise ValueError("Arbitrary RF signal must contain at least one sample.")

        for index, value in enumerate(self.signal):
            if isinstance(value, complex):
                raise NotImplementedError(
                    "Complex arbitrary RF samples are not supported yet in "
                    "SeqStarRFArbitraryShape. Use real-valued samples for now."
                )

            if not isinstance(value, (float, int)):
                raise TypeError(
                    f"signal[{index}] must be numeric. Passed: {value!r}"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"signal[{index}] must be finite. Passed: {value!r}"
                )

        if max(abs(float(value)) for value in self.signal) <= 0:
            raise ValueError("Arbitrary RF signal must not be all zeros.")


def _sinc(x: float) -> float:
    """Return normalized sinc(x) = sin(pi*x)/(pi*x)."""

    if abs(x) < 1e-12:
        return 1.0

    pix = math.pi * x
    return math.sin(pix) / pix