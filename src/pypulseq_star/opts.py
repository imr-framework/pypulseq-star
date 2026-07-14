# src/pypulseq_star/opts.py

from __future__ import annotations

import math
from typing import Any, ClassVar


class Opts:
    """
    System limits and scanner timing options for pypulseq_star.

    This class is intentionally similar in user-facing behavior to PyPulseq's
    Opts class, but it uses explicit defaults instead of None-based defaults.

    User-facing defaults:
        max_grad = 28.0 mT/m
        max_slew = 120.0 mT/m/ms
        rf_ringdown_time = 0.0 s
        rf_dead_time = 0.0 s
        adc_dead_time = 0.0 s
        rf_raster_time = 1e-6 s
        grad_raster_time = 10e-6 s
        block_duration_raster = 10e-6 s
        gamma = 42.575575e6 Hz/T
        max_rf = 21.58e-6 T

    Internal storage:
        max_grad is stored in Hz/m.
        max_slew is stored in Hz/m/s.

    Notes
    -----
    max_rf is stored as a B1 amplitude in tesla by default.
    """

    valid_grad_units: ClassVar[tuple[str, ...]] = (
        "Hz/m",
        "mT/m",
        "rad/ms/mm",
    )

    valid_slew_units: ClassVar[tuple[str, ...]] = (
        "Hz/m/s",
        "mT/m/ms",
        "T/m/s",
        "rad/ms/mm/ms",
    )

    default: ClassVar["Opts"]

    def __init__(
        self,
        max_grad: float = 28.0,
        grad_unit: str = "mT/m",
        max_slew: float = 120.0,
        slew_unit: str = "mT/m/ms",
        rf_ringdown_time: float = 0.0,
        rf_dead_time: float = 0.0,
        adc_dead_time: float = 0.0,
        rf_raster_time: float = 1e-6,
        grad_raster_time: float = 10e-6,
        block_duration_raster: float = 10e-6,
        gamma: float = 42.575575e6,
        max_rf: float = 21.58e-6,
        adc_raster_time: float = 100e-9,
        adc_samples_limit: int = 0,
        adc_samples_divisor: int = 4,
        rise_time: float = 0.0,
        B0: float = 1.5,
        name: str = "system",
        vendor: str = "",
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._validate_grad_unit(grad_unit)
        self._validate_slew_unit(slew_unit)

        self._validate_positive_or_zero("max_grad", max_grad)
        self._validate_positive_or_zero("max_slew", max_slew)
        self._validate_positive_or_zero("rf_ringdown_time", rf_ringdown_time)
        self._validate_positive_or_zero("rf_dead_time", rf_dead_time)
        self._validate_positive_or_zero("adc_dead_time", adc_dead_time)
        self._validate_positive("rf_raster_time", rf_raster_time)
        self._validate_positive("grad_raster_time", grad_raster_time)
        self._validate_positive("block_duration_raster", block_duration_raster)
        self._validate_positive("gamma", gamma)
        self._validate_positive_or_zero("max_rf", max_rf)
        self._validate_positive("adc_raster_time", adc_raster_time)
        self._validate_positive_or_zero("adc_samples_limit", adc_samples_limit)
        self._validate_positive("adc_samples_divisor", adc_samples_divisor)
        self._validate_positive_or_zero("rise_time", rise_time)
        self._validate_positive_or_zero("B0", B0)

        max_grad_hz_per_m = self.convert_gradient_to_hz_per_m(
            value=max_grad,
            unit=grad_unit,
            gamma=abs(gamma),
        )

        max_slew_hz_per_m_per_s = self.convert_slew_to_hz_per_m_per_s(
            value=max_slew,
            unit=slew_unit,
            gamma=abs(gamma),
        )

        if rise_time > 0:
            max_slew_hz_per_m_per_s = max_grad_hz_per_m / rise_time

        self.name = name
        self.vendor = vendor
        self.model = model
        self.metadata = metadata or {}

        # Internally normalized values.
        self.max_grad = max_grad_hz_per_m
        self.max_slew = max_slew_hz_per_m_per_s

        # User-facing values preserved for reporting and JSON export.
        self.input_max_grad = max_grad
        self.input_grad_unit = grad_unit
        self.input_max_slew = max_slew
        self.input_slew_unit = slew_unit

        self.rf_ringdown_time = rf_ringdown_time
        self.rf_dead_time = rf_dead_time
        self.adc_dead_time = adc_dead_time

        self.rf_raster_time = rf_raster_time
        self.grad_raster_time = grad_raster_time
        self.block_duration_raster = block_duration_raster
        self.adc_raster_time = adc_raster_time

        self.adc_samples_limit = adc_samples_limit
        self.adc_samples_divisor = adc_samples_divisor

        self.gamma = gamma
        self.max_rf = max_rf
        self.max_rf_unit = "T"

        self.rise_time = rise_time
        self.B0 = B0


        

    @classmethod
    def _validate_grad_unit(cls, grad_unit: str) -> None:
        if grad_unit not in cls.valid_grad_units:
            raise ValueError(
                f"Invalid gradient unit. Must be one of {cls.valid_grad_units}. "
                f"Passed: {grad_unit}"
            )

    @classmethod
    def _validate_slew_unit(cls, slew_unit: str) -> None:
        if slew_unit not in cls.valid_slew_units:
            raise ValueError(
                f"Invalid slew rate unit. Must be one of {cls.valid_slew_units}. "
                f"Passed: {slew_unit}"
            )

    @staticmethod
    def _validate_positive(name: str, value: float | int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive. Passed: {value}")

    @staticmethod
    def _validate_positive_or_zero(name: str, value: float | int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative. Passed: {value}")

    @staticmethod
    def convert_gradient_to_hz_per_m(
        value: float,
        unit: str,
        gamma: float,
    ) -> float:
        """
        Convert gradient amplitude to Hz/m.

        Parameters
        ----------
        value
            Gradient amplitude.
        unit
            Unit of gradient amplitude.
        gamma
            Gyromagnetic ratio in Hz/T.
        """
        if unit == "Hz/m":
            return value

        if unit == "mT/m":
            return value * 1e-3 * gamma

        if unit == "rad/ms/mm":
            # rad/ms/mm = rad / (1e-3 s * 1e-3 m)
            # Convert rad/s/m to cycles/s/m by dividing by 2*pi.
            return value * 1e6 / (2.0 * math.pi)

        raise ValueError(f"Unsupported gradient unit: {unit}")

    @staticmethod
    def convert_slew_to_hz_per_m_per_s(
        value: float,
        unit: str,
        gamma: float,
    ) -> float:
        """
        Convert slew rate to Hz/m/s.

        Parameters
        ----------
        value
            Slew rate.
        unit
            Unit of slew rate.
        gamma
            Gyromagnetic ratio in Hz/T.
        """
        if unit == "Hz/m/s":
            return value

        if unit == "T/m/s":
            return value * gamma

        if unit == "mT/m/ms":
            # 1 mT/m/ms = 1 T/m/s.
            return value * gamma

        if unit == "rad/ms/mm/ms":
            # rad/ms/mm/ms =
            # rad / (1e-3 s * 1e-3 m * 1e-3 s)
            # Convert rad/s/m/s to cycles/s/m/s by dividing by 2*pi.
            return value * 1e9 / (2.0 * math.pi)

        raise ValueError(f"Unsupported slew unit: {unit}")

    def set_as_default(self) -> None:
        """
        Set this Opts object as the package-wide default.
        """
        Opts.default = self

    @classmethod
    def reset_default(cls) -> None:
        """
        Reset package-wide defaults to the explicit pypulseq_star defaults.
        """
        cls.default = cls()
        

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize system options for pypulseq_star JSON/template export.
        """
        return {
            "name": self.name,
            "vendor": self.vendor,
            "model": self.model,
            "limits": {
                "max_grad": self.max_grad,
                "max_grad_unit": "Hz/m",
                "max_slew": self.max_slew,
                "max_slew_unit": "Hz/m/s",
                "max_rf": self.max_rf,
                "max_rf_unit": self.max_rf_unit,
                "rise_time": self.rise_time,
            },
            "input_limits": {
                "max_grad": self.input_max_grad,
                "grad_unit": self.input_grad_unit,
                "max_slew": self.input_max_slew,
                "slew_unit": self.input_slew_unit,
            },
            "timing": {
                "rf_ringdown_time": self.rf_ringdown_time,
                "rf_dead_time": self.rf_dead_time,
                "adc_dead_time": self.adc_dead_time,
                "rf_raster_time": self.rf_raster_time,
                "grad_raster_time": self.grad_raster_time,
                "adc_raster_time": self.adc_raster_time,
                "block_duration_raster": self.block_duration_raster,
            },
            "adc": {
                "adc_samples_limit": self.adc_samples_limit,
                "adc_samples_divisor": self.adc_samples_divisor,
            },
            "physics": {
                "gamma": self.gamma,
                "B0": self.B0,
            },
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "Opts("
            f"max_grad={self.input_max_grad}, "
            f"grad_unit='{self.input_grad_unit}', "
            f"max_slew={self.input_max_slew}, "
            f"slew_unit='{self.input_slew_unit}', "
            f"rf_ringdown_time={self.rf_ringdown_time}, "
            f"rf_dead_time={self.rf_dead_time}, "
            f"adc_dead_time={self.adc_dead_time}, "
            f"rf_raster_time={self.rf_raster_time}, "
            f"grad_raster_time={self.grad_raster_time}, "
            f"block_duration_raster={self.block_duration_raster}, "
            f"gamma={self.gamma}, "
            f"max_rf={self.max_rf}"
            ")"
        )

    def __str__(self) -> str:
        lines = [
            "System limits:",
            f"name: {self.name}",
            f"vendor: {self.vendor}",
            f"model: {self.model}",
            "",
            "Input limits:",
            f"max_grad: {self.input_max_grad} {self.input_grad_unit}",
            f"max_slew: {self.input_max_slew} {self.input_slew_unit}",
            f"max_rf: {self.max_rf} {self.max_rf_unit}",
            "",
            "Internal normalized limits:",
            f"max_grad: {self.max_grad} Hz/m",
            f"max_slew: {self.max_slew} Hz/m/s",
            "",
            "Timing:",
            f"rf_ringdown_time: {self.rf_ringdown_time}",
            f"rf_dead_time: {self.rf_dead_time}",
            f"adc_dead_time: {self.adc_dead_time}",
            f"rf_raster_time: {self.rf_raster_time}",
            f"grad_raster_time: {self.grad_raster_time}",
            f"adc_raster_time: {self.adc_raster_time}",
            f"block_duration_raster: {self.block_duration_raster}",
            "",
            "Physics:",
            f"gamma: {self.gamma}",
            f"B0: {self.B0}",
        ]
        return "\n".join(lines)
    

    def to_pypulseq_kwargs(self) -> dict[str, float | int | str]:
        """Return PyPulseq-compatible Opts keyword arguments.

        Important:
        pypulseq_star stores max_grad internally in Hz/m and max_slew internally
        in Hz/m/s. Therefore, the exported grad_unit and slew_unit must be the
        normalized internal units, not the original user-input units.
        """

        return {
            "max_grad": self.max_grad,
            "grad_unit": "Hz/m",
            "max_slew": self.max_slew,
            "slew_unit": "Hz/m/s",
            "rf_ringdown_time": self.rf_ringdown_time,
            "rf_dead_time": self.rf_dead_time,
            "adc_dead_time": self.adc_dead_time,
            "rf_raster_time": self.rf_raster_time,
            "grad_raster_time": self.grad_raster_time,
            "block_duration_raster": self.block_duration_raster,
            "gamma": self.gamma,
        }


Opts.reset_default()