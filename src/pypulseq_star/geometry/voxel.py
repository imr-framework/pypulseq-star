"""Geometry for rectangular voxels in logical MRI coordinates.

This module defines a sequence-independent ``Voxel`` abstraction.

A Voxel represents a rectangular 3D region expressed in the logical
read / phase / slice coordinate system used by PyPulseq-Star.

The class intentionally contains geometry only.

It does NOT know about:

- PRESS
- spectroscopy
- imaging
- RF pulses
- RF bandwidth
- gradient amplitudes
- localization frequency offsets
- crushers
- sequence timing

Those concepts should be implemented by higher-level sequence components.

Coordinate convention
---------------------
Voxel dimensions and positions are expressed along the logical axes:

    read
    phase
    slice

Voxel position specifies the CENTER of the voxel relative to scanner
isocenter.

Therefore::

    position_read_m = 0.0
    position_phase_m = 0.0
    position_slice_m = 0.0

places the voxel center at scanner isocenter.

Positions may be positive or negative.

For example::

    position_read_m = +0.010

means +10 mm from isocenter along the positive logical read direction, while::

    position_read_m = -0.010

means -10 mm from isocenter along that direction.

The physical scanner direction corresponding to logical read / phase / slice
is determined by ``EncodingFrame``.

Position guardrail
------------------
By default, voxel-center displacement is restricted to +/- 0.20 m along each
logical axis.

This provides a practical sanity check corresponding approximately to the
radius of a 40-cm-diameter imaging region.

This is NOT intended to represent a scanner-specific DSV model.

In particular, this check:

- limits voxel CENTER position only,
- is applied independently along each logical axis,
- does not guarantee that the entire voxel lies inside a spherical DSV.

The default may be overridden per Voxel, or disabled by setting
``max_abs_position_m=None``.

A future scanner-aware geometry constraint can replace or augment this simple
guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


# =============================================================================
# PUBLIC CONSTANTS / TYPES
# =============================================================================


DEFAULT_MAX_ABS_POSITION_M = 0.20
"""Default maximum absolute voxel-center displacement from isocenter.

Units
-----
meters

The default permits voxel-center positions from:

    -0.20 m to +0.20 m

along each logical axis.

This is a generic safety / sanity guardrail rather than a scanner-specific
DSV definition.
"""


LogicalAxis = Literal[
    "read",
    "phase",
    "slice",
]
"""Logical geometry axis understood by ``Voxel``."""


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Return ``value`` as a finite float."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite; received {value!r}."
        )

    return result


def _validate_positive(
    value: float,
    *,
    name: str,
) -> float:
    """Return ``value`` as a finite, strictly positive float."""

    result = _validate_finite(
        value,
        name=name,
    )

    if result <= 0.0:
        raise ValueError(
            f"{name} must be > 0; received {result!r}."
        )

    return result


def _validate_axis(
    axis: str,
) -> LogicalAxis:
    """Validate and normalize one logical-axis name."""

    normalized = str(
        axis
    ).strip().lower()

    if normalized not in {
        "read",
        "phase",
        "slice",
    }:
        raise ValueError(
            "Logical axis must be one of "
            "'read', 'phase', or 'slice'; "
            f"received {axis!r}."
        )

    return normalized  # type: ignore[return-value]


# =============================================================================
# VOXEL
# =============================================================================


@dataclass(frozen=True, slots=True)
class Voxel:
    """Rectangular voxel in logical read / phase / slice coordinates.

    Parameters
    ----------
    size_read_m
        Voxel extent along the logical read direction, in meters.

    size_phase_m
        Voxel extent along the logical phase direction, in meters.

    size_slice_m
        Voxel extent along the logical slice direction, in meters.

    position_read_m
        Voxel-center displacement from scanner isocenter along the logical
        read direction, in meters.

        May be positive, negative, or zero.

    position_phase_m
        Voxel-center displacement from scanner isocenter along the logical
        phase direction, in meters.

        May be positive, negative, or zero.

    position_slice_m
        Voxel-center displacement from scanner isocenter along the logical
        slice direction, in meters.

        May be positive, negative, or zero.

    max_abs_position_m
        Maximum permitted absolute voxel-center displacement from isocenter
        along each logical axis, in meters.

        Default:

            +/- 0.20 m

        Set to ``None`` to disable this guardrail.

    Notes
    -----
    Position refers to the CENTER of the voxel.

    The Voxel does not map logical axes onto physical scanner axes.
    That responsibility belongs to ``EncodingFrame``.

    The default position constraint is intentionally generic and should not be
    interpreted as a scanner-specific DSV specification.
    """

    # -------------------------------------------------------------------------
    # Voxel dimensions
    # -------------------------------------------------------------------------

    size_read_m: float
    size_phase_m: float
    size_slice_m: float

    # -------------------------------------------------------------------------
    # Voxel-center position relative to scanner isocenter
    # -------------------------------------------------------------------------

    position_read_m: float = 0.0
    position_phase_m: float = 0.0
    position_slice_m: float = 0.0

    # -------------------------------------------------------------------------
    # Generic center-position guardrail
    # -------------------------------------------------------------------------

    max_abs_position_m: float | None = DEFAULT_MAX_ABS_POSITION_M

    def __post_init__(self) -> None:
        """Validate and normalize voxel geometry."""

        # =====================================================================
        # Validate dimensions
        # =====================================================================
        #
        # Dimensions must be finite and strictly positive.
        # =====================================================================

        object.__setattr__(
            self,
            "size_read_m",
            _validate_positive(
                self.size_read_m,
                name="size_read_m",
            ),
        )

        object.__setattr__(
            self,
            "size_phase_m",
            _validate_positive(
                self.size_phase_m,
                name="size_phase_m",
            ),
        )

        object.__setattr__(
            self,
            "size_slice_m",
            _validate_positive(
                self.size_slice_m,
                name="size_slice_m",
            ),
        )

        # =====================================================================
        # Validate positions
        # =====================================================================
        #
        # Position may be:
        #
        #     negative
        #     zero
        #     positive
        #
        # but must always be finite.
        # =====================================================================

        position_read_m = _validate_finite(
            self.position_read_m,
            name="position_read_m",
        )

        position_phase_m = _validate_finite(
            self.position_phase_m,
            name="position_phase_m",
        )

        position_slice_m = _validate_finite(
            self.position_slice_m,
            name="position_slice_m",
        )

        object.__setattr__(
            self,
            "position_read_m",
            position_read_m,
        )

        object.__setattr__(
            self,
            "position_phase_m",
            position_phase_m,
        )

        object.__setattr__(
            self,
            "position_slice_m",
            position_slice_m,
        )

        # =====================================================================
        # Validate optional position guardrail
        # =====================================================================

        if self.max_abs_position_m is not None:

            max_abs_position_m = _validate_positive(
                self.max_abs_position_m,
                name="max_abs_position_m",
            )

            object.__setattr__(
                self,
                "max_abs_position_m",
                max_abs_position_m,
            )

            # -----------------------------------------------------------------
            # Center displacement is bounded symmetrically around isocenter.
            #
            # Example with default:
            #
            #     -0.20 m <= position <= +0.20 m
            #
            # Negative positions are therefore explicitly valid.
            # -----------------------------------------------------------------

            positions = {
                "position_read_m": position_read_m,
                "position_phase_m": position_phase_m,
                "position_slice_m": position_slice_m,
            }

            for name, value in positions.items():

                if abs(value) > max_abs_position_m:
                    raise ValueError(
                        f"{name}={value!r} m exceeds the allowed "
                        f"+/-{max_abs_position_m:g} m voxel-center "
                        "displacement from scanner isocenter."
                    )

    # =========================================================================
    # COMPACT REPRESENTATIONS
    # =========================================================================

    @property
    def size_m(
        self,
    ) -> tuple[float, float, float]:
        """Return voxel size as ``(read, phase, slice)`` in meters."""

        return (
            self.size_read_m,
            self.size_phase_m,
            self.size_slice_m,
        )

    @property
    def position_m(
        self,
    ) -> tuple[float, float, float]:
        """Return voxel-center position relative to isocenter.

        Returns
        -------
        tuple[float, float, float]
            ``(read, phase, slice)`` in meters.
        """

        return (
            self.position_read_m,
            self.position_phase_m,
            self.position_slice_m,
        )

    # =========================================================================
    # AXIS-BASED ACCESS
    # =========================================================================

    def size_along(
        self,
        axis: LogicalAxis | str,
    ) -> float:
        """Return voxel extent along one logical axis.

        Parameters
        ----------
        axis
            ``"read"``, ``"phase"``, or ``"slice"``.

        Returns
        -------
        float
            Voxel dimension along the requested axis, in meters.
        """

        logical_axis = _validate_axis(
            axis
        )

        if logical_axis == "read":
            return self.size_read_m

        if logical_axis == "phase":
            return self.size_phase_m

        return self.size_slice_m

    def position_along(
        self,
        axis: LogicalAxis | str,
    ) -> float:
        """Return voxel-center displacement along one logical axis.

        Parameters
        ----------
        axis
            ``"read"``, ``"phase"``, or ``"slice"``.

        Returns
        -------
        float
            Signed voxel-center displacement from scanner isocenter, in meters.

        Notes
        -----
        Negative values are valid and indicate displacement along the negative
        direction of the corresponding logical axis.
        """

        logical_axis = _validate_axis(
            axis
        )

        if logical_axis == "read":
            return self.position_read_m

        if logical_axis == "phase":
            return self.position_phase_m

        return self.position_slice_m

    # =========================================================================
    # BOUNDS
    # =========================================================================

    def bounds_along(
        self,
        axis: LogicalAxis | str,
    ) -> tuple[float, float]:
        """Return lower and upper voxel bounds along one logical axis.

        For a voxel centered at ``position`` with extent ``size``::

            lower = position - size / 2
            upper = position + size / 2

        The returned coordinates are relative to scanner isocenter.

        Returns
        -------
        tuple[float, float]
            ``(lower_m, upper_m)``.
        """

        size = self.size_along(
            axis
        )

        position = self.position_along(
            axis
        )

        half_size = 0.5 * size

        return (
            position - half_size,
            position + half_size,
        )

    @property
    def read_bounds_m(
        self,
    ) -> tuple[float, float]:
        """Return logical read bounds in meters."""

        return self.bounds_along(
            "read"
        )

    @property
    def phase_bounds_m(
        self,
    ) -> tuple[float, float]:
        """Return logical phase bounds in meters."""

        return self.bounds_along(
            "phase"
        )

    @property
    def slice_bounds_m(
        self,
    ) -> tuple[float, float]:
        """Return logical slice bounds in meters."""

        return self.bounds_along(
            "slice"
        )

    # =========================================================================
    # DERIVED GEOMETRY
    # =========================================================================

    @property
    def volume_m3(
        self,
    ) -> float:
        """Return voxel volume in cubic meters."""

        return (
            self.size_read_m
            * self.size_phase_m
            * self.size_slice_m
        )

    @property
    def is_at_isocenter(
        self,
    ) -> bool:
        """Return whether the voxel CENTER is at scanner isocenter."""

        return (
            self.position_read_m == 0.0
            and self.position_phase_m == 0.0
            and self.position_slice_m == 0.0
        )

    @property
    def max_center_displacement_m(
        self,
    ) -> float:
        """Return largest absolute center displacement along any logical axis."""

        return max(
            abs(self.position_read_m),
            abs(self.position_phase_m),
            abs(self.position_slice_m),
        )

    # =========================================================================
    # CONVENIENCE CONSTRUCTOR
    # =========================================================================

    @classmethod
    def from_size_and_position(
        cls,
        *,
        size_m: tuple[float, float, float],
        position_m: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        max_abs_position_m: float | None = DEFAULT_MAX_ABS_POSITION_M,
    ) -> "Voxel":
        """Create a voxel from compact ``(read, phase, slice)`` tuples.

        Parameters
        ----------
        size_m
            ``(read, phase, slice)`` voxel dimensions in meters.

        position_m
            ``(read, phase, slice)`` signed voxel-center displacement from
            scanner isocenter in meters.

        max_abs_position_m
            Optional maximum absolute center displacement along each logical
            axis.

        Examples
        --------
        A 20-mm isotropic voxel shifted +10 mm in read and -5 mm in slice::

            voxel = Voxel.from_size_and_position(
                size_m=(
                    20e-3,
                    20e-3,
                    20e-3,
                ),
                position_m=(
                    10e-3,
                    0.0,
                    -5e-3,
                ),
            )
        """

        if len(size_m) != 3:
            raise ValueError(
                "size_m must contain exactly three values in "
                "(read, phase, slice) order."
            )

        if len(position_m) != 3:
            raise ValueError(
                "position_m must contain exactly three values in "
                "(read, phase, slice) order."
            )

        return cls(
            size_read_m=size_m[0],
            size_phase_m=size_m[1],
            size_slice_m=size_m[2],
            position_read_m=position_m[0],
            position_phase_m=position_m[1],
            position_slice_m=position_m[2],
            max_abs_position_m=max_abs_position_m,
        )

    # =========================================================================
    # HUMAN-READABLE DESCRIPTION
    # =========================================================================

    def describe(
        self,
    ) -> str:
        """Return a compact human-readable description.

        Internal values remain SI units.

        Millimeters are used here only for human readability.
        """

        size_mm = tuple(
            value * 1e3
            for value in self.size_m
        )

        position_mm = tuple(
            value * 1e3
            for value in self.position_m
        )

        limit_text = (
            "unbounded"
            if self.max_abs_position_m is None
            else f"+/-{self.max_abs_position_m * 1e3:g} mm"
        )

        return (
            "Voxel("
            f"size_mm="
            f"({size_mm[0]:g}, {size_mm[1]:g}, {size_mm[2]:g}), "
            f"position_from_isocenter_mm="
            f"({position_mm[0]:g}, "
            f"{position_mm[1]:g}, "
            f"{position_mm[2]:g}), "
            f"center_position_limit={limit_text}"
            ")"
        )


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================


__all__ = [
    "DEFAULT_MAX_ABS_POSITION_M",
    "LogicalAxis",
    "Voxel",
]