"""RF constructors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from pypulseq_star.events import SeqStarRFBlockEvent
from pypulseq_star.opts import Opts
from pypulseq_star.shapes import SeqStarRFBlockShape
from pypulseq_star.expressions import EventPropertyRef, Expression
from ._symbolic import (
    attach_symbolic_specs,
    evaluate_default,
    resolve_float,
    resolve_optional_float,
    require_positive,
    resolve_string,
    symbolic_specs,
)


def make_block_pulse(
    flip_angle: float | None = None,
    duration: float | None = None,
    *,
    name: str = "rf",
    use: str | None = None,
    phase_offset: float | None = None,
    freq_offset: float | None = None,
    system: Opts | None = None,
    parameters: Mapping[str, Any] | None = None,
    slice_thickness: float | None = None,
    bandwidth: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    return_gz: bool = False,
    channel: str = "z",
) -> SeqStarRFBlockEvent | tuple[SeqStarRFBlockEvent, Any, Any]:
    """Create an enriched rectangular RF pulse.

    If ``return_gz=True`` and ``slice_thickness`` is supplied, return
    ``(rf, gz, gzr)`` where ``gz`` is the slice-select trapezoid and ``gzr`` is the slice-refocusing lobe.

    This intentionally updates only the public API/object return path. It does
    not yet create a gammaSTAR ``ssel`` hierarchy node or an RF-gradient
    relationship; those sequence-hierarchy refinements will be added later.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    constructor_specs = {
        "flip_angle": flip_angle,
        "duration": duration,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "slice_thickness": slice_thickness,
        "bandwidth": bandwidth,
        "use": use,
    }

    symbolic_event_specs = symbolic_specs(
        flip_angle=flip_angle,
        duration=duration,
        phase_offset=phase_offset,
        freq_offset=freq_offset,
        slice_thickness=slice_thickness,
    )

    resolved_slice_thickness = _resolve_slice_thickness_argument(
        slice_thickness
    )

    resolved_flip_angle = _resolve_flip_angle(
        explicit_flip_angle=flip_angle,
        parameters=parameters,
    )

    resolved_duration = _resolve_duration(
        explicit_duration=duration,
        parameters=parameters,
    )

    resolved_phase_offset = _resolve_float_parameter(
        explicit_value=phase_offset,
        parameters=parameters,
        keys=(
            "excitation_phase",
            "phase_offset",
            "phase",
            "rf_phase",
        ),
        default=0.0,
    )

    resolved_freq_offset = _resolve_float_parameter(
        explicit_value=freq_offset,
        parameters=parameters,
        keys=(
            "rf_frequency_offset",
            "freq_offset",
            "frequency_offset",
        ),
        default=0.0,
    )

    resolved_use = _resolve_string_parameter(
        explicit_value=use,
        parameters=parameters,
        keys=(
            "rf_use",
            "use",
        ),
        default=use,
    )

    raster_duration = _round_to_raster(
        value=resolved_duration,
        raster=system.rf_raster_time,
        name="RF duration",
    )

    amplitude_t = _block_rf_amplitude_t(
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        gamma_hz_per_t=system.gamma,
    )

    if amplitude_t > system.max_rf:
        raise ValueError(
            "RF block pulse exceeds system.max_rf. "
            f"Required B1 = {amplitude_t:.6g} T, "
            f"system.max_rf = {system.max_rf:.6g} T. "
            "Increase duration, lower flip_angle, or update system.max_rf."
        )

    shape = SeqStarRFBlockShape(
        name=f"{name}_shape",
        duration=raster_duration,
        rf_raster_time=system.rf_raster_time,
        max_rf=system.max_rf,
    )

    event = SeqStarRFBlockEvent(
        name=name,
        role=resolved_use,
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        phase_offset=resolved_phase_offset,
        freq_offset=resolved_freq_offset,
        use=resolved_use,
        shape=shape,
    )

    _attach_context_if_supported(
        event,
        system=system,
        parameters=parameters,
    )

    _attach_rf_timing_metadata(
        event,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=None,
        pulse_type="block",
        amplitude_t=amplitude_t,
        shape_metadata={
            "family": "block",
            "amplitude_t": amplitude_t,
        },
    )

    _attach_rf_symbolic_constructor(
        event,
        family="block",
        specs=constructor_specs,
    )
    attach_symbolic_specs(event, symbolic_event_specs)

    return _return_rf_with_optional_slice_select_gradient(
        event,
        name=name,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=None,
        pulse_type="block",
        slice_thickness=resolved_slice_thickness,
        bandwidth=bandwidth,
        time_bw_product=None,
        max_grad=max_grad,
        max_slew=max_slew,
        return_gz=return_gz,
        channel=channel,
    )


def make_sinc_pulse(
    flip_angle: float | None = None,
    duration: float | None = None,
    *,
    name: str = "rf",
    use: str | None = None,
    phase_offset: float | None = None,
    freq_offset: float | None = None,
    time_bw_product: float | None = None,
    apodization: float | None = None,
    center_pos: float | None = None,
    delay: float | None = None,
    system: Opts | None = None,
    parameters: Mapping[str, Any] | None = None,
    slice_thickness: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    return_gz: bool = False,
    channel: str = "z",
    gz_name: str | None = None,
    gzr_name: str | None = None,
) -> SeqStarRFBlockEvent:
    """Create an enriched sinc RF pulse.

    If ``return_gz=True`` and ``slice_thickness`` is supplied, return
    ``(rf, gz, gzr)`` where ``gz`` is the slice-select trapezoid and ``gzr`` is
    the slice-refocusing lobe.

    Naming policy
    -------------
    ``name`` names the RF event.

    ``gz_name`` names the slice-select gradient. By default it is named
    ``"ssel_gz"`` rather than ``f"{name}_gz"`` so gradient events are not
    mistaken for RF events downstream.

    ``gzr_name`` names the slice-refocusing gradient. By default it is named
    ``"gz_rephaser"``.

    The slice-select gradient is API-level only in this iteration. The writer
    can export it as an ordinary gradient event, but gammaSTAR ``ssel`` nodes
    and RF-gradient relationships are intentionally deferred.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    constructor_specs = {
        "flip_angle": flip_angle,
        "duration": duration,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "time_bw_product": time_bw_product,
        "apodization": apodization,
        "center_pos": center_pos,
        "delay": delay,
        "slice_thickness": slice_thickness,
        "use": use,
    }

    symbolic_event_specs = symbolic_specs(
        flip_angle=flip_angle,
        duration=duration,
        phase_offset=phase_offset,
        freq_offset=freq_offset,
        time_bw_product=time_bw_product,
        apodization=apodization,
        center_pos=center_pos,
        delay=delay,
        slice_thickness=slice_thickness,
    )

    resolved_slice_thickness = _resolve_slice_thickness_argument(
        slice_thickness
    )

    resolved_flip_angle = _resolve_flip_angle(
        explicit_flip_angle=flip_angle,
        parameters=parameters,
    )

    resolved_duration = _resolve_duration(
        explicit_duration=duration,
        parameters=parameters,
    )

    resolved_phase_offset = _resolve_float_parameter(
        explicit_value=phase_offset,
        parameters=parameters,
        keys=(
            "excitation_phase",
            "phase_offset",
            "phase",
            "rf_phase",
        ),
        default=0.0,
    )

    resolved_freq_offset = _resolve_float_parameter(
        explicit_value=freq_offset,
        parameters=parameters,
        keys=(
            "rf_frequency_offset",
            "freq_offset",
            "frequency_offset",
        ),
        default=0.0,
    )

    resolved_use = _resolve_string_parameter(
        explicit_value=use,
        parameters=parameters,
        keys=(
            "rf_use",
            "use",
        ),
        default=use,
    )

    resolved_time_bw_product = _resolve_float_parameter(
        explicit_value=time_bw_product,
        parameters=parameters,
        keys=(
            "time_bw_product",
            "tbw",
            "rf_time_bw_product",
            "rf_tbw",
        ),
        default=4.0,
    )

    resolved_apodization = _resolve_float_parameter(
        explicit_value=apodization,
        parameters=parameters,
        keys=(
            "apodization",
            "rf_apodization",
        ),
        default=0.0,
    )

    resolved_center_pos = _resolve_float_parameter(
        explicit_value=center_pos,
        parameters=parameters,
        keys=(
            "center_pos",
            "rf_center_pos",
        ),
        default=0.5,
    )

    _validate_shaped_rf_controls(
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
    )

    raster_duration = _round_to_raster(
        value=resolved_duration,
        raster=system.rf_raster_time,
        name="RF duration",
    )

    amplitude_t = _sinc_rf_amplitude_t(
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        gamma_hz_per_t=system.gamma,
        rf_raster_time=system.rf_raster_time,
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
    )

    if amplitude_t > system.max_rf:
        raise ValueError(
            "RF sinc pulse exceeds system.max_rf. "
            f"Required peak B1 = {amplitude_t:.6g} T, "
            f"system.max_rf = {system.max_rf:.6g} T. "
            "Increase duration, lower flip_angle, reduce apodization/TBW, "
            "or update system.max_rf."
        )

    from pypulseq_star.shapes.rf import SeqStarRFSincShape

    shape = SeqStarRFSincShape(
        name=f"{name}_shape",
        duration=raster_duration,
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
        rf_raster_time=system.rf_raster_time,
        max_rf=system.max_rf,
    )

    event = SeqStarRFBlockEvent(
        name=name,
        role=resolved_use,
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        phase_offset=resolved_phase_offset,
        freq_offset=resolved_freq_offset,
        use=resolved_use,
        shape=shape,
    )

    _attach_context_if_supported(
        event,
        system=system,
        parameters=parameters,
    )

    _attach_rf_timing_metadata(
        event,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="sinc",
        amplitude_t=amplitude_t,
        shape_metadata={
            "family": "sinc",
            "time_bw_product": resolved_time_bw_product,
            "apodization": resolved_apodization,
            "center_pos": resolved_center_pos,
            "amplitude_t": amplitude_t,
        },
    )

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        event.parameters.setdefault("time_bw_product", resolved_time_bw_product)
        event.parameters.setdefault("tbw", resolved_time_bw_product)
        event.parameters.setdefault("apodization", resolved_apodization)
        event.parameters.setdefault("center_pos", resolved_center_pos)

    _attach_rf_symbolic_constructor(
        event,
        family="sinc",
        specs=constructor_specs,
    )
    attach_symbolic_specs(event, symbolic_event_specs)

    return _return_rf_with_optional_slice_select_gradient(
        event,
        name=name,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="sinc",
        slice_thickness=resolved_slice_thickness,
        bandwidth=None,
        time_bw_product=resolved_time_bw_product,
        max_grad=max_grad,
        max_slew=max_slew,
        return_gz=return_gz,
        channel=channel,
        gz_name=gz_name or "ssel_gz",
        gzr_name=gzr_name or "gz_rephaser",
    )

def make_gauss_pulse(
    flip_angle: float | None = None,
    duration: float | None = None,
    *,
    name: str = "rf",
    use: str | None = None,
    phase_offset: float | None = None,
    freq_offset: float | None = None,
    time_bw_product: float | None = None,
    apodization: float | None = None,
    center_pos: float | None = None,
    delay: float | None = None,
    system: Opts | None = None,
    parameters: Mapping[str, Any] | None = None,
    slice_thickness: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    return_gz: bool = False,
    channel: str = "z",
) -> SeqStarRFBlockEvent:
    """Create an enriched Gaussian RF pulse.

    This follows the PyPulseq RF-constructor style while returning an enriched
    SeqStar RF event.

    If ``return_gz=True`` and ``slice_thickness`` is supplied, return
    ``(rf, gz, gzr)`` where ``gz`` is the slice-select trapezoid and ``gzr`` is the slice-refocusing lobe.
    The gammaSTAR ``ssel`` hierarchy node is intentionally deferred.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    constructor_specs = {
        "flip_angle": flip_angle,
        "duration": duration,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "time_bw_product": time_bw_product,
        "apodization": apodization,
        "center_pos": center_pos,
        "delay": delay,
        "slice_thickness": slice_thickness,
        "use": use,
    }

    symbolic_event_specs = symbolic_specs(
        flip_angle=flip_angle,
        duration=duration,
        phase_offset=phase_offset,
        freq_offset=freq_offset,
        time_bw_product=time_bw_product,
        apodization=apodization,
        center_pos=center_pos,
        delay=delay,
        slice_thickness=slice_thickness,
    )

    resolved_slice_thickness = _resolve_slice_thickness_argument(
        slice_thickness
    )

    resolved_flip_angle = _resolve_flip_angle(
        explicit_flip_angle=flip_angle,
        parameters=parameters,
    )

    resolved_duration = _resolve_duration(
        explicit_duration=duration,
        parameters=parameters,
    )

    resolved_phase_offset = _resolve_float_parameter(
        explicit_value=phase_offset,
        parameters=parameters,
        keys=(
            "excitation_phase",
            "phase_offset",
            "phase",
            "rf_phase",
        ),
        default=0.0,
    )

    resolved_freq_offset = _resolve_float_parameter(
        explicit_value=freq_offset,
        parameters=parameters,
        keys=(
            "rf_frequency_offset",
            "freq_offset",
            "frequency_offset",
        ),
        default=0.0,
    )

    resolved_use = _resolve_string_parameter(
        explicit_value=use,
        parameters=parameters,
        keys=(
            "rf_use",
            "use",
        ),
        default=use,
    )

    resolved_time_bw_product = _resolve_float_parameter(
        explicit_value=time_bw_product,
        parameters=parameters,
        keys=(
            "time_bw_product",
            "tbw",
            "rf_time_bw_product",
            "rf_tbw",
        ),
        default=4.0,
    )

    resolved_apodization = _resolve_float_parameter(
        explicit_value=apodization,
        parameters=parameters,
        keys=(
            "apodization",
            "rf_apodization",
        ),
        default=0.5,
    )

    resolved_center_pos = _resolve_float_parameter(
        explicit_value=center_pos,
        parameters=parameters,
        keys=(
            "center_pos",
            "rf_center_pos",
        ),
        default=0.5,
    )

    _validate_shaped_rf_controls(
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
    )

    raster_duration = _round_to_raster(
        value=resolved_duration,
        raster=system.rf_raster_time,
        name="RF duration",
    )

    amplitude_t = _gauss_rf_amplitude_t(
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        gamma_hz_per_t=system.gamma,
        rf_raster_time=system.rf_raster_time,
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
    )

    if amplitude_t > system.max_rf:
        raise ValueError(
            "RF Gaussian pulse exceeds system.max_rf. "
            f"Required peak B1 = {amplitude_t:.6g} T, "
            f"system.max_rf = {system.max_rf:.6g} T. "
            "Increase duration, lower flip_angle, reduce TBW/apodization, "
            "or update system.max_rf."
        )

    from pypulseq_star.shapes.rf import SeqStarRFGaussShape

    shape = SeqStarRFGaussShape(
        name=f"{name}_shape",
        duration=raster_duration,
        time_bw_product=resolved_time_bw_product,
        apodization=resolved_apodization,
        center_pos=resolved_center_pos,
        rf_raster_time=system.rf_raster_time,
        max_rf=system.max_rf,
    )

    event = SeqStarRFBlockEvent(
        name=name,
        role=resolved_use,
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        phase_offset=resolved_phase_offset,
        freq_offset=resolved_freq_offset,
        use=resolved_use,
        shape=shape,
    )

    _attach_context_if_supported(
        event,
        system=system,
        parameters=parameters,
    )

    _attach_rf_timing_metadata(
        event,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="gauss",
        amplitude_t=amplitude_t,
        shape_metadata={
            "family": "gauss",
            "time_bw_product": resolved_time_bw_product,
            "apodization": resolved_apodization,
            "center_pos": resolved_center_pos,
            "amplitude_t": amplitude_t,
        },
    )

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        event.parameters.setdefault("time_bw_product", resolved_time_bw_product)
        event.parameters.setdefault("tbw", resolved_time_bw_product)
        event.parameters.setdefault("apodization", resolved_apodization)
        event.parameters.setdefault("center_pos", resolved_center_pos)

    _attach_rf_symbolic_constructor(
        event,
        family="gauss",
        specs=constructor_specs,
    )
    attach_symbolic_specs(event, symbolic_event_specs)

    return _return_rf_with_optional_slice_select_gradient(
        event,
        name=name,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="gauss",
        slice_thickness=resolved_slice_thickness,
        bandwidth=None,
        time_bw_product=resolved_time_bw_product,
        max_grad=max_grad,
        max_slew=max_slew,
        return_gz=return_gz,
        channel=channel,
    )


def make_arbitrary_rf(
    signal: Iterable[float] | None = None,
    flip_angle: float | None = None,
    *,
    duration: float | None = None,
    dwell: float | None = None,
    name: str = "rf",
    use: str | None = None,
    phase_offset: float | None = None,
    freq_offset: float | None = None,
    delay: float | None = None,
    system: Opts | None = None,
    parameters: Mapping[str, Any] | None = None,
    slice_thickness: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    return_gz: bool = False,
    channel: str = "z",
) -> SeqStarRFBlockEvent:
    """Create an enriched arbitrary RF pulse from user-provided samples.

    If ``return_gz=True`` and ``slice_thickness`` is supplied, return
    ``(rf, gz, gzr)`` where ``gz`` is the slice-select trapezoid and ``gzr`` is the slice-refocusing lobe.
    The slice-select bandwidth defaults to ``time_bw_product / duration`` when
    available in parameters, otherwise ``1 / duration``.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    constructor_specs = {
        "signal": tuple(signal) if signal is not None else None,
        "flip_angle": flip_angle,
        "duration": duration,
        "dwell": dwell,
        "phase_offset": phase_offset,
        "freq_offset": freq_offset,
        "delay": delay,
        "slice_thickness": slice_thickness,
        "use": use,
    }

    symbolic_event_specs = symbolic_specs(
        flip_angle=flip_angle,
        duration=duration,
        dwell=dwell,
        phase_offset=phase_offset,
        freq_offset=freq_offset,
        delay=delay,
        slice_thickness=slice_thickness,
    )

    resolved_slice_thickness = _resolve_slice_thickness_argument(
        slice_thickness
    )

    resolved_signal = _resolve_arbitrary_signal(
        explicit_signal=signal,
        parameters=parameters,
    )

    resolved_flip_angle = _resolve_flip_angle(
        explicit_flip_angle=flip_angle,
        parameters=parameters,
    )

    resolved_duration = _resolve_arbitrary_duration(
        explicit_duration=duration,
        dwell=dwell,
        signal_length=len(resolved_signal),
        parameters=parameters,
        system=system,
    )

    resolved_phase_offset = _resolve_float_parameter(
        explicit_value=phase_offset,
        parameters=parameters,
        keys=(
            "excitation_phase",
            "phase_offset",
            "phase",
            "rf_phase",
        ),
        default=0.0,
    )

    resolved_freq_offset = _resolve_float_parameter(
        explicit_value=freq_offset,
        parameters=parameters,
        keys=(
            "rf_frequency_offset",
            "freq_offset",
            "frequency_offset",
        ),
        default=0.0,
    )

    resolved_use = _resolve_string_parameter(
        explicit_value=use,
        parameters=parameters,
        keys=(
            "rf_use",
            "use",
        ),
        default=use,
    )

    raster_duration = _round_to_raster(
        value=resolved_duration,
        raster=system.rf_raster_time,
        name="RF duration",
    )

    raster_signal = _resample_real_signal_to_raster(
        signal=resolved_signal,
        duration=raster_duration,
        rf_raster_time=system.rf_raster_time,
    )

    amplitude_t = _arbitrary_rf_peak_amplitude_t(
        signal=raster_signal,
        flip_angle=resolved_flip_angle,
        rf_raster_time=system.rf_raster_time,
        gamma_hz_per_t=system.gamma,
    )

    if amplitude_t > system.max_rf:
        raise ValueError(
            "Arbitrary RF pulse exceeds system.max_rf. "
            f"Required peak B1 = {amplitude_t:.6g} T, "
            f"system.max_rf = {system.max_rf:.6g} T. "
            "Increase duration, lower flip_angle, reduce signal area/shape, "
            "or update system.max_rf."
        )

    from pypulseq_star.shapes.rf import SeqStarRFArbitraryShape

    shape = SeqStarRFArbitraryShape(
        name=f"{name}_shape",
        duration=raster_duration,
        signal=raster_signal,
        rf_raster_time=system.rf_raster_time,
        max_rf=system.max_rf,
    )

    event = SeqStarRFBlockEvent(
        name=name,
        role=resolved_use,
        flip_angle=resolved_flip_angle,
        duration=raster_duration,
        phase_offset=resolved_phase_offset,
        freq_offset=resolved_freq_offset,
        use=resolved_use,
        shape=shape,
    )

    _attach_context_if_supported(
        event,
        system=system,
        parameters=parameters,
    )

    _attach_rf_timing_metadata(
        event,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="arbitrary",
        amplitude_t=amplitude_t,
        shape_metadata={
            "family": "arbitrary",
            "input_num_samples": len(resolved_signal),
            "raster_num_samples": len(raster_signal),
            "rf_raster_time": system.rf_raster_time,
            "amplitude_t": amplitude_t,
        },
    )

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        event.parameters.setdefault("input_num_samples", len(resolved_signal))
        event.parameters.setdefault("raster_num_samples", len(raster_signal))
        event.parameters.setdefault(
            "arbitrary_signal_label",
            parameters.get("arbitrary_signal_label", "arbitrary"),
        )
        event.parameters.setdefault("arbitrary_input_dwell", dwell)
        event.parameters.setdefault("arbitrary_raster_dwell", system.rf_raster_time)

    _attach_rf_symbolic_constructor(
        event,
        family="arbitrary",
        specs=constructor_specs,
    )
    attach_symbolic_specs(event, symbolic_event_specs)

    return _return_rf_with_optional_slice_select_gradient(
        event,
        name=name,
        system=system,
        parameters=parameters,
        duration=raster_duration,
        delay=delay,
        pulse_type="arbitrary",
        slice_thickness=resolved_slice_thickness,
        bandwidth=None,
        time_bw_product=_resolve_optional_float_parameter(
            parameters=parameters,
            keys=("time_bw_product", "tbw", "rf_time_bw_product", "rf_tbw"),
        ),
        max_grad=max_grad,
        max_slew=max_slew,
        return_gz=return_gz,
        channel=channel,
    )


def _attach_rf_symbolic_constructor(
    event: Any,
    *,
    family: str,
    specs: Mapping[str, Any],
) -> None:
    """Retain RF constructor intent for sequence resolution and gammaSTAR."""

    record = {
        "family": str(family),
        "specs": dict(specs),
    }

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        metadata["symbolic_constructor"] = record

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters["_symbolic_constructor"] = record


def _rf_constructor_specs(event: Any) -> dict[str, Any]:
    """Return the original RF constructor inputs retained on an event."""

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        record = metadata.get("symbolic_constructor")
        if isinstance(record, Mapping):
            specs = record.get("specs")
            if isinstance(specs, Mapping):
                return dict(specs)

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, Mapping):
        record = parameters.get("_symbolic_constructor")
        if isinstance(record, Mapping):
            specs = record.get("specs")
            if isinstance(specs, Mapping):
                return dict(specs)

    return {}


def _set_gradient_symbolic_constructor_specs(
    gradient: Any,
    **updates: Any,
) -> None:
    """Update a numeric gradient with authoritative symbolic constructor inputs.

    The gradient is first constructed numerically so plotting, Pulseq writing,
    and immediate validation remain available. This helper then replaces only
    the authoritative constructor fields with their original expressions for
    sequence resolution and gammaSTAR export.
    """

    symbolic_updates = {
        key: value
        for key, value in updates.items()
        if isinstance(value, Expression)
    }
    if not symbolic_updates:
        return

    for store_name, constructor_key in (
        ("metadata", "symbolic_constructor"),
        ("parameters", "_symbolic_constructor"),
    ):
        store = getattr(gradient, store_name, None)
        if not isinstance(store, dict):
            continue

        record = store.get(constructor_key)
        if not isinstance(record, dict):
            record = {
                "family": "trapezoid",
                "specs": {},
            }
            store[constructor_key] = record

        specs = record.setdefault("specs", {})
        if isinstance(specs, dict):
            specs.update(symbolic_updates)

    metadata = getattr(gradient, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("symbolic_properties", {}).update(
            symbolic_updates
        )
        metadata["has_symbolic_properties"] = True

    parameters = getattr(gradient, "parameters", None)
    if isinstance(parameters, dict):
        parameters.setdefault("_symbolic_properties", {}).update(
            symbolic_updates
        )


def _slice_select_symbolic_inputs(
    event: Any,
    *,
    numeric_duration: float,
    numeric_slice_thickness: float,
    numeric_bandwidth: float,
) -> tuple[Any, Any, Any]:
    """Return authoritative duration, thickness, and bandwidth specifications."""

    specs = _rf_constructor_specs(event)

    duration_spec = specs.get("duration")
    if duration_spec is None:
        duration_spec = numeric_duration

    thickness_spec = specs.get("slice_thickness")
    if thickness_spec is None:
        thickness_spec = numeric_slice_thickness

    bandwidth_spec = specs.get("bandwidth")
    if bandwidth_spec is None:
        tbw_spec = specs.get("time_bw_product")
        if tbw_spec is not None:
            bandwidth_spec = tbw_spec / duration_spec
        elif isinstance(duration_spec, Expression):
            bandwidth_spec = 1.0 / duration_spec
        else:
            bandwidth_spec = numeric_bandwidth

    return duration_spec, thickness_spec, bandwidth_spec


def _resolve_slice_thickness_argument(
    value: Any | None,
) -> float | None:
    """Resolve and validate slice thickness before numeric use."""

    resolved = resolve_optional_float(
        value,
        field_name="slice_thickness",
    )
    require_positive(
        resolved,
        field_name="slice_thickness",
        allow_none=True,
    )
    return resolved


def _return_rf_with_optional_slice_select_gradient(
    event,
    *,
    name,
    system,
    parameters,
    duration,
    delay,
    pulse_type,
    slice_thickness,
    bandwidth,
    time_bw_product,
    max_grad,
    max_slew,
    return_gz,
    channel,
    gz_name: str | None = None,
    gzr_name: str | None = None,
) -> SeqStarRFBlockEvent | tuple[SeqStarRFBlockEvent, Any, Any]:
    """Return RF alone or ``(rf, gz, gzr)`` with slice-select gradients.

    The returned gradients are normal SeqStar trapezoid gradient events:

    - ``gz`` is the slice-select gradient intended to be simultaneous with RF.
    - ``gzr`` is the slice-refocusing/rephasing lobe intended after RF.

    Naming policy:
        ``name`` names the RF event.
        ``gz_name`` names the slice-select gradient.
        ``gzr_name`` names the slice-refocusing gradient.

    This helper updates only the public RF-constructor API and object return
    path. It intentionally does not create a gammaSTAR ``ssel`` node or formal
    RF-gradient relationship object; those hierarchy refinements are deferred.
    """

    slice_select_name = gz_name or "gz_ssel"
    slice_refocus_name = gzr_name or "gz_rephaser"

    if slice_thickness is None:
        if return_gz:
            raise ValueError("slice_thickness must be supplied when return_gz=True.")
        return event

    resolved_slice_thickness = _resolve_slice_thickness_argument(
        slice_thickness
    )
    assert resolved_slice_thickness is not None

    gradient_bandwidth = _resolve_slice_select_bandwidth(
        bandwidth=bandwidth,
        time_bw_product=time_bw_product,
        duration=duration,
        parameters=parameters,
    )

    gradient_amplitude = gradient_bandwidth / resolved_slice_thickness

    (
        symbolic_rf_duration,
        symbolic_slice_thickness,
        symbolic_gradient_bandwidth,
    ) = _slice_select_symbolic_inputs(
        event,
        numeric_duration=float(duration),
        numeric_slice_thickness=resolved_slice_thickness,
        numeric_bandwidth=gradient_bandwidth,
    )
    symbolic_gradient_amplitude = (
        symbolic_gradient_bandwidth / symbolic_slice_thickness
    )

    rf_delay = max(
        float(
            delay
            if delay is not None
            else parameters.get("rf_delay", parameters.get("delay", 0.0))
        ),
        float(system.rf_dead_time),
    )

    from pypulseq_star.make.grad import make_trapezoid

    # Slice-select lobe. The RF pulse is intended to sit on the flat top of
    # this gradient. The constructor chooses the shortest legal ramps based on
    # system limits. The pulse itself carries the RF delay; the gradient uses
    # the same delay so users can add ``rf`` and ``gz`` to the same block.
    gz_metadata = {
        "family": "slice_select",
        "source_rf": name,
        "source_rf_type": pulse_type,
        "slice_thickness": resolved_slice_thickness,
        "bandwidth_hz": gradient_bandwidth,
        "amplitude": gradient_amplitude,
        "flat_time": duration,
        "rf_delay": rf_delay,
        "max_grad": max_grad,
        "max_slew": max_slew,
        "relationship": "slice_select_for_rf",
        "note": (
            "API-level slice-select gradient. gammaSTAR ssel hierarchy and "
            "formal RF-gradient relationships are intentionally deferred."
        ),
    }

    gz = make_trapezoid(
        channel=channel,
        amplitude=gradient_amplitude,
        flat_time=duration,
        delay=rf_delay,
        system=system,
        max_grad=max_grad,
        max_slew=max_slew,
        name=slice_select_name,
        role="slice_select",
        axis_role="slice",
        metadata=gz_metadata,
    )

    _set_gradient_symbolic_constructor_specs(
        gz,
        amplitude=symbolic_gradient_amplitude,
        flat_time=symbolic_rf_duration,
    )

    gz_actual_name = getattr(gz, "name", slice_select_name)
    gz_area = _gradient_event_area(gz)
    refocus_area = -0.5 * gz_area

    # Slice-refocusing lobe. This is intended to be placed after the RF/gz
    # block, so its own delay is zero. Users should typically do:
    #
    #     seq.add_block(rf, gz)
    #     seq.add_block(gzr)
    #
    # rather than placing gz and gzr in the same Pulseq block.
    gzr_metadata = {
        "family": "slice_refocus",
        "source_rf": name,
        "source_rf_type": pulse_type,
        "slice_thickness": resolved_slice_thickness,
        "bandwidth_hz": gradient_bandwidth,
        "slice_select_gradient": gz_actual_name,
        "slice_select_area": gz_area,
        "refocus_area": refocus_area,
        "relationship": "slice_refocus_for_slice_select",
        "refocuses": gz_actual_name,
        "paired_rf": name,
        "note": (
            "API-level slice-refocusing gradient. gammaSTAR ssel hierarchy "
            "and formal RF-gradient relationships are intentionally deferred."
        ),
    }

    gzr = make_trapezoid(
        channel=channel,
        area=refocus_area,
        delay=0.0,
        system=system,
        max_grad=max_grad,
        max_slew=max_slew,
        name=slice_refocus_name,
        role="slice_refocus",
        axis_role="slice",
        metadata=gzr_metadata,
    )

    symbolic_refocus_area = -0.5 * EventPropertyRef(
        gz_actual_name,
        "area",
        metadata={"event": gz},
    )
    _set_gradient_symbolic_constructor_specs(
        gzr,
        area=symbolic_refocus_area,
    )

    gzr_actual_name = getattr(gzr, "name", slice_refocus_name)

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        event.parameters.setdefault("slice_thickness", resolved_slice_thickness)
        event.parameters.setdefault("slice_select_bandwidth", gradient_bandwidth)
        event.parameters.setdefault("slice_select_gradient_amplitude", gradient_amplitude)
        event.parameters.setdefault("slice_select_gradient_area", gz_area)
        event.parameters.setdefault("slice_refocus_gradient_area", refocus_area)
        event.parameters.setdefault("slice_select_channel", channel)
        event.parameters.setdefault("slice_select_gradient_name", gz_actual_name)
        event.parameters.setdefault("slice_refocus_gradient_name", gzr_actual_name)
        event.parameters.setdefault("return_gz", bool(return_gz))
        event.parameters.setdefault("return_gzr", bool(return_gz))

    if hasattr(event, "metadata") and isinstance(event.metadata, dict):
        event.metadata.setdefault("slice_select_gradient", gz_metadata)
        event.metadata.setdefault("slice_refocus_gradient", gzr_metadata)
        event.metadata.setdefault("slice_select_gradient_name", gz_actual_name)
        event.metadata.setdefault("slice_refocus_gradient_name", gzr_actual_name)

    if hasattr(gz, "parameters") and isinstance(gz.parameters, dict):
        gz.parameters.setdefault("paired_rf", name)
        gz.parameters.setdefault("slice_thickness", resolved_slice_thickness)
        gz.parameters.setdefault("slice_select_bandwidth", gradient_bandwidth)
        gz.parameters.setdefault("slice_select_area", gz_area)
        gz.parameters.setdefault("slice_refocus_area", refocus_area)
        gz.parameters.setdefault("slice_refocus_gradient", gzr_actual_name)

    if hasattr(gzr, "parameters") and isinstance(gzr.parameters, dict):
        gzr.parameters.setdefault("paired_rf", name)
        gzr.parameters.setdefault("refocuses", gz_actual_name)
        gzr.parameters.setdefault("slice_thickness", resolved_slice_thickness)
        gzr.parameters.setdefault("slice_select_bandwidth", gradient_bandwidth)
        gzr.parameters.setdefault("slice_select_area", gz_area)
        gzr.parameters.setdefault("slice_refocus_area", refocus_area)
        gzr.parameters.setdefault("slice_select_gradient", gz_actual_name)

    if hasattr(gz, "metadata") and isinstance(gz.metadata, dict):
        gz.metadata.setdefault("paired_rf", name)
        gz.metadata.setdefault("role", "slice_select")
        gz.metadata.setdefault("slice_refocus_gradient", gzr_actual_name)

    if hasattr(gzr, "metadata") and isinstance(gzr.metadata, dict):
        gzr.metadata.setdefault("paired_rf", name)
        gzr.metadata.setdefault("role", "slice_refocus")
        gzr.metadata.setdefault("refocuses", gz_actual_name)
        gzr.metadata.setdefault("slice_select_gradient", gz_actual_name)

    if return_gz:
        return event, gz, gzr

    return event

def _gradient_event_area(gradient: Any) -> float:
    """Return the total gradient area for a SeqStar gradient-like object."""

    for key in ("area", "total_area"):
        if hasattr(gradient, key) and getattr(gradient, key) is not None:
            return float(getattr(gradient, key))

    parameters = getattr(gradient, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in ("area", "total_area"):
            if key in parameters and parameters[key] is not None:
                return float(parameters[key])

    metadata = getattr(gradient, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("area", "total_area"):
            if key in metadata and metadata[key] is not None:
                return float(metadata[key])

    shape = getattr(gradient, "shape", None)
    if shape is not None:
        for key in ("area", "total_area"):
            if hasattr(shape, key) and getattr(shape, key) is not None:
                return float(getattr(shape, key))

        if hasattr(shape, "to_dict"):
            try:
                shape_dict = dict(shape.to_dict())
                for key in ("area", "total_area"):
                    if key in shape_dict and shape_dict[key] is not None:
                        return float(shape_dict[key])
            except Exception:
                pass

    amplitude = _gradient_event_float(gradient, "amplitude", "amp")
    rise_time = _gradient_event_float(gradient, "rise_time")
    flat_time = _gradient_event_float(gradient, "flat_time")
    fall_time = _gradient_event_float(gradient, "fall_time")

    if (
        amplitude is not None
        and rise_time is not None
        and flat_time is not None
        and fall_time is not None
    ):
        return float(amplitude) * (0.5 * rise_time + flat_time + 0.5 * fall_time)

    raise ValueError(f"Could not determine gradient area for {gradient!r}.")


def _gradient_event_float(gradient: Any, *keys: str) -> float | None:
    """Return a float gradient field from event, parameters, metadata, or shape."""

    for key in keys:
        if hasattr(gradient, key) and getattr(gradient, key) is not None:
            try:
                return float(getattr(gradient, key))
            except (TypeError, ValueError):
                pass

    parameters = getattr(gradient, "parameters", None)
    if isinstance(parameters, Mapping):
        for key in keys:
            if key in parameters and parameters[key] is not None:
                try:
                    return float(parameters[key])
                except (TypeError, ValueError):
                    pass

    metadata = getattr(gradient, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping):
            for key in keys:
                if key in timing and timing[key] is not None:
                    try:
                        return float(timing[key])
                    except (TypeError, ValueError):
                        pass
        for key in keys:
            if key in metadata and metadata[key] is not None:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass

    shape = getattr(gradient, "shape", None)
    if shape is not None:
        for key in keys:
            if hasattr(shape, key) and getattr(shape, key) is not None:
                try:
                    return float(getattr(shape, key))
                except (TypeError, ValueError):
                    pass

        if hasattr(shape, "to_dict"):
            try:
                shape_dict = dict(shape.to_dict())
                for key in keys:
                    if key in shape_dict and shape_dict[key] is not None:
                        return float(shape_dict[key])
            except Exception:
                pass

    return None

def _resolve_slice_select_bandwidth(
    *,
    bandwidth: float | None,
    time_bw_product: float | None,
    duration: float,
    parameters: Mapping[str, Any],
) -> float:
    """Resolve RF slice-select bandwidth in Hz."""

    if bandwidth is not None:
        resolved = resolve_float(
            bandwidth,
            field_name="slice-select bandwidth",
        )
    else:
        protocol_bandwidth = _first_present(
            parameters,
            (
                "bandwidth",
                "rf_bandwidth",
                "slice_select_bandwidth",
                "slice_bandwidth",
            ),
        )

        if protocol_bandwidth is not None:
            resolved = float(protocol_bandwidth)
        elif time_bw_product is not None:
            resolved = float(time_bw_product) / float(duration)
        else:
            resolved = 1.0 / float(duration)

    if resolved <= 0:
        raise ValueError(f"Slice-select bandwidth must be positive. Passed: {resolved}")

    return resolved


def _resolve_optional_float_parameter(
    *,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    """Return an optional float parameter if present."""

    value = _first_present(parameters, keys)

    if value is None:
        return None

    return float(value)


def _attach_rf_timing_metadata(
    event: Any,
    *,
    system: Opts,
    parameters: Mapping[str, Any],
    duration: float,
    delay: float | None,
    pulse_type: str,
    amplitude_t: float,
    shape_metadata: dict[str, Any],
) -> None:
    """Attach common RF timing metadata to block and shaped RF events."""

    rf_delay = max(
        float(
            delay
            if delay is not None
            else parameters.get("rf_delay", parameters.get("delay", 0.0))
        ),
        float(system.rf_dead_time),
    )

    rf_ringdown_time = float(system.rf_ringdown_time)
    occupied_duration = rf_delay + duration + rf_ringdown_time

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        event.parameters.setdefault("delay", rf_delay)
        event.parameters.setdefault("rf_dead_time", float(system.rf_dead_time))
        event.parameters.setdefault("rf_ringdown_time", rf_ringdown_time)
        event.parameters.setdefault("occupied_duration", occupied_duration)
        event.parameters.setdefault("shape_duration", duration)
        event.parameters.setdefault("duration", duration)
        event.parameters.setdefault("rf_pulse_type", pulse_type)
        event.parameters.setdefault("pulse_type", pulse_type)
        event.parameters.setdefault("amplitude_t", amplitude_t)

    if hasattr(event, "metadata") and isinstance(event.metadata, dict):
        event.metadata.setdefault(
            "timing",
            {
                "delay": rf_delay,
                "shape_duration": duration,
                "duration": duration,
                "rf_dead_time": float(system.rf_dead_time),
                "rf_ringdown_time": rf_ringdown_time,
                "occupied_duration": occupied_duration,
            },
        )
        event.metadata.setdefault("rf_shape", shape_metadata)


def _resolve_flip_angle(
    *,
    explicit_flip_angle: float | None,
    parameters: Mapping[str, Any],
) -> float:
    """Resolve RF flip angle."""

    if explicit_flip_angle is not None:
        resolved = resolve_float(
            explicit_flip_angle,
            field_name="flip_angle",
        )
        if resolved <= 0:
            raise ValueError(f"flip_angle must be positive. Passed: {resolved}")
        return resolved

    protocol_value = _first_present(
        parameters,
        (
            "flip_angle_excitation",
            "flip_angle",
            "FA",
            "fa",
        ),
    )

    if protocol_value is None:
        raise ValueError(
            "flip_angle must be supplied explicitly or through protocol "
            "parameters such as 'flip_angle_excitation' or 'FA'."
        )

    flip_angle_deg = resolve_float(protocol_value, field_name="flip_angle")

    if flip_angle_deg <= 0:
        raise ValueError(f"Protocol flip angle must be positive. Passed: {flip_angle_deg}")

    return math.radians(flip_angle_deg)


def _resolve_duration(
    *,
    explicit_duration: float | None,
    parameters: Mapping[str, Any],
) -> float:
    """Resolve RF pulse duration in seconds."""

    if explicit_duration is not None:
        resolved = resolve_float(
            explicit_duration,
            field_name="RF duration",
        )
        if resolved <= 0:
            raise ValueError(f"duration must be positive. Passed: {resolved}")
        return resolved

    protocol_value = _first_present(
        parameters,
        (
            "rf_duration",
            "excitation_duration",
            "block_rf_duration",
            "sinc_rf_duration",
            "gauss_rf_duration",
            "gaussian_rf_duration",
            "arbitrary_rf_duration",
            "pulse_duration",
            "duration",
        ),
    )

    if protocol_value is None:
        raise ValueError(
            "duration must be supplied explicitly or through protocol "
            "parameters such as 'rf_duration' or 'excitation_duration'."
        )

    resolved_duration = resolve_float(protocol_value, field_name="RF duration")

    if resolved_duration <= 0:
        raise ValueError(f"Protocol RF duration must be positive. Passed: {resolved_duration}")

    return resolved_duration


def _resolve_arbitrary_duration(
    *,
    explicit_duration: float | None,
    dwell: float | None,
    signal_length: int,
    parameters: Mapping[str, Any],
    system: Opts,
) -> float:
    """Resolve arbitrary RF duration in seconds."""

    if explicit_duration is not None:
        resolved = resolve_float(
            explicit_duration,
            field_name="RF duration",
        )
        if resolved <= 0:
            raise ValueError(f"duration must be positive. Passed: {resolved}")
        return resolved

    protocol_duration = _first_present(
        parameters,
        (
            "arbitrary_rf_duration",
            "rf_duration",
            "pulse_duration",
            "duration",
        ),
    )

    if protocol_duration is not None:
        resolved_duration = resolve_float(
            protocol_duration,
            field_name="arbitrary RF duration",
        )
        if resolved_duration <= 0:
            raise ValueError(
                f"Protocol arbitrary RF duration must be positive. Passed: {resolved_duration}"
            )
        return resolved_duration

    if dwell is not None:
        resolved_dwell = resolve_float(
            dwell,
            field_name="arbitrary RF dwell",
        )
        if resolved_dwell <= 0:
            raise ValueError(
                f"dwell must be positive. Passed: {resolved_dwell}"
            )
        return signal_length * resolved_dwell

    protocol_dwell = _first_present(
        parameters,
        (
            "arbitrary_dwell",
            "rf_dwell",
            "dwell",
        ),
    )

    if protocol_dwell is not None:
        resolved_dwell = resolve_float(
            protocol_dwell,
            field_name="arbitrary RF dwell",
        )
        if resolved_dwell <= 0:
            raise ValueError(f"Protocol dwell must be positive. Passed: {resolved_dwell}")
        return signal_length * resolved_dwell

    return signal_length * float(system.rf_raster_time)


def _resolve_float_parameter(
    *,
    explicit_value: float | None,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    """Resolve a float from explicit value first, protocol second, default third."""

    if explicit_value is not None:
        return resolve_float(explicit_value, field_name=keys[0])

    protocol_value = _first_present(parameters, keys)

    if protocol_value is None:
        return default

    return resolve_float(protocol_value, field_name=keys[0])


def _resolve_string_parameter(
    *,
    explicit_value: str | None,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
    default: str | None,
) -> str | None:
    """Resolve a string from explicit value first, protocol second, default third."""

    if explicit_value is not None:
        return resolve_string(explicit_value, field_name=keys[0])

    protocol_value = _first_present(parameters, keys)

    if protocol_value is None:
        return default

    return resolve_string(protocol_value, field_name=keys[0])


def _resolve_arbitrary_signal(
    *,
    explicit_signal: Iterable[float] | None,
    parameters: Mapping[str, Any],
) -> list[float]:
    """Resolve and validate arbitrary RF envelope samples."""

    signal_source = explicit_signal

    if signal_source is None:
        signal_source = _first_present(
            parameters,
            (
                "signal",
                "rf_signal",
                "arbitrary_signal",
                "arbitrary_rf_signal",
            ),
        )

    if signal_source is None:
        raise ValueError(
            "signal must be supplied explicitly or through protocol parameters "
            "such as 'arbitrary_signal'."
        )

    try:
        signal = list(signal_source)
    except TypeError as exc:
        raise TypeError("signal must be an iterable of real-valued samples.") from exc

    if not signal:
        raise ValueError("signal must contain at least one sample.")

    real_signal: list[float] = []

    for index, value in enumerate(signal):
        if isinstance(value, complex):
            if abs(value.imag) > 1e-12:
                raise NotImplementedError(
                    "Complex arbitrary RF samples are not supported yet. "
                    "Use a real-valued envelope for now. Per-sample RF phase "
                    "modulation will be added later."
                )
            value = value.real

        try:
            real_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"signal[{index}]={value!r} cannot be converted to float."
            ) from exc

        if not math.isfinite(real_value):
            raise ValueError(f"signal[{index}] must be finite. Passed: {real_value}")

        real_signal.append(real_value)

    if max(abs(value) for value in real_signal) <= 0:
        raise ValueError("signal must not be all zeros.")

    return real_signal


def _first_present(
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any | None:
    """Return the first available parameter value for the provided keys."""

    for key in keys:
        if key in parameters:
            return parameters[key]

    return None


def _round_to_raster(
    *,
    value: float,
    raster: float,
    name: str,
) -> float:
    """Round a duration to the nearest positive raster point."""

    if raster <= 0:
        raise ValueError(f"Raster time must be positive. Passed: {raster}")

    n_raster = round(value / raster)

    if n_raster < 1:
        raise ValueError(
            f"{name} is shorter than one raster interval. "
            f"{name} = {value}, raster = {raster}"
        )

    return n_raster * raster


def _validate_shaped_rf_controls(
    *,
    time_bw_product: float,
    apodization: float,
    center_pos: float,
) -> None:
    """Validate controls shared by sinc and Gaussian RF constructors."""

    if time_bw_product <= 0:
        raise ValueError(
            "time_bw_product must be positive. "
            f"Passed: {time_bw_product}"
        )

    if not 0.0 <= apodization <= 1.0:
        raise ValueError(
            "apodization must be between 0 and 1. "
            f"Passed: {apodization}"
        )

    if not 0.0 <= center_pos <= 1.0:
        raise ValueError(
            "center_pos must be between 0 and 1. "
            f"Passed: {center_pos}"
        )


def _block_rf_amplitude_t(
    *,
    flip_angle: float,
    duration: float,
    gamma_hz_per_t: float,
) -> float:
    """Return rectangular RF B1 amplitude in tesla."""

    if duration <= 0:
        raise ValueError(f"RF duration must be positive. Passed: {duration}")

    if gamma_hz_per_t <= 0:
        raise ValueError(f"gamma_hz_per_t must be positive. Passed: {gamma_hz_per_t}")

    return flip_angle / (2.0 * math.pi * gamma_hz_per_t * duration)


def _sinc_rf_amplitude_t(
    *,
    flip_angle: float,
    duration: float,
    gamma_hz_per_t: float,
    rf_raster_time: float,
    time_bw_product: float,
    apodization: float,
    center_pos: float,
) -> float:
    """Return sinc RF peak B1 amplitude in tesla."""

    if duration <= 0:
        raise ValueError(f"RF duration must be positive. Passed: {duration}")

    if gamma_hz_per_t <= 0:
        raise ValueError(f"gamma_hz_per_t must be positive. Passed: {gamma_hz_per_t}")

    if rf_raster_time <= 0:
        raise ValueError(f"rf_raster_time must be positive. Passed: {rf_raster_time}")

    _, envelope = _normalized_sinc_envelope(
        duration=duration,
        rf_raster_time=rf_raster_time,
        time_bw_product=time_bw_product,
        apodization=apodization,
        center_pos=center_pos,
    )

    area = sum(envelope) * rf_raster_time

    if abs(area) <= 0:
        raise ValueError(
            "Sinc RF envelope has zero area and cannot be scaled to the "
            "requested flip angle."
        )

    return abs(flip_angle / (2.0 * math.pi * gamma_hz_per_t * area))


def _gauss_rf_amplitude_t(
    *,
    flip_angle: float,
    duration: float,
    gamma_hz_per_t: float,
    rf_raster_time: float,
    time_bw_product: float,
    apodization: float,
    center_pos: float,
) -> float:
    """Return Gaussian RF peak B1 amplitude in tesla."""

    if duration <= 0:
        raise ValueError(f"RF duration must be positive. Passed: {duration}")

    if gamma_hz_per_t <= 0:
        raise ValueError(f"gamma_hz_per_t must be positive. Passed: {gamma_hz_per_t}")

    if rf_raster_time <= 0:
        raise ValueError(f"rf_raster_time must be positive. Passed: {rf_raster_time}")

    _, envelope = _normalized_gauss_envelope(
        duration=duration,
        rf_raster_time=rf_raster_time,
        time_bw_product=time_bw_product,
        apodization=apodization,
        center_pos=center_pos,
    )

    area = sum(envelope) * rf_raster_time

    if abs(area) <= 0:
        raise ValueError(
            "Gaussian RF envelope has zero area and cannot be scaled to the "
            "requested flip angle."
        )

    return abs(flip_angle / (2.0 * math.pi * gamma_hz_per_t * area))


def _arbitrary_rf_peak_amplitude_t(
    *,
    signal: list[float],
    flip_angle: float,
    rf_raster_time: float,
    gamma_hz_per_t: float,
) -> float:
    """Return arbitrary RF peak B1 amplitude in tesla after flip-angle scaling."""

    if flip_angle <= 0:
        raise ValueError(f"flip_angle must be positive. Passed: {flip_angle}")

    if rf_raster_time <= 0:
        raise ValueError(f"rf_raster_time must be positive. Passed: {rf_raster_time}")

    if gamma_hz_per_t <= 0:
        raise ValueError(f"gamma_hz_per_t must be positive. Passed: {gamma_hz_per_t}")

    if not signal:
        raise ValueError("signal must contain at least one sample.")

    area = sum(signal) * rf_raster_time

    if abs(area) <= 0:
        raise ValueError(
            "Arbitrary RF signal has zero signed area and cannot be scaled to "
            "the requested flip angle. Use a signal with nonzero integral."
        )

    scale_t = flip_angle / (2.0 * math.pi * gamma_hz_per_t * area)
    peak_signal = max(abs(value) for value in signal)

    return abs(scale_t) * peak_signal


def _normalized_sinc_envelope(
    *,
    duration: float,
    rf_raster_time: float,
    time_bw_product: float,
    apodization: float,
    center_pos: float,
) -> tuple[list[float], list[float]]:
    """Return rastered normalized sinc-envelope samples."""

    n_samples = int(round(duration / rf_raster_time))

    if n_samples < 2:
        raise ValueError(
            "Sinc pulse duration must contain at least two RF raster samples. "
            f"duration={duration}, rf_raster_time={rf_raster_time}"
        )

    actual_duration = n_samples * rf_raster_time

    t_values: list[float] = []
    envelope: list[float] = []

    for index in range(n_samples):
        t = (index + 0.5) * rf_raster_time
        relative = (t / actual_duration) - center_pos

        sinc_arg = time_bw_product * relative
        sinc_value = _sinc(sinc_arg)

        window = (1.0 - apodization) + apodization * (
            0.54 + 0.46 * math.cos(2.0 * math.pi * relative)
        )

        t_values.append(t)
        envelope.append(sinc_value * window)

    max_abs = max(abs(value) for value in envelope)

    if max_abs <= 0:
        raise ValueError("Sinc RF envelope is all zeros.")

    envelope = [value / max_abs for value in envelope]

    return t_values, envelope


def _normalized_gauss_envelope(
    *,
    duration: float,
    rf_raster_time: float,
    time_bw_product: float,
    apodization: float,
    center_pos: float,
) -> tuple[list[float], list[float]]:
    """Return rastered normalized Gaussian-envelope samples.

    The Gaussian is controlled using PyPulseq-like arguments:

    ``time_bw_product``
        Controls the width of the Gaussian. Larger values create a narrower
        pulse.

    ``apodization``
        Controls the strength of the Gaussian taper. A value of 0 gives a flat
        envelope; a value of 1 gives the strongest taper.

    ``center_pos``
        Places the Gaussian center within the pulse duration.
    """

    n_samples = int(round(duration / rf_raster_time))

    if n_samples < 2:
        raise ValueError(
            "Gaussian pulse duration must contain at least two RF raster samples. "
            f"duration={duration}, rf_raster_time={rf_raster_time}"
        )

    actual_duration = n_samples * rf_raster_time
    taper_strength = max(apodization, 1e-12)

    t_values: list[float] = []
    envelope: list[float] = []

    for index in range(n_samples):
        t = (index + 0.5) * rf_raster_time
        relative = (t / actual_duration) - center_pos

        gauss_arg = time_bw_product * relative
        gauss_value = math.exp(-taper_strength * gauss_arg * gauss_arg)

        t_values.append(t)
        envelope.append(gauss_value)

    max_abs = max(abs(value) for value in envelope)

    if max_abs <= 0:
        raise ValueError("Gaussian RF envelope is all zeros.")

    envelope = [value / max_abs for value in envelope]

    return t_values, envelope


def _resample_real_signal_to_raster(
    *,
    signal: list[float],
    duration: float,
    rf_raster_time: float,
) -> list[float]:
    """Resample a real arbitrary RF envelope to the RF raster."""

    if duration <= 0:
        raise ValueError(f"duration must be positive. Passed: {duration}")

    if rf_raster_time <= 0:
        raise ValueError(f"rf_raster_time must be positive. Passed: {rf_raster_time}")

    if not signal:
        raise ValueError("signal must contain at least one sample.")

    target_count = int(round(duration / rf_raster_time))

    if target_count < 1:
        raise ValueError(
            "RF duration is shorter than one RF raster sample. "
            f"duration={duration}, rf_raster_time={rf_raster_time}"
        )

    if len(signal) == target_count:
        return list(signal)

    if len(signal) == 1:
        return [signal[0] for _ in range(target_count)]

    if target_count == 1:
        return [signal[0]]

    input_last = len(signal) - 1
    output_last = target_count - 1
    resampled: list[float] = []

    for out_index in range(target_count):
        x = out_index * input_last / output_last
        left = int(math.floor(x))
        right = min(left + 1, input_last)
        frac = x - left
        value = (1.0 - frac) * signal[left] + frac * signal[right]
        resampled.append(value)

    return resampled


def _sinc(x: float) -> float:
    """Return normalized sinc(x) = sin(pi*x)/(pi*x)."""

    if abs(x) < 1e-12:
        return 1.0

    pix = math.pi * x
    return math.sin(pix) / pix


def _attach_context_if_supported(
    event: Any,
    *,
    system: Opts,
    parameters: Mapping[str, Any],
) -> None:
    """Attach system/protocol context without requiring every event to support it."""

    if hasattr(event, "bind_system"):
        event.bind_system(system)
    elif hasattr(event, "system"):
        try:
            event.system = system
        except AttributeError:
            pass

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        for key, value in parameters.items():
            event.parameters.setdefault(key, value)

        event.parameters.setdefault("rf_raster_time", system.rf_raster_time)
        event.parameters.setdefault("gamma", system.gamma)
        event.parameters.setdefault("max_rf", system.max_rf)

    if hasattr(event, "metadata") and isinstance(event.metadata, dict):
        event.metadata.setdefault(
            "system",
            system.to_dict() if hasattr(system, "to_dict") else {},
        )