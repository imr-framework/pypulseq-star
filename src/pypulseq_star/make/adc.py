"""ADC constructors.

All symbolic constructor values are resolved before numerical validation while
the original expressions remain attached to the event.

Public constructors
-------------------
make_adc(...)
    PyPulseq-style single ADC constructor.

make_adc_train(...)
    SeqStar/gammaSTAR-native ADC train constructor.

Compatibility rule
------------------
The enriched ADC train is semantic. It should later be lowered to individual
standard Pulseq ADC events/blocks immediately before Pulseq .seq writing.

That future preprocessing step should use:

    adc_event.to_pulseq_windows()

and create standard one-window ADC blocks from those returned dictionaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from pypulseq_star.events.adc import SeqStarADCEvent
from pypulseq_star.opts import Opts
from pypulseq_star.shapes.adc import SeqStarADCTrainShape, SeqStarADCWindow

from ._symbolic import (
    attach_symbolic_specs,
    resolve_float,
    resolve_int,
    resolve_string,
    symbolic_specs,
)


def make_adc(
    num_samples: int | None = None,
    *,
    delay: float | None = None,
    duration: float | None = None,
    dwell: float | None = None,
    freq_offset: float | None = None,
    phase_offset: float | None = None,
    system: Opts | None = None,
    freq_ppm: float | None = None,
    phase_ppm: float | None = None,
    phase_modulation: Iterable[float] | None = None,
    name: str = "adc",
    role: str | None = "readout",
    parameters: Mapping[str, Any] | None = None,
) -> SeqStarADCEvent:
    """Create a PyPulseq-style single ADC event.

    This stays intentionally close to ``pypulseq.make_adc`` usage, but returns
    an enriched SeqStar ADC train with exactly one window.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    symbolic_event_specs = symbolic_specs(
        num_samples=num_samples,
        delay=delay,
        duration=duration,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
    )

    resolved_num_samples = _resolve_int_parameter(
        explicit_value=num_samples,
        parameters=parameters,
        keys=("num_samples", "adc_samples", "number_of_samples"),
        default=None,
        name="num_samples",
    )

    resolved_delay = _resolve_float_parameter(
        explicit_value=delay,
        parameters=parameters,
        keys=("adc_delay", "delay", "first_delay"),
        default=0.0,
    )

    # PyPulseq-style frontend protection. For train-native usage, make_adc_train
    # can keep dead time as train-level metadata.
    adc_dead_time = float(getattr(system, "adc_dead_time", 0.0))
    resolved_delay = max(resolved_delay, adc_dead_time)

    event = make_adc_train(
        num_samples=resolved_num_samples,
        dwell=dwell,
        duration=duration,
        num_echoes=1,
        first_delay=resolved_delay,
        echo_spacing=None,
        mode="single",
        name=name,
        role=role,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        phase_modulation=phase_modulation,
        dead_time=adc_dead_time,
        dead_time_policy="pypulseq_single_adc",
        pulseq_export_mode="single_if_possible",
        system=system,
        parameters=parameters,
    )
    attach_symbolic_specs(event, symbolic_event_specs)
    return event



def make_adc_train(
    *,
    num_samples: int | None = None,
    dwell: float | None = None,
    duration: float | None = None,
    windows: Iterable[Mapping[str, Any]] | None = None,
    num_echoes: int | None = None,
    first_delay: float | None = None,
    echo_spacing: float | None = None,
    mode: str | None = None,
    polarity: str | int | None = None,
    trajectory: str | None = None,
    name: str = "adc",
    role: str | None = "readout",
    freq_offset: float | None = None,
    phase_offset: float | None = None,
    freq_ppm: float | None = None,
    phase_ppm: float | None = None,
    phase_modulation: Iterable[float] | None = None,
    dead_time: float | None = None,
    window_guard_time: float | None = None,
    dead_time_policy: str = "train_level_once",
    pulseq_export_mode: str = "windows",
    adc_raster_time: float | None = None,
    system: Opts | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> SeqStarADCEvent:
    """Create an enriched ADC train.

    ``windows`` is the most explicit form and wins over regular-train
    generation. If ``windows`` is omitted, a regular train is generated from
    ``num_samples``, ``dwell``/``duration``, ``num_echoes``, ``first_delay``,
    and ``echo_spacing``.
    """

    if system is None:
        system = Opts.default

    parameters = dict(parameters or {})

    constructor_specs = {
        "num_samples": num_samples,
        "dwell": dwell,
        "duration": duration,
        "windows": tuple(dict(window) for window in windows) if windows is not None else None,
        "num_echoes": num_echoes,
        "first_delay": first_delay,
        "echo_spacing": echo_spacing,
        "mode": mode,
        "polarity": polarity,
        "trajectory": trajectory,
        "freq_offset": freq_offset,
        "phase_offset": phase_offset,
        "freq_ppm": freq_ppm,
        "phase_ppm": phase_ppm,
        "dead_time": dead_time,
        "window_guard_time": window_guard_time,
        "adc_raster_time": adc_raster_time,
    }

    symbolic_train_specs = symbolic_specs(
        num_samples=num_samples,
        dwell=dwell,
        duration=duration,
        num_echoes=num_echoes,
        first_delay=first_delay,
        echo_spacing=echo_spacing,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        dead_time=dead_time,
        window_guard_time=window_guard_time,
        adc_raster_time=adc_raster_time,
    )

    resolved_mode = _resolve_string_parameter(
        explicit_value=mode,
        parameters=parameters,
        keys=("adc_mode", "mode", "readout_mode"),
        default="single",
    )

    resolved_trajectory = _resolve_string_parameter(
        explicit_value=trajectory,
        parameters=parameters,
        keys=("trajectory", "readout_trajectory"),
        default=None,
    )

    resolved_freq_offset = _resolve_float_parameter(
        explicit_value=freq_offset,
        parameters=parameters,
        keys=("adc_frequency_offset", "freq_offset", "frequency_offset"),
        default=0.0,
    )

    resolved_phase_offset = _resolve_float_parameter(
        explicit_value=phase_offset,
        parameters=parameters,
        keys=("adc_phase_offset", "phase_offset", "phase"),
        default=0.0,
    )

    resolved_freq_ppm = _resolve_float_parameter(
        explicit_value=freq_ppm,
        parameters=parameters,
        keys=("adc_freq_ppm", "freq_ppm", "frequency_shift_ppm"),
        default=0.0,
    )

    resolved_phase_ppm = _resolve_float_parameter(
        explicit_value=phase_ppm,
        parameters=parameters,
        keys=("adc_phase_ppm", "phase_ppm"),
        default=0.0,
    )

    resolved_dead_time = _resolve_float_parameter(
        explicit_value=dead_time,
        parameters=parameters,
        keys=("adc_dead_time", "frontend_dead_time"),
        default=float(getattr(system, "adc_dead_time", 0.0)),
    )

    resolved_window_guard_time = _resolve_float_parameter(
        explicit_value=window_guard_time,
        parameters=parameters,
        keys=("window_guard_time", "adc_window_guard_time"),
        default=0.0,
    )

    resolved_adc_raster_time = _resolve_adc_raster_time(
        explicit_adc_raster_time=adc_raster_time,
        parameters=parameters,
        system=system,
    )

    if windows is not None:
        adc_windows = _build_windows_from_explicit_list(
            windows=windows,
            train_num_samples=num_samples,
            train_dwell=dwell,
            train_duration=duration,
            train_freq_offset=resolved_freq_offset,
            train_phase_offset=resolved_phase_offset,
            train_freq_ppm=resolved_freq_ppm,
            train_phase_ppm=resolved_phase_ppm,
            train_polarity=polarity,
            train_trajectory=resolved_trajectory,
            adc_raster_time=resolved_adc_raster_time,
        )
    else:
        adc_windows = _build_regular_windows(
            num_samples=num_samples,
            dwell=dwell,
            duration=duration,
            num_echoes=num_echoes,
            first_delay=first_delay,
            echo_spacing=echo_spacing,
            polarity=polarity,
            trajectory=resolved_trajectory,
            parameters=parameters,
            adc_raster_time=resolved_adc_raster_time,
            freq_offset=resolved_freq_offset,
            phase_offset=resolved_phase_offset,
            freq_ppm=resolved_freq_ppm,
            phase_ppm=resolved_phase_ppm,
        )

    if not adc_windows:
        raise ValueError("make_adc_train must create at least one ADC window.")

    adc_windows.sort(key=lambda window: window.delay)

    # Use a two-level timing model:
    #   event.delay      = first ADC-window delay in the parent block
    #   window.delay     = delay relative to the ADC event start
    #
    # This keeps PyPulseq-style single ADC timing intuitive:
    #
    #   adc = make_adc(..., delay=gx.rise_time)
    #   seq.add_block(gx, adc)
    #
    # and also keeps multi-window ADC trains compact and internally relative.
    # Without this normalization, event.delay stays 0 while the first window
    # carries the frontend/readout delay; timing checks then report false
    # ADC_DEAD_TIME errors even though the plotted/window timing is correct.
    event_delay = float(adc_windows[0].delay)
    adc_windows = _normalize_windows_to_event_delay(
        windows=adc_windows,
        event_delay=event_delay,
    )

    for index, window in enumerate(adc_windows):
        window.index = index

    event_duration = max(window.tend for window in adc_windows)
    first_window = adc_windows[0]

    resolved_phase_modulation = _resolve_phase_modulation(
        explicit_phase_modulation=phase_modulation,
        parameters=parameters,
        num_samples=first_window.num_samples,
    )

    shape = SeqStarADCTrainShape(
        name=f"{name}_shape",
        duration=event_duration,
        windows=adc_windows,
        mode=resolved_mode,
        trajectory=resolved_trajectory,
        adc_raster_time=resolved_adc_raster_time,
        frontend_dead_time=resolved_dead_time,
        window_guard_time=resolved_window_guard_time,
        dead_time_policy=dead_time_policy,
        pulseq_export_mode=pulseq_export_mode,
    )

    event = SeqStarADCEvent(
        name=name,
        role=role,
        num_samples=first_window.num_samples,
        dwell=first_window.dwell,
        duration=event_duration,
        delay=event_delay,
        freq_offset=resolved_freq_offset,
        phase_offset=resolved_phase_offset,
        freq_ppm=resolved_freq_ppm,
        phase_ppm=resolved_phase_ppm,
        phase_modulation=resolved_phase_modulation,
        use=role,
        shape=shape,
        dead_time=resolved_dead_time,
        window_guard_time=resolved_window_guard_time,
        dead_time_policy=dead_time_policy,
        pulseq_export_mode=pulseq_export_mode,
        mode=resolved_mode,
        trajectory=resolved_trajectory,
        parameters=dict(parameters),
        metadata={
            "adc_train": {
                "mode": resolved_mode,
                "trajectory": resolved_trajectory,
                "num_windows": len(adc_windows),
                "total_num_samples": sum(window.num_samples for window in adc_windows),
                "dead_time_policy": dead_time_policy,
                "pulseq_export_mode": pulseq_export_mode,
                "event_delay": event_delay,
                "event_duration": event_duration,
            }
        },
    )

    _set_adc_event_timing(
        event,
        delay=event_delay,
        duration=event_duration,
        first_window=first_window,
        windows=adc_windows,
    )

    _attach_context_if_supported(
        event,
        system=system,
        parameters=parameters,
    )

    # Re-apply timing after context attachment because protocol parameters are
    # merged into event.parameters with setdefault() above. This makes the ADC
    # event, its timing object, parameters, and windows agree.
    _set_adc_event_timing(
        event,
        delay=event_delay,
        duration=event_duration,
        first_window=first_window,
        windows=adc_windows,
    )

    constructor_record = {
        "family": (
            "explicit_windows"
            if constructor_specs.get("windows") is not None
            else "regular_train"
        ),
        "specs": constructor_specs,
    }
    if isinstance(getattr(event, "metadata", None), dict):
        event.metadata["symbolic_constructor"] = constructor_record
    if isinstance(getattr(event, "parameters", None), dict):
        event.parameters["_symbolic_constructor"] = constructor_record

    attach_symbolic_specs(event, symbolic_train_specs)
    event.validate()

    return event



def _normalize_windows_to_event_delay(
    *,
    windows: list[SeqStarADCWindow],
    event_delay: float,
) -> list[SeqStarADCWindow]:
    """Shift ADC-window delays into the ADC event's local frame.

    Public/event-level timing uses ``event.delay`` to place the ADC event in
    its parent block. Individual windows are then relative to that event. This
    preserves the absolute timing of all windows while allowing generic timing
    checks to see the ADC pre-delay.
    """

    event_delay = float(event_delay)

    for window in windows:
        new_delay = max(float(window.delay) - event_delay, 0.0)

        try:
            window.delay = new_delay
        except Exception:
            object.__setattr__(window, "delay", new_delay)

        metadata = getattr(window, "metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("absolute_delay", event_delay + new_delay)
            metadata.setdefault("event_delay", event_delay)

    return windows


def _set_adc_event_timing(
    event: SeqStarADCEvent,
    *,
    delay: float,
    duration: float,
    first_window: SeqStarADCWindow,
    windows: list[SeqStarADCWindow],
) -> None:
    """Synchronize ADC event delay/duration across all public stores.

    SeqStar currently has several timing-facing representations: dataclass
    fields/properties, ``event.timing``, ``event.parameters``, and the ADC train
    shape/windows. This helper keeps them consistent so generic timing checks,
    plotters, writers, and gammaSTAR export do not disagree.
    """

    delay = float(delay)
    duration = float(duration)

    try:
        setattr(event, "delay", delay)
    except Exception:
        pass

    try:
        setattr(event, "duration", duration)
    except Exception:
        pass

    timing = getattr(event, "timing", None)
    if timing is not None:
        try:
            setattr(timing, "tstart", delay)
        except Exception:
            pass
        try:
            setattr(timing, "duration", duration)
        except Exception:
            pass

    parameters = getattr(event, "parameters", None)
    if isinstance(parameters, dict):
        parameters["delay"] = delay
        parameters["tstart"] = delay
        parameters["duration"] = duration
        parameters["adc_duration"] = duration
        parameters["num_samples"] = int(first_window.num_samples)
        parameters["number_of_samples"] = int(first_window.num_samples)
        parameters["dwell"] = float(first_window.dwell)
        parameters["sample_time"] = float(first_window.dwell)
        parameters["first_window_delay"] = float(first_window.delay)
        parameters["num_windows"] = len(windows)

    shape = getattr(event, "shape", None)
    if shape is not None:
        try:
            setattr(shape, "duration", duration)
        except Exception:
            pass
        try:
            setattr(shape, "windows", windows)
        except Exception:
            pass

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        adc_train = metadata.setdefault("adc_train", {})
        if isinstance(adc_train, dict):
            adc_train["event_delay"] = delay
            adc_train["event_duration"] = duration
            adc_train["first_window_delay"] = float(first_window.delay)
            adc_train["num_windows"] = len(windows)

def _build_regular_windows(
    *,
    num_samples: int | None,
    dwell: float | None,
    duration: float | None,
    num_echoes: int | None,
    first_delay: float | None,
    echo_spacing: float | None,
    polarity: str | int | None,
    trajectory: str | None,
    parameters: Mapping[str, Any],
    adc_raster_time: float | None,
    freq_offset: float,
    phase_offset: float,
    freq_ppm: float,
    phase_ppm: float,
) -> list[SeqStarADCWindow]:
    """Build one regular single- or multi-echo ADC train."""

    resolved_num_samples = _resolve_int_parameter(
        explicit_value=num_samples,
        parameters=parameters,
        keys=("num_samples", "adc_samples", "number_of_samples"),
        default=None,
        name="num_samples",
    )

    resolved_dwell, resolved_duration = _resolve_dwell_duration(
        dwell=dwell,
        duration=duration,
        num_samples=resolved_num_samples,
        parameters=parameters,
        adc_raster_time=adc_raster_time,
    )

    resolved_num_echoes = _resolve_int_parameter(
        explicit_value=num_echoes,
        parameters=parameters,
        keys=("num_echoes", "echo_train_length", "etl", "ETL"),
        default=1,
        name="num_echoes",
    )

    resolved_first_delay = _resolve_float_parameter(
        explicit_value=first_delay,
        parameters=parameters,
        keys=("first_delay", "adc_delay", "delay"),
        default=0.0,
    )

    if resolved_num_echoes == 1:
        resolved_echo_spacing = resolved_duration
    else:
        resolved_echo_spacing = _resolve_float_parameter(
            explicit_value=echo_spacing,
            parameters=parameters,
            keys=("echo_spacing", "adc_echo_spacing"),
            default=resolved_duration,
        )

    if resolved_echo_spacing < resolved_duration - 1e-12:
        raise ValueError(
            "echo_spacing must be at least ADC duration for non-overlapping windows. "
            f"echo_spacing={resolved_echo_spacing}, duration={resolved_duration}"
        )

    windows: list[SeqStarADCWindow] = []

    for echo_index in range(resolved_num_echoes):
        polarity_value = _resolve_window_polarity(
            polarity=polarity,
            index=echo_index,
        )

        windows.append(
            SeqStarADCWindow(
                index=echo_index,
                delay=resolved_first_delay + echo_index * resolved_echo_spacing,
                num_samples=resolved_num_samples,
                dwell=resolved_dwell,
                duration=resolved_duration,
                label=f"adc_{echo_index:03d}",
                role="imaging",
                freq_offset=freq_offset,
                phase_offset=phase_offset,
                freq_ppm=freq_ppm,
                phase_ppm=phase_ppm,
                polarity=polarity_value,
                echo_index=echo_index,
                line_index=echo_index,
                trajectory=trajectory,
            )
        )

    return windows


def _build_windows_from_explicit_list(
    *,
    windows: Iterable[Mapping[str, Any]],
    train_num_samples: int | None,
    train_dwell: float | None,
    train_duration: float | None,
    train_freq_offset: float,
    train_phase_offset: float,
    train_freq_ppm: float,
    train_phase_ppm: float,
    train_polarity: str | int | None,
    train_trajectory: str | None,
    adc_raster_time: float | None,
) -> list[SeqStarADCWindow]:
    """Build ADC windows from explicit dictionaries."""

    built_windows: list[SeqStarADCWindow] = []

    for index, window in enumerate(windows):
        window_dict = dict(window)

        resolved_num_samples = _resolve_int_parameter(
            explicit_value=window_dict.get("num_samples", train_num_samples),
            parameters=window_dict,
            keys=("number_of_samples", "adc_samples"),
            default=None,
            name=f"windows[{index}].num_samples",
        )

        resolved_dwell, resolved_duration = _resolve_dwell_duration(
            dwell=window_dict.get("dwell", train_dwell),
            duration=window_dict.get("duration", train_duration),
            num_samples=resolved_num_samples,
            parameters=window_dict,
            adc_raster_time=adc_raster_time,
        )

        delay = float(window_dict.get("delay", window_dict.get("tstart", 0.0)))

        polarity = _resolve_window_polarity(
            polarity=window_dict.get("polarity", train_polarity),
            index=index,
        )

        metadata = {
            key: value
            for key, value in window_dict.items()
            if key
            not in {
                "num_samples",
                "number_of_samples",
                "adc_samples",
                "dwell",
                "duration",
                "delay",
                "tstart",
                "label",
                "role",
                "freq_offset",
                "phase_offset",
                "freq_ppm",
                "phase_ppm",
                "polarity",
                "echo_index",
                "shot_index",
                "segment_index",
                "line_index",
                "line_index_in_shot",
                "spin_echo_index",
                "gradient_echo_index",
                "spoke_index",
                "navigator_index",
                "center_sample",
                "echo_time",
                "trajectory",
            }
        }

        built_windows.append(
            SeqStarADCWindow(
                index=index,
                delay=delay,
                num_samples=resolved_num_samples,
                dwell=resolved_dwell,
                duration=resolved_duration,
                label=window_dict.get("label"),
                role=str(window_dict.get("role", "imaging")),
                freq_offset=float(window_dict.get("freq_offset", train_freq_offset)),
                phase_offset=float(window_dict.get("phase_offset", train_phase_offset)),
                freq_ppm=float(window_dict.get("freq_ppm", train_freq_ppm)),
                phase_ppm=float(window_dict.get("phase_ppm", train_phase_ppm)),
                polarity=polarity,
                echo_index=_optional_int(window_dict.get("echo_index")),
                shot_index=_optional_int(window_dict.get("shot_index")),
                segment_index=_optional_int(window_dict.get("segment_index")),
                line_index=_optional_int(window_dict.get("line_index")),
                line_index_in_shot=_optional_int(window_dict.get("line_index_in_shot")),
                spin_echo_index=_optional_int(window_dict.get("spin_echo_index")),
                gradient_echo_index=_optional_int(window_dict.get("gradient_echo_index")),
                spoke_index=_optional_int(window_dict.get("spoke_index")),
                navigator_index=_optional_int(window_dict.get("navigator_index")),
                center_sample=_optional_int(window_dict.get("center_sample")),
                echo_time=_optional_float(window_dict.get("echo_time")),
                trajectory=window_dict.get("trajectory", train_trajectory),
                metadata=metadata,
            )
        )

    return built_windows


def _resolve_dwell_duration(
    *,
    dwell: float | None,
    duration: float | None,
    num_samples: int,
    parameters: Mapping[str, Any],
    adc_raster_time: float | None,
) -> tuple[float, float]:
    """Resolve dwell and duration with PyPulseq-like semantics.

    Exactly one of dwell or duration should be provided. The missing quantity is
    inferred from ``num_samples``.
    """

    protocol_dwell = _first_present(parameters, ("dwell", "adc_dwell", "sample_time"))
    protocol_duration = _first_present(parameters, ("duration", "adc_duration", "readout_duration"))

    resolved_dwell = resolve_float(dwell if dwell is not None else protocol_dwell, field_name="dwell") if (dwell is not None or protocol_dwell is not None) else None
    resolved_duration = resolve_float(duration if duration is not None else protocol_duration, field_name="ADC duration") if (duration is not None or protocol_duration is not None) else None

    if resolved_dwell is None and resolved_duration is None:
        raise ValueError("Exactly one of dwell or duration must be supplied for ADC.")

    if resolved_dwell is not None and resolved_duration is not None:
        expected_duration = resolved_num_samples_duration(
            num_samples=num_samples,
            dwell=resolved_dwell,
        )

        if not math.isclose(resolved_duration, expected_duration, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "When both dwell and duration are supplied, duration must equal "
                "num_samples*dwell. "
                f"duration={resolved_duration}, expected={expected_duration}"
            )

    elif resolved_dwell is not None:
        resolved_duration = resolved_num_samples_duration(
            num_samples=num_samples,
            dwell=resolved_dwell,
        )

    else:
        resolved_dwell = resolved_duration / num_samples

    if resolved_dwell <= 0:
        raise ValueError(f"ADC dwell must be positive. Passed: {resolved_dwell}")

    if resolved_duration <= 0:
        raise ValueError(f"ADC duration must be positive. Passed: {resolved_duration}")

    if adc_raster_time is not None:
        _assert_raster_aligned(value=resolved_dwell, raster=adc_raster_time, name="ADC dwell")
        _assert_raster_aligned(value=resolved_duration, raster=adc_raster_time, name="ADC duration")

    return resolved_dwell, resolved_duration


def resolved_num_samples_duration(*, num_samples: int, dwell: float) -> float:
    """Return ADC duration from samples and dwell."""

    return int(num_samples) * float(dwell)


def _resolve_window_polarity(*, polarity: str | int | None, index: int) -> int:
    """Resolve readout polarity metadata."""

    if polarity is None:
        return 1

    if isinstance(polarity, str):
        polarity_lower = polarity.lower()

        if polarity_lower in {"constant", "positive", "+", "+1"}:
            return 1

        if polarity_lower in {"negative", "-", "-1"}:
            return -1

        if polarity_lower == "alternating":
            return 1 if index % 2 == 0 else -1

        raise ValueError(
            "Unsupported ADC polarity string. "
            f"polarity={polarity!r}. Use 'constant', 'alternating', '+1', or '-1'."
        )

    polarity_int = int(polarity)

    if polarity_int not in (-1, 1):
        raise ValueError(f"ADC polarity must be +1 or -1. Passed: {polarity}")

    return polarity_int


def _resolve_phase_modulation(
    *,
    explicit_phase_modulation: Iterable[float] | None,
    parameters: Mapping[str, Any],
    num_samples: int,
) -> list[float] | None:
    """Resolve optional ADC phase modulation."""

    source = explicit_phase_modulation

    if source is None:
        source = _first_present(parameters, ("phase_modulation", "adc_phase_modulation"))

    if source is None:
        return None

    values = [float(value) for value in source]

    if len(values) != num_samples:
        raise ValueError(
            "phase_modulation length must equal num_samples. "
            f"len={len(values)}, num_samples={num_samples}"
        )

    return values


def _resolve_adc_raster_time(
    *,
    explicit_adc_raster_time: float | None,
    parameters: Mapping[str, Any],
    system: Opts,
) -> float | None:
    """Resolve ADC raster time if available.

    Some current ``Opts`` implementations may not define adc_raster_time yet.
    In that case, validation skips ADC-raster alignment.
    """

    if explicit_adc_raster_time is not None:
        value = float(explicit_adc_raster_time)
    elif "adc_raster_time" in parameters:
        value = float(parameters["adc_raster_time"])
    else:
        value = getattr(system, "adc_raster_time", None)

    if value is None:
        return None

    value = float(value)

    if value <= 0:
        raise ValueError(f"adc_raster_time must be positive. Passed: {value}")

    return value


def _resolve_int_parameter(
    *,
    explicit_value: int | None,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
    default: int | None,
    name: str,
) -> int:
    """Resolve an integer from explicit value, protocol keys, or default."""

    value = explicit_value

    if value is None:
        protocol_value = _first_present(parameters, keys)
        value = protocol_value if protocol_value is not None else default

    if value is None:
        raise ValueError(f"{name} must be supplied.")

    value_int = resolve_int(value, field_name=name)

    if value_int <= 0:
        raise ValueError(f"{name} must be positive. Passed: {value_int}")

    return value_int


def _resolve_float_parameter(
    *,
    explicit_value: float | None,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    """Resolve a float from explicit value, protocol keys, or default."""

    if explicit_value is not None:
        return resolve_float(explicit_value, field_name=keys[0])

    protocol_value = _first_present(parameters, keys)

    if protocol_value is None:
        return float(default)

    return resolve_float(protocol_value, field_name=keys[0])


def _resolve_string_parameter(
    *,
    explicit_value: str | None,
    parameters: Mapping[str, Any],
    keys: tuple[str, ...],
    default: str | None,
) -> str | None:
    """Resolve a string from explicit value, protocol keys, or default."""

    if explicit_value is not None:
        return resolve_string(explicit_value, field_name=keys[0])

    protocol_value = _first_present(parameters, keys)

    if protocol_value is None:
        return default

    return resolve_string(protocol_value, field_name=keys[0])


def _first_present(parameters: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return the first available value from a parameter dictionary."""

    for key in keys:
        if key in parameters:
            return parameters[key]

    return None


def _optional_int(value: Any) -> int | None:
    """Convert optional value to int."""

    if value is None:
        return None

    return int(value)


def _optional_float(value: Any) -> float | None:
    """Convert optional value to float."""

    if value is None:
        return None

    return float(value)


def _assert_raster_aligned(*, value: float, raster: float, name: str) -> None:
    """Validate approximate raster alignment."""

    if raster <= 0:
        raise ValueError(f"Raster must be positive. Passed: {raster}")

    n = value / raster

    if not math.isclose(n, round(n), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{name} is not raster-aligned. value={value}, raster={raster}"
        )


def _attach_context_if_supported(
    event: SeqStarADCEvent,
    *,
    system: Opts,
    parameters: Mapping[str, Any],
) -> None:
    """Attach system/protocol context to the ADC event."""

    if hasattr(event, "bind_system"):
        event.bind_system(system)

    if hasattr(event, "parameters") and isinstance(event.parameters, dict):
        for key, value in parameters.items():
            event.parameters.setdefault(key, value)

        event.parameters.setdefault("adc_dead_time", float(getattr(system, "adc_dead_time", 0.0)))
        event.parameters.setdefault("adc_raster_time", getattr(system, "adc_raster_time", None))

    if hasattr(event, "metadata") and isinstance(event.metadata, dict):
        event.metadata.setdefault(
            "system",
            system.to_dict() if hasattr(system, "to_dict") else {},
        )