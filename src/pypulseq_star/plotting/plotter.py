"""gammaSTAR-like plotting for pypulseq_star sequences.

This module provides a PyPulseq-like user API:

    seq.plot()
    seq.plot(time_range=(0, 2.0))
    seq.plot(save="sequence.png")

but renders a gammaSTAR-like stacked timeline with RF, ADC, GX, GY, and GZ
channels from the enriched SeqStar object model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from pypulseq_star.calc_duration import calc_duration

CHANNEL_COLORS = {
    "rf": "#d6aa00",
    "adc": "#202020",
    "x": "#ff3030",
    "y": "#00b020",
    "z": "#3f5cff",
}

class SeqStarPlotter:
    """Plotter for enriched SeqStar sequences."""

    def __init__(self, seq: Any) -> None:
        self.seq = seq
        self.system = seq.system

    def plot(
        self,
        time_range: tuple[float, float] | None = None,
        *,
        title: str | None = None,
        show: bool = True,
        save: str | Path | None = None,
        one_tr: bool = False,
        rf_scale: str = "normalized",
        gradient_scale: str = "auto",
        figsize: tuple[float, float] = (15.0, 8.5),
        dpi: int = 140,
        debug: bool = False,
        show_blocks: bool = True,
    ):
        """Create a gammaSTAR-like sequence plot."""

        render = self.render_sequence(
            time_range=time_range,
            one_tr=one_tr,
            rf_scale=rf_scale,
            gradient_scale=gradient_scale,
        )

        if debug:
            print(
                "Plot render counts:",
                {
                    "blocks": len(list(_iter_blocks(self.seq))),
                    "rf": len(render["rf"]),
                    "adc": len(render["adc"]),
                    "gx": len(render["gradients"]["x"]),
                    "gy": len(render["gradients"]["y"]),
                    "gz": len(render["gradients"]["z"]),
                    "time_range": render["time_range"],
                },
            )
            debug_render_summary(render)
        fig = self._plot_render(
            render,
            title=title or getattr(self.seq, "name", "SeqStar sequence"),
            figsize=figsize,
            show_blocks=show_blocks,
        )

        if save is not None:
            output_path = Path(save)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def render_sequence(
        self,
        time_range: tuple[float, float] | None = None,
        *,
        one_tr: bool = False,
        rf_scale: str = "normalized",
        gradient_scale: str = "auto",
    ) -> dict[str, Any]:
        """Convert a SeqStarSequence into a plot-ready render dictionary.

        Plotting policy
        ---------------
        Top-level sequence blocks are interpreted as sequential blocks unless
        they explicitly carry an absolute timing field.

        This mirrors PyPulseq-style construction:

            seq.add_block(...)
            seq.add_block(...)
            seq.add_block(...)

        where each block starts after the previous block ends. Event-level
        delays are still honored within each block.
        """

        blocks = list(_iter_blocks(self.seq))

        if one_tr:
            time_range = self._first_tr_range(blocks)

        # Resolve relationships before plotting when the sequence exposes the
        # relationship API. Plotting remains read-only; failures fall back to
        # the materialized timeline rather than preventing visualization.
        _try_resolve_relationships(self.seq)

        resolved_starts = _resolved_block_starts(self.seq)

        rendered: dict[str, Any] = {
            "selected_path": getattr(self.seq, "path", getattr(self.seq, "name", "root")),
            "selected_tstart": 0.0,
            "selected_duration": None,
            "rf": [],
            "adc": [],
            "gradients": {
                "x": [],
                "y": [],
                "z": [],
            },
            "metadata": {
                "rf_scale": rf_scale,
                "gradient_scale": gradient_scale,
                "timing_source": "resolved_relationships" if resolved_starts else "timeline",
            },
        }
        rendered["blocks"] = []
        current_time = 0.0

        for block_index, block in enumerate(blocks):
            # Important:
            # Do not treat block.delay == 0 as an absolute start time.
            # In PyPulseq-style add_block() usage, blocks are sequential.
            explicit_block_start = _get_block_tstart(block)
            relationship_start = resolved_starts.get(block_index)
            if relationship_start is not None:
                block_start = relationship_start
            elif (
                explicit_block_start is not None
                and (
                    block_index == 0
                    or explicit_block_start > 0
                    or _block_has_explicit_absolute_start(block)
                )
            ):
                block_start = explicit_block_start
            else:
                block_start = current_time

            block_duration = float(calc_duration(block))
            rendered["blocks"].append(
                {
                    "index": block_index,
                    "node": _block_node_name(block, block_index),
                    "repeat_parent": _block_repeat_parent(block),
                    "role": _block_role(block),
                    "tstart": block_start,
                    "duration": block_duration,
                    "tend": block_start + block_duration,
                }
            )

            # Executable blocks are rendered once at their timeline position.
            # Sequence-level parameters such as ``average`` and
            # ``repetition_time`` may be mirrored into every block's parameter
            # dictionary, but they do not make each block independently repeat.
            #
            # Logical-node repetition is expanded after the complete base motif
            # has been rendered. This preserves:
            #
            #     RF -> delay -> ADC -> recovery
            #
            # instead of incorrectly rendering:
            #
            #     RF x N -> delay x N -> ADC x N -> recovery x N.
            events = list(_iter_events(block))

            for event_index, event in enumerate(events):
                event_start = block_start + _get_event_tstart(event)

                self._add_event_to_render(
                    render=rendered,
                    event=event,
                    event_start=event_start,
                    block_index=block_index,
                    event_index=event_index,
                    rf_scale=rf_scale,
                    gradient_scale=gradient_scale,
                )

            current_time = max(current_time, block_start + block_duration)

        _expand_rendered_node_repetitions(
            self.seq,
            rendered,
        )

        # Normalize RF amplitudes once across the complete rendered sequence.
        # Per-event normalization makes every pulse peak at 1.0 and therefore
        # hides relative flip-angle scaling (for example, 90-degree excitation
        # versus 180-degree refocusing pulses in TSE).
        if rf_scale == "normalized":
            _normalize_rendered_rf(rendered)

        inferred_duration = _render_duration(rendered)

        if time_range is None:
            if inferred_duration <= 0:
                time_range = (0.0, 1e-3)
            else:
                right = inferred_duration
                margin = max(0.02 * right, 1e-6)
                time_range = (0.0, right + margin)

        rendered["time_range"] = time_range
        rendered["selected_duration"] = time_range[1] - time_range[0]

        _clip_render_to_time_range(rendered, time_range)

        return rendered
    
    def _add_event_to_render(
        self,
        *,
        render: dict[str, Any],
        event: Any,
        event_start: float,
        block_index: int,
        event_index: int,
        rf_scale: str,
        gradient_scale: str,
    ) -> None:
        """Route one event into RF, ADC, or gradient plot channels."""

        # Route explicit families first. Gradient before RF prevents paired
        # slice-select gradients from being misclassified as RF-like events.
        if _is_adc_event(event):
            adc_events = self._render_adc_event(
                event=event,
                event_start=event_start,
                block_index=block_index,
                event_index=event_index,
            )

            if adc_events:
                render["adc"].extend(adc_events)

            return

        if _is_gradient_event(event):
            grad_event = self._render_gradient_event(
                event=event,
                event_start=event_start,
                block_index=block_index,
                event_index=event_index,
                gradient_scale=gradient_scale,
            )
            if grad_event is not None:
                axis = grad_event.pop("axis")
                render["gradients"][axis].append(grad_event)
            return

        if _is_rf_event(event):
            rf_event = self._render_rf_event(
                event=event,
                event_start=event_start,
                block_index=block_index,
                event_index=event_index,
                rf_scale=rf_scale,
            )
            if rf_event is not None:
                render["rf"].append(rf_event)
            return

    def _render_rf_event(
        self,
        *,
        event: Any,
        event_start: float,
        block_index: int,
        event_index: int,
        rf_scale: str,
    ) -> dict[str, Any] | None:
        """Render an RF event."""

        flip_angle = getattr(event, "flip_angle", None)
        shape = getattr(event, "shape", None)

        t_values: list[float] = []
        v_values: list[float] = []

        if shape is not None and hasattr(shape, "to_gammastar_samples") and flip_angle is not None:
            try:
                samples = shape.to_gammastar_samples(
                    flip_angle=float(flip_angle),
                    gamma_hz_per_t=float(self.system.gamma),
                    max_rf=None,
                )
            except TypeError:
                samples = shape.to_gammastar_samples(
                    flip_angle=float(flip_angle),
                    gamma_hz_per_t=float(self.system.gamma),
                )

            t_values = [event_start + float(t) for t in samples.get("t", [])]

            v_container = samples.get("v", [])
            if (
                isinstance(v_container, list)
                and v_container
                and isinstance(v_container[0], Mapping)
            ):
                am_values = v_container[0].get("am", [])
                v_values = [float(v) for v in am_values]

        else:
            duration = _get_event_duration(event)
            if duration is None or duration <= 0:
                return None

            t_values = [event_start, event_start + duration]

            if flip_angle is not None:
                amplitude_t = float(flip_angle) / (
                    2.0 * math.pi * float(self.system.gamma) * duration
                )
            else:
                amplitude_t = 1.0

            v_values = [amplitude_t, amplitude_t]

        if not t_values or not v_values:
            return None

        count = min(len(t_values), len(v_values))
        t_values = t_values[:count]
        v_values = v_values[:count]

        if rf_scale == "normalized":
            # Preserve physical relative amplitudes here. The complete RF
            # channel is normalized after all events and logical repetitions
            # have been rendered.
            y_label = "RF"
        elif rf_scale == "tesla":
            y_label = "RF (T)"
        else:
            raise ValueError("rf_scale must be either 'normalized' or 'tesla'")

        return {
            "path": _event_path(event, block_index, event_index),
            "block_index": block_index,
            "block_node": _event_block_node(self.seq, block_index),
            "t": t_values,
            "v": v_values,
            "label": y_label,
        }

    def _render_adc_event(
        self,
        *,
        event: Any,
        event_start: float,
        block_index: int,
        event_index: int,
    ) -> list[dict[str, Any]]:
        """Render an ADC event as one or more on/off gates.

        A SeqStar ADC event may be a single-window PyPulseq-like ADC or an
        enriched ADC train with many windows. The plotter should show every ADC
        window, not just the representative event-level num_samples*dwell.
        """

        windows = _adc_windows_for_plot(event)

        rendered_windows: list[dict[str, Any]] = []

        if windows:
            for window_index, window in enumerate(windows):
                delay = float(window.get("delay", window.get("tstart", 0.0)))

                duration = window.get("duration", window.get("adc_duration", None))

                if duration is None:
                    num_samples = window.get(
                        "num_samples",
                        window.get("number_of_samples", None),
                    )
                    dwell = window.get("dwell", window.get("sample_time", None))

                    if num_samples is None or dwell is None:
                        continue

                    duration = int(num_samples) * float(dwell)

                duration = float(duration)

                if duration <= 0:
                    continue

                rendered_windows.append(
                    {
                        "path": (
                            f"{_event_path(event, block_index, event_index)}."
                            f"window_{window_index:03d}"
                        ),
                        "block_index": block_index,
                        "block_node": _event_block_node(self.seq, block_index),
                        "span": (event_start + delay, duration),
                        "metadata": {
                            "window_index": window_index,
                            "role": window.get("role"),
                            "label": window.get("label"),
                            "trajectory": window.get("trajectory"),
                            "echo_index": window.get("echo_index"),
                            "line_index": window.get("line_index"),
                            "spin_echo_index": window.get("spin_echo_index"),
                            "gradient_echo_index": window.get("gradient_echo_index"),
                            "polarity": window.get("polarity"),
                        },
                    }
                )

            return rendered_windows

        # Fallback for old/simple ADC events with no explicit windows.
        duration = _get_adc_duration(event)

        if duration is None or duration <= 0:
            return []

        return [
            {
                "path": _event_path(event, block_index, event_index),
                "block_index": block_index,
                "block_node": _event_block_node(self.seq, block_index),
                "span": (event_start, duration),
            }
        ]

    def _render_gradient_event(
        self,
        *,
        event: Any,
        event_start: float,
        block_index: int,
        event_index: int,
        gradient_scale: str,
    ) -> dict[str, Any] | None:
        """Render a gradient event."""

        axis = _gradient_axis(event, sequence=self.seq)

        if axis not in {"x", "y", "z"}:
            return None

        t_values, v_values = _gradient_waveform(event, event_start)

        if not t_values or not v_values:
            return None

        count = min(len(t_values), len(v_values))
        t_values = t_values[:count]
        v_values = v_values[:count]

        if gradient_scale == "auto":
            requested_unit = str(getattr(self.system, "grad_unit", "") or "").lower()
            if requested_unit in {"mt/m", "mt_per_m", "mtm"}:
                gradient_scale = "mt_per_m"
            else:
                gradient_scale = "hz_per_m"

        if gradient_scale == "mt_per_m":
            v_values = [v / float(self.system.gamma) * 1e3 for v in v_values]
            unit = "mT/m"
        elif gradient_scale == "hz_per_m":
            unit = "Hz/m"
        else:
            raise ValueError("gradient_scale must be 'auto', 'hz_per_m', or 'mt_per_m'")

        rendered_event = {
            "axis": axis,
            "path": _event_path(event, block_index, event_index),
            "block_index": block_index,
            "block_node": _event_block_node(self.seq, block_index),
            "t": t_values,
            "v": v_values,
            "unit": unit,
            "_event_start": float(event_start),
        }

        # Counter-indexed gradient variants are attached by loop-native
        # sequences through event.metadata. Store their relative waveforms once
        # so nested-loop plotting can select the proper variant without
        # modifying the source event or invoking either writer.
        metadata = getattr(event, "metadata", None)
        variants = (
            metadata.get("seqstar_repetition_variants")
            if isinstance(metadata, Mapping)
            else None
        )

        if isinstance(variants, list) and variants:
            variant_waveforms: list[dict[str, Any]] = []

            for variant in variants:
                t_relative, v_variant = _gradient_waveform(variant, 0.0)

                if not t_relative or not v_variant:
                    variant_waveforms.append({"t": [], "v": []})
                    continue

                count_variant = min(
                    len(t_relative),
                    len(v_variant),
                )
                t_relative = [
                    float(value)
                    for value in t_relative[:count_variant]
                ]
                v_variant = [
                    float(value)
                    for value in v_variant[:count_variant]
                ]

                if gradient_scale == "mt_per_m":
                    v_variant = [
                        value / float(self.system.gamma) * 1e3
                        for value in v_variant
                    ]

                variant_waveforms.append(
                    {
                        "t": t_relative,
                        "v": v_variant,
                    }
                )

            rendered_event["_variant_waveforms"] = variant_waveforms

        return rendered_event

    def _first_tr_range(self, blocks: list[Any]) -> tuple[float, float]:
        """Return a useful first-TR range."""

        for block in blocks:
            repetitions = _get_int_from_parameters(
                block,
                keys=("repetitions", "average", "averages", "n_avg", "num_averages"),
                default=_get_int_from_parameters(
                    self.seq,
                    keys=("repetitions", "average", "averages", "n_avg", "num_averages"),
                    default=1,
                ),
            )
            repetition_time = _get_float_from_parameters(
                block,
                keys=("repetition_time", "TR", "tr"),
                default=_get_float_from_parameters(
                    self.seq,
                    keys=("repetition_time", "TR", "tr"),
                    default=None,
                ),
            )
            if repetitions > 1 and repetition_time is not None and repetition_time > 0:
                start = _get_block_tstart(block)
                if start is None:
                    start = 0.0
                return (start, start + repetition_time)

        sequence_tr = _get_float_from_parameters(
            self.seq,
            keys=("repetition_time", "TR", "tr"),
            default=None,
        )
        if sequence_tr is not None and sequence_tr > 0:
            return (0.0, sequence_tr)

        duration = calc_duration(blocks[0]) if blocks else 1e-3
        return (0.0, max(duration, 1e-6))

  
    @staticmethod
    def _plot_render(
        render: dict[str, Any],
        *,
        title: str,
        figsize: tuple[float, float],
        show_blocks: bool = True,
    ):
        """Plot a render dictionary in a gammaSTAR-like stacked layout."""

        channel_specs = [
            ("rf", "RF"),
            ("adc", "ADC"),
            ("gx", "GX"),
            ("gy", "GY"),
            ("gz", "GZ"),
        ]

        fig, axes = plt.subplots(
            5,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 0.8, 1.0, 1.0, 1.0]},
        )

        fig.patch.set_facecolor("#fcfcfd")
        time_range = render["time_range"]

        for ax, (channel, label) in zip(axes, channel_specs):
            if show_blocks:
                _draw_block_guides(ax, render.get("blocks", []), time_range)

            if channel == "rf":
                _draw_waveform_events(
                    ax,
                    render["rf"],
                    time_range=time_range,
                    color=CHANNEL_COLORS["rf"],
                    min_visible_fraction=0.002,
                )
                ax.set_ylim(_symmetric_or_positive_ylim(render["rf"], positive=True))

            elif channel == "adc":
                _draw_adc_events(
                    ax,
                    render["adc"],
                    time_range=time_range,
                    color=CHANNEL_COLORS["adc"],
                    min_visible_fraction=0.002,
                )
                ax.set_ylim(-0.05, 1.05)

            else:
                axis = channel[-1]
                events = render["gradients"][axis]
                _draw_waveform_events(
                    ax,
                    events,
                    time_range=time_range,
                    color=CHANNEL_COLORS[axis],
                    min_visible_fraction=0.002,
                )
                ax.set_ylim(_symmetric_or_positive_ylim(events, positive=False))

                unit = _first_event_unit(events)
                if unit:
                    label = f"{label} ({unit})"

            ax.axhline(0, color="#666666", linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(axis="x", color="#d8dce3", linewidth=0.8, alpha=0.7)
            ax.grid(axis="y", color="#eef1f5", linewidth=0.6, alpha=0.8)

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        left, right = time_range
        axes[-1].set_xlim(left, right)
        axes[-1].set_xlabel("Time within selected node (s)")

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        return fig
    
def plot(
    seq: Any,
    time_range: tuple[float, float] | None = None,
    *,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    one_tr: bool = False,
    rf_scale: str = "normalized",
    gradient_scale: str = "auto",
    figsize: tuple[float, float] = (15.0, 8.5),
    dpi: int = 140,
    debug: bool = False,
    show_blocks: bool = True,
):
    """Convenience function matching PyPulseq-style plotting."""

    return SeqStarPlotter(seq).plot(
        time_range=time_range,
        title=title,
        show=show,
        save=save,
        one_tr=one_tr,
        rf_scale=rf_scale,
        gradient_scale=gradient_scale,
        figsize=figsize,
        dpi=dpi,
        debug=debug,
        show_blocks=show_blocks,
    )




def _expand_rendered_node_repetitions(
    seq: Any,
    render: dict[str, Any],
) -> None:
    """Expand nested logical Loop nodes recursively for visualization.

    The source sequence may contain only one representative motif. This
    function expands the logical node hierarchy solely in the render
    dictionary:

        shot Loop
            excitation
            echo_train Loop
                refocusing
                phase encode
                ADC/readout
                rewind

    No writer is invoked and the source sequence remains compact.
    """

    registry = _plot_node_registry(seq)
    repeated = {
        str(name): dict(record)
        for name, record in registry.items()
        if (
            isinstance(record, Mapping)
            and str(record.get("repeat_mode") or "")
            .strip()
            .lower()
            == "loop"
            and (
                record.get("repeat_count") is not None
                or record.get("repeat_every") is not None
            )
        )
    }

    if not repeated:
        return

    repeated_names = sorted(
        repeated,
        key=lambda name: (_plot_node_depth(name), name),
    )

    def repeated_parent(node_name: str) -> str | None:
        ancestors = [
            candidate
            for candidate in repeated_names
            if (
                candidate != node_name
                and _plot_node_within(node_name, candidate)
            )
        ]
        if not ancestors:
            return None
        return max(ancestors, key=_plot_node_depth)

    children: dict[str | None, list[str]] = {}
    for node_name in repeated_names:
        children.setdefault(repeated_parent(node_name), []).append(
            node_name
        )

    for child_list in children.values():
        child_list.sort(
            key=lambda name: (_plot_node_depth(name), name)
        )

    base_blocks = [dict(block) for block in render.get("blocks", [])]
    base_rf = [dict(event) for event in render.get("rf", [])]
    base_adc = [dict(event) for event in render.get("adc", [])]
    base_gradients = {
        axis: [
            dict(event)
            for event in render.get("gradients", {}).get(axis, [])
        ]
        for axis in ("x", "y", "z")
    }

    repeated_block_indices: set[int] = set()
    for block in base_blocks:
        index = block.get("index")
        if not isinstance(index, int):
            continue
        if any(
            _rendered_block_belongs_to_node(block, node_name)
            for node_name in repeated_names
        ):
            repeated_block_indices.add(index)

    # Preserve content that lies outside all repeated roots.
    render["blocks"] = [
        block
        for block in base_blocks
        if block.get("index") not in repeated_block_indices
    ]
    render["rf"] = [
        event
        for event in base_rf
        if event.get("block_index") not in repeated_block_indices
    ]
    render["adc"] = [
        event
        for event in base_adc
        if event.get("block_index") not in repeated_block_indices
    ]
    render["gradients"] = {
        axis: [
            event
            for event in base_gradients[axis]
            if event.get("block_index") not in repeated_block_indices
        ]
        for axis in ("x", "y", "z")
    }

    ordered_repeated = sorted(
        repeated_names,
        key=_plot_node_depth,
    )
    dimensions = {
        node_name: max(
            1,
            int(
                round(
                    float(
                        _plot_resolve_parameter(
                            seq,
                            repeated[node_name].get("repeat_count"),
                            default=1,
                        )
                    )
                )
            ),
        )
        for node_name in ordered_repeated
    }

    counters = {
        node_name: str(
            repeated[node_name].get("counter")
            or f"{node_name.rsplit('.', 1)[-1]}_index"
        )
        for node_name in ordered_repeated
    }

    def direct_blocks(node_name: str) -> list[dict[str, Any]]:
        child_nodes = children.get(node_name, [])
        out_blocks: list[dict[str, Any]] = []

        for block in base_blocks:
            if not _rendered_block_belongs_to_node(
                block,
                node_name,
            ):
                continue

            if any(
                _rendered_block_belongs_to_node(block, child)
                for child in child_nodes
            ):
                continue

            out_blocks.append(block)

        return out_blocks

    def node_motif_start(node_name: str) -> float:
        owned = [
            block
            for block in base_blocks
            if _rendered_block_belongs_to_node(block, node_name)
        ]
        if not owned:
            return 0.0
        return min(float(block.get("tstart", 0.0)) for block in owned)

    def repeat_period(node_name: str) -> float:
        record = repeated[node_name]
        period_ref = record.get("repeat_every")

        if period_ref is not None:
            period = float(
                _plot_resolve_parameter(
                    seq,
                    period_ref,
                    default=0.0,
                )
            )
            if period > 0:
                return period

        owned = [
            block
            for block in base_blocks
            if _rendered_block_belongs_to_node(block, node_name)
        ]
        if not owned:
            return 0.0

        start_time = min(
            float(block.get("tstart", 0.0))
            for block in owned
        )
        end_time = max(
            float(block.get("tend", start_time))
            for block in owned
        )
        return max(end_time - start_time, 0.0)

    def clone_direct_content(
        node_name: str,
        *,
        offset: float,
        context: dict[str, int],
        instance_tag: str,
    ) -> None:
        blocks_here = direct_blocks(node_name)
        indices = {
            int(block["index"])
            for block in blocks_here
            if isinstance(block.get("index"), int)
        }

        for block in blocks_here:
            clone = dict(block)
            clone["index"] = (
                f"{block.get('index')}:{instance_tag}"
            )
            clone["tstart"] = (
                float(block.get("tstart", 0.0)) + offset
            )
            clone["tend"] = (
                float(block.get("tend", 0.0)) + offset
            )
            clone["loop_context"] = dict(context)
            render["blocks"].append(clone)

        for event in base_rf:
            if _rendered_event_belongs_to_blocks(
                event,
                indices,
                node_name,
            ):
                render["rf"].append(
                    _shift_plot_event_with_context(
                        event,
                        offset=offset,
                        instance_tag=instance_tag,
                        context=context,
                        ordered_nodes=ordered_repeated,
                        dimensions=dimensions,
                        counters=counters,
                    )
                )

        for event in base_adc:
            if _rendered_event_belongs_to_blocks(
                event,
                indices,
                node_name,
            ):
                render["adc"].append(
                    _shift_plot_event_with_context(
                        event,
                        offset=offset,
                        instance_tag=instance_tag,
                        context=context,
                        ordered_nodes=ordered_repeated,
                        dimensions=dimensions,
                        counters=counters,
                    )
                )

        for axis in ("x", "y", "z"):
            for event in base_gradients[axis]:
                if _rendered_event_belongs_to_blocks(
                    event,
                    indices,
                    node_name,
                ):
                    render["gradients"][axis].append(
                        _shift_plot_event_with_context(
                            event,
                            offset=offset,
                            instance_tag=instance_tag,
                            context=context,
                            ordered_nodes=ordered_repeated,
                            dimensions=dimensions,
                            counters=counters,
                        )
                    )

    def expand_node(
        node_name: str,
        *,
        parent_offset: float,
        context: dict[str, int],
        parent_tag: str,
    ) -> None:
        count = dimensions[node_name]
        period = repeat_period(node_name)
        counter_name = counters[node_name]

        if period <= 0 and count > 1:
            raise ValueError(
                f"Repeated node {node_name!r} has no positive "
                "plot repeat period."
            )

        for iteration in range(count):
            iteration_offset = parent_offset + iteration * period
            iteration_context = dict(context)
            iteration_context[counter_name] = iteration
            instance_tag = (
                f"{parent_tag}."
                f"{_safe_plot_token(node_name)}_{iteration:03d}"
            )

            clone_direct_content(
                node_name,
                offset=iteration_offset,
                context=iteration_context,
                instance_tag=instance_tag,
            )

            for child in children.get(node_name, []):
                expand_node(
                    child,
                    parent_offset=iteration_offset,
                    context=iteration_context,
                    parent_tag=instance_tag,
                )

    roots = children.get(None, [])
    for root_index, root in enumerate(roots):
        # The representative source motif is already positioned relative to
        # zero. Repetition offsets are therefore added from zero rather than
        # subtracting the motif's first event time.
        expand_node(
            root,
            parent_offset=0.0,
            context={},
            parent_tag=f"root_{root_index:03d}",
        )


def _safe_plot_token(value: str) -> str:
    """Return a compact token for generated plot-instance identifiers."""

    token = str(value or "node").strip(".").replace(".", "_")
    return token or "node"


def _shift_plot_event_with_context(
    event: Mapping[str, Any],
    *,
    offset: float,
    instance_tag: str,
    context: dict[str, int],
    ordered_nodes: list[str],
    dimensions: dict[str, int],
    counters: dict[str, str],
) -> dict[str, Any]:
    """Clone one event, shift it, and select its loop-indexed variant."""

    clone = dict(event)
    clone["path"] = (
        f"{event.get('path', 'event')}.{instance_tag}"
    )
    clone["loop_context"] = dict(context)

    variant_waveforms = event.get("_variant_waveforms")

    if (
        isinstance(variant_waveforms, list)
        and variant_waveforms
    ):
        flat_index = 0
        stride = 1

        for node_name in reversed(ordered_nodes):
            counter_name = counters[node_name]
            index = int(context.get(counter_name, 0))
            flat_index += index * stride
            stride *= int(dimensions[node_name])

        if 0 <= flat_index < len(variant_waveforms):
            waveform = variant_waveforms[flat_index]
            relative_t = waveform.get("t", [])
            values = waveform.get("v", [])
            source_start = float(
                event.get(
                    "_event_start",
                    min(event.get("t", [0.0]) or [0.0]),
                )
            )
            clone["t"] = [
                source_start + offset + float(value)
                for value in relative_t
            ]
            clone["v"] = [float(value) for value in values]
        elif "t" in clone:
            clone["t"] = [
                float(value) + offset
                for value in clone["t"]
            ]
    elif "t" in clone:
        clone["t"] = [
            float(value) + offset
            for value in clone["t"]
        ]

    if "span" in clone:
        start, duration = clone["span"]
        clone["span"] = (
            float(start) + offset,
            float(duration),
        )

    metadata = dict(clone.get("metadata", {}))
    metadata["loop_context"] = dict(context)
    clone["metadata"] = metadata

    return clone



def _rendered_block_belongs_to_node(
    block: Mapping[str, Any],
    node_name: str,
) -> bool:
    """Return True when one rendered block belongs to a logical node."""

    block_node = str(block.get("node", "") or "")
    repeat_parent = str(block.get("repeat_parent", "") or "")

    return (
        _plot_node_within(block_node, node_name)
        or repeat_parent == node_name
        or _plot_node_within(repeat_parent, node_name)
    )


def _rendered_event_belongs_to_blocks(
    event: Mapping[str, Any],
    block_indices: set[int],
    node_name: str,
) -> bool:
    """Return True when a rendered event belongs to a repeated motif."""

    block_index = event.get("block_index")

    if isinstance(block_index, int) and block_index in block_indices:
        return True

    block_node = str(event.get("block_node", "") or "")
    return _plot_node_within(block_node, node_name)


def _event_block_node(seq: Any, block_index: int) -> str:
    """Return the logical node path for one executable block."""

    blocks = list(_iter_blocks(seq))

    if 0 <= block_index < len(blocks):
        return _block_node_name(blocks[block_index], block_index)

    return f"block_{block_index:03d}"



def _shift_plot_event(
    event: Mapping[str, Any],
    offset: float,
    repeat_index: int,
) -> dict[str, Any]:
    """Return a time-shifted copy of one rendered event."""

    clone = dict(event)
    clone["path"] = f"{event.get('path', 'event')}.repeat_{repeat_index:03d}"

    if "t" in clone:
        clone["t"] = [float(value) + offset for value in clone["t"]]

    if "span" in clone:
        start, duration = clone["span"]
        clone["span"] = (float(start) + offset, float(duration))

    metadata = dict(clone.get("metadata", {}))
    metadata["repeat_index"] = repeat_index
    clone["metadata"] = metadata

    return clone


def _plot_node_registry(seq: Any) -> dict[str, Any]:
    """Return the logical-node registry from common sequence locations."""

    candidates = (
        getattr(seq, "nodes", None),
        getattr(getattr(seq, "timeline", None), "nodes", None),
        getattr(getattr(seq, "timeline", None), "node_registry", None),
    )

    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, Mapping):
        candidates = candidates + (
            metadata.get("seqstar_nodes"),
            metadata.get("nodes"),
            metadata.get("node_registry"),
        )

    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)

    return {}


def _plot_resolve_parameter(
    seq: Any,
    value: Any,
    *,
    default: Any,
) -> Any:
    """Resolve a literal or exact sequence-parameter reference."""

    if value is None:
        return default

    if not isinstance(value, str):
        return value

    aliases = {
        "TR": ("TR", "repetition_time"),
        "repetition_time": ("repetition_time", "TR"),
        "n_slices": ("n_slices", "num_slices"),
        "num_slices": ("num_slices", "n_slices"),
    }
    candidates = aliases.get(value, (value,))

    parameters = getattr(seq, "parameters", None)
    if isinstance(parameters, Mapping):
        for candidate in candidates:
            if candidate in parameters:
                return parameters[candidate]

    metadata = getattr(seq, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("parameters", "protocol_parameters", "protocol"):
            container = metadata.get(key)
            if isinstance(container, Mapping):
                for candidate in candidates:
                    if candidate in container:
                        return container[candidate]

    try:
        return float(value)
    except ValueError:
        return default


def _plot_node_within(node: str, ancestor: str) -> bool:
    node = str(node or "").strip(".")
    ancestor = str(ancestor or "").strip(".")
    return bool(
        node
        and ancestor
        and (node == ancestor or node.startswith(f"{ancestor}."))
    )


def _plot_node_depth(node: str) -> int:
    node = str(node or "").strip(".")
    return len(node.split(".")) if node else 0


def _try_resolve_relationships(seq: Any) -> None:
    """Resolve timing relationships if the public API is available."""

    try:
        import pypulseq_star as ppstar

        relationships = getattr(ppstar, "relationships", None)
        resolver = getattr(relationships, "resolve", None)
        if callable(resolver):
            resolver(seq)
    except Exception:
        # Plotting should remain available while a relationship is being
        # developed or debugged. The materialized timeline is the fallback.
        return


def _resolved_block_starts(seq: Any) -> dict[int, float]:
    """Return unambiguous block-index -> resolved absolute starts.

    Logical node paths are not necessarily unique. An expanded sequence may
    contain many blocks named ``kernel.readout`` or ``kernel.spoiling``. A
    relationship resolved for one concrete occurrence must never be broadcast
    to every block sharing that logical path.

    Resolution policy
    -----------------
    1. Prefer explicit concrete block indices carried by relationship metadata.
    2. Accept integer references in resolved records.
    3. Accept a logical node/path only when it maps to exactly one block.
    4. Ignore ambiguous logical references and let the executable sequential
       timeline determine those starts.

    This policy is generic for loop and expanded timelines and does not depend
    on FID, GRE, EPI, or any particular node name.
    """

    out: dict[int, float] = {}
    relationships: list[Any] = []
    blocks = list(_iter_blocks(seq))

    node_to_indices: dict[str, list[int]] = {}
    object_id_to_index: dict[int, int] = {}

    for index, block in enumerate(blocks):
        object_id_to_index[id(block)] = index

        candidates: set[str] = {
            _block_node_name(block, index),
        }

        metadata = getattr(block, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in (
                "seqstar_node",
                "node",
                "path",
                "name",
            ):
                value = metadata.get(key)
                if value:
                    candidates.add(str(value))

        for attr in ("node", "path", "name"):
            value = getattr(block, attr, None)
            if value:
                candidates.add(str(value))

        for candidate in candidates:
            node_to_indices.setdefault(candidate, []).append(index)

    for owner in (seq, getattr(seq, "timeline", None)):
        if owner is None:
            continue

        for attr in ("relationships", "seqstar_relationships"):
            value = getattr(owner, attr, None)
            if value:
                relationships.extend(
                    list(value.values())
                    if isinstance(value, Mapping)
                    else list(value)
                )

    try:
        from pypulseq_star.relationships.relationship import get_relationships

        relationships.extend(get_relationships(seq))
    except Exception:
        pass

    # Preserve order while removing duplicate relationship objects.
    unique_relationships: list[Any] = []
    seen_relationship_ids: set[int] = set()

    for relationship in relationships:
        relationship_id = id(relationship)
        if relationship_id in seen_relationship_ids:
            continue
        seen_relationship_ids.add(relationship_id)
        unique_relationships.append(relationship)

    def integer_index(reference: Any) -> int | None:
        if reference is None:
            return None

        if isinstance(reference, bool):
            return None

        try:
            index = int(reference)
        except (TypeError, ValueError):
            return None

        if 0 <= index < len(blocks):
            return index

        return None

    def unique_logical_index(reference: Any) -> int | None:
        if reference is None:
            return None

        direct_index = integer_index(reference)
        if direct_index is not None:
            return direct_index

        reference_text = str(reference)

        direct_matches = node_to_indices.get(reference_text, [])
        if len(direct_matches) == 1:
            return direct_matches[0]
        if len(direct_matches) > 1:
            return None

        suffix_matches: set[int] = set()

        for node_name, indices in node_to_indices.items():
            if (
                node_name.endswith(f".{reference_text}")
                or reference_text.endswith(f".{node_name}")
            ):
                suffix_matches.update(indices)

        if len(suffix_matches) == 1:
            return next(iter(suffix_matches))

        return None

    def metadata_block_index(
        relationship: Any,
        endpoint: str,
    ) -> int | None:
        metadata = getattr(relationship, "metadata", None)
        if not isinstance(metadata, Mapping):
            return None

        # Concrete object references are authoritative.
        for key in (
            f"{endpoint}_block_object",
            f"{endpoint}_block",
        ):
            candidate = metadata.get(key)
            if candidate is not None and not isinstance(
                candidate,
                (str, int, float, bool),
            ):
                index = object_id_to_index.get(id(candidate))
                if index is not None:
                    return index

        # Explicit object IDs are also authoritative.
        for key in (
            f"{endpoint}_block_object_id",
            f"{endpoint}_block_id",
        ):
            candidate_id = metadata.get(key)
            if isinstance(candidate_id, int):
                index = object_id_to_index.get(candidate_id)
                if index is not None:
                    return index

        # Explicit timeline/global indices.
        for key in (
            f"{endpoint}_block_index",
            f"{endpoint}_global_block_index",
            f"{endpoint}_timeline_index",
        ):
            index = integer_index(metadata.get(key))
            if index is not None:
                return index

        return None

    for relationship in unique_relationships:
        resolved = getattr(relationship, "resolved", None)
        if not isinstance(resolved, Mapping):
            continue

        for endpoint in ("target", "reference"):
            start_key = f"{endpoint}_block_start"

            try:
                start = float(resolved.get(start_key))
            except (TypeError, ValueError):
                continue

            # First use concrete relationship metadata.
            index = metadata_block_index(relationship, endpoint)

            # Then accept explicit resolved indices.
            if index is None:
                for key in (
                    f"{endpoint}_block_index",
                    f"{endpoint}_global_block_index",
                    f"{endpoint}_timeline_index",
                ):
                    index = integer_index(resolved.get(key))
                    if index is not None:
                        break

            # Finally accept a logical reference only if unique.
            if index is None:
                for key in (
                    f"{endpoint}_block",
                    f"{endpoint}_block_node",
                ):
                    index = unique_logical_index(resolved.get(key))
                    if index is not None:
                        break

            if index is not None:
                out[index] = start

    return out



def _block_node_name(block: Any, index: int) -> str:
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("seqstar_node") or metadata.get("node")
        if value:
            return str(value)
    for attr in ("node", "path", "name"):
        value = getattr(block, attr, None)
        if value:
            return str(value)
    return f"block_{index:03d}"


def _block_repeat_parent(block: Any) -> str | None:
    """Return explicit logical repeat-parent metadata for one block."""

    metadata = getattr(block, "metadata", None)

    if isinstance(metadata, Mapping):
        for key in (
            "seqstar_repeat_parent",
            "repeat_parent",
        ):
            value = metadata.get(key)
            if value:
                return str(value)

    for attr in (
        "seqstar_repeat_parent",
        "repeat_parent",
    ):
        value = getattr(block, attr, None)
        if value:
            return str(value)

    return None


def _block_role(block: Any) -> str | None:
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("seqstar_role") or metadata.get("role")
        if value is not None:
            return str(value)
    value = getattr(block, "role", None)
    return None if value is None else str(value)


def _draw_block_guides(ax: Any, blocks: list[dict[str, Any]], time_range: tuple[float, float]) -> None:
    """Draw subtle block boundaries without changing waveform scaling."""

    left, right = time_range
    visible = [
        block for block in blocks
        if float(block.get("tend", 0.0)) >= left and float(block.get("tstart", 0.0)) <= right
    ]
    for position, block in enumerate(visible):
        start = float(block.get("tstart", 0.0))
        end = float(block.get("tend", start))
        if position % 2 == 1 and end > start:
            ax.axvspan(max(start, left), min(end, right), color="#eef2f7", alpha=0.22, linewidth=0)
        ax.axvline(start, color="#b9c2cf", linewidth=0.45, alpha=0.55, zorder=0)


def _iter_blocks(seq: Any) -> Iterable[Any]:
    """Iterate over blocks from a SeqStar sequence."""

    if hasattr(seq, "timeline") and hasattr(seq.timeline, "blocks"):
        blocks = seq.timeline.blocks

        if isinstance(blocks, Mapping):
            yield from blocks.values()
            return

        yield from blocks
        return

    if hasattr(seq, "blocks"):
        blocks = seq.blocks

        if isinstance(blocks, Mapping):
            yield from blocks.values()
            return

        yield from blocks
        return

    raise TypeError("Sequence does not expose timeline.blocks or blocks.")


def _iter_events(block: Any) -> Iterable[Any]:
    """Iterate over plottable events from a SeqStar block.

    This intentionally searches both ``events`` and ``children`` and does not
    return early just because one of those containers exists. This matters for
    hierarchy-rich SeqStar blocks where timing may pass because children are
    valid, while block.events itself may be empty or incomplete.
    """

    visited: set[int] = set()
    yield from _iter_node_events(block, visited=visited, is_root=True)


def _iter_node_events(
    node: Any,
    *,
    visited: set[int],
    is_root: bool = False,
) -> Iterable[Any]:
    """Recursively yield renderable events from a node or container."""

    if node is None or isinstance(node, (float, int, str, bool)):
        return

    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if not is_root and _looks_renderable_event(node):
        yield node
        return

    # Mapping container.
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _iter_node_events(value, visited=visited)
        return

    # List/tuple/set container.
    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from _iter_node_events(value, visited=visited)
        return

    # Preferred explicit event container.
    if hasattr(node, "events"):
        events = getattr(node, "events")
        yield from _iter_node_events(events, visited=visited)

    # Preferred hierarchy container.
    if hasattr(node, "children"):
        children = getattr(node, "children")
        yield from _iter_node_events(children, visited=visited)

    # Slotted dataclass fallback.
    if is_dataclass(node):
        for field_info in fields(node):
            if field_info.name in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                # parameters/metadata are checked by render helpers, not
                # recursively traversed as events. shape is not an event.
                continue

            try:
                value = getattr(node, field_info.name)
            except AttributeError:
                continue

            yield from _iter_node_events(value, visited=visited)

    # Normal object fallback.
    elif hasattr(node, "__dict__"):
        for key, value in vars(node).items():
            if key in {
                "parent",
                "relationships",
                "metadata",
                "parameters",
                "shape",
                "system",
            }:
                continue

            yield from _iter_node_events(value, visited=visited)


def _looks_renderable_event(obj: Any) -> bool:
    """Return True if object looks like a plottable RF/ADC/gradient event."""

    if obj is None or isinstance(obj, (float, int, str, bool)):
        return False

    class_name = obj.__class__.__name__.lower()

    # Avoid plotting waveform-shape objects as events.
    if "shape" in class_name and not any(
        hasattr(obj, marker)
        for marker in (
            "flip_angle",
            "num_samples",
            "dwell",
            "channel",
            "axis",
            "rise_time",
            "flat_time",
            "fall_time",
            "area",
            "amplitude",
            "flat_area",
        )
    ):
        return False

    return _is_adc_event(obj) or _is_gradient_event(obj) or _is_rf_event(obj)


def _block_has_explicit_absolute_start(block: Any) -> bool:
    """Return True when a block explicitly declares absolute timing."""

    metadata = getattr(block, "metadata", None)

    if isinstance(metadata, Mapping):
        for key in (
            "absolute_timing",
            "absolute_start",
            "has_absolute_tstart",
            "seqstar_absolute_timing",
        ):
            value = metadata.get(key)
            if value is not None:
                return bool(value)

    for attr in (
        "absolute_timing",
        "has_absolute_tstart",
    ):
        value = getattr(block, attr, None)
        if value is not None:
            return bool(value)

    return False


def _get_block_tstart(block: Any) -> float | None:
    """Return explicit absolute block start time, if present.

    A block-level ``delay=0`` should not reset the plot timeline to zero.
    In PyPulseq-style ``seq.add_block(...)`` construction, blocks are
    sequential unless an explicit absolute tstart/start field is provided.
    """

    for attr in ("tstart", "start", "start_s", "absolute_tstart", "absolute_start"):
        value = _get_optional_float(block, attr)
        if value is not None:
            return value

    for key in ("tstart", "start", "start_s", "absolute_tstart", "absolute_start"):
        value = _get_timing_value(block, key)
        if value is not None:
            return value

    return None


def _get_event_tstart(event: Any) -> float:
    """Return event local start time."""

    for attr in ("delay", "tstart", "start", "start_s"):
        value = _get_optional_float(event, attr)
        if value is not None:
            return value

    for key in ("delay", "tstart", "start", "start_s"):
        value = _get_timing_value(event, key)
        if value is not None:
            return value

    return 0.0


def _get_event_duration(event: Any) -> float | None:
    """Return event active duration if available."""

    # ADC duration.
    if _get_optional_float(event, "num_samples") is not None and _get_optional_float(event, "dwell") is not None:
        return _get_optional_float(event, "num_samples") * _get_optional_float(event, "dwell")

    num_samples = _get_timing_value(event, "num_samples")
    dwell = _get_timing_value(event, "dwell")
    if num_samples is not None and dwell is not None:
        return num_samples * dwell

    # Trapezoid duration.
    rise_time = _get_timing_value(event, "rise_time")
    flat_time = _get_timing_value(event, "flat_time")
    fall_time = _get_timing_value(event, "fall_time")
    if rise_time is not None and flat_time is not None and fall_time is not None:
        return rise_time + flat_time + fall_time

    if hasattr(event, "duration") and getattr(event, "duration") is not None:
        return float(getattr(event, "duration"))

    duration = _get_timing_value(event, "duration")
    if duration is not None:
        return duration

    shape_duration = _get_timing_value(event, "shape_duration")
    if shape_duration is not None:
        return shape_duration

    shape = getattr(event, "shape", None)
    if shape is not None and hasattr(shape, "duration"):
        return float(shape.duration)

    if hasattr(event, "t"):
        t = getattr(event, "t")
        try:
            return float(t[-1])
        except (TypeError, IndexError):
            return None

    return None


def _get_adc_duration(event: Any) -> float | None:
    """Return ADC event duration."""

    num_samples = _get_optional_float(event, "num_samples")
    dwell = _get_optional_float(event, "dwell")

    if num_samples is not None and dwell is not None:
        return num_samples * dwell

    num_samples = _get_timing_value(event, "num_samples")
    dwell = _get_timing_value(event, "dwell")

    if num_samples is not None and dwell is not None:
        return num_samples * dwell

    return _get_event_duration(event)


def _adc_windows_for_plot(event: Any) -> list[dict[str, Any]]:
    """Return ADC windows from an enriched ADC event, if available.

    This deliberately supports several possible ADC object shapes:

        event.to_gammastar_windows()
        event.to_gammastar_samples()
        event.to_pulseq_windows()
        event.shape.to_gammastar_windows()
        event.shape.to_pulseq_windows()
        event.windows

    The plotter is a visualization layer, so it should be permissive and avoid
    depending on only one writer/export representation.
    """

    candidate_data: Any = None

    for method_name in (
        "to_gammastar_windows",
        "to_gammastar_samples",
        "to_pulseq_windows",
    ):
        if hasattr(event, method_name):
            method = getattr(event, method_name)

            try:
                candidate_data = method(include_sample_times=False)
            except TypeError:
                candidate_data = method()

            windows = _extract_windows_from_candidate(candidate_data)
            if windows:
                return windows

    shape = getattr(event, "shape", None)

    if shape is not None:
        for method_name in (
            "to_gammastar_windows",
            "to_gammastar_samples",
            "to_pulseq_windows",
        ):
            if hasattr(shape, method_name):
                method = getattr(shape, method_name)

                try:
                    candidate_data = method(include_sample_times=False)
                except TypeError:
                    candidate_data = method()

                windows = _extract_windows_from_candidate(candidate_data)
                if windows:
                    return windows

    windows_attr = getattr(event, "windows", None)

    if windows_attr is not None:
        windows: list[dict[str, Any]] = []

        for window in windows_attr:
            if isinstance(window, Mapping):
                windows.append(dict(window))
            elif hasattr(window, "to_dict"):
                windows.append(dict(window.to_dict()))
            elif hasattr(window, "to_pulseq_dict"):
                windows.append(dict(window.to_pulseq_dict()))
            else:
                windows.append(
                    {
                        "delay": getattr(window, "delay", 0.0),
                        "duration": getattr(window, "duration", None),
                        "num_samples": getattr(window, "num_samples", None),
                        "dwell": getattr(window, "dwell", None),
                        "label": getattr(window, "label", None),
                        "role": getattr(window, "role", None),
                        "trajectory": getattr(window, "trajectory", None),
                    }
                )

        return [_normalize_adc_window_for_plot(window) for window in windows]

    return []


def _extract_windows_from_candidate(candidate: Any) -> list[dict[str, Any]]:
    """Extract ADC windows from a method return value."""

    if candidate is None:
        return []

    if isinstance(candidate, Mapping):
        windows = candidate.get("windows", None)

        if windows is None:
            # Some simple representations may themselves be one window.
            if "num_samples" in candidate or "number_of_samples" in candidate:
                return [_normalize_adc_window_for_plot(dict(candidate))]

            return []

        return [_normalize_adc_window_for_plot(window) for window in windows]

    if isinstance(candidate, list):
        return [_normalize_adc_window_for_plot(window) for window in candidate]

    return []


def _normalize_adc_window_for_plot(window: Any) -> dict[str, Any]:
    """Normalize one ADC window into delay/duration/num_samples/dwell fields."""

    if isinstance(window, Mapping):
        window_dict = dict(window)
    elif hasattr(window, "to_dict"):
        window_dict = dict(window.to_dict())
    elif hasattr(window, "to_pulseq_dict"):
        window_dict = dict(window.to_pulseq_dict())
    else:
        window_dict = {
            "delay": getattr(window, "delay", 0.0),
            "duration": getattr(window, "duration", None),
            "num_samples": getattr(window, "num_samples", None),
            "dwell": getattr(window, "dwell", None),
        }

    if "delay" not in window_dict and "tstart" in window_dict:
        window_dict["delay"] = window_dict["tstart"]

    if "tstart" not in window_dict and "delay" in window_dict:
        window_dict["tstart"] = window_dict["delay"]

    if "dwell" not in window_dict and "sample_time" in window_dict:
        window_dict["dwell"] = window_dict["sample_time"]

    if "sample_time" not in window_dict and "dwell" in window_dict:
        window_dict["sample_time"] = window_dict["dwell"]

    if "num_samples" not in window_dict and "number_of_samples" in window_dict:
        window_dict["num_samples"] = window_dict["number_of_samples"]

    if "number_of_samples" not in window_dict and "num_samples" in window_dict:
        window_dict["number_of_samples"] = window_dict["num_samples"]

    if window_dict.get("duration") is None:
        num_samples = window_dict.get("num_samples")
        dwell = window_dict.get("dwell")

        if num_samples is not None and dwell is not None:
            window_dict["duration"] = int(num_samples) * float(dwell)

    return window_dict


def _is_rf_event(event: Any) -> bool:
    """Return True if event is RF-like.

    Deliberately avoids matching on event name or generic roles. Slice-select
    gradients can be paired with RF pulses and should still route as gradients.
    """

    event_type = str(getattr(event, "event_type", "") or "").lower()
    type_value = str(getattr(event, "type", "") or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    use = str(getattr(event, "use", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    return (
        event_type in {"rf", "radiofrequency"}
        or type_value in {"rf", "radiofrequency"}
        or kind in {"rf", "radiofrequency", "rf_block", "rf_sinc", "rf_gauss"}
        or class_name.startswith("seqstarrf")
        or "rfevent" in class_name
        or "rfblock" in class_name
        or "rfpulse" in class_name
        or hasattr(event, "flip_angle")
        or (
            use in {
                "excitation",
                "excite",
                "refocusing",
                "refocus",
                "inversion",
                "invert",
            }
            and hasattr(event, "phase_offset")
            and hasattr(event, "freq_offset")
        )
    )


def _is_adc_event(event: Any) -> bool:
    """Return True if event is ADC-like."""

    event_type = str(getattr(event, "event_type", "") or "").lower()
    type_value = str(getattr(event, "type", "") or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    return (
        event_type == "adc"
        or type_value == "adc"
        or kind.startswith("adc")
        or "adcevent" in class_name
        or "adctrain" in class_name
        or hasattr(event, "num_samples")
        or hasattr(event, "to_pulseq_windows")
        or hasattr(event, "to_gammastar_windows")
    )


def _is_gradient_event(event: Any) -> bool:
    """Return True if event is gradient-like."""

    event_type = str(getattr(event, "event_type", "") or "").lower()
    type_value = str(getattr(event, "type", "") or "").lower()
    kind = str(getattr(event, "kind", "") or "").lower()
    class_name = event.__class__.__name__.lower()

    return (
        event_type in {"grad", "gradient", "trap", "trapezoid"}
        or type_value in {"grad", "gradient", "trap", "trapezoid"}
        or kind in {
            "grad",
            "gradient",
            "trap",
            "trapezoid",
            "arbitrary",
            "arbitrary_grad",
            "arbitrary_gradient",
            "split",
            "split_gradient",
        }
        or "gradientevent" in class_name
        or "gradientshape" in class_name
        or "trapezoid" in class_name
        or "splitgradient" in class_name
        or (
            hasattr(event, "channel")
            and (
                hasattr(event, "area")
                or hasattr(event, "amplitude")
                or hasattr(event, "flat_area")
                or all(
                    hasattr(event, attr)
                    for attr in ("rise_time", "flat_time", "fall_time")
                )
            )
        )
    )

def _gradient_axis(
    event: Any,
    *,
    sequence: Any | None = None,
) -> str | None:
    """Return the physical gradient axis x/y/z.

    Logical ``axis_role`` metadata is mapped through the sequence encoding
    frame when available. Falling back to the event channel preserves legacy
    behavior for sequences that do not use logical imaging axes.
    """

    logical_role = None
    for attr in ("logical_axis", "axis_role", "encoding_role"):
        value = str(getattr(event, attr, "") or "").strip().lower()
        if value in {"read", "phase", "slice"}:
            logical_role = value
            break
    metadata = getattr(event, "metadata", None)
    if logical_role is None and isinstance(metadata, Mapping):
        for key in ("logical_axis", "axis_role", "encoding_role"):
            value = str(metadata.get(key) or "").strip().lower()
            if value in {"read", "phase", "slice"}:
                logical_role = value
                break

    frame = getattr(sequence, "encoding_frame", None)
    if logical_role is not None and frame is not None:
        direction = getattr(frame, f"{logical_role}_dir", None)
        if isinstance(direction, (list, tuple)) and len(direction) >= 3:
            magnitudes = [abs(float(direction[i])) for i in range(3)]
            if max(magnitudes) > 0:
                return ("x", "y", "z")[magnitudes.index(max(magnitudes))]

    for attr in ("axis", "channel"):
        if hasattr(event, attr) and getattr(event, attr) is not None:
            axis = str(getattr(event, attr)).lower()
            return _normalize_axis(axis)

    if hasattr(event, "parameters") and isinstance(event.parameters, Mapping):
        for key in ("axis", "channel"):
            if key in event.parameters:
                return _normalize_axis(str(event.parameters[key]).lower())

    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("axis", "channel"):
            if key in metadata:
                return _normalize_axis(str(metadata[key]).lower())

    name = str(getattr(event, "name", "")).lower()
    for axis in ("x", "y", "z"):
        if name.endswith(axis) or name in {
            f"g{axis}",
            f"grad_{axis}",
            f"gradient_{axis}",
            f"gradient{axis}",
        }:
            return axis

    return None


def _normalize_axis(axis: str) -> str | None:
    """Normalize gradient axis names."""

    axis = axis.lower()

    if axis in {"x", "gx", "read", "readout", "ro"}:
        return "x"

    if axis in {"y", "gy", "phase", "phase_encode", "pe"}:
        return "y"

    if axis in {"z", "gz", "slice", "slice_select", "ss"}:
        return "z"

    return None


def _gradient_waveform(event: Any, event_start: float) -> tuple[list[float], list[float]]:
    """Return gradient waveform time/value arrays.

    The gradient object model currently has two useful representations:

    1. Trapezoids expose rise_time / flat_time / fall_time / amplitude.
    2. Arbitrary and split-gradient parts expose waveform samples through
       event.waveform + event.tt, shape.waveform() + shape.time_axis(), or
       event.to_gammastar_waveform().

    The previous plotter only handled the trapezoid path and a non-existent
    shape.to_gammastar_samples() path, so arbitrary and split-gradient parts
    rendered as empty channels. This function is deliberately permissive and
    plotter-only; it does not change the gradient objects or writers.
    """

    rise_time = _get_gradient_value(event, "rise_time")
    flat_time = _get_gradient_value(event, "flat_time")
    fall_time = _get_gradient_value(event, "fall_time")
    amplitude = _get_gradient_value(event, "amplitude")

    if amplitude is None:
        amplitude = _get_gradient_value(event, "amp")

    if (
        rise_time is not None
        and flat_time is not None
        and fall_time is not None
        and amplitude is not None
    ):
        rise_time = float(rise_time)
        flat_time = float(flat_time)
        fall_time = float(fall_time)
        amplitude = float(amplitude)

        t0 = event_start
        t1 = t0 + rise_time
        t2 = t1 + flat_time
        t3 = t2 + fall_time

        return [t0, t1, t2, t3], [0.0, amplitude, amplitude, 0.0]

    # Generic arbitrary-gradient path. This covers:
    #   - SeqStarArbitraryGradientEvent
    #   - split-gradient ramp_up / flat_top / ramp_down parts
    #   - future gradient shapes that expose waveform/time_axis methods
    t_relative, v_values = _gradient_sample_arrays(event)

    if t_relative and v_values:
        count = min(len(t_relative), len(v_values))
        t_relative = [float(t) for t in t_relative[:count]]
        v_values = [float(v) for v in v_values[:count]]

        # Add explicit edge samples when the object exposes first/last values.
        # This makes two-sample split-gradient parts visually meaningful and
        # preserves the intended start/end values rather than only plotting
        # center samples.
        active_duration = _gradient_active_duration(event, t_relative=t_relative)
        first_value = _get_gradient_value(event, "first")
        last_value = _get_gradient_value(event, "last")

        if first_value is not None and active_duration is not None:
            if not t_relative or abs(t_relative[0]) > 1e-15:
                t_relative = [0.0, *t_relative]
                v_values = [float(first_value), *v_values]

        if last_value is not None and active_duration is not None:
            if not t_relative or abs(t_relative[-1] - active_duration) > 1e-15:
                t_relative = [*t_relative, float(active_duration)]
                v_values = [*v_values, float(last_value)]

        t_values = [event_start + float(t) for t in t_relative]
        return t_values, v_values

    duration = _gradient_active_duration(event, t_relative=None)

    if amplitude is None:
        amplitude = _get_optional_float(event, "amplitude")
    if amplitude is None:
        amplitude = _get_optional_float(event, "amp")

    if duration is not None and amplitude is not None:
        return [event_start, event_start + duration], [float(amplitude), float(amplitude)]

    return [], []


def _gradient_sample_arrays(event: Any) -> tuple[list[float], list[float]]:
    """Extract arbitrary-gradient sample times and values for plotting."""

    # 1. Preferred explicit event representation used by SeqStarGradientEvent.
    if hasattr(event, "tt") and hasattr(event, "waveform"):
        try:
            t_values = _as_float_list(getattr(event, "tt"))
            v_values = _as_float_list(getattr(event, "waveform"))
            if t_values and v_values:
                return t_values, v_values
        except Exception:
            pass

    # 2. Common simple attribute spellings.
    for time_name, value_name in (
        ("t", "waveform"),
        ("tt", "samples"),
        ("times", "amplitudes"),
        ("time", "amplitude"),
    ):
        if hasattr(event, time_name) and hasattr(event, value_name):
            try:
                t_values = _as_float_list(getattr(event, time_name))
                v_values = _as_float_list(getattr(event, value_name))
                if t_values and v_values:
                    return t_values, v_values
            except Exception:
                pass

    # 3. gammaSTAR-oriented event payload.
    if hasattr(event, "to_gammastar_waveform"):
        try:
            payload = event.to_gammastar_waveform()
            t_values, v_values = _gradient_arrays_from_payload(payload)
            if t_values and v_values:
                return t_values, v_values
        except Exception:
            pass

    # 4. Shape-level waveform/time_axis methods.
    shape = getattr(event, "shape", None)
    if shape is not None:
        if hasattr(shape, "time_axis") and hasattr(shape, "waveform"):
            try:
                t_values = _as_float_list(shape.time_axis())
                v_values = _as_float_list(shape.waveform())
                if t_values and v_values:
                    return t_values, v_values
            except Exception:
                pass

        if hasattr(shape, "to_dict"):
            try:
                payload = shape.to_dict()
                t_values, v_values = _gradient_arrays_from_payload(payload)
                if t_values and v_values:
                    return t_values, v_values
            except Exception:
                pass

    # 5. Event dictionary fallback.
    if hasattr(event, "to_dict"):
        try:
            payload = event.to_dict()
            t_values, v_values = _gradient_arrays_from_payload(payload)
            if t_values and v_values:
                return t_values, v_values
        except Exception:
            pass

    return [], []


def _gradient_arrays_from_payload(payload: Any) -> tuple[list[float], list[float]]:
    """Extract gradient arrays from a dict-like payload."""

    if not isinstance(payload, Mapping):
        return [], []

    t_raw = (
        payload.get("tt")
        or payload.get("t")
        or payload.get("times")
        or payload.get("time")
    )
    v_raw = (
        payload.get("waveform")
        or payload.get("samples")
        or payload.get("amplitudes")
        or payload.get("values")
    )

    # gammaSTAR samples sometimes use {t=[...], v=[{am=[...]}]}.
    if v_raw is None and "v" in payload:
        v_container = payload.get("v")
        if (
            isinstance(v_container, list)
            and v_container
            and isinstance(v_container[0], Mapping)
        ):
            v_raw = v_container[0].get("am")
        else:
            v_raw = v_container

    if t_raw is None and v_raw is not None:
        grad_raster = payload.get("grad_raster_time") or payload.get("raster_time")
        if grad_raster is not None:
            try:
                raster = float(grad_raster)
                n = len(v_raw)
                t_raw = [(i + 0.5) * raster for i in range(n)]
            except Exception:
                t_raw = None

    try:
        t_values = _as_float_list(t_raw)
        v_values = _as_float_list(v_raw)
    except Exception:
        return [], []

    return t_values, v_values


def _gradient_active_duration(
    event: Any,
    *,
    t_relative: list[float] | None,
) -> float | None:
    """Return active gradient duration excluding event delay."""

    for key in ("active_duration", "shape_duration"):
        value = _get_gradient_value(event, key)
        if value is not None:
            return float(value)

    shape = getattr(event, "shape", None)
    if shape is not None:
        for key in ("active_duration", "shape_duration"):
            value = _get_gradient_value(shape, key)
            if value is not None:
                return float(value)

    # duration on SeqStarGradientEvent includes delay, so subtract the event
    # delay when possible.
    duration = _get_gradient_value(event, "duration")
    if duration is not None:
        delay = _get_event_tstart(event)
        return max(float(duration) - float(delay), 0.0)

    if t_relative:
        return max(float(t) for t in t_relative)

    return None


def _get_gradient_value(obj: Any, key: str) -> Any:
    """Read a gradient value from attrs, parameters, metadata, shape, or dicts."""

    if obj is None:
        return None

    if isinstance(obj, Mapping):
        if key in obj and obj[key] is not None:
            return obj[key]
        shape_payload = obj.get("shape")
        if isinstance(shape_payload, Mapping) and key in shape_payload:
            return shape_payload[key]
        return None

    if hasattr(obj, key):
        try:
            value = getattr(obj, key)
        except Exception:
            value = None
        if value is not None:
            return value

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping) and key in parameters and parameters[key] is not None:
        return parameters[key]

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping) and key in timing and timing[key] is not None:
            return timing[key]
        if key in metadata and metadata[key] is not None:
            return metadata[key]

    shape = getattr(obj, "shape", None)
    if shape is not None and shape is not obj:
        value = _get_gradient_value(shape, key)
        if value is not None:
            return value

    if hasattr(obj, "to_dict"):
        try:
            payload = obj.to_dict()
            value = _get_gradient_value(payload, key)
            if value is not None:
                return value
        except Exception:
            pass

    return None


def _as_float_list(values: Any) -> list[float]:
    """Convert a list/tuple/numpy-like object to a flat list of floats."""

    if values is None:
        return []

    if isinstance(values, Mapping):
        return []

    if isinstance(values, (str, bytes)):
        return []

    try:
        return [float(v) for v in values]
    except TypeError:
        return [float(values)]


def _event_path(event: Any, block_index: int, event_index: int) -> str:
    """Return best available event path for plot metadata."""

    if hasattr(event, "path") and getattr(event, "path") is not None:
        return str(getattr(event, "path"))

    name = getattr(event, "name", f"event_{event_index}")
    return f"block_{block_index}.{name}"


def _get_int_from_parameters(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: int,
) -> int:
    """Resolve an integer from object parameters."""

    value = _get_from_parameters(obj, keys=keys, default=default)
    return int(value)


def _get_float_from_parameters(
    obj: Any,
    *,
    keys: tuple[str, ...],
    default: float | None,
) -> float | None:
    """Resolve a float from object parameters."""

    value = _get_from_parameters(obj, keys=keys, default=default)

    if value is None:
        return None

    return float(value)


def _get_from_parameters(obj: Any, *, keys: tuple[str, ...], default: Any) -> Any:
    """Return first matching value from obj.parameters."""

    if hasattr(obj, "parameters") and isinstance(obj.parameters, Mapping):
        for key in keys:
            if key in obj.parameters:
                return obj.parameters[key]

    return default


def _get_timing_value(obj: Any, key: str) -> float | None:
    """Read a timing-like value from attribute, parameters, or metadata.timing."""

    value = _get_optional_float(obj, key)
    if value is not None:
        return value

    parameters = getattr(obj, "parameters", None)
    if isinstance(parameters, Mapping) and key in parameters:
        try:
            return float(parameters[key])
        except (TypeError, ValueError):
            return None

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        timing = metadata.get("timing")
        if isinstance(timing, Mapping) and key in timing:
            try:
                return float(timing[key])
            except (TypeError, ValueError):
                return None

        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                return None

    return None


def _get_optional_float(obj: Any, attr: str) -> float | None:
    """Return object attribute as float if available."""

    if not hasattr(obj, attr):
        return None

    value = getattr(obj, attr)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _draw_waveform_events(
    ax: Any,
    events: list[dict[str, Any]],
    *,
    time_range: tuple[float, float],
    color: str,
    min_visible_fraction: float = 0.002,
) -> None:
    """Draw RF or gradient waveform events using a fixed channel color."""

    left, right = time_range
    x_span = max(right - left, 1e-12)
    min_visible_width = x_span * min_visible_fraction

    for event in events:
        t_values = event.get("t", [])
        v_values = event.get("v", [])

        if not t_values or not v_values:
            continue

        count = min(len(t_values), len(v_values))
        t_values = [float(t) for t in t_values[:count]]
        v_values = [float(v) for v in v_values[:count]]

        event_start = min(t_values)
        event_end = max(t_values)
        event_width = event_end - event_start

        ax.plot(t_values, v_values, linewidth=1.8, color=color)
        ax.fill_between(t_values, v_values, 0, alpha=0.18, color=color)

        if 0 < event_width < min_visible_width:
            center = 0.5 * (event_start + event_end)
            max_value = max(v_values, key=abs)

            ax.vlines(
                center,
                0,
                max_value,
                linewidth=1.4,
                alpha=0.95,
                color=color,
            )

            ax.plot(
                [center],
                [max_value],
                marker="o",
                markersize=3,
                color=color,
            )

def _draw_adc_events(
    ax: Any,
    events: list[dict[str, Any]],
    *,
    time_range: tuple[float, float],
    color: str,
    min_visible_fraction: float = 0.002,
) -> None:
    """Draw ADC gate spans using a fixed gammaSTAR-like color."""

    left, right = time_range
    x_span = max(right - left, 1e-12)
    min_visible_width = x_span * min_visible_fraction

    for event in events:
        span = event.get("span")

        if not span:
            continue

        start, duration = span
        start = float(start)
        duration = float(duration)
        end = start + duration

        t_values = [start, start, end, end]
        v_values = [0.0, 1.0, 1.0, 0.0]

        ax.plot(t_values, v_values, linewidth=1.8, color=color)
        ax.fill_between(t_values, v_values, 0, alpha=0.22, step="pre", color=color)

        if 0 < duration < min_visible_width:
            center = 0.5 * (start + end)
            ax.vlines(center, 0, 1.0, linewidth=1.2, alpha=0.95, color=color)
            ax.plot([center], [1.0], marker="o", markersize=3, color=color)

def _symmetric_or_positive_ylim(
    events: list[dict[str, Any]],
    *,
    positive: bool,
) -> tuple[float, float]:
    """Return reasonable y-limits for a channel."""

    values: list[float] = []

    for event in events:
        values.extend(float(v) for v in event.get("v", []))

    if not values:
        return (-1.0, 1.0) if not positive else (-0.05, 1.05)

    max_abs = max(abs(v) for v in values)

    if max_abs <= 0:
        return (-1.0, 1.0) if not positive else (-0.05, 1.05)

    if positive:
        return (-0.05 * max_abs, 1.15 * max_abs)

    return (-1.15 * max_abs, 1.15 * max_abs)


def _first_event_unit(events: list[dict[str, Any]]) -> str | None:
    """Return the first available unit string from rendered events."""

    for event in events:
        unit = event.get("unit")
        if unit:
            return str(unit)

    return None


def _normalize_rendered_rf(rendered: dict[str, Any]) -> None:
    """Normalize the RF channel with one scale factor for the full sequence.

    A single channel-wide scale preserves relative RF amplitudes. In
    particular, otherwise-identical 90-degree and 180-degree pulses render at
    approximately 0.5 and 1.0 rather than both being independently normalized
    to 1.0.
    """

    rf_events = rendered.get("rf", [])
    max_abs = max(
        (abs(float(value)) for event in rf_events for value in event.get("v", [])),
        default=0.0,
    )
    if max_abs <= 0.0:
        return

    for event in rf_events:
        event["v"] = [float(value) / max_abs for value in event.get("v", [])]


def _render_duration(render: dict[str, Any]) -> float:
    """Return maximum plotted time."""

    max_time = 0.0

    for event in render["rf"]:
        if event.get("t"):
            max_time = max(max_time, max(event["t"]))

    for event in render["adc"]:
        if event.get("span"):
            start, duration = event["span"]
            max_time = max(max_time, start + duration)

    for axis in ("x", "y", "z"):
        for event in render["gradients"][axis]:
            if event.get("t"):
                max_time = max(max_time, max(event["t"]))

    return max_time


def _clip_render_to_time_range(
    render: dict[str, Any],
    time_range: tuple[float, float],
) -> None:
    """Clip plotted events to the requested time range."""

    left, right = time_range

    render["rf"] = [
        event for event in render["rf"]
        if _event_intersects_range(event, left, right)
    ]

    render["adc"] = [
        event for event in render["adc"]
        if _event_intersects_range(event, left, right)
    ]

    for axis in ("x", "y", "z"):
        render["gradients"][axis] = [
            event for event in render["gradients"][axis]
            if _event_intersects_range(event, left, right)
        ]


def _event_intersects_range(event: dict[str, Any], left: float, right: float) -> bool:
    """Return True if event intersects the time range."""

    if event.get("span"):
        start, duration = event["span"]
        return start + duration >= left and start <= right

    if event.get("t"):
        return max(event["t"]) >= left and min(event["t"]) <= right

    return False

def debug_render_summary(render: dict[str, Any]) -> None:
    """Print a compact sorted summary of rendered events."""

    rows: list[tuple[float, float, str, str]] = []

    for event in render.get("rf", []):
        t = event.get("t", [])
        if t:
            rows.append((min(t), max(t), "RF", event.get("path", "")))

    for event in render.get("adc", []):
        span = event.get("span")
        if span:
            start, duration = span
            rows.append((float(start), float(start) + float(duration), "ADC", event.get("path", "")))

    for axis in ("x", "y", "z"):
        for event in render.get("gradients", {}).get(axis, []):
            t = event.get("t", [])
            if t:
                rows.append((min(t), max(t), f"G{axis.upper()}", event.get("path", "")))

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    print("\nRendered event summary")
    print("----------------------")
    for start, end, channel, path in rows:
        print(f"{start:10.6f}  {end:10.6f}  {channel:>3s}  {path}")