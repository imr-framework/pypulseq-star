#!/usr/bin/env python3
"""Generate publication-ready timing diagrams for the four reference sequences.

This script is intended to serve two purposes:

1. create consistently formatted FID, GRE, EPI, and TSE timing diagrams for
   the SoftwareX manuscript; and
2. act as a lightweight integration test by building, resolving, and checking
   the timing of all four current demo sequences.

Typical use
-----------
From the repository root:

    python examples/plot_reference_sequences.py

Selected overrides:

    python examples/plot_reference_sequences.py \
        --orientation sagittal \
        --gre-n-y 128 \
        --epi-etl 8 \
        --tse-etl 16 \
        --tse-phase-order centric

Generated files
---------------
By default, outputs are written to:

    out/reference_sequences/
        fid_timing.png
        gre_timing.png
        epi_timing.png
        tse_timing.png
        reference_sequence_capabilities.png
        reference_sequence_summary.txt

The combined PNG is a 2 × 2 manuscript-oriented layout. The four individual
PNGs remain available for later manual refinement.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Use a non-interactive backend by default so that automated test runs do not
# open four GUI windows. ``--show`` leaves the current backend untouched.
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import demo_EPI
import demo_FID
import demo_GRE
import demo_TSE


@dataclass(frozen=True)
class PlotResult:
    """Summary of one generated reference-sequence panel."""

    name: str
    sequence: Any
    realization: Any
    timing_ok: bool
    timing_report: tuple[str, ...]
    png_path: Path
    summary_lines: tuple[str, ...]


def _resolve_and_check(sequence: Any) -> tuple[Any, bool, tuple[str, ...]]:
    """Resolve one symbolic sequence and normalize its timing report."""

    realization = sequence.resolve()
    ok, report = realization.check_timing()
    normalized = tuple(str(item) for item in report)
    return realization, bool(ok), normalized


def _save_sequence_plot(
    *,
    name: str,
    sequence: Any,
    realization: Any,
    output_path: Path,
    title: str,
    repetitions: int,
    dpi: int,
    width: float,
    height: float,
    gradient_scale: str | None = None,
    one_tr: bool = True,
    show: bool = False,
) -> None:
    """Render and save one timing diagram with common figure dimensions."""

    plot_kwargs: dict[str, Any] = {
        "realization": realization,
        "repetitions": repetitions,
        "title": title,
        "one_tr": one_tr,
        "figsize": (width, height),
        "dpi": dpi,
        "debug": False,
    }
    if gradient_scale is not None:
        plot_kwargs["gradient_scale"] = gradient_scale

    # seq.plot() already displays the figure internally in interactive
    # environments. Assigning the returned object prevents Jupyter-style
    # duplicate display and gives us a stable object to save.
    figure = sequence.plot(**plot_kwargs)
    if figure is None:
        figure = plt.gcf()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    if show:
        plt.show()
    plt.close(figure)


def _build_fid(args: argparse.Namespace) -> tuple[Any, tuple[str, ...]]:
    overrides = {
        "average": args.fid_averages,
        "num_samples": args.fid_num_samples,
        "echo_time": args.fid_te,
        "repetition_time": args.fid_tr,
    }
    system = demo_FID.define_system_limits()
    protocol = demo_FID.define_protocol(overrides=overrides)
    sequence, te_fill, tr_fill = demo_FID.build_sequence(system, protocol)
    notes = (
        f"averages={args.fid_averages}",
        f"samples={args.fid_num_samples}",
        f"TE={args.fid_te * 1e3:.3f} ms",
        f"TR={args.fid_tr * 1e3:.3f} ms",
        f"TE fill={te_fill.eval(sequence) * 1e3:.3f} ms",
        f"TR fill={tr_fill.eval(sequence) * 1e3:.3f} ms",
    )
    return sequence, notes


def _build_gre(args: argparse.Namespace) -> tuple[Any, tuple[str, ...]]:
    overrides = {
        "n_y": args.gre_n_y,
        "fov": args.gre_fov,
        "echo_time": args.gre_te,
        "repetition_time": args.gre_tr,
        "rf_spoiling_increment": args.gre_rf_spoiling_increment,
    }
    system = demo_GRE.define_system_limits()
    protocol = demo_GRE.define_protocol(
        orientation=args.orientation,
        overrides=overrides,
    )
    sequence, te_fill, tr_fill = demo_GRE.build_sequence(
        system,
        protocol,
        orientation=args.orientation,
    )
    notes = (
        f"orientation={args.orientation}",
        f"n_y={args.gre_n_y}",
        f"FOV={args.gre_fov * 1e3:.1f} mm",
        f"TE={args.gre_te * 1e3:.3f} ms",
        f"TR={args.gre_tr * 1e3:.3f} ms",
        f"RF spoiling={args.gre_rf_spoiling_increment:.1f} deg",
        f"TE fill={te_fill.eval(sequence) * 1e3:.3f} ms",
        f"TR fill={tr_fill.eval(sequence) * 1e3:.3f} ms",
    )
    return sequence, notes


def _build_epi(args: argparse.Namespace) -> tuple[Any, tuple[str, ...]]:
    overrides = {
        "n_y": args.epi_n_y,
        "echo_train_length": args.epi_etl,
        "fov_phase": args.epi_fov_phase,
        "readout_duration": args.epi_readout_duration,
        "echo_time": args.epi_te,
        "repetition_time": args.epi_tr,
    }
    system = demo_EPI.define_system_limits()
    protocol = demo_EPI.define_protocol(
        orientation=args.orientation,
        overrides=overrides,
    )
    sequence = demo_EPI.build_sequence(
        system,
        protocol,
        orientation=args.orientation,
    )
    notes = (
        f"orientation={args.orientation}",
        f"n_y={args.epi_n_y}",
        f"ETL={args.epi_etl}",
        f"FOV phase={args.epi_fov_phase * 1e3:.1f} mm",
        f"readout={args.epi_readout_duration * 1e6:.1f} us",
        f"TE={args.epi_te * 1e3:.3f} ms",
        f"TR={args.epi_tr * 1e3:.3f} ms",
    )
    return sequence, notes


def _build_tse(args: argparse.Namespace) -> tuple[Any, tuple[str, ...]]:
    overrides = {
        "n_y": args.tse_n_y,
        "echo_train_length": args.tse_etl,
        "phase_encode_order": args.tse_phase_order,
        "TE_effective": args.tse_te_effective,
        "repetition_time": args.tse_tr,
        "refocusing_flip_angle": args.tse_refocusing_flip,
    }
    system = demo_TSE.define_system_limits()
    protocol = demo_TSE.define_protocol(
        orientation=args.orientation,
        overrides=overrides,
    )
    sequence, first_refocus_fill, pre_readout_fill, tr_fill = (
        demo_TSE.build_sequence(
            system,
            protocol,
            orientation=args.orientation,
        )
    )
    notes = (
        f"orientation={args.orientation}",
        f"n_y={args.tse_n_y}",
        f"ETL={args.tse_etl}",
        f"order={args.tse_phase_order}",
        f"effective TE={args.tse_te_effective * 1e3:.3f} ms",
        f"TR={args.tse_tr * 1e3:.3f} ms",
        f"refocusing={args.tse_refocusing_flip:.1f} deg",
        f"first-refocus fill={first_refocus_fill.eval(sequence) * 1e3:.3f} ms",
        f"pre-readout fill={pre_readout_fill.eval(sequence) * 1e3:.3f} ms",
        f"TR fill={tr_fill.eval(sequence) * 1e3:.3f} ms",
    )
    return sequence, notes


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Return a broadly available font with a safe fallback."""

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_composite(
    results: list[PlotResult],
    output_path: Path,
    *,
    gap: int = 42,
    outer_margin: int = 45,
    label_height: int = 72,
) -> None:
    """Arrange the four sequence PNGs into a consistent 2 × 2 layout."""

    if len(results) != 4:
        raise ValueError("The manuscript composite requires exactly four panels.")

    loaded = [Image.open(result.png_path).convert("RGB") for result in results]

    # Normalize all panels to the same pixel dimensions without changing their
    # aspect ratio. White padding is preferable to distorting waveform timing.
    target_width = max(image.width for image in loaded)
    target_height = max(image.height for image in loaded)

    normalized: list[Image.Image] = []
    for image in loaded:
        scale = min(target_width / image.width, target_height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        panel = Image.new("RGB", (target_width, target_height), "white")
        panel.paste(
            resized,
            (
                (target_width - resized.width) // 2,
                (target_height - resized.height) // 2,
            ),
        )
        normalized.append(panel)

    canvas_width = outer_margin * 2 + target_width * 2 + gap
    canvas_height = (
        outer_margin * 2
        + (label_height + target_height) * 2
        + gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    panel_font = _font(36, bold=True)

    labels = ("a)  FID", "b)  Spoiled GRE", "c)  Segmented EPI", "d)  TSE")

    for index, (panel, label) in enumerate(zip(normalized, labels)):
        row, col = divmod(index, 2)
        x = outer_margin + col * (target_width + gap)
        y = outer_margin + row * (label_height + target_height + gap)
        draw.text((x, y), label, font=panel_font, fill="black")
        canvas.paste(panel, (x, y + label_height))

    canvas.save(output_path, dpi=(300, 300))


def _write_summary(results: list[PlotResult], path: Path) -> None:
    lines = [
        "PyPulseq-Star reference-sequence timing-diagram run",
        "=" * 58,
        "",
    ]
    for result in results:
        lines.append(f"{result.name}: {'PASS' if result.timing_ok else 'FAIL'}")
        for note in result.summary_lines:
            lines.append(f"  {note}")
        if result.timing_report:
            lines.append("  timing report:")
            lines.extend(f"    - {item}" for item in result.timing_report)
        lines.append(f"  figure: {result.png_path}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, and plot the FID, GRE, EPI, and TSE "
            "reference sequences in a consistent manuscript layout."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/reference_sequences"),
        help="Directory for individual and composite PNGs.",
    )
    parser.add_argument(
        "--orientation",
        choices=("axial", "coronal", "sagittal"),
        default="sagittal",
        help="Common orientation for GRE, EPI, and TSE.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--panel-width", type=float, default=12.0)
    parser.add_argument("--panel-height", type=float, default=7.4)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively in addition to saving it.",
    )
    parser.add_argument(
        "--no-composite",
        action="store_true",
        help="Generate individual panels only.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Generate all figures but do not exit nonzero after timing failures.",
    )

    fid = parser.add_argument_group("FID")
    fid.add_argument("--fid-averages", type=int, default=4)
    fid.add_argument("--fid-num-samples", type=int, default=2048)
    fid.add_argument("--fid-te", type=float, default=20e-3)
    fid.add_argument("--fid-tr", type=float, default=100e-3)

    gre = parser.add_argument_group("GRE")
    gre.add_argument("--gre-n-y", type=int, default=64)
    gre.add_argument("--gre-fov", type=float, default=256e-3)
    gre.add_argument("--gre-te", type=float, default=5e-3)
    gre.add_argument("--gre-tr", type=float, default=12e-3)
    gre.add_argument("--gre-rf-spoiling-increment", type=float, default=117.0)

    epi = parser.add_argument_group("EPI")
    epi.add_argument("--epi-n-y", type=int, default=64)
    epi.add_argument("--epi-etl", type=int, default=16)
    epi.add_argument("--epi-fov-phase", type=float, default=220e-3)
    epi.add_argument("--epi-readout-duration", type=float, default=640e-6)
    epi.add_argument("--epi-te", type=float, default=30e-3)
    epi.add_argument("--epi-tr", type=float, default=50e-3)

    tse = parser.add_argument_group("TSE")
    tse.add_argument("--tse-n-y", type=int, default=64)
    tse.add_argument("--tse-etl", type=int, default=8)
    tse.add_argument(
        "--tse-phase-order",
        choices=("linear", "centric", "reverse"),
        default="linear",
    )
    tse.add_argument("--tse-te-effective", type=float, default=80e-3)
    tse.add_argument("--tse-tr", type=float, default=200e-3)
    tse.add_argument("--tse-refocusing-flip", type=float, default=180.0)

    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: tuple[
        tuple[str, Callable[[argparse.Namespace], tuple[Any, tuple[str, ...]]], str, int, str | None],
        ...,
    ] = (
        ("FID", _build_fid, "PyPulseq-Star FID", 1, None),
        (
            "Spoiled GRE",
            _build_gre,
            f"PyPulseq-Star spoiled GRE ({args.orientation})",
            1,
            "mt_per_m",
        ),
        (
            "Segmented EPI",
            _build_epi,
            f"PyPulseq-Star segmented EPI ({args.orientation})",
            1,
            "mt_per_m",
        ),
        (
            "TSE",
            _build_tse,
            f"PyPulseq-Star TSE ({args.orientation})",
            1,
            "mt_per_m",
        ),
    )

    results: list[PlotResult] = []
    failures: list[str] = []

    for display_name, builder, title, repetitions, gradient_scale in jobs:
        slug = (
            display_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        png_path = output_dir / f"{slug}_timing.png"

        print(f"\n{'=' * 72}")
        print(f"Building {display_name}")
        print("=" * 72)

        # Keep console output readable while retaining all diagnostics from
        # sequence resolution and plotting.
        sequence, notes = builder(args)
        realization, timing_ok, timing_report = _resolve_and_check(sequence)

        print(f"Timing check: {'PASS' if timing_ok else 'FAIL'}")
        for item in timing_report:
            print(f"  {item}")

        _save_sequence_plot(
            name=display_name,
            sequence=sequence,
            realization=realization,
            output_path=png_path,
            title=title,
            repetitions=repetitions,
            dpi=args.dpi,
            width=args.panel_width,
            height=args.panel_height,
            gradient_scale=gradient_scale,
            one_tr=True,
            show=args.show,
        )
        print(f"Wrote timing diagram: {png_path}")

        result = PlotResult(
            name=display_name,
            sequence=sequence,
            realization=realization,
            timing_ok=timing_ok,
            timing_report=timing_report,
            png_path=png_path,
            summary_lines=notes,
        )
        results.append(result)

        if not timing_ok:
            failures.append(display_name)

    summary_path = output_dir / "reference_sequence_summary.txt"
    _write_summary(results, summary_path)
    print(f"\nWrote run summary: {summary_path}")

    if not args.no_composite:
        composite_path = output_dir / "reference_sequence_capabilities.png"
        _make_composite(results, composite_path)
        print(f"Wrote 2 × 2 composite: {composite_path}")

    if failures:
        print("\nTiming failures:")
        for name in failures:
            print(f"  - {name}")
        if not args.continue_on_failure:
            return 1

    print("\nAll requested reference-sequence figures were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
