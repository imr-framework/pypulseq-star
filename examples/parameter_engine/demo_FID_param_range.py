#!/usr/bin/env python3
"""Parameter-engine demo for the Phase 2 FID example.

This example intentionally lives outside ``demo_FID.py`` so the base sequence
demo stays focused on sequence construction/export.  The goal here is to
exercise the sequence-agnostic parameter engine:

    protocol values
        -> relationship resolution
        -> generic constraints
        -> feasibility report
        -> selected-parameter feasible ranges

Run from the repository root:

    python examples/parameter_engine/demo_FID_param_range.py

Useful stress tests:

    python examples/parameter_engine/demo_FID_param_range.py --bad-te
    python examples/parameter_engine/demo_FID_param_range.py --bad-tr
    python examples/parameter_engine/demo_FID_param_range.py --dense

This script does not write Pulseq or gammaSTAR files.  It prints terminal
diagnostics only.  The same outputs can later be exported to gammaSTAR as
dashboard/relationship metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the neighboring examples/demo_FID.py without requiring examples to be
# a Python package.
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
EXAMPLES_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
DEMO_FID_PATH = EXAMPLES_DIR / "demo_FID.py"

# Make local src/ imports work when running directly from a source checkout.
SRC_DIR = REPO_ROOT / "src"
for path in (str(SRC_DIR), str(EXAMPLES_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_demo_fid_module():
    spec = importlib.util.spec_from_file_location("demo_FID_for_param_engine", DEMO_FID_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {DEMO_FID_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo_FID = _load_demo_fid_module()

try:
    from pypulseq_star.feasibility import evaluate_feasibility, feasible_range
except Exception as exc:  # pragma: no cover - user-facing setup message
    raise SystemExit(
        "Could not import pypulseq_star.feasibility. "
        "Please install/copy the feasibility foundation patch first.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_seconds(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if abs(v) >= 1.0:
        return f"{v:.6g} s"
    if abs(v) >= 1e-3:
        return f"{v * 1e3:.6g} ms"
    if abs(v) >= 1e-6:
        return f"{v * 1e6:.6g} us"
    return f"{v:.6g} s"


def _fmt_value(value: Any, *, unit: str | None = None) -> str:
    if unit == "s":
        return _fmt_seconds(value)
    if unit == "Hz":
        try:
            return f"{float(value):.6g} Hz"
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _print_report(report: Any, *, max_diagnostics: int = 12) -> None:
    status = "OK" if getattr(report, "ok", False) else "FAIL"
    errors = len(getattr(report, "errors", []) or [])
    warnings = len(getattr(report, "warnings", []) or [])
    print(f"Feasibility status: {status}  ({errors} error(s), {warnings} warning(s))")

    diagnostics = list(getattr(report, "diagnostics", []) or [])
    if not diagnostics:
        print("Diagnostics: none")
        return

    print("Diagnostics:")
    for diag in diagnostics[:max_diagnostics]:
        code = getattr(diag, "code", "DIAGNOSTIC")
        severity = getattr(diag, "severity", "info")
        target = getattr(diag, "target", None)
        message = getattr(diag, "message", "")
        limit = getattr(diag, "limiting_constraint", None)
        prefix = f"  - [{severity}] {code}"
        if target:
            prefix += f" @ {target}"
        if limit:
            prefix += f" ({limit})"
        print(prefix)
        if message:
            print(f"      {message}")
    if len(diagnostics) > max_diagnostics:
        print(f"  ... {len(diagnostics) - max_diagnostics} additional diagnostic(s)")


def _print_bounds(bound: Any, *, unit: str | None = None) -> None:
    print(bound.summary_text())
    intervals = getattr(bound, "allowed_intervals", []) or []
    if not intervals:
        print("  Allowed interval(s): <none found>")
    else:
        print("  Allowed interval(s):")
        for interval in intervals:
            start = _fmt_value(getattr(interval, "start", None), unit=unit)
            stop = _fmt_value(getattr(interval, "stop", None), unit=unit)
            step = getattr(interval, "step", None)
            if step is None:
                print(f"    - {start} to {stop}")
            else:
                print(f"    - {start} to {stop}  step {step}")

    limiting = list(getattr(bound, "limiting_diagnostics", []) or [])
    if limiting:
        print("  Example limiting diagnostic(s):")
        for diag in limiting[:3]:
            print(f"    - {getattr(diag, 'code', 'DIAGNOSTIC')}: {getattr(diag, 'message', '')}")


# ---------------------------------------------------------------------------
# Parameter-engine demo
# ---------------------------------------------------------------------------

def build_fid_sequence(*, overrides: dict[str, Any] | None = None, debug: bool = False):
    """Build the FID sequence, optionally modifying protocol defaults first."""

    system = demo_FID.define_system_limits()
    protocol = demo_FID.define_protocol()

    if overrides:
        # Keep this intentionally simple: this demo edits the protocol defaults
        # before sequence construction.  The feasible_range() function below
        # then uses protocol_overrides internally to test candidate values.
        for key, value in overrides.items():
            canonical = key
            canonical_name = getattr(protocol, "canonical_name", None)
            if callable(canonical_name):
                try:
                    canonical = canonical_name(key)
                except Exception:
                    canonical = key
            protocol.parameters[canonical] = value

    seq, te_fill_duration, tr_fill_duration = demo_FID.build_sequence(
        system,
        protocol,
        debug=debug,
    )
    return seq, te_fill_duration, tr_fill_duration


def print_current_fid_state(seq: Any, te_fill_duration: Any, tr_fill_duration: Any) -> None:
    _print_section("Current FID protocol and symbolic fill checks")

    values = getattr(seq, "parameters", {}) or {}
    for key in (
        "flip_angle",
        "rf_duration",
        "echo_time",
        "repetition_time",
        "num_averages",
        "num_samples",
        "dwell",
    ):
        value = values.get(key, None)
        unit = "s" if key in {"rf_duration", "echo_time", "repetition_time", "dwell"} else None
        print(f"{key:>18s}: {_fmt_value(value, unit=unit)}")

    try:
        print(f"{'requested TE fill':>18s}: {_fmt_seconds(te_fill_duration.eval(seq))}")
    except Exception as exc:
        print(f"{'requested TE fill':>18s}: <could not evaluate: {exc}>")

    try:
        print(f"{'requested TR fill':>18s}: {_fmt_seconds(tr_fill_duration.eval(seq))}")
    except Exception as exc:
        print(f"{'requested TR fill':>18s}: <could not evaluate: {exc}>")


def run_current_feasibility(seq: Any) -> None:
    _print_section("Current-protocol feasibility")
    report = evaluate_feasibility(seq)
    _print_report(report)


def run_stress_tests(seq: Any) -> None:
    _print_section("Deliberately bad candidate protocols")

    cases = [
        ("too-short echo_time", {"echo_time": 100e-6}),
        ("too-short repetition_time", {"repetition_time": 1e-3}),
        ("non-positive num_samples", {"num_samples": 0}),
        ("negative dwell", {"dwell": -10e-6}),
    ]

    for label, overrides in cases:
        print()
        print(f"Candidate: {label}  overrides={overrides}")
        report = evaluate_feasibility(seq, protocol_overrides=overrides)
        _print_report(report, max_diagnostics=5)


def run_parameter_ranges(seq: Any, *, dense: bool = False) -> None:
    _print_section("Selected-parameter feasible ranges")

    steps = 96 if dense else 48

    requests = [
        {
            "parameter": "echo_time",
            "declared_min": 0.0,
            "declared_max": 80e-3,
            "steps": steps,
            "integer": False,
            "unit": "s",
        },
        {
            "parameter": "repetition_time",
            "declared_min": 0.0,
            "declared_max": 2.0,
            "steps": steps,
            "integer": False,
            "unit": "s",
        },
        {
            "parameter": "num_samples",
            "declared_min": 1,
            "declared_max": 4096,
            "steps": steps,
            "integer": True,
            "unit": None,
        },
        {
            "parameter": "dwell",
            "declared_min": 1e-6,
            "declared_max": 100e-6,
            "steps": steps,
            "integer": False,
            "unit": "s",
        },
        {
            "parameter": "num_averages",
            "declared_min": 1,
            "declared_max": 32,
            "steps": 32,
            "integer": True,
            "unit": None,
        },
    ]

    for request in requests:
        print()
        print(f"Parameter: {request['parameter']}")
        bound = feasible_range(
            seq,
            request["parameter"],
            declared_min=request["declared_min"],
            declared_max=request["declared_max"],
            steps=request["steps"],
            integer=request["integer"],
        )
        _print_bounds(bound, unit=request["unit"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal-only parameter-engine/range demo for demo_FID.py"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use more samples for continuous feasible-range sweeps.",
    )
    parser.add_argument(
        "--bad-te",
        action="store_true",
        help="Build the sequence with an intentionally too-short TE.",
    )
    parser.add_argument(
        "--bad-tr",
        action="store_true",
        help="Build the sequence with an intentionally too-short TR.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable SeqStar debug mode while building the FID sequence.",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.bad_te:
        overrides["echo_time"] = 100e-6
    if args.bad_tr:
        overrides["repetition_time"] = 1e-3

    seq, te_fill_duration, tr_fill_duration = build_fid_sequence(
        overrides=overrides,
        debug=args.debug,
    )

    print_current_fid_state(seq, te_fill_duration, tr_fill_duration)
    run_current_feasibility(seq)
    run_parameter_ranges(seq, dense=args.dense)
    run_stress_tests(seq)

    print()
    print("Done. This example intentionally does not write .seq or gammaSTAR JSON.")


if __name__ == "__main__":
    main()
