"""Unified PyPulseq-Star test and coverage runner.

This replaces the former ``tests/run_tests.sh`` workflow.

Examples
--------
Run the complete suite::

    python tests/run_tests.py
    python tests/run_tests.py test

Run selected groups::

    python tests/run_tests.py unit
    python tests/run_tests.py integration
    python tests/run_tests.py sequences
    python tests/run_tests.py semantic
    python tests/run_tests.py smoke

Coverage reports::

    python tests/run_tests.py coverage
    python tests/run_tests.py html
    python tests/run_tests.py all --fail-under 75
    python tests/run_tests.py report

Pass additional arguments to pytest after ``--``::

    python tests/run_tests.py sequences -- -x -vv
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "tests"
PYTEST_CONFIG = TEST_ROOT / "pytest.ini"
COVERAGE_DIR = REPO_ROOT / ".coverage_data"
COVERAGE_FILE = COVERAGE_DIR / ".coverage"
HTML_DIR = REPO_ROOT / "htmlcov"
XML_REPORT = REPO_ROOT / "coverage.xml"
JSON_REPORT = REPO_ROOT / "coverage.json"

VALID_MODES = (
    "test",
    "unit",
    "integration",
    "sequences",
    "semantic",
    "smoke",
    "coverage",
    "html",
    "report",
    "clean",
    "all",
)


def _configure_environment() -> None:
    """Configure deterministic paths and a headless plotting backend."""
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["COVERAGE_FILE"] = str(COVERAGE_FILE)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _clean_reports() -> None:
    """Remove generated coverage data and reports."""
    shutil.rmtree(COVERAGE_DIR, ignore_errors=True)
    shutil.rmtree(HTML_DIR, ignore_errors=True)

    for report in (XML_REPORT, JSON_REPORT):
        report.unlink(missing_ok=True)


def _pytest_base_args() -> list[str]:
    return [
        "-c",
        str(PYTEST_CONFIG),
        str(TEST_ROOT),
        "-ra",
        "--strict-markers",
    ]


def _selection_args(mode: str) -> list[str]:
    """Return marker/path selection arguments for a test mode."""
    if mode == "unit":
        return ["-m", "unit"]
    if mode == "integration":
        return ["-m", "integration"]
    if mode == "sequences":
        return [
            str(TEST_ROOT / "sequences"),
            "-m",
            "sequence or semantic",
        ]
    if mode == "semantic":
        return ["-m", "semantic"]
    if mode == "smoke":
        return ["-m", "smoke"]
    return []


def _coverage_args(
    *,
    terminal: bool,
    html: bool,
    xml: bool,
    json_report: bool,
    fail_under: float | None,
) -> list[str]:
    args = [
        "--cov=pypulseq_star",
        "--cov-branch",
    ]

    if terminal:
        args.append("--cov-report=term-missing:skip-covered")
    else:
        args.append("--cov-report=")

    if html:
        args.append(f"--cov-report=html:{HTML_DIR}")
    if xml:
        args.append(f"--cov-report=xml:{XML_REPORT}")
    if json_report:
        args.append(f"--cov-report=json:{JSON_REPORT}")
    if fail_under is not None:
        args.append(f"--cov-fail-under={fail_under:g}")

    return args


def _run_pytest(
    mode: str,
    *,
    verbose: bool,
    fail_under: float,
    extra: Sequence[str],
) -> int:
    args = _pytest_base_args()

    # Avoid specifying TEST_ROOT twice for the sequence-folder mode.
    selection = _selection_args(mode)
    if mode == "sequences":
        args = [
            "-c",
            str(PYTEST_CONFIG),
            *selection,
            "-ra",
            "--strict-markers",
        ]
    else:
        args.extend(selection)

    if mode in {"coverage", "html", "all"}:
        args.extend(
            _coverage_args(
                terminal=True,
                html=mode in {"html", "all"},
                xml=mode in {"html", "all"},
                json_report=mode in {"html", "all"},
                fail_under=fail_under if mode == "all" else None,
            )
        )

    if verbose:
        args.append("-vv")

    args.extend(extra)
    return pytest.main(args)


def _run_coverage_command(arguments: Sequence[str]) -> int:
    """Run ``python -m coverage`` using the active interpreter."""
    command = [sys.executable, "-m", "coverage", *arguments]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


def _build_reports_from_existing_data() -> int:
    """Recreate terminal, HTML, XML, and JSON reports without rerunning tests."""
    if not COVERAGE_FILE.exists():
        print(
            f"No coverage data found at {COVERAGE_FILE}.\n"
            "Run `python tests/run_tests.py html` first.",
            file=sys.stderr,
        )
        return 2

    commands = (
        ["report", "--show-missing", "--skip-covered"],
        ["html", "-d", str(HTML_DIR)],
        ["xml", "-o", str(XML_REPORT)],
        ["json", "-o", str(JSON_REPORT)],
    )
    for command in commands:
        status = _run_coverage_command(command)
        if status != 0:
            return status

    _print_report_locations()
    return 0


def _print_report_locations() -> None:
    print()
    print(f"HTML coverage report: {HTML_DIR / 'index.html'}")
    print(f"Coverage XML:         {XML_REPORT}")
    print(f"Coverage JSON:        {JSON_REPORT}")
    print(f"Coverage data:        {COVERAGE_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PyPulseq-Star tests and generate coverage reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""\
            Examples:
              python tests/run_tests.py
              python tests/run_tests.py test
              python tests/run_tests.py sequences
              python tests/run_tests.py coverage
              python tests/run_tests.py html
              python tests/run_tests.py all --fail-under 75
              python tests/run_tests.py report
              python tests/run_tests.py clean
              python tests/run_tests.py sequences -- -x -vv

            HTML report:
              {HTML_DIR / "index.html"}
            """
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=VALID_MODES,
        default="test",
        help="Test or coverage mode. Default: test.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=75.0,
        help="Coverage threshold used by the 'all' mode. Default: 75.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbose pytest output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    extra: list[str] = []
    if "--" in raw_args:
        split = raw_args.index("--")
        extra = raw_args[split + 1 :]
        raw_args = raw_args[:split]

    args = build_parser().parse_args(raw_args)
    _configure_environment()

    if args.mode == "clean":
        _clean_reports()
        print("Removed coverage data and generated reports.")
        return 0

    if args.mode == "report":
        return _build_reports_from_existing_data()

    if args.mode == "all":
        _clean_reports()
        _configure_environment()

    status = _run_pytest(
        args.mode,
        verbose=args.verbose,
        fail_under=args.fail_under,
        extra=extra,
    )

    if args.mode in {"html", "all"}:
        if status == 0:
            _print_report_locations()
            if args.mode == "all":
                print(f"\nCoverage gate passed at {args.fail_under:g}%.")
        else:
            print(
                "\nTests or the coverage gate failed. Generated reports, when "
                "available, remain in the paths below:"
            )
            _print_report_locations()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
