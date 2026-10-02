"""Unified PyPulseq-Star validation runner.

By default, running this script with no arguments performs the complete local
pre-push validation gate:

    Ruff
      ->
    GitHub Actions workflow validation
      ->
    full pytest suite
      ->
    branch-aware coverage gate
      ->
    wheel / sdist build

Examples
--------
Run the complete local validation gate::

    python tests/run_tests.py

Equivalent explicit form::

    python tests/run_tests.py ci

Run selected groups::

    python tests/run_tests.py test
    python tests/run_tests.py unit
    python tests/run_tests.py integration
    python tests/run_tests.py sequences
    python tests/run_tests.py semantic
    python tests/run_tests.py smoke

Run static checks::

    python tests/run_tests.py lint
    python tests/run_tests.py workflow

Coverage reports::

    python tests/run_tests.py coverage
    python tests/run_tests.py html
    python tests/run_tests.py all --fail-under 75
    python tests/run_tests.py report

Build package artifacts::

    python tests/run_tests.py build

Pass additional arguments to pytest after ``--``::

    python tests/run_tests.py sequences -- -x -vv

Notes
-----
GitHub Actions workflow validation uses ``actionlint`` when available.
If actionlint is not installed, that check is reported as skipped rather
than failing the local run.

On macOS::

    brew install actionlint
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import pytest

# =============================================================================
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_ROOT = REPO_ROOT / "tests"
PYTEST_CONFIG = TEST_ROOT / "pytest.ini"

GITHUB_ROOT = REPO_ROOT / ".github"
WORKFLOW_DIR = GITHUB_ROOT / "workflows"

COVERAGE_DIR = REPO_ROOT / ".coverage_data"
COVERAGE_FILE = COVERAGE_DIR / ".coverage"

HTML_DIR = REPO_ROOT / "htmlcov"
XML_REPORT = REPO_ROOT / "coverage.xml"
JSON_REPORT = REPO_ROOT / "coverage.json"

DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"


# =============================================================================
# Supported modes
# =============================================================================

VALID_MODES = (
    "ci",
    "test",
    "unit",
    "integration",
    "sequences",
    "semantic",
    "smoke",
    "lint",
    "workflow",
    "coverage",
    "html",
    "report",
    "build",
    "clean",
    "all",
)


# =============================================================================
# Environment
# =============================================================================


def _configure_environment() -> None:
    """Configure deterministic paths and a headless plotting backend."""

    COVERAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["COVERAGE_FILE"] = str(
        COVERAGE_FILE
    )

    os.environ.setdefault(
        "MPLBACKEND",
        "Agg",
    )


# =============================================================================
# Cleaning
# =============================================================================


def _clean_reports() -> None:
    """Remove generated coverage data and reports."""

    shutil.rmtree(
        COVERAGE_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        HTML_DIR,
        ignore_errors=True,
    )

    for report in (
        XML_REPORT,
        JSON_REPORT,
    ):
        report.unlink(
            missing_ok=True
        )


def _clean_build_artifacts() -> None:
    """Remove generated package-build output."""

    shutil.rmtree(
        DIST_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        BUILD_DIR,
        ignore_errors=True,
    )


# =============================================================================
# Console helpers
# =============================================================================


def _print_header(
    title: str,
) -> None:
    """Print a visible validation-step header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_command(
    command: Sequence[str],
) -> None:
    """Print a command in copy/paste-friendly form."""

    print(
        " ".join(
            str(part)
            for part in command
        )
    )

    print()


# =============================================================================
# Generic subprocess runner
# =============================================================================


def _run_command(
    command: Sequence[str],
    *,
    label: str,
) -> int:
    """Run one repository-level validation command."""

    _print_header(
        label
    )

    _print_command(
        command
    )

    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
    )

    return completed.returncode


# =============================================================================
# Pytest configuration
# =============================================================================


def _pytest_base_args() -> list[str]:
    """Return common pytest arguments."""

    return [
        "-c",
        str(PYTEST_CONFIG),
        str(TEST_ROOT),
        "-ra",
        "--strict-markers",
    ]


def _selection_args(
    mode: str,
) -> list[str]:
    """Return marker/path selection arguments for a test mode."""

    if mode == "unit":
        return [
            "-m",
            "unit",
        ]

    if mode == "integration":
        return [
            "-m",
            "integration",
        ]

    if mode == "sequences":
        return [
            str(
                TEST_ROOT / "sequences"
            ),
            "-m",
            "sequence or semantic",
        ]

    if mode == "semantic":
        return [
            "-m",
            "semantic",
        ]

    if mode == "smoke":
        return [
            "-m",
            "smoke",
        ]

    return []


# =============================================================================
# Coverage configuration
# =============================================================================


def _coverage_args(
    *,
    terminal: bool,
    html: bool,
    xml: bool,
    json_report: bool,
    fail_under: float | None,
) -> list[str]:
    """Return pytest-cov arguments."""

    args = [
        "--cov=pypulseq_star",
        "--cov-branch",
    ]

    if terminal:
        args.append(
            "--cov-report=term-missing:skip-covered"
        )
    else:
        args.append(
            "--cov-report="
        )

    if html:
        args.append(
            f"--cov-report=html:{HTML_DIR}"
        )

    if xml:
        args.append(
            f"--cov-report=xml:{XML_REPORT}"
        )

    if json_report:
        args.append(
            f"--cov-report=json:{JSON_REPORT}"
        )

    if fail_under is not None:
        args.append(
            f"--cov-fail-under={fail_under:g}"
        )

    return args


# =============================================================================
# Pytest runner
# =============================================================================


def _run_pytest(
    mode: str,
    *,
    verbose: bool,
    fail_under: float,
    extra: Sequence[str],
) -> int:
    """Run pytest for the requested mode."""

    args = _pytest_base_args()

    selection = _selection_args(
        mode
    )

    if mode == "sequences":
        args = [
            "-c",
            str(PYTEST_CONFIG),
            *selection,
            "-ra",
            "--strict-markers",
        ]
    else:
        args.extend(
            selection
        )

    if mode in {
        "coverage",
        "html",
        "all",
    }:
        args.extend(
            _coverage_args(
                terminal=True,
                html=mode in {
                    "html",
                    "all",
                },
                xml=mode in {
                    "html",
                    "all",
                },
                json_report=mode in {
                    "html",
                    "all",
                },
                fail_under=(
                    fail_under
                    if mode == "all"
                    else None
                ),
            )
        )

    if verbose:
        args.append(
            "-vv"
        )

    args.extend(
        extra
    )

    _print_header(
        f"pytest: {mode}"
    )

    print(
        "pytest "
        + " ".join(
            args
        )
    )

    print()

    return pytest.main(
        args
    )


# =============================================================================
# Ruff
# =============================================================================


def _run_ruff() -> int:
    """Run the same Ruff gate used by GitHub Actions."""

    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "src",
        "tests",
        "examples",
        "--select",
        "E4,E7,E9,F,I",
        "--ignore",
        "E501",
    ]

    return _run_command(
        command,
        label="Ruff lint check",
    )


# =============================================================================
# GitHub Actions workflow validation
# =============================================================================


def _workflow_files() -> list[Path]:
    """Return repository GitHub Actions workflow files."""

    if not WORKFLOW_DIR.exists():
        return []

    return sorted(
        [
            *WORKFLOW_DIR.glob(
                "*.yml"
            ),
            *WORKFLOW_DIR.glob(
                "*.yaml"
            ),
        ]
    )


def _run_actionlint() -> int:
    """Validate GitHub Actions workflow files."""

    workflow_files = _workflow_files()

    _print_header(
        "GitHub Actions workflow validation"
    )

    if not workflow_files:
        print(
            "No workflow files found under "
            f"{WORKFLOW_DIR}."
        )
        return 0

    print(
        "Workflow files:"
    )

    for path in workflow_files:
        print(
            f"  - {path.relative_to(REPO_ROOT)}"
        )

    print()

    executable = shutil.which(
        "actionlint"
    )

    if executable is None:
        print(
            "WARNING: actionlint is not installed."
        )
        print(
            "GitHub Actions workflow validation was skipped."
        )
        print()
        print(
            "Install on macOS with:"
        )
        print()
        print(
            "    brew install actionlint"
        )
        print()

        return 0

    command = [
        executable,
        *[
            str(path)
            for path in workflow_files
        ],
    ]

    _print_command(
        command
    )

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    )

    return completed.returncode


# =============================================================================
# Coverage report utilities
# =============================================================================


def _run_coverage_command(
    arguments: Sequence[str],
) -> int:
    """Run ``python -m coverage`` using the active interpreter."""

    command = [
        sys.executable,
        "-m",
        "coverage",
        *arguments,
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    )

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
        [
            "report",
            "--show-missing",
            "--skip-covered",
        ],
        [
            "html",
            "-d",
            str(HTML_DIR),
        ],
        [
            "xml",
            "-o",
            str(XML_REPORT),
        ],
        [
            "json",
            "-o",
            str(JSON_REPORT),
        ],
    )

    for command in commands:
        status = _run_coverage_command(
            command
        )

        if status != 0:
            return status

    _print_report_locations()

    return 0


def _print_report_locations() -> None:
    """Print coverage output locations."""

    print()

    print(
        f"HTML coverage report: "
        f"{HTML_DIR / 'index.html'}"
    )

    print(
        f"Coverage XML:         "
        f"{XML_REPORT}"
    )

    print(
        f"Coverage JSON:        "
        f"{JSON_REPORT}"
    )

    print(
        f"Coverage data:        "
        f"{COVERAGE_FILE}"
    )


# =============================================================================
# Package build validation
# =============================================================================


def _run_build() -> int:
    """Build wheel and source distribution."""

    _clean_build_artifacts()

    command = [
        sys.executable,
        "-m",
        "build",
    ]

    return _run_command(
        command,
        label="Package build",
    )


# =============================================================================
# Complete local CI gate
# =============================================================================


def _run_ci(
    *,
    fail_under: float,
    verbose: bool,
    extra: Sequence[str],
) -> int:
    """Run the complete local pre-push validation gate."""

    _print_header(
        "PyPulseq-Star complete local validation"
    )

    print(
        "Running:"
    )
    print(
        "  1. Ruff"
    )
    print(
        "  2. GitHub Actions workflow validation"
    )
    print(
        "  3. Full pytest suite"
    )
    print(
        "  4. Branch-aware coverage gate"
    )
    print(
        "  5. Package build"
    )
    print()

    # -------------------------------------------------------------------------
    # Ruff
    # -------------------------------------------------------------------------

    status = _run_ruff()

    if status != 0:
        print(
            "\nVALIDATION FAILED: Ruff"
        )
        return status

    # -------------------------------------------------------------------------
    # GitHub Actions workflows
    # -------------------------------------------------------------------------

    status = _run_actionlint()

    if status != 0:
        print(
            "\nVALIDATION FAILED: GitHub Actions workflow validation"
        )
        return status

    # -------------------------------------------------------------------------
    # Tests + coverage
    # -------------------------------------------------------------------------

    _clean_reports()
    _configure_environment()

    status = _run_pytest(
        "all",
        verbose=verbose,
        fail_under=fail_under,
        extra=extra,
    )

    if status != 0:
        print(
            "\nVALIDATION FAILED: tests or coverage gate"
        )
        return status

    # -------------------------------------------------------------------------
    # Package build
    # -------------------------------------------------------------------------

    status = _run_build()

    if status != 0:
        print(
            "\nVALIDATION FAILED: package build"
        )
        return status

    # -------------------------------------------------------------------------
    # Success
    # -------------------------------------------------------------------------

    _print_header(
        "ALL LOCAL VALIDATION CHECKS PASSED"
    )

    print(
        "Ruff:               PASS"
    )

    print(
        "Workflow check:     PASS / SKIPPED if actionlint unavailable"
    )

    print(
        "Tests:              PASS"
    )

    print(
        f"Coverage threshold: PASS "
        f"(>= {fail_under:g}%)"
    )

    print(
        "Package build:      PASS"
    )

    _print_report_locations()

    return 0


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run PyPulseq-Star validation checks. "
            "With no mode specified, the complete local CI gate is run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""\
            Complete validation:
              python tests/run_tests.py
              python tests/run_tests.py ci

            Selected checks:
              python tests/run_tests.py test
              python tests/run_tests.py unit
              python tests/run_tests.py integration
              python tests/run_tests.py sequences
              python tests/run_tests.py semantic
              python tests/run_tests.py smoke
              python tests/run_tests.py lint
              python tests/run_tests.py workflow
              python tests/run_tests.py build

            Coverage:
              python tests/run_tests.py coverage
              python tests/run_tests.py html
              python tests/run_tests.py all --fail-under 75
              python tests/run_tests.py report

            Cleanup:
              python tests/run_tests.py clean

            Forward pytest arguments:
              python tests/run_tests.py sequences -- -x -vv

            HTML report:
              {HTML_DIR / "index.html"}

            Recommended before pushing:
              python tests/run_tests.py
            """
        ),
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=VALID_MODES,
        default="ci",
        help=(
            "Validation mode. "
            "Default: ci (complete local validation)."
        ),
    )

    parser.add_argument(
        "--fail-under",
        type=float,
        default=65.0,
        help=(
            "Coverage threshold used by "
            "'all' and default CI validation. "
            "Default: 65."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Use verbose pytest output."
        ),
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run requested validation mode."""

    raw_args = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    # -------------------------------------------------------------------------
    # Forward arguments following "--" to pytest.
    # -------------------------------------------------------------------------

    extra: list[str] = []

    if "--" in raw_args:
        split = raw_args.index(
            "--"
        )

        extra = raw_args[
            split + 1 :
        ]

        raw_args = raw_args[
            :split
        ]

    args = build_parser().parse_args(
        raw_args
    )

    _configure_environment()

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    if args.mode == "clean":
        _clean_reports()
        _clean_build_artifacts()

        print(
            "Removed coverage data, reports, "
            "and package-build artifacts."
        )

        return 0

    # -------------------------------------------------------------------------
    # Existing coverage report
    # -------------------------------------------------------------------------

    if args.mode == "report":
        return _build_reports_from_existing_data()

    # -------------------------------------------------------------------------
    # Ruff
    # -------------------------------------------------------------------------

    if args.mode == "lint":
        return _run_ruff()

    # -------------------------------------------------------------------------
    # Workflow validation
    # -------------------------------------------------------------------------

    if args.mode == "workflow":
        return _run_actionlint()

    # -------------------------------------------------------------------------
    # Package build
    # -------------------------------------------------------------------------

    if args.mode == "build":
        return _run_build()

    # -------------------------------------------------------------------------
    # Complete local validation
    # -------------------------------------------------------------------------

    if args.mode == "ci":
        return _run_ci(
            fail_under=args.fail_under,
            verbose=args.verbose,
            extra=extra,
        )

    # -------------------------------------------------------------------------
    # Coverage/all setup
    # -------------------------------------------------------------------------

    if args.mode == "all":
        _clean_reports()
        _configure_environment()

    # -------------------------------------------------------------------------
    # Pytest / coverage modes
    # -------------------------------------------------------------------------

    status = _run_pytest(
        args.mode,
        verbose=args.verbose,
        fail_under=args.fail_under,
        extra=extra,
    )

    # -------------------------------------------------------------------------
    # Coverage-report information
    # -------------------------------------------------------------------------

    if args.mode in {
        "html",
        "all",
    }:
        if status == 0:
            _print_report_locations()

            if args.mode == "all":
                print(
                    f"\nCoverage gate passed at "
                    f"{args.fail_under:g}%."
                )

        else:
            print(
                "\nTests or the coverage gate failed. "
                "Generated reports, when available, "
                "remain in the paths below:"
            )

            _print_report_locations()

    return status


if __name__ == "__main__":
    raise SystemExit(
        main()
    )