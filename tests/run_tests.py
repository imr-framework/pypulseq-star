"""Run the complete PyPulseq-Star test suite without typing a pytest command.

Examples
--------
Run the normal suite:

    python run_tests.py

Run fast unit tests only:

    python run_tests.py --unit

Run integration tests only:

    python run_tests.py --integration

Run with coverage:

    python run_tests.py --coverage

Pass additional pytest arguments after ``--``:

    python run_tests.py --coverage -- -x -vv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def build_pytest_args(args: argparse.Namespace, extra: list[str]) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    test_root = repo_root / "tests"

    pytest_args = [str(test_root), "-ra"]

    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.smoke:
        pytest_args.extend(["-m", "smoke"])

    if args.coverage:
        pytest_args.extend(
            [
                "--cov=pypulseq_star",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--cov-report=xml",
            ]
        )

    if args.verbose:
        pytest_args.append("-vv")

    pytest_args.extend(extra)
    return pytest_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PyPulseq-Star test suite.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unit", action="store_true", help="Run unit tests only.")
    group.add_argument("--integration", action="store_true", help="Run integration tests only.")
    group.add_argument("--smoke", action="store_true", help="Run smoke tests only.")
    parser.add_argument("--coverage", action="store_true", help="Collect package coverage.")
    parser.add_argument("--verbose", action="store_true", help="Use verbose pytest output.")

    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        extra = argv[split + 1 :]
        argv = argv[:split]

    args = parser.parse_args(argv)
    return pytest.main(build_pytest_args(args, extra))


if __name__ == "__main__":
    raise SystemExit(main())
