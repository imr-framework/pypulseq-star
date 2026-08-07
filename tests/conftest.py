"""Repository-wide pytest fixtures.

This file intentionally lives directly under ``tests/`` so the fixtures and
non-interactive plotting configuration apply to unit, integration, and
sequence-level tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

import pypulseq_star as ppstar


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def system() -> ppstar.Opts:
    return ppstar.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        rf_raster_time=2e-6,
        grad_raster_time=10e-6,
    )


@pytest.fixture
def basic_sequence(system: ppstar.Opts) -> Any:
    return ppstar.Sequence(
        system=system,
        name="test_sequence",
        parameters={
            "Name": "test_sequence",
            "sequence_type": "TEST",
            "TE": 5e-3,
            "TR": 20e-3,
            "averages": 2,
        },
        debug=False,
    )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def clean_debug_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PYPULSEQ_STAR_DEBUG",
        "PYPULSEQ_STAR_SEQUENCE_DEBUG",
        "PYPULSEQ_STAR_RELATIONSHIP_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)
