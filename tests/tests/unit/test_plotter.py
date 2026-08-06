"""Unit tests for noninteractive plotting and logical repetition."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from tests.helpers.builders import build_fid_sequence, build_gradient_sequence

pytestmark = pytest.mark.unit


def test_plot_runs_without_interactive_backend(system, monkeypatch) -> None:
    seq = build_fid_sequence(system)
    monkeypatch.setattr(plt, "show", lambda: None)

    seq.plot(title="FID test", debug=False)
    assert plt.get_fignums()


def test_plot_accepts_gradient_scale(system, monkeypatch) -> None:
    seq = build_gradient_sequence(system)
    monkeypatch.setattr(plt, "show", lambda: None)

    seq.plot(title="Gradient test", gradient_scale="mt_per_m", debug=False)
    assert plt.get_fignums()


def test_loop_plot_expands_representative_kernel(system, monkeypatch) -> None:
    seq = build_fid_sequence(system, averages=3, tr=20e-3)
    monkeypatch.setattr(plt, "show", lambda: None)

    seq.plot(time_range=(0.0, 60e-3), title="Repeated FID", debug=False)
    figure = plt.gcf()
    assert figure.axes


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_normalized_rf_uses_one_sequence_wide_scale() -> None:
    """Relative RF amplitudes must survive normalized plotting."""

    from pypulseq_star.plotting.plotter import _normalize_rendered_rf

    rendered = {
        "rf": [
            {"v": [0.0, 0.5, 0.0]},
            {"v": [0.0, 1.0, 0.0]},
        ]
    }

    _normalize_rendered_rf(rendered)

    assert max(rendered["rf"][0]["v"]) == pytest.approx(0.5)
    assert max(rendered["rf"][1]["v"]) == pytest.approx(1.0)
