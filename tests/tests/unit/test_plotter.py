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

def test_publication_mode_applies_manuscript_style(system, monkeypatch) -> None:
    """Publication mode should apply the stable manuscript-facing plot contract."""

    seq = build_gradient_sequence(system)
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = seq.plot(
        title="Publication gradient test",
        gradient_scale="mt_per_m",
        publication=True,
        show=False,
        debug=False,
    )

    assert len(figure.axes) == 5

    # Publication mode keeps the gradient channel name compact and moves the
    # shared unit out of the vertically stacked y-axis labels.
    ylabels = [axis.get_ylabel() for axis in figure.axes]
    assert "GX" in ylabels
    assert "GX (mT/m)" not in ylabels

    unit_annotations = [
        annotation
        for axis in figure.axes
        for annotation in axis.texts
        if annotation.get_text() == "mT/m"
    ]
    assert len(unit_annotations) == 1

    # Axis-number typography is part of the publication-mode contract.
    visible_tick_labels = [
        label
        for axis in figure.axes
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
        if label.get_visible()
    ]
    assert visible_tick_labels
    assert all(label.get_fontsize() == pytest.approx(15) for label in visible_tick_labels)


def test_publication_false_preserves_legacy_gradient_label(system, monkeypatch) -> None:
    """Disabling publication mode must preserve existing plot-label behavior."""

    seq = build_gradient_sequence(system)
    monkeypatch.setattr(plt, "show", lambda: None)

    figure = seq.plot(
        title="Legacy gradient test",
        gradient_scale="mt_per_m",
        publication=False,
        show=False,
        debug=False,
    )

    ylabels = [axis.get_ylabel() for axis in figure.axes]
    assert "GX (mT/m)" in ylabels

    unit_annotations = [
        annotation
        for axis in figure.axes
        for annotation in axis.texts
        if annotation.get_text() == "mT/m"
    ]
    assert unit_annotations == []

