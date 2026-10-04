from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_rf_waveform(
    rf_waveform,
    *,
    title: str | None = None,
    show_phase: bool = True,
):
    """Plot an RF waveform and its effective timing reference."""

    signal = np.asarray(rf_waveform.signal)
    time_s = np.arange(signal.size) * rf_waveform.dwell_s

    if show_phase:
        fig, axes = plt.subplots(2, 1, sharex=True)
        ax_mag, ax_phase = axes

        ax_phase.plot(
            1e3 * time_s,
            np.unwrap(np.angle(signal)),
        )
        ax_phase.axvline(
            1e3 * rf_waveform.time_ref_s,
            linestyle="--",
        )
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.set_xlabel("Time (ms)")
    else:
        fig, ax_mag = plt.subplots()

    ax_mag.plot(
        1e3 * time_s,
        np.abs(signal),
    )
    ax_mag.axvline(
        1e3 * rf_waveform.time_ref_s,
        linestyle="--",
    )

    ax_mag.set_ylabel("RF magnitude (a.u.)")

    if not show_phase:
        ax_mag.set_xlabel("Time (ms)")

    if title is None:
        title = f"RF waveform: {rf_waveform.source}"

    fig.suptitle(title)
    fig.tight_layout()

    return fig