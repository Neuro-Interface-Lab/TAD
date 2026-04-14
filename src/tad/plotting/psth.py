from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tad.metrics.psth import PSTHResult


def plot_psth_lines(
    psth: PSTHResult,
    *,
    ax: Optional[plt.Axes] = None,
    channels: Optional[Sequence[int]] = None,
    kind: str = "rate",
    show: bool = True,
) -> plt.Axes:
    """
    Plot PSTH as lines for selected channels.

    Parameters
    ----------
    psth
        PSTHResult from `compute_psth`.
    ax
        Matplotlib axes.
    channels
        Row indices into psth.counts/psth.rate_hz. If None, plots up to first 10.
    kind
        "rate" (Hz) or "counts" (counts/bin/stim).
    show
        If True, calls plt.show().

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    Y = psth.rate_hz if kind == "rate" else psth.counts
    if channels is None:
        idx = np.arange(min(10, Y.shape[0]))
    else:
        idx = np.asarray(list(channels), dtype=int)

    for i in idx:
        label = str(psth.channels[i]) if i < len(psth.channels) else str(i)
        ax.plot(psth.t, Y[i, :], label=label)

    ax.axvline(0.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time from stimulus (s)")
    ax.set_ylabel("Rate (Hz)" if kind == "rate" else "Counts / bin / stim")
    ax.set_title("PSTH (per channel)")
    if idx.size <= 12:
        ax.legend(ncol=2, fontsize=8)

    if show:
        plt.show()

    return ax


def plot_psth_heatmap(
    psth: PSTHResult,
    *,
    ax: Optional[plt.Axes] = None,
    kind: str = "rate",
    robust: bool = True,
    show_colorbar: bool = True,
    show: bool = True,
) -> plt.Axes:
    """
    Plot PSTH as a channel × time heatmap.

    Parameters
    ----------
    psth
        PSTHResult from `compute_psth`.
    ax
        Matplotlib axes.
    kind
        "rate" or "counts".
    robust
        If True, clip color range to 2–98 percentiles for readability.
    show_colorbar
        Add a colorbar.
    show
        If True, calls plt.show().

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    Y = psth.rate_hz if kind == "rate" else psth.counts
    vmin = vmax = None
    if robust and Y.size:
        vmin, vmax = np.percentile(Y, [2, 98])

    # Use edges for extent
    x0 = float(psth.bin_edges[0])
    x1 = float(psth.bin_edges[-1])

    im = ax.imshow(
        Y,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, 0, Y.shape[0]],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvline(0.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time from stimulus (s)")
    ax.set_ylabel("Channel (row index)")
    ax.set_title("PSTH heatmap")

    if show_colorbar:
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Rate (Hz)" if kind == "rate" else "Counts / bin / stim")

    if show:
        plt.show()

    return ax
