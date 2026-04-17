from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from tad.metrics import BurstDetectionResult
from tad.raster import ChannelId


def plot_bursts_spans(
    bursts: BurstDetectionResult,
    *,
    ax: Optional[plt.Axes] = None,
    channels: Optional[Sequence[ChannelId]] = None,
    ymin: float = 0.0,
    ymax: float = 1.0,
    alpha: float = 0.15,
    show: bool = False,
) -> plt.Axes:
    """
    Overlay burst intervals as shaded spans on an axis.

    Parameters
    ----------
    bursts
        Output of `tad.metrics.bursts.detect_bursts`.
    ax
        Matplotlib Axes to draw on. If None, creates a new figure/axes.
    channels
        Optional subset of channels to plot. If None, plot all channels present
        in `bursts.channels`.
    ymin, ymax
        Vertical span fraction in axis coordinates, passed to `ax.axvspan`.
        Default (0,1) shades full vertical extent.
    alpha
        Span transparency.
    show
        If True, calls plt.show(). Default False so you can compose plots.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with spans added.

    Notes
    -----
    - This helper intentionally draws spans in axis coordinates so it can be used
      both on raster plots and on rate plots.
    - Coloring is uniform by default for clarity; if you later want per-channel
      colors, we can add an option.
    """
    if ax is None:
        _, ax = plt.subplots()

    if channels is None:
        ch_list = bursts.channels
    else:
        ch_list = list(channels)

    for ch in ch_list:
        cr = bursts.per_channel.get(ch, None)
        if cr is None:
            continue
        for b in cr.bursts:
            ax.axvspan(b.start, b.end, ymin=ymin, ymax=ymax, alpha=alpha)

    if show:
        plt.show()

    return ax


def plot_bursts_on_raster(
    r,
    bursts: BurstDetectionResult,
    *,
    ax: Optional[plt.Axes] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    channels: Optional[Sequence[ChannelId]] = None,
    alpha: float = 0.15,
    show: bool = False,
) -> plt.Axes:
    """
    Convenience: plot a raster then overlay burst spans.

    Parameters
    ----------
    r
        Raster instance (expects `.plot(...)` method).
    bursts
        BurstDetectionResult.
    ax
        Axes to draw on. If None, creates a new figure/axes.
    tstart, tstop
        Time window forwarded to Raster.plot.
    channels
        Channels to overlay bursts for (burst detection channels, not raster plot channels).
        If None, overlays all channels present in bursts.
    alpha
        Span transparency.
    show
        If True, calls plt.show().

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing raster + burst overlays.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    r.plot(ax=ax, tstart=tstart, tstop=tstop, show=False)
    plot_bursts_spans(bursts, ax=ax, channels=channels, alpha=alpha, show=False)

    if show:
        plt.show()

    return ax
