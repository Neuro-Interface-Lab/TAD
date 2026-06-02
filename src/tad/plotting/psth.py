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

    This function is intended for per-channel PSTH results. If a pooled
    PSTH is passed, it is forwarded to `plot_psth_lines_pooled`.

    Parameters
    ----------
    psth
        PSTHResult from `compute_psth`.
    ax
        Matplotlib axes.
    channels
        Row indices into psth.counts/psth.rate_hz. If None, plots up to first 10.
    kind
        "rate" (Hz), "counts" (counts/bin/stim), or "total_counts" (total counts/bin).
    show
        If True, calls plt.show().

    Returns
    -------
    Axes
    """
    if getattr(psth, "mode", None) == "pooled":
        return plot_psth_lines_pooled(psth, ax=ax, kind=kind, show=show)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    if kind == "rate":
        Y = psth.rate_hz
    elif kind == "counts":
        Y = psth.counts
    elif kind == "total_counts":
        Y = psth.total_counts
    else:
        raise ValueError("kind must be 'rate', 'counts', or 'total_counts'.")

    if channels is None:
        idx = np.arange(min(10, Y.shape[0]))
    else:
        idx = np.asarray(list(channels), dtype=int)

    for i in idx:
        label = str(psth.channels[i]) if i < len(psth.channels) else str(i)
        ax.plot(psth.t, Y[i, :], label=label)

    ax.axvline(0.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time from stimulus (s)")
    if kind == "rate":
        ylabel = "Rate (Hz)"
    elif kind == "counts":
        ylabel = "Counts / bin / stim"
    else:
        ylabel = "Total counts / bin"
    ax.set_ylabel(ylabel)
    ax.set_title("PSTH (per channel)")
    if idx.size <= 12:
        ax.legend(ncol=2, fontsize=8)

    if show:
        plt.show()

    return ax


def plot_psth_lines_pooled(
    psth: PSTHResult,
    *,
    ax: Optional[plt.Axes] = None,
    kind: str = "rate",
    label: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a pooled PSTH as a single line.

    Parameters
    ----------
    psth
        PSTHResult from `compute_psth` with mode="pooled".
    ax
        Matplotlib axes.
    kind
        "rate", "counts", or "total_counts".
    label
        Optional label for the pooled line.
    show
        If True, calls plt.show().

    Returns
    -------
    Axes
    """
    if getattr(psth, "mode", None) != "pooled":
        raise ValueError("plot_psth_lines_pooled() requires a pooled PSTHResult.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    if kind == "rate":
        Y = psth.rate_hz[0]
    elif kind == "counts":
        Y = psth.counts[0]
    elif kind == "total_counts":
        Y = psth.total_counts[0]
    else:
        raise ValueError("kind must be 'rate', 'counts', or 'total_counts'.")
    ax.plot(psth.t, Y, label=label or "pooled")
    ax.axvline(0.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time from stimulus (s)")
    if kind == "rate":
        ylabel = "Rate (Hz)"
    elif kind == "counts":
        ylabel = "Counts / bin / stim"
    else:
        ylabel = "Total counts / bin"
    ax.set_ylabel(ylabel)
    ax.set_title("PSTH (pooled)")
    if label is not None:
        ax.legend(ncol=1, fontsize=8)

    if show:
        plt.show()

    return ax

def plot_PSTH_and_raster(
        r: tad.metrics.raster.RasterResult,
        psth: tad.metrics.psth.PSTHResult,
        kind: str = "rate",
        show: bool = True,
    ) -> int:
        """
        Plot PSTH for each channel in an 8x8 grid layout.

        Parameters
        ----------
        r : tad.metrics.raster.RasterResult
            Per-channel raster result from `tad.metrics.raster.get_raster`.
        psth : tad.metrics.psth.PSTHResult
            Per-channel PSTH result from `tad.metrics.psth.compute_psth`.
        kind : {'rate', 'counts', 'total_counts'}, default='rate'
            Which PSTH quantity to plot.
        show : bool, default=True
            Show plot.

        Returns
        -------
        int
            Always returns 1 (backward compatible).

        Notes
        -----
        The layout follows the same 8x8 electrode grid used by
        :meth:`plot_traces_in_grid`. Channels that are not present in the PSTH
        result remain empty in the grid.
        """
        from matplotlib import gridspec

        if getattr(psth, "mode", None) != "per_channel":
            raise ValueError("plot_PSTH_in_grid requires a per-channel PSTHResult.")

        if kind == "rate":
            values = psth.rate_hz
        elif kind == "counts":
            values = psth.counts
        elif kind == "total_counts":
            values = psth.total_counts
        else:
            raise ValueError("kind must be 'rate', 'counts', or 'total_counts'.")

        def _normalize_channel_id(channel_id):
            if isinstance(channel_id, str):
                if channel_id.startswith("Ch"):
                    try:
                        return int(channel_id[2:])
                    except ValueError:
                        pass
                digits = "".join([c for c in channel_id if c.isdigit()])
                if digits:
                    return int(digits)
            return channel_id

        channel_to_index = {}
        for idx, ch in enumerate(psth.channels):
            channel_to_index[ch] = idx
            normalized = _normalize_channel_id(ch)
            if normalized != ch:
                channel_to_index[normalized] = idx

        
        
        for ch in psth.channels:
            fig = plt.figure(figsize = (16, 8))
            spec = gridspec.GridSpec(ncols=1, nrows = 4)

            ax_raster = fig.add_subplot(spec[0:3, 0])
            ax_psth = fig.add_subplot(spec[3, 0])
            #print(f"Plotting channel {ch}...")
            #print(channel_to_index)
            if ch in channel_to_index:
                psth_idx = channel_to_index[ch]
            else:
                normalized = _normalize_channel_id(ch)
                psth_idx = channel_to_index.get(normalized, None)
            if psth_idx is not None:
                y = values[psth_idx]
                ax_psth.plot(psth.t, y, lw=0.8, color="C0")
                ax_psth.axvline(0.0, linestyle="--", alpha=0.6)
                ax_psth.set_xlim(float(psth.t[0]), float(psth.t[-1]))
                if y.size:
                    y_min = 0
                    y_max = float(np.max(y))+0.2*float(np.max(y))
                    #print(y_min, y_max)
                    ax_psth.set_ylim(y_min, y_max)
                ax_psth.axis("off")

            #ax_psth.set_title(lab, fontsize=6)
            #print(ch)
            #print(r.events[ch])
            
            for i,t in enumerate(psth.stim_times_used):
                # find raster of that channel between stim_time and stim_time + t_post
                # and plot spikes in that window as scatter points in the raster plot
                raster_ch = r.events[ch]
                # find the indices of raster_ch that are between t and t + psth.t[-1]
                raster_idxs = np.where((raster_ch >= t) & (raster_ch <= t + float(psth.t[-1])))[0]
                raster = raster_ch[raster_idxs] - t
                if raster.size > 0:
                    ax_raster.scatter(raster, i* np.ones(len(raster)), s=1, color="black")
            ax_raster.set_xlabel("Time from stimulus (s)")
            ax_raster.set_ylabel("Stimulation event")
            ax_raster.set_title("Raster plot")
            ax_raster.axvline(0.0, linestyle="--", alpha=0.6)
            ax_raster.set_xlim(float(psth.t[0]), float(psth.t[-1]))
            #ax_raster.set_ylim(float(self.electrode_labels.min())-1, float(self.electrode_labels.max())+1)
            #ax_raster.set_yticks(self.electrode_labels)
        # ax_raster.set_yticklabels(self.electrode_labels, fontsize=6)
            # show the plot for each channel:
            ax_psth.set_title(f"Channel {ch}", fontsize=6)
            plt.tight_layout()
            plt.show()
        return 1


        if show:
            plt.show()
        return 1

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
        "rate", "counts", or "total_counts".
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

    if kind == "rate":
        Y = psth.rate_hz
    elif kind == "counts":
        Y = psth.counts
    elif kind == "total_counts":
        Y = psth.total_counts
    else:
        raise ValueError("kind must be 'rate', 'counts', or 'total_counts'.")
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
    if getattr(psth, "mode", None) == "pooled":
        ax.set_yticks([0.5])
        ax.set_yticklabels(["pooled"])
        ax.set_ylabel("Pooled PSTH")
    else:
        ax.set_ylabel("Channel (row index)")
    ax.set_title("PSTH heatmap (pooled)" if getattr(psth, "mode", None) == "pooled" else "PSTH heatmap")

    if show_colorbar:
        cb = plt.colorbar(im, ax=ax)
        if kind == "rate":
            cb_label = "Rate (Hz)"
        elif kind == "counts":
            cb_label = "Counts / bin / stim"
        else:
            cb_label = "Total counts / bin"
        cb.set_label(cb_label)

    if show:
        plt.show()

    return ax
