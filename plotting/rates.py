from __future__ import annotations

from typing import Optional, Sequence, Tuple, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt

from tad.metrics.rates import FiringRateCurveResult


def plot_firing_rate_stack(
    fr: FiringRateCurveResult,
    *,
    ax: Optional[plt.Axes] = None,
    channels: Optional[Sequence[int]] = None,
    mode: str = "stack",
    spacing: Optional[float] = None,
    normalize: str = "none",
    color: str = "b",
    show: bool = False,
) -> plt.Axes:
    """
    Plot per-channel firing-rate curves in an EEG-like view.

    Parameters
    ----------
    fr
        Output of `tad.metrics.rates.firing_rate_curve`.
    ax
        Axes to draw on. If None, creates a new figure and axes.
    channels
        Optional subset of channels by row indices in `fr.fr_ch`.
        If None, plots all rows (in stored order).
    mode
        - "stack": stacked traces with vertical offsets (EEG-like)
        - "overlay": all traces overlaid without offsets (butterfly)
    spacing
        Vertical spacing between traces (only for mode="stack").
        If None, uses 1.2 * median(max(fr)-min(fr)) across selected channels.
    normalize
        - "none": plot raw Hz
        - "zscore": z-score each channel across time
        - "max": scale each channel by its own max (unit peak)
    show
        If True, calls plt.show().

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Notes
    -----
    This is a plotting helper; it does not change metric definitions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    X = fr.fr_ch
    t = fr.t

    if channels is None:
        idx = np.arange(X.shape[0])
    else:
        idx = np.asarray(list(channels), dtype=int)

    Y = X[idx, :].astype(np.float64, copy=True)

    # Optional normalization for visualization
    if normalize == "none":
        pass
    elif normalize == "zscore":
        mu = Y.mean(axis=1, keepdims=True)
        sd = Y.std(axis=1, keepdims=True)
        sd[sd == 0.0] = 1.0
        Y = (Y - mu) / sd
    elif normalize == "max":
        m = np.max(np.abs(Y), axis=1, keepdims=True)
        m[m == 0.0] = 1.0
        Y = Y / m
    else:
        raise ValueError(f"normalize must be 'none', 'zscore', or 'max', got {normalize!r}.")

    if mode == "overlay":
        for k in range(Y.shape[0]):
            ax.plot(t, Y[k, :])
        ax.set_ylabel("FR (Hz)" if normalize == "none" else "normalized FR")
        ax.set_title("Per-channel firing rates (overlay)")
    elif mode == "stack":
        # Compute a default spacing based on data scale
        if spacing is None:
            ptp = np.ptp(Y, axis=1)  # max-min per channel
            typical = float(np.median(ptp)) if ptp.size else 1.0
            spacing = 1.2 * typical if typical > 0 else 1.0

        offsets = spacing * np.arange(Y.shape[0])[::-1]  # top channel first
        for k in range(Y.shape[0]):
            ax.plot(t, Y[k, :] + offsets[k], color=color)

        ax.set_yticks(offsets)
        # Label with channel IDs if available; otherwise index
        labels = []
        for k in range(Y.shape[0]):
            ch_id = fr.channels[idx[k]] if idx[k] < len(fr.channels) else idx[k]
            labels.append(str(ch_id))
        ax.set_yticklabels(labels)
        ax.set_ylabel("Channel")
        ax.set_title("Per-channel firing rates (stacked)")
    else:
        raise ValueError(f"mode must be 'stack' or 'overlay', got {mode!r}.")

    ax.set_xlabel("Time")

    if show:
        plt.show()

    return ax

def plot_firing_rate_heatmap(
    fr: FiringRateCurveResult,
    *,
    ax: Optional[plt.Axes] = None,
    channels: Optional[Sequence[int]] = None,
    normalize: str = "none",
    robust: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    order: Optional[str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot per-channel firing rate as a channels × time heatmap.

    Parameters
    ----------
    fr
        Output of `tad.metrics.rates.firing_rate_curve`.
    ax
        Axes to draw on. If None, creates a new figure and axes.
    channels
        Optional subset of channels by row indices in `fr.fr_ch`.
        If None, uses all channels in stored order.
    normalize
        - "none": show raw FR (Hz)
        - "zscore": z-score each channel across time (highlights synchrony patterns)
        - "max": divide each channel by its max (unit peak per channel)
        - "log1p": show log(1 + FR) (useful when FR has rare large peaks)
    robust
        If True and vmin/vmax are not provided, choose color limits using percentiles
        (2nd, 98th) to avoid a few extreme bins dominating the colormap.
    vmin, vmax
        Explicit color limits. Overrides `robust` scaling.
    cmap
        Matplotlib colormap name.
    show_colorbar
        If True, add a colorbar.
    order
        Optional method to reorder channels by activity pattern similarity for better visualization.
        See `order_channels_by_activity` for available methods.
    show
        If True, calls plt.show().

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Notes
    -----
    This is a visualization helper. It does not alter the metric computation.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    X = fr.fr_ch
    t = fr.t

    if channels is None:
        idx = np.arange(X.shape[0])
    else:
        idx = np.asarray(list(channels), dtype=int)

    Y = X[idx, :].astype(np.float64, copy=True)

    # Normalization / transform for visualization
    if normalize == "none":
        label = "FR (Hz)"
    elif normalize == "zscore":
        mu = Y.mean(axis=1, keepdims=True)
        sd = Y.std(axis=1, keepdims=True)
        sd[sd == 0.0] = 1.0
        Y = (Y - mu) / sd
        label = "z-scored FR"
    elif normalize == "max":
        m = np.max(np.abs(Y), axis=1, keepdims=True)
        m[m == 0.0] = 1.0
        Y = Y / m
        label = "FR / max(FR)"
    elif normalize == "log1p":
        Y = np.log1p(Y)
        label = "log(1 + FR)"
    else:
        raise ValueError(
            "normalize must be one of {'none','zscore','max','log1p'}, "
            f"got {normalize!r}."
        )

    # OPTIONAL: reorder channels for visualization
    if order is not None:
        ord_idx = order_channels_by_activity(Y, method=order)
        Y = Y[ord_idx, :]
        idx = idx[ord_idx]  # keep channel labels consistent

    # Time bounds for imshow extent
    if t.size >= 2:
        dt_est = float(np.median(np.diff(t)))
    else:
        dt_est = float(fr.dt)
    t0 = float(t[0] - 0.5 * dt_est) if t.size else 0.0
    t1 = float(t[-1] + 0.5 * dt_est) if t.size else 0.0

    # Choose color limits
    if vmin is None or vmax is None:
        if robust and Y.size:
            lo, hi = np.nanpercentile(Y, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    im = ax.imshow(
        Y,
        aspect="auto",
        origin="lower",
        extent=[t0, t1, 0, Y.shape[0]],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    # Y ticks labeled with channel IDs (not indices), if available
    yticks = np.arange(Y.shape[0]) + 0.5
    ax.set_yticks(yticks)
    labels = []
    for k in range(Y.shape[0]):
        ch_id = fr.channels[idx[k]] if idx[k] < len(fr.channels) else idx[k]
        labels.append(str(ch_id))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    ax.set_title("Firing-rate heatmap")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)

    if show:
        plt.show()

    return ax

# ------------------------------------------------------------------------------------------------
# helper function to sort channels by activity level, for better visualization of “sync” patterns
# ------------------------------------------------------------------------------------------------
def order_channels_by_activity(
    Y: np.ndarray,
    *,
    method: Literal["population_corr", "pca1", "greedy_corr_chain"] = "population_corr",
) -> np.ndarray:
    """
    Compute an ordering of channels given a (n_channels, n_time) activity matrix.

    Parameters
    ----------
    Y
        Activity matrix (e.g., firing rate), shape (n_channels, n_time).
    method
        Ordering method:
        - "population_corr": sort by corr(channel, population_mean)
        - "pca1": sort by projection on first principal component
        - "greedy_corr_chain": build a chain where adjacent channels are highly correlated

    Returns
    -------
    order : ndarray, shape (n_channels,)
        Indices that reorder channels.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_ch = Y.shape[0]
    if n_ch <= 1:
        return np.arange(n_ch)

    # Z-score per channel for correlation-based methods
    Yc = Y - Y.mean(axis=1, keepdims=True)
    sd = Yc.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    Z = Yc / sd

    if method == "population_corr":
        pop = Z.mean(axis=0, keepdims=True)
        # corr = mean(Z * pop) since both are z-scored
        corr = (Z * pop).mean(axis=1)
        return np.argsort(corr)  # low->high; reverse if you prefer

    if method == "pca1":
        # first PC of channels x time: use SVD on Z
        # Z = U S Vt, first component scores are U[:,0]*S[0]
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        scores = U[:, 0] * S[0]
        return np.argsort(scores)

    if method == "greedy_corr_chain":
        # Compute correlation matrix quickly: corr = Z @ Z.T / T
        T = Z.shape[1]
        C = (Z @ Z.T) / float(T)
        np.fill_diagonal(C, -np.inf)

        # Start from the channel with largest correlation to population (stable start)
        pop = Z.mean(axis=0, keepdims=True)
        corr_pop = (Z * pop).mean(axis=1)
        start = int(np.argmax(corr_pop))

        order = [start]
        used = np.zeros(n_ch, dtype=bool)
        used[start] = True

        for _ in range(n_ch - 1):
            last = order[-1]
            # pick best unused correlated with last
            candidates = np.where(~used)[0]
            nxt = candidates[int(np.argmax(C[last, candidates]))]
            order.append(int(nxt))
            used[nxt] = True

        return np.asarray(order, dtype=int)

    raise ValueError(f"Unknown method {method!r}")