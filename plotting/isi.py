from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from tad.metrics.burst import LogISIThresholdDiagnostics


def plot_logisih_threshold(
    diag: LogISIThresholdDiagnostics,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot log-ISIH histogram/smoothed curve and the selected threshold.

    Parameters
    ----------
    diag
        Diagnostics from choose_isi_threshold_logisih.
    ax
        Matplotlib axes.
    show
        Whether to call plt.show().

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    if diag.centers_log10.size == 0:
        ax.set_title(f"log-ISIH: {diag.method} (no data)")
        if show:
            plt.show()
        return ax

    ax.plot(diag.centers_log10, diag.hist, label="hist", alpha=0.5)
    ax.plot(diag.centers_log10, diag.hist_smooth, label="smooth")

    if diag.peak1_idx >= 0:
        ax.axvline(diag.centers_log10[diag.peak1_idx], linestyle="--", alpha=0.6)
    if diag.peak2_idx >= 0:
        ax.axvline(diag.centers_log10[diag.peak2_idx], linestyle="--", alpha=0.6)
    if diag.valley_idx >= 0:
        ax.axvline(diag.centers_log10[diag.valley_idx], linestyle="-", alpha=0.9)

    if np.isfinite(diag.isi_th):
        ax.set_title(f"log-ISIH threshold: ISI_th={diag.isi_th:.4g}s ({diag.method})")
    else:
        ax.set_title(f"log-ISIH threshold: ISI_th=NaN ({diag.method})")

    ax.set_xlabel("log10(ISI)")
    ax.set_ylabel("count" if diag.method else "value")
    ax.legend()

    if show:
        plt.show()

    return ax
