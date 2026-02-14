from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from tad.raster import Raster, ChannelId


@dataclass(frozen=True)
class FiringRateCurveResult:
    """
    Result of a binned firing-rate curve computation.

    Parameters
    ----------
    t
        Bin times according to `time_mode` (centers by default), shape (n_bins,).
    fr_ch
        Per-channel firing rate per bin (Hz), shape (n_channels, n_bins).
    fr_pop
        Population firing rate per bin (Hz), i.e. sum over channels, shape (n_bins,).
    dt
        Bin width (seconds or your time unit).
    tstart
        Analysis window start (actual binning start).
    tstop
        Analysis window stop (actual binning stop = last edge).
    channels
        Channel IDs corresponding to fr_ch rows.
    bin_edges
        Bin edges used, shape (n_bins+1,).
    """
    t: np.ndarray
    fr_ch: np.ndarray
    fr_pop: np.ndarray
    dt: float
    tstart: float
    tstop: float
    channels: List[ChannelId]
    bin_edges: np.ndarray


def firing_rate_curve(
    r: Raster,
    dt: float,
    *,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    channels: Optional[Sequence[ChannelId]] = None,
    inclusive_stop: bool = False,
    time_mode: str = "center",
) -> FiringRateCurveResult:
    """
    Compute binned firing-rate curves from a Raster.

    This is the natural “time-resolved” extension of FR = N/T from the references:
    counts in bins divided by bin width.

    Parameters
    ----------
    r
        Input raster.
    dt
        Bin width.
    tstart, tstop
        Optional time window. If None, inferred by Raster.bin_counts.
    channels
        Optional subset/order of channels.
    inclusive_stop
        If True, include events exactly at tstop.
    time_mode
        How to represent time points:
        - "left": left bin edges (starts), shape (n_bins,)
        - "center": bin centers, shape (n_bins,) (default)

    Returns
    -------
    FiringRateCurveResult
        Contains per-channel and population firing-rate curves.

    Raises
    ------
    ValueError
        If `time_mode` is not recognized.
    """
    edges, counts, ch_list = r.bin_counts(
        dt,
        tstart=tstart,
        tstop=tstop,
        channels=channels,
        inclusive_stop=inclusive_stop,
    )

    dt_f = float(dt)
    fr_ch = counts.astype(np.float64, copy=False) / dt_f                 # (n_ch, n_bins)
    fr_pop = counts.sum(axis=0).astype(np.float64, copy=False) / dt_f    # (n_bins,)

    if time_mode == "left":
        t = edges[:-1].astype(np.float64, copy=False)
    elif time_mode == "center":
        t = (0.5 * (edges[:-1] + edges[1:])).astype(np.float64, copy=False)
    else:
        raise ValueError(f"time_mode must be 'left' or 'center', got {time_mode!r}.")

    tstart_used = float(edges[0]) if edges.size else 0.0
    tstop_used = float(edges[-1]) if edges.size else 0.0

    return FiringRateCurveResult(
        t=t,
        fr_ch=fr_ch,
        fr_pop=fr_pop,
        dt=dt_f,
        tstart=tstart_used,
        tstop=tstop_used,
        channels=list(ch_list),
        bin_edges=edges,
    )
