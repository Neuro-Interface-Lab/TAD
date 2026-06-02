from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np

from tad.raster import Raster, ChannelId
from tad.metrics.utils import _select_channels, _infer_window


@dataclass(frozen=True)
class PSTHResult:
    """
    Post-Stimulus Time Histogram (PSTH) result.

    The PSTH is computed by aligning spikes to each stimulus time, binning the
    relative spike times in a window [-t_pre, t_post), and averaging across stimuli.

    Parameters
    ----------
    t
        Bin times (centers if time_mode="center", else left edges), shape (n_bins,).
    bin_edges
        Bin edges in relative time (seconds), shape (n_bins+1,).
    counts
        Mean spike counts per bin per stimulus, shape (n_channels, n_bins).
    total_counts
        Total spike counts per bin across all stimuli, shape (n_channels, n_bins).
    rate_hz
        Mean firing rate per bin (Hz), i.e. counts / dt, shape (n_channels, n_bins).
    dt
        Bin width (seconds).
    t_pre
        Pre-stimulus window length (seconds).
    t_post
        Post-stimulus window length (seconds).
    stim_times_used
        Stimulus times actually used (after filtering), shape (n_stim_used,).
    channels
        Channel IDs corresponding to counts rows.
        If `mode == "pooled"`, this is ["pooled"] and counts/rate_hz have shape (1, n_bins).
    """
    t: np.ndarray
    bin_edges: np.ndarray
    counts: np.ndarray
    rate_hz: np.ndarray
    total_counts: np.ndarray
    dt: float
    t_pre: float
    t_post: float
    stim_times_used: np.ndarray
    channels: List[ChannelId]
    mode: Literal["per_channel", "pooled"]

    @property
    def population_counts(self) -> np.ndarray:
        """
        Population PSTH in counts/bin/stimulus (sum across channels).

        Returns
        -------
        ndarray, shape (n_bins,)
        """
        return self.counts.sum(axis=0)

    @property
    def population_total_counts(self) -> np.ndarray:
        """
        Population total spike counts per bin (sum across channels and stimuli).

        Returns
        -------
        ndarray, shape (n_bins,)
        """
        return self.total_counts.sum(axis=0)

    @property
    def population_rate_hz(self) -> np.ndarray:
        """
        Population PSTH in Hz (sum across channels).

        Returns
        -------
        ndarray, shape (n_bins,)
        """
        return self.rate_hz.sum(axis=0)


def compute_psth(
    r: Raster,
    stim_times: Sequence[float],
    *,
    dt: float,
    t_pre: float,
    t_post: float,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    inclusive_zero: bool = True,
    time_mode: str = "center",
    mode: Literal["per_channel", "pooled"] = "per_channel",
    dtype: np.dtype = np.dtype(np.float64),
) -> PSTHResult:
    """
    Compute PSTH aligned to stimulus onsets.

    This implements the paper definition:
        P(t) = (1/N) * sum_i ST(t - t_i^stim)
    in discrete form by binning relative spike times per stimulus and averaging.

    Parameters
    ----------
    r
        Input raster.
    stim_times
        Iterable of stimulus onset times (seconds).
    dt
        Bin width (seconds).
    t_pre
        Pre-stimulus window length (seconds). Window starts at -t_pre.
    t_post
        Post-stimulus window length (seconds). Window ends at +t_post (exclusive).
    channels
        Optional subset/order of channels to include.
    tstart, tstop
        Optional global time window for admissible stimuli. If None, inferred from raster.
        A stimulus is used only if its full PSTH window lies within [tstart, tstop)
        (or [tstart, tstop] if inclusive_stop=True).
    inclusive_stop
        Right-bound policy for the admissible stimulus window.
    inclusive_zero
        If True, spikes exactly at stimulus time (relative time 0) are included.
        This affects the left/right searchsorted side used for slicing.
    time_mode
        - "center": return bin centers in `t`
        - "left": return left bin edges in `t`
    mode
        - "per_channel": compute a separate PSTH for each selected channel
        - "pooled": sum counts and rates across all selected channels into one PSTH
    dtype
        Floating dtype for output arrays.

    Returns
    -------
    PSTHResult
        PSTH counts and rates per channel or pooled across channels.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    dt_f = float(dt)
    if dt_f <= 0:
        raise ValueError("dt must be > 0.")
    t_pre_f = float(t_pre)
    t_post_f = float(t_post)
    if t_pre_f < 0 or t_post_f <= 0:
        raise ValueError("t_pre must be >= 0 and t_post must be > 0.")
    if mode not in ("per_channel", "pooled"):
        raise ValueError("mode must be 'per_channel' or 'pooled'.")

    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    stim = np.asarray(list(stim_times), dtype=np.float64)
    stim = stim[np.isfinite(stim)]
    if stim.size == 0:
        # Return empty PSTH with correct shapes
        edges = np.arange(-t_pre_f, t_post_f + 1e-12, dt_f, dtype=np.float64)
        if edges[-1] < t_post_f:
            edges = np.append(edges, t_post_f)
        n_bins = edges.size - 1
        if mode == "pooled":
            counts = np.zeros((1, n_bins), dtype=dtype)
            total_counts = np.zeros((1, n_bins), dtype=dtype)
            channels_out = ["pooled"]
        else:
            counts = np.zeros((len(ch_list), n_bins), dtype=dtype)
            total_counts = np.zeros((len(ch_list), n_bins), dtype=dtype)
            channels_out = list(ch_list)
        rate = counts / dt_f
        t = 0.5 * (edges[:-1] + edges[1:]) if time_mode == "center" else edges[:-1].copy()
        return PSTHResult(
            t=t.astype(np.float64, copy=False),
            bin_edges=edges,
            counts=counts,
            total_counts=total_counts,
            rate_hz=rate,
            dt=dt_f,
            t_pre=t_pre_f,
            t_post=t_post_f,
            stim_times_used=np.asarray([], dtype=np.float64),
            channels=channels_out,
            mode=mode,
        )

    # Keep only stimuli whose whole window lies in [tstart, tstop) (or inclusive stop)
    win_left = stim - t_pre_f
    win_right = stim + t_post_f
    if inclusive_stop:
        keep = (win_left >= tstart_f) & (win_right <= tstop_f)
    else:
        keep = (win_left >= tstart_f) & (win_right < tstop_f)
    stim_used = stim[keep]
    stim_used.sort()

    # Bin edges in relative time
    edges = np.arange(-t_pre_f, t_post_f + 1e-12, dt_f, dtype=np.float64)
    if edges[-1] < t_post_f:
        edges = np.append(edges, t_post_f)
    n_bins = edges.size - 1

    counts_acc = np.zeros((len(ch_list), n_bins), dtype=np.float64)

    # Searchsorted sides for windowing
    # - left bound: include spikes at stim - t_pre always
    # - right bound: exclude stim + t_post unless inclusive_stop True in raster-window sense
    # - spike at stim time: controlled by inclusive_zero
    side_left = "left" if inclusive_zero else "right"
    side_right = "right" if inclusive_stop else "left"

    # Compute PSTH
    for s in stim_used:
        #print(s)
        abs_left = s - t_pre_f
        abs_right = s + t_post_f

        for ci, ch in enumerate(ch_list):
            arr = r.events[ch]
            if arr.size == 0:
                continue

            # Slice spikes in absolute time window, then shift to relative time
            left_idx = np.searchsorted(arr, abs_left, side=side_left)
            right_idx = np.searchsorted(arr, abs_right, side=side_right)
            w = arr[left_idx:right_idx]
            if w.size == 0:
                continue

            rel = w.astype(np.float64, copy=False) - s
            histogram = np.zeros(edges.size-1)
            for i, time in enumerate(edges[:-1]):
                histogram[i] = np.count_nonzero((rel > time) & (rel < time+dt_f))
            counts_acc[ci, :] += histogram

    n_stim = int(stim_used.size)
    if n_stim > 0:
        if mode == "pooled":
            # counts (Psth) = \Delta_t [ms] / N_stim [#] * total_spikes [spk/\Delta_t [ms]]
            counts = counts_acc.sum(axis=0, keepdims=True) / float(n_stim) * dt_f 
            total_counts = counts_acc.sum(axis=0, keepdims=True)
        else:
            counts = counts_acc / float(n_stim) * dt_f
            total_counts = counts_acc
    else:
        if mode == "pooled":
            counts = np.zeros((1, n_bins), dtype=np.float64)
            total_counts = np.zeros((1, n_bins), dtype=np.float64)
        else:
            counts = counts_acc  # all zeros
            total_counts = counts_acc

    counts = counts.astype(dtype, copy=False)
    total_counts = total_counts.astype(dtype, copy=False)
    rate = (counts / dt_f).astype(dtype, copy=False)

    if time_mode == "center":
        t = 0.5 * (edges[:-1] + edges[1:])
    elif time_mode == "left":
        t = edges[:-1].copy()
    else:
        raise ValueError(f"time_mode must be 'center' or 'left', got {time_mode!r}.")

    return PSTHResult(
        t=t.astype(np.float64, copy=False),
        bin_edges=edges,
        counts=counts,
        total_counts=total_counts,
        rate_hz=rate,
        dt=dt_f,
        t_pre=t_pre_f,
        t_post=t_post_f,
        stim_times_used=stim_used,
        channels=["pooled"] if mode == "pooled" else list(ch_list),
        mode=mode,
    )
