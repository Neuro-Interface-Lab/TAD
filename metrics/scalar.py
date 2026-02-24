from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from tad.raster import Raster, ChannelId
from tad.metrics.utils import _select_channels, _infer_window, pooled_spike_times


def spike_count(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    per_channel: bool = True,
) -> Union[int, np.ndarray]:
    """
    Count spikes in a window.

    This is the base quantity used by firing rate metrics in the references
    (FR = N / T).

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels.
    tstart, tstop
        Optional time window. If None, inferred from data.
    inclusive_stop
        If False, use [tstart, tstop); if True, use [tstart, tstop].
    per_channel
        If True, return an array of counts per channel; otherwise return total count.

    Returns
    -------
    counts : ndarray or int
        Spike counts per channel (shape (n_channels,)) or pooled total count.
    """
    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    counts = np.zeros(len(ch_list), dtype=np.int64)
    for i, ch in enumerate(ch_list):
        arr = r.events[ch]
        if arr.size == 0:
            continue
        left = np.searchsorted(arr, tstart_f, side="left")
        right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
        counts[i] = right - left

    if per_channel:
        return counts
    return int(counts.sum())


def firing_rate(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    per_channel: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute firing rate FR = N / T, as in the references.

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels.
    tstart, tstop
        Optional time window. If None, inferred from data.
    inclusive_stop
        If False, use [tstart, tstop); if True, use [tstart, tstop].
    per_channel
        If True, return FR per channel; otherwise return pooled FR.

    Returns
    -------
    fr : ndarray or float
        Firing rate per channel (Hz) or pooled rate (Hz).

    Raises
    ------
    ValueError
        If window duration is non-positive.
    """
    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)
    T = tstop_f - tstart_f
    if T <= 0.0:
        raise ValueError(f"Window duration must be positive, got {T}.")

    counts = spike_count(
        r,
        channels=ch_list,
        tstart=tstart_f,
        tstop=tstop_f,
        inclusive_stop=inclusive_stop,
        per_channel=True,
    ).astype(np.float64, copy=False)

    fr = counts / T
    if per_channel:
        return fr
    return float(fr.sum())


def mean_firing_rate_across_channels(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    active_threshold_hz: Optional[float] = None,
) -> float:
    """
    Compute mean firing rate across channels (optionally over "active channels").

    This matches the typical workflow described in the references where per-channel
    rates are computed and an "active channel" criterion may be applied.

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include.
    tstart, tstop
        Optional time window.
    inclusive_stop
        If False, use [tstart, tstop); if True, use [tstart, tstop].
    active_threshold_hz
        If provided, only channels with FR >= threshold are included in the mean.

    Returns
    -------
    float
        Mean firing rate (Hz) over selected channels.

    Raises
    ------
    ValueError
        If no channels remain after thresholding.
    """
    fr = firing_rate(
        r,
        channels=channels,
        tstart=tstart,
        tstop=tstop,
        inclusive_stop=inclusive_stop,
        per_channel=True,
    )

    if active_threshold_hz is not None:
        thr = float(active_threshold_hz)
        mask = fr >= thr
        if not np.any(mask):
            raise ValueError("No channels remain after applying active_threshold_hz.")
        fr = fr[mask]

    return float(np.mean(fr)) if fr.size else 0.0


def mean_inter_event_interval(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
) -> float:
    """
    Compute the mean inter-event interval (IEI) of pooled activity.

    This corresponds to the pooled IEI analysis described in Pasquale et al. 2008
    (IEIs computed from the globally ordered spike times across electrodes).

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels.
    tstart, tstop
        Optional time window.
    inclusive_stop
        Window right-bound policy.

    Returns
    -------
    float
        Mean IEI. Returns np.nan if fewer than 2 pooled events exist.
    """
    t = pooled_spike_times(
        r,
        channels=channels,
        tstart=tstart,
        tstop=tstop,
        inclusive_stop=inclusive_stop,
    )
    if t.size < 2:
        return float("nan")
    return float(np.mean(np.diff(t)))


def percent_random_spiking(
    r: Raster,
    burst_intervals: Dict[ChannelId, np.ndarray],
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
) -> float:
    """
    Compute percentage of random spiking activity (%RS), i.e. fraction of spikes outside bursts.

    In Pasquale et al. 2008, "% random spiking activity" is defined as the fraction
    of spikes that do not belong to bursts.

    Parameters
    ----------
    r
        Input raster.
    burst_intervals
        Mapping channel -> array of burst intervals with shape (B, 2), where each row
        is (burst_start, burst_end). Intervals are assumed to be within the analysis window.
    channels
        Channels to include. If None, uses all raster channels.
    tstart, tstop
        Optional analysis window.
    inclusive_stop
        Window right-bound policy.

    Returns
    -------
    float
        Percentage (0..1) of spikes outside bursts. Returns np.nan if no spikes exist.

    Notes
    -----
    This function assumes bursts are already detected elsewhere (next step in your pipeline).
    """
    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    total_spikes = 0
    in_burst_spikes = 0

    for ch in ch_list:
        arr = r.events[ch]
        if arr.size == 0:
            continue

        left = np.searchsorted(arr, tstart_f, side="left")
        right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
        w = arr[left:right]
        n = int(w.size)
        if n == 0:
            continue

        total_spikes += n

        intervals = burst_intervals.get(ch, None)
        if intervals is None or len(intervals) == 0:
            continue

        intervals = np.asarray(intervals, dtype=float)
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError("burst_intervals[ch] must have shape (B, 2).")

        # Count spikes that fall inside any burst interval: [start, end]
        # For efficiency, bursts are assumed non-overlapping; if they overlap, this still works.
        for (bs, be) in intervals:
            bs_f = float(bs)
            be_f = float(be)
            if be_f < bs_f:
                continue
            li = np.searchsorted(w, bs_f, side="left")
            ri = np.searchsorted(w, be_f, side="right")
            in_burst_spikes += int(ri - li)

    if total_spikes == 0:
        return float("nan")

    outside = total_spikes - in_burst_spikes
    return float(outside / total_spikes)

    # --------------------------- Saving and Loading scalar metrics -------------------------

def save_scalar_metric(
        list_of_metrics: Dict[str, Union[int, float, List[float], List[int]]],
        fname: str,
        output_folder: str = "results/",

        ):
    """
    Function to save the list_of_metrics passed as an argument here to a json file

    Parameters
    ----------
    list_of_metrics
        A dictionary containing the metrics to be saved.
    fname
        The name of the file to save the metrics to (without extension).
    output_folder
        The folder to save the results in (default "results/").
    
    """

    import json
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        # creates .keepholder file for results folder:
        with open(os.path.join(output_folder, ".keepholder"), 'w') as f:
            f.write("This folder is used to store results. Do not delete.")
        
        
    with open(f"{output_folder}{fname}.json", 'w') as f:
        json.dump(list_of_metrics, f, indent=4)