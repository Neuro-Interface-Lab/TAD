from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from tad.raster import Raster, ChannelId


def _select_channels(r: Raster, channels: Optional[Sequence[ChannelId]]) -> List[ChannelId]:
    """
    Select a channel list for computations.

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels in the raster.

    Returns
    -------
    list
        Channel IDs in the order that will be used.
    """
    ch_list = list(channels) if channels is not None else r.channels()
    for ch in ch_list:
        r._require_channel(ch)
    return ch_list


def _infer_window(
    r: Raster,
    ch_list: Sequence[ChannelId],
    tstart: Optional[float],
    tstop: Optional[float],
) -> Tuple[float, float]:
    """
    Infer (tstart, tstop) if missing, based on min/max event times.

    Parameters
    ----------
    r
        Input raster.
    ch_list
        Channels considered.
    tstart, tstop
        Optional window bounds.

    Returns
    -------
    tstart_f, tstop_f
        Window bounds as floats.

    Notes
    -----
    - If no events exist, defaults to (0.0, 0.0) unless a bound is provided.
    """
    mins = []
    maxs = []
    for ch in ch_list:
        arr = r.events[ch]
        if arr.size:
            mins.append(arr[0])
            maxs.append(arr[-1])

    if tstart is None:
        tstart_f = float(np.min(mins)) if mins else 0.0
    else:
        tstart_f = r._validate_time(tstart)

    if tstop is None:
        tstop_f = float(np.max(maxs)) if maxs else tstart_f
    else:
        tstop_f = r._validate_time(tstop)

    if tstop_f < tstart_f:
        raise ValueError(f"tstop ({tstop_f}) must be >= tstart ({tstart_f}).")

    return tstart_f, tstop_f


def pooled_spike_times(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
) -> np.ndarray:
    """
    Pool spike times across channels into a single sorted array.

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels.
    tstart, tstop
        Optional time window.
    inclusive_stop
        Window right-bound policy: [tstart,tstop) if False, [tstart,tstop] if True.

    Returns
    -------
    ndarray, shape (N,)
        Sorted pooled times in the requested window.
    """
    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    times = []
    for ch in ch_list:
        arr = r.events[ch]
        if arr.size == 0:
            continue
        left = np.searchsorted(arr, tstart_f, side="left")
        right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
        w = arr[left:right]
        if w.size:
            times.append(w)

    if not times:
        return np.asarray([], dtype=r.dtype)

    out = np.concatenate(times).astype(r.dtype, copy=False)
    out.sort()
    return out
