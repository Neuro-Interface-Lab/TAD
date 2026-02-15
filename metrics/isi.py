from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from tad.raster import Raster, ChannelId
from tad.metrics.utils import _select_channels, _infer_window, pooled_spike_times


@dataclass(frozen=True)
class ISIResult:
    """
    Result container for ISI extraction.

    Parameters
    ----------
    isi
        ISIs. If mode="per_channel", this is a dict {channel_id: isi_array}.
        If mode="pooled", this is a single 1D array.
    mode
        "per_channel" or "pooled".
    tstart, tstop
        Window used for extraction.
    channels
        Channel IDs included in extraction (order).
    """
    isi: Union[Dict[ChannelId, np.ndarray], np.ndarray]
    mode: Literal["per_channel", "pooled"]
    tstart: float
    tstop: float
    channels: List[ChannelId]


def isi(
    r: Raster,
    *,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    mode: Literal["per_channel", "pooled"] = "per_channel",
) -> ISIResult:
    """
    Compute inter-spike intervals (ISIs) from a raster.

    Parameters
    ----------
    r
        Input raster.
    channels
        Channels to include. If None, uses all channels.
    tstart, tstop
        Optional analysis window. If None, inferred from data.
    inclusive_stop
        Right-bound policy: [tstart,tstop) if False, [tstart,tstop] if True.
    mode
        - "per_channel": compute ISIs separately for each channel
        - "pooled": pool spike times across channels (global ordering) then compute ISIs

    Returns
    -------
    ISIResult
        ISIs and metadata.

    Notes
    -----
    - Per-channel ISIs are computed from each channel’s sorted spike times.
    - Pooled ISIs correspond to the IEI concept used in the references when pooling
      across electrodes (global event train).
    """
    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    if mode == "pooled":
        t = pooled_spike_times(
            r,
            channels=ch_list,
            tstart=tstart_f,
            tstop=tstop_f,
            inclusive_stop=inclusive_stop,
        )
        if t.size < 2:
            out = np.asarray([], dtype=r.dtype)
        else:
            out = np.diff(t).astype(r.dtype, copy=False)

        return ISIResult(
            isi=out,
            mode="pooled",
            tstart=tstart_f,
            tstop=tstop_f,
            channels=list(ch_list),
        )

    if mode == "per_channel":
        out: Dict[ChannelId, np.ndarray] = {}
        for ch in ch_list:
            arr = r.events[ch]
            if arr.size < 2:
                out[ch] = np.asarray([], dtype=r.dtype)
                continue

            left = np.searchsorted(arr, tstart_f, side="left")
            right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
            w = arr[left:right]

            if w.size < 2:
                out[ch] = np.asarray([], dtype=r.dtype)
            else:
                out[ch] = np.diff(w).astype(r.dtype, copy=False)

        return ISIResult(
            isi=out,
            mode="per_channel",
            tstart=tstart_f,
            tstop=tstop_f,
            channels=list(ch_list),
        )

    raise ValueError(f"mode must be 'per_channel' or 'pooled', got {mode!r}.")


def isih(
    isi_values: np.ndarray,
    *,
    bins: Union[int, np.ndarray] = 50,
    density: bool = False,
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an ISI histogram (ISIH), optionally in log domain.

    Parameters
    ----------
    isi_values
        1D array of ISIs (>0).
    bins
        Number of bins or explicit bin edges (passed to np.histogram).
    density
        If True, return a probability density (area = 1). If False, return counts.
    log
        If True, histogram log10(ISI) rather than ISI.

    Returns
    -------
    centers : ndarray
        Bin centers (in ISI units if log=False; in log10 units if log=True).
    hist : ndarray
        Histogram values (counts or density).
    """
    x = np.asarray(isi_values, dtype=np.float64).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    if log:
        x = np.log10(x)

    hist, edges = np.histogram(x, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist.astype(np.float64, copy=False)
