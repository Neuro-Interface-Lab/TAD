from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from tad.raster import Raster, ChannelId


@dataclass(frozen=True)
class AvalancheResult:
    """
    Result of avalanche extraction on a raster.

    Parameters
    ----------
    sizes
        Avalanche sizes, one per avalanche. Definition depends on `size_definition`.
    lifetimes
        Avalanche lifetimes in number of bins (Δt units), one per avalanche.
    dt
        Bin width used to define avalanches.
    tstart
        Start time of analysis window used for binning.
    tstop
        Stop time of analysis window used for binning.
    channels
        Channel IDs included in the analysis (row order of internal matrices).
    size_definition
        Size definition used:
        - 1: sum over bins of (# active electrodes in bin)
        - 2: number of distinct electrodes active at least once in avalanche
    bin_edges
        Bin edges used, shape (n_bins+1,).
    active_bins
        Boolean array indicating which bins were active (at least one spike in any channel),
        shape (n_bins,).
    intervals_bins
        List of (start_bin, stop_bin) bin indices for each avalanche, where stop_bin is exclusive.
    """
    sizes: np.ndarray
    lifetimes: np.ndarray
    dt: float
    tstart: float
    tstop: float
    channels: List[ChannelId]
    size_definition: int
    bin_edges: np.ndarray
    active_bins: np.ndarray
    intervals_bins: List[Tuple[int, int]]


def extract_avalanches(
    r: Raster,
    dt: float,
    *,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    channels: Optional[Sequence[ChannelId]] = None,
    inclusive_stop: bool = False,
    size_definition: int = 1,
) -> AvalancheResult:
    """
    Extract neuronal avalanches (sizes and lifetimes) from a raster.

    This follows the definitions in the provided references:

    - Bin time into windows of width Δt.
    - A bin is "active" if at least one electrode fires ≥1 spike in that bin; "blank" otherwise.
    - An avalanche is a maximal contiguous sequence of active bins bounded by blank bins.
    - Lifetime = number of consecutive active bins.
    - Size:
        (1) sum over bins of number of active electrodes in that bin (multiple activations counted)
        (2) number of distinct electrodes that were active at least once during the avalanche

    Parameters
    ----------
    r
        Input raster.
    dt
        Bin width (Δt). Must be positive.
    tstart, tstop
        Optional analysis window. If None, inferred from the data in the selected channels.
        (Inference follows your Raster.bin_counts behavior.)
    channels
        Channels to include. If None, uses all channels in the raster.
    inclusive_stop
        Whether to include events at exactly tstop in binning window.
    size_definition
        1 or 2, as above.

    Returns
    -------
    AvalancheResult
        Extracted avalanche sizes, lifetimes, and supporting metadata.

    Raises
    ------
    ValueError
        If `size_definition` is not 1 or 2.
    """
    if size_definition not in (1, 2):
        raise ValueError(f"size_definition must be 1 or 2, got {size_definition!r}.")

    # Use the canonical binning primitive from Raster
    bin_edges, counts, ch_list = r.bin_counts(
        dt,
        tstart=tstart,
        tstop=tstop,
        channels=channels,
        inclusive_stop=inclusive_stop,
    )

    # counts: shape (n_channels, n_bins)
    n_ch, n_bins = counts.shape

    # Per-channel activity mask per bin
    active_ch_bin = counts > 0                       # bool, (n_ch, n_bins)
    active_bins = np.any(active_ch_bin, axis=0)      # bool, (n_bins,)

    # Avalanche segments: contiguous True regions bounded by False.
    # Treat outside-window as blank by padding with False on both sides.
    padded = np.concatenate([[False], active_bins, [False]]).astype(bool, copy=False)
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    # transitions come in pairs: start (False->True), stop (True->False), in padded coordinates
    # Convert to bin indices in [0, n_bins)
    intervals_bins: List[Tuple[int, int]] = []
    for k in range(0, transitions.size, 2):
        start_p = int(transitions[k])       # index in padded where change occurs
        stop_p = int(transitions[k + 1])
        start_bin = start_p                 # because padded has one leading element
        stop_bin = stop_p                   # exclusive
        # In terms of original active_bins indexing: [start_bin, stop_bin)
        intervals_bins.append((start_bin, stop_bin))

    # Compute lifetimes and sizes
    lifetimes = np.zeros(len(intervals_bins), dtype=np.int64)
    sizes = np.zeros(len(intervals_bins), dtype=np.int64)

    for i, (b0, b1) in enumerate(intervals_bins):
        seg_len = b1 - b0
        lifetimes[i] = seg_len

        seg = active_ch_bin[:, b0:b1]  # bool (n_ch, seg_len)

        if size_definition == 1:
            # sum over bins of # active electrodes in bin
            # i.e., sum_{bin} sum_{ch} 1[ch active in bin]
            sizes[i] = int(np.sum(seg))
        else:
            # number of distinct electrodes active at least once in avalanche
            sizes[i] = int(np.sum(np.any(seg, axis=1)))

    # Determine actual window used for metadata from edges
    tstart_used = float(bin_edges[0]) if bin_edges.size else 0.0
    tstop_used = float(bin_edges[-1]) if bin_edges.size else 0.0

    return AvalancheResult(
        sizes=sizes,
        lifetimes=lifetimes,
        dt=float(dt),
        tstart=tstart_used,
        tstop=tstop_used,
        channels=list(ch_list),
        size_definition=size_definition,
        bin_edges=bin_edges,
        active_bins=active_bins,
        intervals_bins=intervals_bins,
    )
