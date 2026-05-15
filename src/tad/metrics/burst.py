from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

from tad.raster import Raster, ChannelId
from tad.metrics.utils import _select_channels, _infer_window


@dataclass(frozen=True)
class Burst:
    """
    Single burst description.

    Parameters
    ----------
    start
        Burst start time (time of first spike in burst).
    end
        Burst end time (time of last spike in burst).
    n_spikes
        Number of spikes in the burst.
    duration
        Burst duration defined as end - start.
    intra_rate_hz
        Intra-burst firing rate (Hz), defined here as (n_spikes - 1) / duration
        if duration > 0, else np.inf.
    """
    start: float
    end: float
    n_spikes: int
    duration: float
    intra_rate_hz: float


@dataclass(frozen=True)
class LogISIThresholdDiagnostics:
    """
    Diagnostics for adaptive ISI threshold selection using the log-ISIH.

    Parameters
    ----------
    isi_th
        Chosen ISI threshold (seconds).
    centers_log10
        Histogram bin centers in log10(ISI) domain.
    hist
        Raw histogram values (counts or density depending on call).
    hist_smooth
        Smoothed histogram used for peak/valley detection.
    peak1_idx
        Index of the left peak (short-ISI regime) in `centers_log10`, or -1.
    peak2_idx
        Index of the right peak (long-ISI regime) in `centers_log10`, or -1.
    valley_idx
        Index of the valley between peaks used as threshold, or -1 when fallback used.
    method
        Selection path, e.g. "valley_between_peaks" or "fallback_quantile_*".
    """
    isi_th: float
    centers_log10: np.ndarray
    hist: np.ndarray
    hist_smooth: np.ndarray
    peak1_idx: int
    peak2_idx: int
    valley_idx: int
    method: str


@dataclass(frozen=True)
class BurstChannelResult:
    """
    Bursts for one channel.

    Parameters
    ----------
    channel
        Channel id.
    isi_th
        ISI threshold used for this channel.
    bursts
        List of detected bursts.
    diagnostics
        Threshold-selection diagnostics if method="logisih", otherwise None.
        If threshold_scope="pooled", this may be the same object for all channels.
    """
    channel: ChannelId
    isi_th: float
    bursts: List[Burst]
    diagnostics: Optional[LogISIThresholdDiagnostics] = None


@dataclass(frozen=True)
class BurstDetectionResult:
    """
    Burst detection result across channels.

    Parameters
    ----------
    per_channel
        Mapping channel -> BurstChannelResult.
    method
        A human-readable description of the method used.
    tstart, tstop
        Time window used for detection.
    channels
        Channel IDs included (order).
    """
    per_channel: Dict[ChannelId, BurstChannelResult]
    method: str
    tstart: float
    tstop: float
    channels: List[ChannelId]


def _detect_bursts_from_times(
    times: np.ndarray,
    *,
    isi_th: float,
    min_spikes: int = 3,
) -> List[Burst]:
    """
    Detect bursts from sorted spike times using a fixed ISI threshold rule.

    A burst is a maximal consecutive sequence of spikes such that each consecutive
    inter-spike interval (ISI) satisfies ISI <= isi_th. The sequence is accepted as
    a burst if it contains at least `min_spikes` spikes.

    Parameters
    ----------
    times
        Sorted spike times, shape (N,).
    isi_th
        ISI threshold (seconds).
    min_spikes
        Minimum number of spikes required to accept a burst.

    Returns
    -------
    list of Burst
        Detected bursts.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    if t.size < min_spikes:
        return []

    isi = np.diff(t)
    if isi.size == 0:
        return []

    link = isi <= float(isi_th)  # True means spike i and i+1 are within the same burst

    bursts: List[Burst] = []
    i = 0
    n = t.size

    while i < n - 1:
        if not link[i]:
            i += 1
            continue

        start_idx = i
        j = i
        while j < n - 1 and link[j]:
            j += 1
        end_idx = j

        nsp = end_idx - start_idx + 1
        if nsp >= min_spikes:
            start = float(t[start_idx])
            end = float(t[end_idx])
            dur = float(end - start)
            intra = float((nsp - 1) / dur) if dur > 0.0 else float("inf")
            bursts.append(Burst(start=start, end=end, n_spikes=int(nsp), duration=dur, intra_rate_hz=intra))

        i = end_idx

    return bursts


def choose_isi_threshold_logisih(
    isi_values: Optional[np.ndarray] = None,
    r: Optional[Raster] = None,
    channel: Optional[ChannelId] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    *,
    bins: str | int = "pasquale",
    smooth_window: str | int = "from_bins",
    density: bool = False,
    min_isi: float = 1e-6,
    fallback: float = 0.1,
) -> LogISIThresholdDiagnostics:
    """
    Choose an ISI threshold from the log10(ISI) histogram (log-ISIH).

    This is an adaptive threshold selection strategy commonly used for burst detection
    in MEA analyses: short ISIs (within bursts) and long ISIs (between bursts) can
    create a structured distribution in log10(ISI). When two regimes are detectable,
    the threshold is taken as the valley between the two main peaks.

    Algorithm (robust, NumPy-only):
    1) Compute y = log10(ISI) for ISI > 0.
    2) Compute histogram of y with `bins` bins.
    3) Smooth histogram with moving average window `smooth_window` using 20% of bins as window.
    4) Detect local maxima (peaks) on smoothed histogram.
    5) If two sufficiently separated peaks exist, pick the valley (minimum) between them
       and set ISI_th = 10**(center_at_valley).
    6) Otherwise fallback to a conservative quantile of the ISI distribution.

    Parameters
    ----------
    isi_values
        1D array of ISIs (>0).
    bins
        Number of bins in log10(ISI) space, or method from np.histogram.
    smooth_window
        Moving average smoothing window length, or method from number of bins.
    density
        If True, histogram is density; otherwise counts.
    min_isi
        Minimum ISI value before log transform (prevents log(0)).
    fallback
        ISI used if peak/valley detection fails.

    Returns
    -------
    LogISIThresholdDiagnostics
        Contains chosen ISI_th and intermediate arrays/indices for inspection.
    """
    if isi_values is None and r is None:
        raise ValueError("Pass isi_values or r.")
    if isi_values is not None and r is not None:
        raise ValueError("Pass only isi_values or r.")
    if r is not None:
        ch = channel
        tstart_f, tstop_f = _infer_window(r, [ch], tstart, tstop)
        arr = r.events[channel]
        left = np.searchsorted(arr, tstart_f, side="left")
        right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
        w = arr[left:right]
        isi_values = np.diff(w).astype(np.float64, copy=False) if w.size >= 2 else np.asarray([], dtype=np.float64)
    x = np.asarray(isi_values, dtype=np.float64).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return LogISIThresholdDiagnostics(
            isi_th=float("nan"),
            centers_log10=np.asarray([], dtype=np.float64),
            hist=np.asarray([], dtype=np.float64),
            hist_smooth=np.asarray([], dtype=np.float64),
            peak1_idx=-1,
            peak2_idx=-1,
            valley_idx=-1,
            method="no_data",
        )

    x = np.maximum(x, float(min_isi))
    y = np.log10(x)

    if bins == "pasquale":
        y_max = np.max(y)
        y_min = np.min(y)
        # set bin size to 0.1 in log10 space:
        bins = int(np.ceil((y_max - y_min) / 0.1))
    if bins < 1:
        isi_th = float("nan")
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=None,
            hist=None,
            hist_smooth=None,
            peak1_idx=-1,
            peak2_idx=-1,
            valley_idx=-1,
            method="not_able_to_find_bins",
        )
    hist, edges = np.histogram(y, bins=bins, density=bool(density))
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = hist.astype(np.float64, copy=False)

    # Smooth (moving average)
    if isinstance(smooth_window,str) and smooth_window == "from_bins":
        w = max(1, int(0.20 * centers.size))  # 20% of bins
    else:
        w = int(smooth_window)
    if w < 1:
        w = 1
    if w > 1:
        kernel = np.ones(w, dtype=np.float64) / float(w)
        h_s = np.convolve(h, kernel, mode="same")
    else:
        h_s = h.copy()

    if h_s.size < 3:
        isi_th = float("nan")
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=-1,
            peak2_idx=-1,
            valley_idx=-1,
            method="not_able_to_find_bursts",
        )


    # Local maxima indices
    peaks = find_peaks(h_s)[0]
    isi_peaks = centers[peaks] if peaks.size > 0 else np.array([])
    if isi_peaks.size == 0:
        isi_th = float("nan")
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=-1,
            peak2_idx=-1,
            valley_idx=-1,
            method="no_burst_regime_no_peaks",
        )
    if isi_peaks[0] > -1.0:
        isi_th = float("nan")
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=int(peaks[0]) if peaks.size > 0 else -1,
            peak2_idx=int(peaks[1]) if peaks.size > 1 else -1,
            valley_idx=-1,
            method="no_burst_regime_first_peak_high",
        )
    if isi_peaks[-1] < -1.0:
        isi_th = float(fallback)
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=int(peaks[0]) if peaks.size > 0 else -1,
            peak2_idx=int(peaks[1]) if peaks.size > 1 else -1,
            valley_idx=-1,
            method="fallback_returned",
        )

    if peaks.size < 2 and isi_peaks[0] < -1.0:
        isi_th = float(fallback)
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=int(peaks[0]) if peaks.size == 1 else -1,
            peak2_idx=-1,
            valley_idx=-1,
            method="fallback_returned",
        )
        
    # Find last peak with log10(isi) < -1.0 and maximum peak with log10(isi) > -1.0
    intra_burst_peaks = isi_peaks[isi_peaks < -1.0]
    idx_last_intra_burst_peak = peaks[int(len(intra_burst_peaks) - 1)]
    inter_burst_peaks = isi_peaks[isi_peaks > -1.0]
    argmax_inter = np.argmax(inter_burst_peaks)
    idx_in_isi_peaks = np.where(isi_peaks > -1.0)[0][argmax_inter]
    idx_first_inter_burst_peak = peaks[idx_in_isi_peaks]

    peak1_idx, peak2_idx = (idx_last_intra_burst_peak, idx_first_inter_burst_peak)

    if h_s[peak1_idx] < 0.05 * h_s[peak2_idx]:
        isi_th = float("nan")
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=int(peak1_idx),
            peak2_idx=int(peak2_idx),
            valley_idx=-1,
            method="no_burst_regime_small_first_peak",
        )

    seg = h_s[peak1_idx:peak2_idx + 1]
    if seg.size < 3:
        isi_th = float(fallback)
        return LogISIThresholdDiagnostics(
            isi_th=isi_th,
            centers_log10=centers,
            hist=h,
            hist_smooth=h_s,
            peak1_idx=int(peak1_idx),
            peak2_idx=int(peak2_idx),
            valley_idx=-1,
            method="fallback_segment_too_small",
        )

    valley_rel = int(np.argmin(seg))
    valley_idx = int(peak1_idx + valley_rel)
    isi_th = float(10.0 ** centers[valley_idx])

    return LogISIThresholdDiagnostics(
        isi_th=isi_th,
        centers_log10=centers,
        hist=h,
        hist_smooth=h_s,
        peak1_idx=int(peak1_idx),
        peak2_idx=int(peak2_idx),
        valley_idx=int(valley_idx),
        method="valley_between_peaks",
    )


def detect_bursts(
    r: Raster,
    *,
    method: Literal["fixed", "logisih", "fixed_from_logisih"] = "fixed",
    isi_th: float | np.ndarray = 0.1,
    min_spikes: int = 3,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    threshold_scope: Literal["per_channel", "pooled"] = "per_channel",
    logisih_bins: str | int = "auto",
    logisih_smooth_window: str | int = "from_bins",
    fallback: float = 0.1,
) -> BurstDetectionResult:
    """
    Detect single-channel bursts in a Raster.

    Three threshold-selection methods are supported:

    1) method="fixed"
        Uses the constant threshold `isi_th` for all channels.

    2) method="logisih"
        Derives the ISI threshold from the log10(ISI) histogram (log-ISIH) either:
        - per channel (threshold_scope="per_channel"), or
        - from pooled ISIs across selected channels (threshold_scope="pooled").
    
    3) method="fixed_from_logisih"
        Uses pre-calculated ISI thresholds from log-ISIH analysis.
        For this method, `isi_th` must be an array with one threshold value per channel
        in the same order as `channels`.

    Burst definition (segmentation):
    A burst is a maximal consecutive sequence of spikes such that every consecutive
    inter-spike interval satisfies ISI <= ISI_th, and the sequence contains at least
    `min_spikes` spikes.

    Parameters
    ----------
    r
        Input raster.
    method
        Threshold selection method: "fixed", "logisih", or "fixed_from_logisih".
    isi_th
        ISI threshold(s) in seconds. For "fixed" and "logisih", a single float.
        For "fixed_from_logisih", an array of floats with one per channel.
    min_spikes
        Minimum number of spikes required to accept a burst.
    channels
        Channels to include. If None, uses all channels in the raster.
    tstart, tstop
        Optional time window. If None, inferred from data across selected channels.
    inclusive_stop
        If True, include events exactly at tstop; otherwise window is [tstart, tstop).
    threshold_scope
        Only used when method="logisih":
        - "per_channel": compute a separate ISI_th per channel from that channel's ISIs
        - "pooled": compute a single ISI_th from pooled ISIs across channels and reuse it
    logisih_bins
        Number of bins in log10(ISI) histogram, or method for numpy.histogram ("auto", "fd", "doane", "sqrt" etc) (method="logisih").
    logisih_smooth_window
        Smoothing window length for histogram, or method to determine it from number of bins (method="logisih").
    fallback
        ISI threshold value used when log-ISIH peak/valley detection fails.

    Returns
    -------
    BurstDetectionResult
        Per-channel burst lists and metadata. For method="logisih", each channel result
        includes a `diagnostics` object that can be plotted/inspected; when
        threshold_scope="pooled", this diagnostic may be shared across channels.

    Raises
    ------
    ValueError
        If parameters are inconsistent.
    """
    if method not in ("fixed", "logisih", "fixed_from_logisih"):
        raise ValueError("method must be 'fixed', 'logisih', or 'fixed_from_logisih'.")

    if threshold_scope not in ("per_channel", "pooled"):
        raise ValueError("threshold_scope must be 'per_channel' or 'pooled'.")

    if min_spikes < 2:
        raise ValueError("min_spikes must be >= 2.")

    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    # Validate isi_th for fixed_from_logisih method
    if method == "fixed_from_logisih":
        isi_th_array = np.asarray(isi_th)
        if isi_th_array.ndim != 1 or isi_th_array.size != len(ch_list):
            raise ValueError(
                f"For method='fixed_from_logisih', isi_th must be a 1D array with "
                f"exactly {len(ch_list)} elements (one per channel), but got shape {isi_th_array.shape}"
            )

    out: Dict[ChannelId, BurstChannelResult] = {}

    pooled_diag: Optional[LogISIThresholdDiagnostics] = None
    pooled_isi_th: Optional[float] = None

    if method == "logisih" and threshold_scope == "pooled":
        pooled_isis: List[np.ndarray] = []
        for ch in ch_list:
            arr = r.events[ch]
            if arr.size < 2:
                continue
            left = np.searchsorted(arr, tstart_f, side="left")
            right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
            w = arr[left:right]
            if w.size >= 2:
                pooled_isis.append(np.diff(w).astype(np.float64, copy=False))

        pooled_concat = np.concatenate(pooled_isis) if pooled_isis else np.asarray([], dtype=np.float64)

        pooled_diag = choose_isi_threshold_logisih(
            pooled_concat,
            bins=logisih_bins,
            smooth_window=logisih_smooth_window,
            fallback=fallback,
        )
        pooled_isi_th = pooled_diag.isi_th

    for i, ch in enumerate(ch_list):
        arr = r.events[ch]
        left = np.searchsorted(arr, tstart_f, side="left")
        right = np.searchsorted(arr, tstop_f, side=("right" if inclusive_stop else "left"))
        w = arr[left:right]

        diag: Optional[LogISIThresholdDiagnostics] = None

        if method == "fixed":
            isi_th_ch = float(isi_th)
        elif method == "fixed_from_logisih":
            isi_th_array = np.asarray(isi_th)
            isi_th_ch = float(isi_th_array[i])
        else:  # method == "logisih"
            if threshold_scope == "pooled":
                diag = pooled_diag
                isi_th_ch = float(pooled_isi_th) if pooled_isi_th is not None else float("nan")
            else:
                isi_vals = np.diff(w).astype(np.float64, copy=False) if w.size >= 2 else np.asarray([], dtype=np.float64)
                diag = choose_isi_threshold_logisih(
                    isi_vals,
                    bins=logisih_bins,
                    smooth_window=logisih_smooth_window,
                    fallback=fallback,
                )
                isi_th_ch = float(diag.isi_th)

        # If isi_th_ch is NaN (e.g., no data), no bursts will be detected.
        if not np.isfinite(isi_th_ch):
            bursts = []
        else:
            bursts = _detect_bursts_from_times(w, isi_th=isi_th_ch, min_spikes=int(min_spikes))

        out[ch] = BurstChannelResult(channel=ch, isi_th=float(isi_th_ch), bursts=bursts, diagnostics=diag)

    method_str = (
        f"fixed_isi_th={float(isi_th)}"
        if method == "fixed"
        else "fixed_from_logisih(per_channel_thresholds)"
        if method == "fixed_from_logisih"
        else f"logisih(scope={threshold_scope}, bins={logisih_bins}, smooth={logisih_smooth_window}, fallback_q={float(fallback)})"
    )

    return BurstDetectionResult(
        per_channel=out,
        method=method_str,
        tstart=float(tstart_f),
        tstop=float(tstop_f),
        channels=list(ch_list),
    )
