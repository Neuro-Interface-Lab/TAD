from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from tad.metrics.psth import PSTHResult
from tad.raster import Raster, ChannelId
from tad.metrics.utils import _select_channels, _infer_window


@dataclass(frozen=True)
class EvokedPeakResult:
    """
    Scalar evoked-response metrics derived from a PSTH.

    Parameters
    ----------
    channels
        Channel IDs corresponding to rows.
    baseline_hz
        Baseline rate in Hz (mean over baseline window).
    peak_hz
        Peak rate in Hz (max over response window).
    peak_latency_s
        Time of peak in seconds (relative to stimulus).
    peak_minus_baseline_hz
        Peak above baseline (Hz).
    auc_above_baseline
        Area above baseline over response window (spikes per stimulus),
        computed as sum(max(rate-baseline,0))*dt.
    """
    channels: List[ChannelId]
    baseline_hz: np.ndarray
    peak_hz: np.ndarray
    peak_latency_s: np.ndarray
    peak_minus_baseline_hz: np.ndarray
    auc_above_baseline: np.ndarray


def evoked_peak_metrics(
    psth: PSTHResult,
    *,
    baseline_window: Tuple[float, float] = (-0.050, 0.0),
    response_window: Tuple[float, float] = (0.0, 0.050),
    rectify: bool = True,
) -> EvokedPeakResult:
    """
    Compute per-channel evoked-response scalar metrics from PSTH.

    Parameters
    ----------
    psth
        PSTHResult as returned by `compute_psth`.
    baseline_window
        (t0, t1) window for baseline mean rate (seconds, relative to stimulus).
    response_window
        (t0, t1) window where peak/latency/AUC are computed.
    rectify
        If True, AUC uses max(rate-baseline, 0). If False, uses signed (rate-baseline).

    Returns
    -------
    EvokedPeakResult
        Per-channel scalar metrics.

    Raises
    ------
    ValueError
        If windows are invalid or contain no bins.
    """
    t = psth.t
    R = psth.rate_hz
    dt = float(psth.dt)

    b0, b1 = map(float, baseline_window)
    r0, r1 = map(float, response_window)
    if not (b0 < b1 and r0 < r1):
        raise ValueError("baseline_window and response_window must satisfy t0 < t1.")

    bmask = (t >= b0) & (t < b1)
    rmask = (t >= r0) & (t < r1)

    if not np.any(bmask):
        raise ValueError("baseline_window contains no PSTH bins. Adjust baseline_window or dt.")
    if not np.any(rmask):
        raise ValueError("response_window contains no PSTH bins. Adjust response_window or dt.")

    baseline = R[:, bmask].mean(axis=1)

    R_resp = R[:, rmask]
    t_resp = t[rmask]

    # Peak and latency (argmax)
    peak_idx = np.argmax(R_resp, axis=1)
    peak = R_resp[np.arange(R_resp.shape[0]), peak_idx]
    latency = t_resp[peak_idx]

    delta = R_resp - baseline[:, None]
    if rectify:
        delta = np.maximum(delta, 0.0)
    auc = delta.sum(axis=1) * dt  # spikes per stimulus (since rate*sec)

    return EvokedPeakResult(
        channels=list(psth.channels),
        baseline_hz=baseline.astype(np.float64, copy=False),
        peak_hz=peak.astype(np.float64, copy=False),
        peak_latency_s=latency.astype(np.float64, copy=False),
        peak_minus_baseline_hz=(peak - baseline).astype(np.float64, copy=False),
        auc_above_baseline=auc.astype(np.float64, copy=False),
    )


def response_probability(
    r: Raster,
    stim_times: Sequence[float],
    *,
    t0: float = 0.0,
    t1: float = 0.050,
    channels: Optional[Sequence[ChannelId]] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    inclusive_stop: bool = False,
    inclusive_zero: bool = True,
) -> Tuple[List[ChannelId], np.ndarray]:
    """
    Estimate per-channel response probability to a stimulus.

    Definition used here:
    For each channel c and each stimulus i, define an indicator:
        I_{c,i} = 1 if channel c has >=1 spike in [stim_i + t0, stim_i + t1)
                else 0
    Response probability per channel:
        p_c = (1/N) * sum_i I_{c,i}

    This complements PSTH peak metrics and is a standard evoked-response quantity.

    Parameters
    ----------
    r
        Input raster.
    stim_times
        Stimulus onset times.
    t0, t1
        Response window bounds relative to stimulus (seconds), with [t0, t1).
    channels
        Optional channels subset/order.
    tstart, tstop
        Optional admissible stimulus window; stimuli are kept only if the full
        [stim+t0, stim+t1) lies within [tstart, tstop) (or inclusive_stop).
    inclusive_stop
        Right-bound policy for admissible stimulus window.
    inclusive_zero
        If True, include spikes exactly at stim+t0 when t0==0 (uses searchsorted side).

    Returns
    -------
    channels_list, p
        channels_list is list of ChannelId, p is ndarray shape (n_channels,) with probabilities.
    """
    if not (float(t0) < float(t1)):
        raise ValueError("Require t0 < t1.")

    ch_list = _select_channels(r, channels)
    tstart_f, tstop_f = _infer_window(r, ch_list, tstart, tstop)

    stim = np.asarray(list(stim_times), dtype=np.float64)
    stim = stim[np.isfinite(stim)]
    if stim.size == 0:
        return list(ch_list), np.zeros((len(ch_list),), dtype=np.float64)

    win_left = stim + float(t0)
    win_right = stim + float(t1)
    if inclusive_stop:
        keep = (win_left >= tstart_f) & (win_right <= tstop_f)
    else:
        keep = (win_left >= tstart_f) & (win_right < tstop_f)
    stim_used = stim[keep]
    stim_used.sort()

    n_stim = int(stim_used.size)
    if n_stim == 0:
        return list(ch_list), np.zeros((len(ch_list),), dtype=np.float64)

    side_left = "left" if inclusive_zero else "right"
    side_right = "right" if inclusive_stop else "left"

    resp = np.zeros((len(ch_list),), dtype=np.float64)

    for ci, ch in enumerate(ch_list):
        arr = r.events[ch]
        if arr.size == 0:
            continue

        hits = 0
        for s in stim_used:
            a = s + float(t0)
            b = s + float(t1)
            l = np.searchsorted(arr, a, side=side_left)
            rr = np.searchsorted(arr, b, side=side_right)
            if rr > l:
                hits += 1
        resp[ci] = hits / float(n_stim)

    return list(ch_list), resp
