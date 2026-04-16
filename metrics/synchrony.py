from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from tad.metrics.rates import FiringRateCurveResult


@dataclass(frozen=True)
class PearsonSynchronyResult:
    """
    Pearson correlation-based synchrony result computed from firing-rate curves.

    Parameters
    ----------
    corr
        Pearson correlation matrix, shape (n_channels, n_channels).
    channels
        Channel IDs corresponding to corr rows/cols (same order as input FR).
    method
        Description of computation.
    n_time
        Number of time bins used.
    masked_channels
        Channels removed due to zero variance (if drop_constant=True). Each entry is
        (index_in_input, channel_id).
    """
    corr: np.ndarray
    channels: List
    method: str
    n_time: int
    masked_channels: List[Tuple[int, object]]


def pearson_corr_firing_rate(
    fr: FiringRateCurveResult,
    *,
    channels: Optional[Sequence[int]] = None,
    zscore: bool = True,
    drop_constant: bool = True,
    ddof: int = 0,
) -> PearsonSynchronyResult:
    """
    Compute Pearson correlation matrix between channels using firing-rate curves.

    This is intended as an exploratory synchrony metric:
    - Take per-channel rate signals r_i(t) (from binning)
    - Optionally z-score each channel across time
    - Compute corr(i,j) = corr(r_i(t), r_j(t))

    Parameters
    ----------
    fr
        Firing rate curves result (from `tad.metrics.rates.firing_rate_curve`).
    channels
        Optional subset of channels specified as row indices into fr.fr_ch.
        If None, uses all channels in stored order.
    zscore
        If True, z-score each channel across time before correlation.
        If False, correlation is still Pearson (mean-centered, variance-normalized),
        but keeping this option helps debugging.
    drop_constant
        If True, drop channels with (near-)zero temporal variance to avoid NaNs.
        If False, those channels will produce NaNs in correlation.
    ddof
        Degrees of freedom for std calculation in z-scoring (0 is fine for signals).

    Returns
    -------
    PearsonSynchronyResult
        Correlation matrix and metadata.

    Notes
    -----
    - Correlation is computed as (Z @ Z.T) / T where Z is per-channel standardized data.
    - If zscore=True, Z has mean 0 and std 1 (per channel), up to numerical tolerance.
    """
    X = np.asarray(fr.fr_ch, dtype=np.float64)
    n_ch_total, n_time = X.shape

    if channels is None:
        idx = np.arange(n_ch_total)
    else:
        idx = np.asarray(list(channels), dtype=int)

    Y = X[idx, :].copy()
    ch_ids = [fr.channels[i] for i in idx]

    masked: List[Tuple[int, object]] = []

    # Standardize / z-score per channel
    mu = Y.mean(axis=1, keepdims=True)
    Yc = Y - mu

    sd = Yc.std(axis=1, keepdims=True, ddof=ddof)

    # Identify constant channels
    eps = 0.0
    const_mask = (sd.squeeze(axis=1) <= eps)

    if np.any(const_mask):
        if drop_constant:
            # Record and remove
            bad_idx = np.where(const_mask)[0]
            for bi in bad_idx:
                masked.append((int(idx[bi]), ch_ids[bi]))

            keep = ~const_mask
            Yc = Yc[keep, :]
            sd = sd[keep, :]
            ch_ids = [ch for k, ch in enumerate(ch_ids) if keep[k]]

        else:
            # Will lead to division by zero -> NaNs
            pass

    if Yc.shape[0] == 0:
        return PearsonSynchronyResult(
            corr=np.zeros((0, 0), dtype=np.float64),
            channels=[],
            method="pearson_corr(fr_ch over time): empty after dropping constants",
            n_time=int(n_time),
            masked_channels=masked,
        )

    # If requested zscore, divide by sd; otherwise still do Pearson normalization
    # (Pearson requires variance normalization anyway).
    # We keep the switch mainly for debug clarity.
    if zscore:
        Z = Yc / sd
    else:
        Z = Yc / sd

    # Compute corr = (Z Z^T) / T
    T = float(Z.shape[1])
    corr = (Z @ Z.T) / T

    # Numerical safety: clip slight roundoff outside [-1,1]
    corr = np.clip(corr, -1.0, 1.0)

    return PearsonSynchronyResult(
        corr=corr,
        channels=ch_ids,
        method="pearson_corr(zscored firing rate curves across time)",
        n_time=int(n_time),
        masked_channels=masked,
    )
