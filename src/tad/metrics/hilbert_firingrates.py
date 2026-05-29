from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from tad.metrics.rates import FiringRateCurveResult


@dataclass(frozen=True)
class HilbertFiringRateResult:
    """
    Analytic-signal representation of firing-rate curves.

    Attributes
    ----------
    phase_ch:
        Per-channel instantaneous phase, shape (n_channels, n_time), radians.
    amplitude_ch:
        Per-channel analytic amplitude, shape (n_channels, n_time).
    filtered_ch:
        Per-channel demeaned and optionally band-pass filtered rate signals,
        shape (n_channels, n_time).
    analytic_ch:
        Per-channel analytic signals, shape (n_channels, n_time).
    phase_pop, amplitude_pop, filtered_pop, analytic_pop:
        Same quantities for the population firing-rate curve.
    valid_mask_ch:
        Boolean mask indicating where per-channel amplitude is above threshold.
    valid_mask_pop:
        Boolean mask indicating where population amplitude is above threshold.
    channels:
        Channel IDs corresponding to rows.
    fs:
        Sampling frequency of the firing-rate curves, equal to 1 / fr.dt.
    band:
        Frequency band `(f_low, f_high)` used for filtering, or None.
    method:
        Human-readable method description.
    """
    phase_ch: np.ndarray
    amplitude_ch: np.ndarray
    filtered_ch: np.ndarray
    analytic_ch: np.ndarray
    phase_pop: np.ndarray
    amplitude_pop: np.ndarray
    filtered_pop: np.ndarray
    analytic_pop: np.ndarray
    valid_mask_ch: np.ndarray
    valid_mask_pop: np.ndarray
    channels: List
    fs: float
    band: Optional[Tuple[float, float]]
    method: str


@dataclass(frozen=True)
class PhaseLockingValueResult:
    """
    Pairwise phase-locking value computed from Hilbert phases.

    Attributes
    ----------
    plv:
        Pairwise PLV matrix, shape (n_channels, n_channels).
    mean_phase_lag:
        Circular mean of phase differences, shape (n_channels, n_channels).
    channels:
        Channel IDs corresponding to matrix rows/columns.
    n_time:
        Number of valid time bins used.
    method:
        Human-readable method description.
    """
    plv: np.ndarray
    mean_phase_lag: np.ndarray
    channels: List
    n_time: int
    method: str


@dataclass(frozen=True)
class KuramotoOrderResult:
    """
    Kuramoto order parameter from Hilbert phases.

    Attributes
    ----------
    order:
        Synchronization magnitude R(t), shape (n_time,).
    mean_phase:
        Mean population phase psi(t), shape (n_time,).
    channels:
        Channels included in the estimate.
    method:
        Human-readable method description.
    """
    order: np.ndarray
    mean_phase: np.ndarray
    channels: List
    method: str


def _selected_rate_matrix(
    fr: FiringRateCurveResult,
    channels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List]:
    x = np.asarray(fr.fr_ch, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("fr.fr_ch must be a 2D array with shape (n_channels, n_time).")

    if channels is None:
        idx = np.arange(x.shape[0])
    else:
        idx = np.asarray(list(channels), dtype=int)

    return x[idx, :], [fr.channels[i] for i in idx]


def _sampling_frequency(fr: FiringRateCurveResult) -> float:
    dt = float(fr.dt)
    if dt <= 0:
        raise ValueError("fr.dt must be strictly positive.")
    return 1.0 / dt


def _demean(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - np.mean(x, axis=axis, keepdims=True)


def _bandpass_or_demean(
    x: np.ndarray,
    *,
    fs: float,
    f_low: Optional[float],
    f_high: Optional[float],
    order: int,
    axis: int,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    y = _demean(np.asarray(x, dtype=np.float64), axis=axis)

    if f_low is None and f_high is None:
        return y, None

    if f_low is None or f_high is None:
        raise ValueError("f_low and f_high must either both be provided or both be None.")

    f_low_f = float(f_low)
    f_high_f = float(f_high)
    nyq = fs / 2.0

    if not (0 < f_low_f < f_high_f < nyq):
        raise ValueError(
            f"Expected 0 < f_low < f_high < Nyquist ({nyq:.6g} Hz), "
            f"got f_low={f_low_f}, f_high={f_high_f}."
        )

    sos = signal.butter(
        int(order),
        [f_low_f / nyq, f_high_f / nyq],
        btype="bandpass",
        output="sos",
    )
    return signal.sosfiltfilt(sos, y, axis=axis), (f_low_f, f_high_f)


def _analytic_quantities(x: np.ndarray, *, axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    analytic = signal.hilbert(x, axis=axis)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)
    return analytic, phase, amplitude


def _amplitude_mask(
    amplitude: np.ndarray,
    *,
    min_amplitude: Optional[float],
    amplitude_quantile: Optional[float],
    axis: int = -1,
) -> np.ndarray:
    if min_amplitude is not None and amplitude_quantile is not None:
        raise ValueError("Use either min_amplitude or amplitude_quantile, not both.")

    if min_amplitude is not None:
        return amplitude >= float(min_amplitude)

    if amplitude_quantile is not None:
        q = float(amplitude_quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError("amplitude_quantile must be between 0 and 1.")
        threshold = np.quantile(amplitude, q, axis=axis, keepdims=True)
        return amplitude >= threshold

    return np.ones_like(amplitude, dtype=bool)


def hilbert_firing_rate(
    fr: FiringRateCurveResult,
    *,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    channels: Optional[Sequence[int]] = None,
    order: int = 3,
    min_amplitude: Optional[float] = None,
    amplitude_quantile: Optional[float] = 0.05,
) -> HilbertFiringRateResult:
    """
    Compute a Hilbert analytic-signal representation of firing-rate curves.

    Scientific note
    ---------------
    A Hilbert phase is interpretable mainly when applied to a sufficiently
    narrow-band component. Prefer passing `f_low` and `f_high` after inspecting
    the firing-rate spectrum.

    Parameters
    ----------
    fr:
        Firing-rate curves returned by `tad.metrics.rates.firing_rate_curve`.
    f_low, f_high:
        Optional band-pass limits in Hz. If both are None, only demeaning is
        applied before Hilbert transform.
    channels:
        Optional subset of channel row indices into `fr.fr_ch`.
    order:
        Butterworth band-pass filter order.
    min_amplitude:
        Optional absolute amplitude threshold for valid phase samples.
    amplitude_quantile:
        Optional per-signal quantile threshold for valid phase samples.
        Default is 0.05, masking the lowest 5% amplitude samples.

    Returns
    -------
    HilbertFiringRateResult
        Filtered signals, analytic signals, phase, amplitude, and masks.
    """
    x, ch_ids = _selected_rate_matrix(fr, channels=channels)
    fs = _sampling_frequency(fr)

    filtered_ch, band = _bandpass_or_demean(
        x,
        fs=fs,
        f_low=f_low,
        f_high=f_high,
        order=order,
        axis=1,
    )
    filtered_pop, _ = _bandpass_or_demean(
        np.asarray(fr.fr_pop, dtype=np.float64),
        fs=fs,
        f_low=f_low,
        f_high=f_high,
        order=order,
        axis=0,
    )

    analytic_ch, phase_ch, amplitude_ch = _analytic_quantities(filtered_ch, axis=1)
    analytic_pop, phase_pop, amplitude_pop = _analytic_quantities(filtered_pop, axis=0)

    valid_mask_ch = _amplitude_mask(
        amplitude_ch,
        min_amplitude=min_amplitude,
        amplitude_quantile=amplitude_quantile,
        axis=1,
    )
    valid_mask_pop = _amplitude_mask(
        amplitude_pop,
        min_amplitude=min_amplitude,
        amplitude_quantile=amplitude_quantile,
        axis=0,
    )

    method = "hilbert analytic signal of "
    method += "band-pass filtered firing-rate curves" if band is not None else "demeaned firing-rate curves"

    return HilbertFiringRateResult(
        phase_ch=phase_ch,
        amplitude_ch=amplitude_ch,
        filtered_ch=filtered_ch,
        analytic_ch=analytic_ch,
        phase_pop=phase_pop,
        amplitude_pop=amplitude_pop,
        filtered_pop=filtered_pop,
        analytic_pop=analytic_pop,
        valid_mask_ch=valid_mask_ch,
        valid_mask_pop=valid_mask_pop,
        channels=ch_ids,
        fs=fs,
        band=band,
        method=method,
    )


def phase_locking_value_firing_rate(
    h: HilbertFiringRateResult,
    *,
    use_amplitude_mask: bool = True,
    min_valid_fraction: float = 0.5,
) -> PhaseLockingValueResult:
    """
    Compute pairwise phase-locking value between Hilbert firing-rate phases.

    Parameters
    ----------
    h:
        Result returned by `hilbert_firing_rate`.
    use_amplitude_mask:
        If True, only samples valid in both channels are used for each pair.
    min_valid_fraction:
        Minimum fraction of time bins required for a pairwise PLV estimate.
        Pairs below the threshold are returned as NaN.

    Returns
    -------
    PhaseLockingValueResult
        PLV and circular mean phase lag matrices.
    """
    phase = np.asarray(h.phase_ch, dtype=np.float64)
    n_ch, n_time = phase.shape

    if not (0 <= min_valid_fraction <= 1):
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    plv = np.full((n_ch, n_ch), np.nan, dtype=np.float64)
    mean_lag = np.full((n_ch, n_ch), np.nan, dtype=np.float64)
    min_valid = int(np.ceil(min_valid_fraction * n_time))

    for i in range(n_ch):
        for j in range(n_ch):
            if use_amplitude_mask:
                mask = h.valid_mask_ch[i] & h.valid_mask_ch[j]
            else:
                mask = np.ones(n_time, dtype=bool)

            if int(np.sum(mask)) < min_valid:
                continue

            dphi = phase[i, mask] - phase[j, mask]
            z = np.mean(np.exp(1j * dphi))
            plv[i, j] = np.abs(z)
            mean_lag[i, j] = np.angle(z)

    return PhaseLockingValueResult(
        plv=plv,
        mean_phase_lag=mean_lag,
        channels=list(h.channels),
        n_time=int(n_time),
        method="pairwise phase-locking value from Hilbert firing-rate phases",
    )


def kuramoto_order_firing_rate(
    h: HilbertFiringRateResult,
    *,
    use_amplitude_mask: bool = True,
) -> KuramotoOrderResult:
    """
    Compute the Kuramoto order parameter across channels.

    Parameters
    ----------
    h:
        Result returned by `hilbert_firing_rate`.
    use_amplitude_mask:
        If True, invalid low-amplitude phase samples are excluded at each time.
        If all channels are invalid at a time point, the result is NaN.

    Returns
    -------
    KuramotoOrderResult
        R(t) and mean phase psi(t).
    """
    phase = np.asarray(h.phase_ch, dtype=np.float64)

    if use_amplitude_mask:
        unit = np.exp(1j * phase)
        unit = np.where(h.valid_mask_ch, unit, np.nan + 1j * np.nan)
        z = np.nanmean(unit, axis=0)
    else:
        z = np.mean(np.exp(1j * phase), axis=0)

    return KuramotoOrderResult(
        order=np.abs(z),
        mean_phase=np.angle(z),
        channels=list(h.channels),
        method="Kuramoto order parameter from Hilbert firing-rate phases",
    )
