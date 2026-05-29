from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from tad.metrics.rates import FiringRateCurveResult


@dataclass(frozen=True)
class FiringRatePowerSpectrumResult:
    """
    Power spectral density estimated from firing-rate curves.

    Attributes
    ----------
    freqs:
        Frequency vector, shape (n_freqs,).
    psd_ch:
        Per-channel PSD, shape (n_channels, n_freqs).
    psd_pop:
        Population firing-rate PSD, shape (n_freqs,).
    channels:
        Channel IDs corresponding to rows of `psd_ch`.
    fs:
        Sampling frequency of the firing-rate curves, equal to 1 / fr.dt.
    method:
        Human-readable method description.
    """
    freqs: np.ndarray
    psd_ch: np.ndarray
    psd_pop: np.ndarray
    channels: List
    fs: float
    method: str


@dataclass(frozen=True)
class FiringRateSpectrogramResult:
    """
    Time-frequency representation estimated from firing-rate curves.

    Attributes
    ----------
    freqs:
        Frequency vector, shape (n_freqs,).
    t:
        Spectrogram time vector in the same time units as `fr.t`, shape (n_windows,).
    sxx_ch:
        Per-channel spectrogram, shape (n_channels, n_freqs, n_windows).
    sxx_pop:
        Population firing-rate spectrogram, shape (n_freqs, n_windows).
    channels:
        Channel IDs corresponding to rows of `sxx_ch`.
    fs:
        Sampling frequency of the firing-rate curves, equal to 1 / fr.dt.
    method:
        Human-readable method description.
    """
    freqs: np.ndarray
    t: np.ndarray
    sxx_ch: np.ndarray
    sxx_pop: np.ndarray
    channels: List
    fs: float
    method: str


@dataclass(frozen=True)
class FiringRateAutocorrelationResult:
    """
    Autocorrelation of firing-rate curves.

    Attributes
    ----------
    lags:
        Lag vector in the same time units as `fr.t`, shape (n_lags,).
    autocorr_ch:
        Per-channel autocorrelation, shape (n_channels, n_lags).
    autocorr_pop:
        Population firing-rate autocorrelation, shape (n_lags,).
    channels:
        Channel IDs corresponding to rows of `autocorr_ch`.
    method:
        Human-readable method description.
    """
    lags: np.ndarray
    autocorr_ch: np.ndarray
    autocorr_pop: np.ndarray
    channels: List
    method: str


def _selected_rate_matrix(
    fr: FiringRateCurveResult,
    channels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List]:
    """Return selected per-channel firing-rate matrix and channel IDs."""
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


def _default_nperseg(n_time: int, nperseg: Optional[int]) -> int:
    if n_time < 2:
        raise ValueError("At least two time bins are required.")
    if nperseg is None:
        return min(256, n_time)
    nperseg_i = int(nperseg)
    if nperseg_i < 2:
        raise ValueError("nperseg must be >= 2.")
    return min(nperseg_i, n_time)


def power_spectrum_firing_rate(
    fr: FiringRateCurveResult,
    *,
    channels: Optional[Sequence[int]] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: str = "constant",
    scaling: str = "density",
) -> FiringRatePowerSpectrumResult:
    """
    Estimate the power spectral density of firing-rate curves using Welch's method.

    Parameters
    ----------
    fr:
        Firing-rate curves returned by `tad.metrics.rates.firing_rate_curve`.
    channels:
        Optional subset of channel row indices into `fr.fr_ch`.
        If None, all channels are used.
    nperseg, noverlap, window, detrend, scaling:
        Passed to `scipy.signal.welch`.

    Returns
    -------
    FiringRatePowerSpectrumResult
        PSD for each selected channel and for the population firing-rate curve.
    """
    x, ch_ids = _selected_rate_matrix(fr, channels=channels)
    fs = _sampling_frequency(fr)
    nperseg_i = _default_nperseg(x.shape[1], nperseg)

    freqs, psd_ch = signal.welch(
        x,
        fs=fs,
        axis=1,
        nperseg=nperseg_i,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        scaling=scaling,
    )
    freqs_pop, psd_pop = signal.welch(
        np.asarray(fr.fr_pop, dtype=np.float64),
        fs=fs,
        nperseg=nperseg_i,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        scaling=scaling,
    )

    if not np.allclose(freqs, freqs_pop):
        raise RuntimeError("Channel and population PSD frequency grids differ.")

    return FiringRatePowerSpectrumResult(
        freqs=freqs,
        psd_ch=psd_ch,
        psd_pop=psd_pop,
        channels=ch_ids,
        fs=fs,
        method="welch power spectral density of firing-rate curves",
    )


def spectrogram_firing_rate(
    fr: FiringRateCurveResult,
    *,
    channels: Optional[Sequence[int]] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: str = "constant",
    scaling: str = "density",
    mode: str = "psd",
) -> FiringRateSpectrogramResult:
    """
    Estimate a spectrogram of firing-rate curves.

    The returned spectrogram times are shifted by `fr.tstart` so that they are
    expressed in the same time reference as `fr.t`.

    Parameters
    ----------
    fr:
        Firing-rate curves returned by `tad.metrics.rates.firing_rate_curve`.
    channels:
        Optional subset of channel row indices into `fr.fr_ch`.
    nperseg, noverlap, window, detrend, scaling, mode:
        Passed to `scipy.signal.spectrogram`.

    Returns
    -------
    FiringRateSpectrogramResult
        Spectrogram for each selected channel and for the population curve.
    """
    x, ch_ids = _selected_rate_matrix(fr, channels=channels)
    fs = _sampling_frequency(fr)
    nperseg_i = _default_nperseg(x.shape[1], nperseg)

    freqs, times, sxx_ch = signal.spectrogram(
        x,
        fs=fs,
        axis=1,
        nperseg=nperseg_i,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        scaling=scaling,
        mode=mode,
    )
    freqs_pop, times_pop, sxx_pop = signal.spectrogram(
        np.asarray(fr.fr_pop, dtype=np.float64),
        fs=fs,
        nperseg=nperseg_i,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
        scaling=scaling,
        mode=mode,
    )

    if not np.allclose(freqs, freqs_pop):
        raise RuntimeError("Channel and population spectrogram frequency grids differ.")
    if not np.allclose(times, times_pop):
        raise RuntimeError("Channel and population spectrogram time grids differ.")

    return FiringRateSpectrogramResult(
        freqs=freqs,
        t=times + float(fr.tstart),
        sxx_ch=sxx_ch,
        sxx_pop=sxx_pop,
        channels=ch_ids,
        fs=fs,
        method="spectrogram of firing-rate curves",
    )


def _autocorr_1d(x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Full autocorrelation for non-negative lags only."""
    y = np.asarray(x, dtype=np.float64)
    y = y - np.mean(y)

    corr = signal.correlate(y, y, mode="full", method="auto")
    corr = corr[y.size - 1 :]

    if normalize:
        zero_lag = corr[0]
        if zero_lag > 0:
            corr = corr / zero_lag
        else:
            corr = np.full_like(corr, np.nan, dtype=np.float64)

    return corr


def autocorrelation_firing_rate(
    fr: FiringRateCurveResult,
    *,
    channels: Optional[Sequence[int]] = None,
    max_lag: Optional[float] = None,
    normalize: bool = True,
) -> FiringRateAutocorrelationResult:
    """
    Compute autocorrelation of firing-rate curves.

    Parameters
    ----------
    fr:
        Firing-rate curves returned by `tad.metrics.rates.firing_rate_curve`.
    channels:
        Optional subset of channel row indices into `fr.fr_ch`.
    max_lag:
        Optional maximum lag in the same time unit as `fr.dt`.
    normalize:
        If True, normalize by the zero-lag autocorrelation, giving 1 at lag 0
        for non-constant signals.

    Returns
    -------
    FiringRateAutocorrelationResult
        Autocorrelation for selected channels and the population curve.
    """
    x, ch_ids = _selected_rate_matrix(fr, channels=channels)
    n_time = x.shape[1]

    if n_time < 2:
        raise ValueError("At least two time bins are required.")

    if max_lag is None:
        n_lags = n_time
    else:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        n_lags = min(n_time, int(np.floor(float(max_lag) / float(fr.dt))) + 1)

    autocorr_ch = np.vstack(
        [_autocorr_1d(row, normalize=normalize)[:n_lags] for row in x]
    )
    autocorr_pop = _autocorr_1d(
        np.asarray(fr.fr_pop, dtype=np.float64),
        normalize=normalize,
    )[:n_lags]

    lags = np.arange(n_lags, dtype=np.float64) * float(fr.dt)

    return FiringRateAutocorrelationResult(
        lags=lags,
        autocorr_ch=autocorr_ch,
        autocorr_pop=autocorr_pop,
        channels=ch_ids,
        method="autocorrelation of firing-rate curves",
    )
