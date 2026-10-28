"""Spike-time tiling coefficient (STTC) for pairs of spike trains."""
from __future__ import annotations

import numpy as np

from tad.metrics.correlation import CorrelationResult
from tad.raster import Raster


def _spikes_within_tolerance(source: np.ndarray, target: np.ndarray, dt: float) -> float:
    """Fraction of source spikes with any target spike within ``dt``."""
    if source.size == 0 or target.size == 0:
        return np.nan

    right = np.searchsorted(target, source, side="left")
    has_match = np.zeros(source.size, dtype=bool)

    has_left = right > 0
    has_right = right < target.size
    has_match[has_left] |= (
        np.abs(source[has_left] - target[right[has_left] - 1]) <= dt
    )
    has_match[has_right] |= (
        np.abs(source[has_right] - target[right[has_right]]) <= dt
    )
    return float(np.mean(has_match))


def _time_coverage_fraction(
    spike_times: np.ndarray,
    tstart: float,
    tstop: float,
    dt: float,
) -> float:
    """Fraction of the recording covered by the union of spike tiles."""
    if spike_times.size == 0:
        return 0.0

    starts = np.maximum(spike_times - dt, tstart)
    stops = np.minimum(spike_times + dt, tstop)

    covered = 0.0
    current_start = starts[0]
    current_stop = stops[0]
    for interval_start, interval_stop in zip(starts[1:], stops[1:]):
        if interval_start <= current_stop:
            current_stop = max(current_stop, interval_stop)
        else:
            covered += current_stop - current_start
            current_start = interval_start
            current_stop = interval_stop
    covered += current_stop - current_start

    return float(covered / (tstop - tstart))


def _sttc_value(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    coverage_a: float,
    coverage_b: float,
    dt: float,
) -> float:
    """Calculate STTC for one channel pair."""
    if spikes_a.size == 0 or spikes_b.size == 0:
        # Encode a silent pair as no observed edge for network construction.
        return 0.0

    p_a = _spikes_within_tolerance(spikes_a, spikes_b, dt)
    p_b = _spikes_within_tolerance(spikes_b, spikes_a, dt)
    denominator_a = 1.0 - p_a * coverage_b
    denominator_b = 1.0 - p_b * coverage_a

    # The formula is undefined when a train's tiles cover all recording time
    # and every spike from the other train lies inside those tiles.
    epsilon = 100.0 * np.finfo(float).eps
    if denominator_a <= epsilon or denominator_b <= epsilon:
        return np.nan

    term_a = (p_a - coverage_b) / denominator_a
    term_b = (p_b - coverage_a) / denominator_b
    return float(0.5 * (term_a + term_b))


def _channel_sort_key(channel):
    """Sort standard ``ChN`` and integer channel IDs numerically."""
    if isinstance(channel, (int, np.integer)):
        return (0, int(channel))
    label = str(channel)
    numeric_label = label[2:] if label.startswith("Ch") else label
    try:
        return (0, int(numeric_label))
    except ValueError:
        return (1, label)


def compute_sttc(
    raster: Raster,
    tstart: float = 0.0,
    tstop: float | None = None,
    dt: float = 0.01,
) -> CorrelationResult:
    """Compute the pairwise Spike-Time Tiling Coefficient matrix.

    For each pair of channels A and B, STTC is

    ``0.5 * ((P_A - T_B) / (1 - P_A*T_B)
             + (P_B - T_A) / (1 - P_B*T_A))``

    where ``P_A`` is the fraction of A spikes within ``±dt`` of any B spike,
    and ``T_A`` is the fraction of the recording covered by the union of
    ``±dt`` intervals around A spikes. The corresponding quantities for B are
    defined analogously. Overlapping tiles count only once.

    Parameters
    ----------
    raster
        Raster containing sorted spike times by channel.
    tstart, tstop
        Recording interval ``[tstart, tstop)`` in seconds. ``tstop`` must be
        supplied: STTC's tile-coverage correction depends on the actual
        recording duration, including silent time after the final spike.
    dt
        Coincidence tolerance in seconds. This is the STTC timescale
        ``Δt``; use one fixed value when comparing conditions.

    Returns
    -------
    CorrelationResult
        Symmetric matrix with STTC values in ``[-1, 1]`` and ``tau=None``.
        Pairs where either train is empty are represented by zero for network
        construction. A nonempty train's diagonal value is 1; a silent
        channel's diagonal is 0. A nonempty pair for which the STTC formula is
        mathematically undefined remains ``NaN``.

    Notes
    -----
    STTC is a measure of pairwise spike-time association, not evidence of a
    direct or causal synaptic connection. With very few spikes, estimates can
    still be unstable; the coefficient does not replace minimum-activity or
    statistical-significance criteria.
    """
    if not isinstance(raster, Raster):
        raise TypeError("raster must be a tad.Raster instance")
    if not np.isfinite(tstart):
        raise ValueError("tstart must be finite")
    if tstop is None or not np.isfinite(tstop):
        raise ValueError("tstop must be supplied and finite for STTC")
    if tstop <= tstart:
        raise ValueError("tstop must be greater than tstart")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    channels = sorted(raster.events, key=_channel_sort_key)
    spike_times = [
        np.asarray(
            raster.events[channel][
                (raster.events[channel] >= tstart)
                & (raster.events[channel] < tstop)
            ],
            dtype=np.float64,
        )
        for channel in channels
    ]

    n_channels = len(channels)
    sttc = np.zeros((n_channels, n_channels), dtype=np.float64)
    coverage = [
        _time_coverage_fraction(times, tstart, tstop, dt)
        for times in spike_times
    ]

    for i in range(n_channels):
        sttc[i, i] = 1.0 if spike_times[i].size else 0.0
        for j in range(i + 1, n_channels):
            value = _sttc_value(
                spike_times[i],
                spike_times[j],
                coverage[i],
                coverage[j],
                dt,
            )
            sttc[i, j] = value
            sttc[j, i] = value

    info = {
        "method": "spike_time_tiling_coefficient",
        "tstart": float(tstart),
        "tstop": float(tstop),
        "recording_duration": float(tstop - tstart),
        "dt": float(dt),
    }
    return CorrelationResult(
        correlation=sttc,
        tau=None,
        channels=channels,
        info=info,
    )
