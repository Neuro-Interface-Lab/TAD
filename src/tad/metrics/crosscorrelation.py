from __future__ import annotations
from importlib import import_module
import numpy as np

from tad.metrics.correlation import CorrelationResult

try:
    _numba = import_module("numba")
except ImportError:
    _numba = None
else:
    @_numba.njit(parallel=True)
    def _crosscorrelation_numba(times, lengths, tau_range, binsize):
        n_channels = len(lengths)
        correlation = np.zeros((n_channels, n_channels))
        tau = np.zeros_like(correlation)
        half_bin = binsize / 2

        for i in _numba.prange(n_channels):
            nx = lengths[i]
            if nx == 0:
                continue
            for j in range(n_channels):
                ny = lengths[j]
                if ny == 0:
                    continue
                best_value = 0.0
                best_lag = 0
                for k in range(len(tau_range)):
                    count = 0
                    for x_index in range(nx):
                        spike_x = times[i, x_index]
                        lower = spike_x - tau_range[k] - half_bin
                        upper = spike_x - tau_range[k] + half_bin
                        for y_index in range(ny):
                            spike_y = times[j, y_index]
                            if spike_y >= lower and spike_y < upper:
                                count += 1
                    value = count / np.sqrt(nx * ny)
                    if k == 0 or value > best_value:
                        best_value = value
                        best_lag = k
                correlation[i, j] = best_value
                tau[i, j] = tau_range[best_lag]

        return correlation, tau


def _crosscorrelation_from_spike_times(spike_times, tau_range, binsize):
    """Calculate the maximum normalized coincidence count for each pair."""
    lengths = np.asarray([len(times) for times in spike_times], dtype=np.int64)
    n_channels = len(spike_times)
    correlation = np.zeros((n_channels, n_channels), dtype=np.float64)
    tau = np.zeros_like(correlation)
    if not n_channels:
        return correlation, tau

    total_spikes = int(lengths.sum())
    estimated_operations = len(tau_range) * total_spikes**2
    if _crosscorrelation_numba is not None and estimated_operations >= 2_000_000:
        max_spikes = max(lengths, default=0)
        padded_times = np.zeros((n_channels, max_spikes), dtype=np.float64)
        for index, times in enumerate(spike_times):
            padded_times[index, :len(times)] = times
        return _crosscorrelation_numba(
            padded_times, lengths, tau_range, binsize
        )

    half_bin = binsize / 2
    for i, spike_times_x in enumerate(spike_times):
        n_x = len(spike_times_x)
        if n_x == 0:
            continue
        lower = spike_times_x[:, None] - tau_range[None, :] - half_bin
        upper = spike_times_x[:, None] - tau_range[None, :] + half_bin
        for j, spike_times_y in enumerate(spike_times):
            n_y = len(spike_times_y)
            if n_y == 0:
                continue
            counts = (
                np.searchsorted(spike_times_y, upper, side="left")
                - np.searchsorted(spike_times_y, lower, side="left")
            ).sum(axis=0)
            values = counts / np.sqrt(n_x * n_y)
            best_lag = int(np.argmax(values))
            correlation[i, j] = values[best_lag]
            tau[i, j] = tau_range[best_lag]

    return correlation, tau


def _shuffle_spike_times_by_bin(spike_times, tstart, tstop, binsize, rng):
    """Permute time-bin blocks independently, preserving within-bin spike times.

    For every spike, its offset from the source bin's left edge is preserved
    when the contents of that bin are moved to a destination bin. A final
    partial bin, if present, remains at the end of the interval because it
    cannot be exchanged with full-width bins without changing spike times or
    moving spikes outside the analyzed interval.
    """
    duration = tstop - tstart
    n_full_bins = int(np.floor(duration / binsize))
    if n_full_bins < 2:
        raise ValueError(
            "At least two full bins are required for bin-shuffle surrogates"
        )
    full_edges = tstart + np.arange(n_full_bins + 1, dtype=float) * binsize
    full_stop = full_edges[-1]

    shuffled_trains = []
    for times in spike_times:
        # Determine each spike's source bin and within-bin offset once. This
        # avoids scanning the full spike vector separately for every bin.
        source_bins = np.searchsorted(full_edges, times, side="right") - 1
        in_full_bins = (source_bins >= 0) & (source_bins < n_full_bins)
        full_source_bins = source_bins[in_full_bins]
        within_bin_offsets = times[in_full_bins] - full_edges[full_source_bins]

        # source_for_destination[destination] gives the source bin moved there.
        source_for_destination = rng.permutation(n_full_bins)
        destination_for_source = np.empty(n_full_bins, dtype=np.int64)
        destination_for_source[source_for_destination] = np.arange(n_full_bins)
        shuffled_full_times = (
            full_edges[destination_for_source[full_source_bins]]
            + within_bin_offsets
        )

        # Keep spikes in the shorter final bin fixed, as before.
        partial_bin_spikes = times[~in_full_bins]
        shuffled_trains.append(
            np.sort(np.concatenate((shuffled_full_times, partial_bin_spikes)))
        )

    return shuffled_trains


def compute_crosscorrelation(
    raster,
    tstart: float = 0,
    tstop: float = None,
    taumax: float = None,
    fsample: float = None,
    binsize: float = None,
    n_surrogates: int = 0,
    surrogate_percentile: float = 95.0,
    random_state: int | None = None,
) -> CorrelationResult:
    """
    This function receives a raster object between tstart and tstop, and computes the cross-correlation between all pairs of electrodes in the raster object. The function returns a CrossCorrelationResult dataclass containing the cross-correlation matrix, the tau matrix, and additional information about the analysis.

    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - taumax: Maximum lag in seconds. This implementation tests lags from
      zero up to, but not including, taumax.
    - fsample: Sampling frequency of the data.
    - binsize: Spike-coincidence tolerance and surrogate bin width.
    - n_surrogates: Number of independently shuffled surrogate datasets. Zero
      (the default) disables surrogate threshold calculation.
    - surrogate_percentile: Percentile of surrogate maximum crosscorrelations
      used as the pair-specific adaptive threshold.
    - random_state: Optional seed for reproducible surrogate shuffling.

    When surrogates are enabled, each surrogate independently permutes the
    full-width time-bin blocks in every channel, preserving each spike's
    offset within its bin. The observed correlation remains calculated from
    the original spike timestamps. ``result.info`` also contains
    ``surrogate_threshold`` (a pair-specific threshold matrix),
    ``surrogate_pvalue`` (empirical upper-tail p-values), and
    ``surrogate_significant`` (observed values greater than the threshold).
    The observed ``result.correlation`` matrix is not thresholded.
    """
    if tstop is None:
        non_empty_times = [times for times in raster.events.values() if len(times)]
        if not non_empty_times:
            raise ValueError("Cannot infer tstop from a raster with no spikes")
        tstop = max(float(np.max(times)) for times in non_empty_times)
    if tstop <= tstart:
        raise ValueError("tstop must be greater than tstart")

    # reduce events between tstart and tstop
    events = {
            channel_id: times[(times>=tstart) & (times <tstop)] 
            for channel_id, times in raster.events.items()    
        }

    events = dict(
        sorted(
            events.items(),
            key = lambda item: int(item[0][2:])
        )
    )

    channels = list(events.keys())

    if fsample is None:
        fsample = 20000 # Hz

    if binsize is None:
        binsize =200*1/fsample # s
    if taumax is None:
        taumax = 0.2 # 200 ms

    if binsize <= 0:
        raise ValueError("binsize must be positive")
    if taumax < 0:
        raise ValueError("taumax must be non-negative")
    if int(n_surrogates) != n_surrogates or n_surrogates < 0:
        raise ValueError("n_surrogates must be a non-negative integer")
    if not 0 < surrogate_percentile < 100:
        raise ValueError("surrogate_percentile must be between 0 and 100")

    spike_times = [np.sort(events[channel]) for channel in channels]
    tau_range = np.arange(0, taumax, step=binsize)
    if not tau_range.size:
        tau_range = np.array([0.0])
    cross_correlation, tau = _crosscorrelation_from_spike_times(
        spike_times, tau_range, binsize
    )

    info = {
        "binsize": binsize,
        "tstart": tstart,
        "tstop": tstop,
        "taumax": taumax,
    }

    if n_surrogates:
        rng = np.random.default_rng(random_state)
        surrogate_correlations = np.empty(
            (n_surrogates, len(channels), len(channels)), dtype=np.float64
        )
        for surrogate_index in range(n_surrogates):
            shuffled_spike_times = _shuffle_spike_times_by_bin(
                spike_times, tstart, tstop, binsize, rng
            )
            surrogate_correlations[surrogate_index], _ = (
                _crosscorrelation_from_spike_times(
                    shuffled_spike_times, tau_range, binsize
                )
            )

        adaptive_threshold = np.percentile(
            surrogate_correlations, surrogate_percentile, axis=0
        )
        empirical_pvalue = (
            1
            + np.sum(
                surrogate_correlations >= cross_correlation[None, :, :], axis=0
            )
        ) / (n_surrogates + 1)
        info.update(
            {
                "surrogate_method": "independent_permutation_of_time_bins_with_within_bin_offsets_preserved",
                "correlation_representation": "original_spike_timestamps",
                "n_surrogates": int(n_surrogates),
                "surrogate_percentile": float(surrogate_percentile),
                "random_state": random_state,
                "surrogate_threshold": adaptive_threshold,
                "surrogate_pvalue": empirical_pvalue,
                "surrogate_significant": cross_correlation > adaptive_threshold,
            }
        )

    return CorrelationResult(
        correlation=cross_correlation,
        tau=tau,
        channels=channels,
        info=info,
    )
