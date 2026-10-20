from __future__ import annotations
import numpy as np

from tad.metrics.correlation import CorrelationResult
from tad.metrics.utils import bin_spike_trains
from tad.metrics.crosscorrelation import _shuffle_spike_times_by_bin


def _pearson_matrix_from_binned(binned):
    """Calculate a Pearson matrix from channel-by-time binned spike counts."""
    n_electrodes = binned.shape[0]
    correlation_matrix = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)
    if binned.shape[1]:
        centered = binned - binned.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        variable = norms > 0
        if variable.any():
            variable_traces = centered[variable]
            variable_norms = norms[variable]
            correlation_matrix[np.ix_(variable, variable)] = (
                variable_traces @ variable_traces.T
                / np.outer(variable_norms, variable_norms)
            )
            np.clip(correlation_matrix, -1.0, 1.0, out=correlation_matrix)
    return correlation_matrix


def compute_pearson_correlation(
    raster,
    tstart: float = 0,
    tstop: float = None,
    binsize: float = 0.001,
    n_surrogates: int = 0,
    surrogate_half_percentile: float = 90.0,
    random_state: int | None = None,
) -> CorrelationResult:
    """
    This function receives a raster object, and selects the events between tstart and tstop.
    Then, it computes the Pearson correlation coefficient between all pairs of electrodes in the raster object.
    
    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - binsize: The size of the bins (in seconds) for the spike train binning
    - n_surrogates: Number of independently shuffled time-bin datasets. Zero
      (the default) disables surrogate thresholds.
    - surrogate_half_percentile: Percentile of the distance from the median,
      calculated separately below and above the median. A value of 90 sets
      the lower bound 90% of the lower-half spread below the median and the
      upper bound 90% of the upper-half spread above it (about global 5th and
      95th percentiles for an even, continuous sample).
    - random_state: Optional seed for reproducible surrogate shuffling.
    - Returns:
    - A CorrelationResult dataclass containing the correlation matrix, the correlated channels, and additional information about the analysis.
    
    """
    if tstop == None:
        tstop = max(np.max(times) for times in raster.events.values() if len(times)>0)

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
    n_electrodes = len(channels)
    if binsize <= 0:
        raise ValueError("binsize must be positive")
    if int(n_surrogates) != n_surrogates or n_surrogates < 0:
        raise ValueError("n_surrogates must be a non-negative integer")
    if not 50.0 < surrogate_half_percentile < 100.0:
        raise ValueError("surrogate_half_percentile must be between 50 and 100")
    if n_surrogates and n_surrogates < 2:
        raise ValueError("At least two surrogates are required for median splitting")

    # Bin all channels once, then calculate all pairwise correlations together.
    binned = bin_spike_trains(events, tstart, tstop, binsize=binsize)[2]
    correlation_matrix = _pearson_matrix_from_binned(binned)

    info = {
        "tstart": tstart,
        "tstop": tstop,
        "binsize": binsize,
    }

    if n_surrogates:
        rng = np.random.default_rng(random_state)
        spike_times = [np.sort(events[channel]) for channel in channels]
        surrogate_correlations = np.empty(
            (n_surrogates, n_electrodes, n_electrodes), dtype=np.float64
        )
        for surrogate_index in range(n_surrogates):
            shuffled_times = _shuffle_spike_times_by_bin(
                spike_times, tstart, tstop, binsize, rng
            )
            shuffled_events = dict(zip(channels, shuffled_times, strict=True))
            shuffled_binned = bin_spike_trains(
                shuffled_events, tstart, tstop, binsize=binsize
            )[2]
            surrogate_correlations[surrogate_index] = (
                _pearson_matrix_from_binned(shuffled_binned)
            )

        surrogate_median = np.median(surrogate_correlations, axis=0)
        # Sort and split by rank so values tied at the median are not counted
        # in both halves. For an odd surrogate count, omit the single central
        # value from the two tails; the median remains the center reference.
        ordered_surrogates = np.sort(surrogate_correlations, axis=0)
        half_count = n_surrogates // 2
        lower_half = ordered_surrogates[:half_count]
        upper_half = ordered_surrogates[-half_count:]
        lower_deviation = surrogate_median[None, :, :] - lower_half
        upper_deviation = upper_half - surrogate_median[None, :, :]
        lower_threshold = surrogate_median - np.percentile(
            lower_deviation, surrogate_half_percentile, axis=0
        )
        upper_threshold = surrogate_median + np.percentile(
            upper_deviation, surrogate_half_percentile, axis=0
        )
        info.update(
            {
                "surrogate_method": (
                    "independent_permutation_of_time_bins_with_within_bin_offsets_preserved"
                ),
                "correlation_representation": "binned_spike_counts",
                "n_surrogates": int(n_surrogates),
                "surrogate_half_percentile": float(surrogate_half_percentile),
                "random_state": random_state,
                "surrogate_median": surrogate_median,
                "surrogate_lower_threshold": lower_threshold,
                "surrogate_upper_threshold": upper_threshold,
                "surrogate_significant": (
                    (correlation_matrix < lower_threshold)
                    | (correlation_matrix > upper_threshold)
                ),
            }
        )

    return CorrelationResult(
        correlation=correlation_matrix,
        tau=None,
        channels=channels,
        info=info
    )

def compute_delayed_pearson_correlation(raster, tstart: float = 0.0, tstop :float = None, binsize : float = 0.001, taumax : float = 0.2) -> CorrelationResult:
    """
    This function receives a raster object, and selects the events between tstart and tstop.
    Then, it computes delayed Pearson correlations between all electrode pairs
    and selects the coefficient with the largest absolute magnitude. The
    selected coefficient keeps its sign, and the corresponding signed delay
    is returned in ``CorrelationResult.tau``. A positive tau means that the
    second channel is evaluated later than the first channel.
    
    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - binsize: The size of the bins (in seconds) for the spike train binning
    - taumax: The maximum delay (in seconds) to consider for the delayed correlation
    - Returns:
    - A CorrelationResult dataclass containing the correlation matrix, the correlated channels, and additional information about the analysis.
    
    """
    if binsize <= 0:
        raise ValueError("binsize must be positive")
    if taumax < 0:
        raise ValueError("taumax must be non-negative")

    if tstop is None:
        non_empty_times = [times for times in raster.events.values() if len(times) > 0]
        if not non_empty_times:
            raise ValueError("Cannot infer tstop from a raster with no spikes")
        tstop = max(np.max(times) for times in non_empty_times)

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

    channels, _, binned = bin_spike_trains(
        events, tstart, tstop, binsize=binsize
    )
    n_electrodes, n_bins = binned.shape
    correlation_matrix = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)
    tau_matrix = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)

    # Use only overlapping bins for each lag. Matrix multiplication calculates
    # all channel pairs at once, avoiding the Python loop over channel pairs.
    # Unlike np.roll, these slices do not wrap activity across the window edge.
    max_lag_bins = min(
        int(np.floor(taumax / binsize + 1e-12)),
        max(0, n_bins - 2),
    )
    lag_bins = np.arange(-max_lag_bins, max_lag_bins + 1)
    best_abs_correlation = np.full(
        (n_electrodes, n_electrodes), -1.0, dtype=np.float64
    )

    for lag in lag_bins:
        if lag > 0:
            x = binned[:, :-lag]
            y = binned[:, lag:]
        elif lag < 0:
            x = binned[:, -lag:]
            y = binned[:, :lag]
        else:
            x = binned
            y = binned

        n_samples = x.shape[1]
        if n_samples < 2:
            continue

        # Pearson r = centered cross-product / product of centered norms.
        # This computes a full channel-by-channel matrix for this lag.
        sum_x = x.sum(axis=1)
        sum_y = y.sum(axis=1)
        covariance = x @ y.T - np.outer(sum_x, sum_y) / n_samples
        variance_x = np.maximum(
            np.einsum("ij,ij->i", x, x) - sum_x**2 / n_samples, 0.0
        )
        variance_y = np.maximum(
            np.einsum("ij,ij->i", y, y) - sum_y**2 / n_samples, 0.0
        )
        denominator = np.sqrt(np.outer(variance_x, variance_y))

        correlations = np.zeros_like(covariance)
        valid = denominator > 0
        np.divide(covariance, denominator, out=correlations, where=valid)
        np.clip(correlations, -1.0, 1.0, out=correlations)

        stronger = valid & (np.abs(correlations) > best_abs_correlation)
        correlation_matrix[stronger] = correlations[stronger]
        tau_matrix[stronger] = lag * binsize
        best_abs_correlation[stronger] = np.abs(correlations[stronger])

    # A variable channel has self-correlation 1 at zero delay.
    variable_channels = np.std(binned, axis=1) > 0
    diagonal = np.flatnonzero(variable_channels)
    correlation_matrix[diagonal, diagonal] = 1.0
    tau_matrix[diagonal, diagonal] = 0.0

    info = {
        "tstart": tstart,
        "tstop": tstop,
        "binsize": binsize,
        "taumax": taumax,
        "lag_selection": "maximum_absolute_correlation",
    }

    return CorrelationResult(
        correlation=correlation_matrix,
        tau=tau_matrix,
        channels=channels,
        info=info
    )
