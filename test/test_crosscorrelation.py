import numpy as np

from tad import Raster
from tad.metrics.crosscorrelation import compute_crosscorrelation


def reference_crosscorrelation(raster, tstart, tstop, binsize, taumax):
    events = {
        channel: times[(times >= tstart) & (times < tstop)]
        for channel, times in raster.events.items()
    }
    events = dict(sorted(events.items(), key=lambda item: int(item[0][2:])))
    channels = list(events)
    tau_range = np.arange(0, taumax, step=binsize)
    correlation = np.zeros((len(channels), len(channels)))
    tau = np.zeros_like(correlation)

    for i, channel_x in enumerate(channels):
        for j, channel_y in enumerate(channels):
            spike_times_x = events[channel_x]
            spike_times_y = events[channel_y]
            counts = np.zeros(len(tau_range))
            for k, lag in enumerate(tau_range):
                for spike_time in spike_times_x:
                    counts[k] += np.sum(
                        (spike_times_y >= spike_time - lag - binsize / 2)
                        & (spike_times_y < spike_time - lag + binsize / 2)
                    )
            if len(spike_times_x) and len(spike_times_y):
                counts /= np.sqrt(len(spike_times_x) * len(spike_times_y))
            best_lag = np.argmax(counts)
            correlation[i, j] = counts[best_lag]
            tau[i, j] = tau_range[best_lag]

    return channels, correlation, tau


def test_vectorized_crosscorrelation_matches_reference():
    rng = np.random.default_rng(23)
    events = {
        f"Ch{channel:02d}": np.sort(rng.uniform(-0.1, 1.1, int(rng.integers(0, 35))))
        for channel in range(1, 13)
    }
    events["Ch01"] = np.array([0.005, 0.105, 0.205, 0.305, 0.995])
    events["Ch02"] = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    raster = Raster(events=events)

    expected_channels, expected, expected_tau = reference_crosscorrelation(
        raster, 0, 1, 0.01, 0.2
    )
    actual = compute_crosscorrelation(
        raster, tstart=0, tstop=1, binsize=0.01, taumax=0.2
    )

    assert actual.channels == expected_channels
    np.testing.assert_array_equal(actual.correlation, expected)
    np.testing.assert_array_equal(actual.tau, expected_tau)
