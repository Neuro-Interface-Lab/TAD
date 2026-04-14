import numpy as np

from tad.raster import Raster
from tad.metrics.rates import firing_rate_curve
from tad.metrics.scalar import spike_count


def make_poisson_raster(
    n_channels: int = 10,
    duration: float = 2.0,
    rate_hz: float = 15.0,
    seed: int = 0,
) -> Raster:
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))
    for ch in range(n_channels):
        n = rng.poisson(rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)
    return r


def test_firing_rate_curve_identity() -> None:
    duration = 2.0
    dt = 0.02
    r = make_poisson_raster(n_channels=10, duration=duration, rate_hz=15.0, seed=0)

    res = firing_rate_curve(r, dt=dt, tstart=0.0, tstop=duration)

    # Basic shapes
    assert res.fr_ch.ndim == 2
    n_ch, n_bins = res.fr_ch.shape
    assert n_ch == 10
    assert res.t.shape == (n_bins,)
    assert res.fr_pop.shape == (n_bins,)
    assert res.bin_edges.shape == (n_bins + 1,)

    # Identity 1: population rate is sum of per-channel rates
    assert np.allclose(res.fr_pop, res.fr_ch.sum(axis=0))

    # Identity 2: integrating population FR over time returns total spike count in window
    # Discrete integral: sum(fr_pop * dt) == total spikes in window
    total_spikes = spike_count(r, tstart=0.0, tstop=duration, per_channel=False)
    approx_spikes = float(np.sum(res.fr_pop) * dt)
    assert abs(approx_spikes - total_spikes) < 1e-9

    # Identity 3: per-channel integrals match per-channel counts
    counts_per_ch = spike_count(r, tstart=0.0, tstop=duration, per_channel=True)
    approx_counts = np.sum(res.fr_ch, axis=1) * dt
    assert np.allclose(approx_counts, counts_per_ch.astype(float), atol=1e-9, rtol=0.0)


if __name__ == "__main__":
    # Run the test directly without pytest
    test_firing_rate_curve_identity()
    print("OK: firing_rate_curve assert tests passed.")
