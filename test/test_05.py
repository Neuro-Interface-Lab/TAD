import numpy as np

from tad.raster import Raster
from tad.metrics.avalanches import extract_avalanches


def test_extract_avalanches_poisson_invariants():
    rng = np.random.default_rng(0)

    n_channels = 10
    duration = 2.0
    rate_hz = 15.0

    r = Raster.empty(channels=range(n_channels))
    for ch in range(n_channels):
        n_spikes = rng.poisson(rate_hz * duration)
        times = rng.uniform(0.0, duration, size=n_spikes)
        r.insert_timestamparray(ch, times, assume_sorted=False)

    dt = 0.01
    res1 = extract_avalanches(r, dt=dt, tstart=0.0, tstop=duration, size_definition=1)
    res2 = extract_avalanches(r, dt=dt, tstart=0.0, tstop=duration, size_definition=2)

    # Lifetimes should match regardless of size definition
    assert np.array_equal(res1.lifetimes, res2.lifetimes)

    # If there are avalanches, lifetimes and sizes must be >= 1
    if res1.lifetimes.size > 0:
        assert np.all(res1.lifetimes >= 1)
        assert np.all(res1.sizes >= 1)
        assert np.all(res2.sizes >= 1)

        # size_def2 <= n_channels
        assert np.all(res2.sizes <= n_channels)

        # size_def1 >= size_def2 (since def1 counts active-ch per bin, def2 counts unique channels)
        assert np.all(res1.sizes >= res2.sizes)

    # Active bins are boolean with correct length
    assert res1.active_bins.dtype == np.bool_
    assert res1.active_bins.shape[0] == (res1.bin_edges.shape[0] - 1)

if __name__ == "__main__":
    test_extract_avalanches_poisson_invariants()
    print("All tests passed!")
