import numpy as np

from tad.raster import Raster
from tad.metrics.avalanches import extract_avalanches


def test_extract_avalanches_handcrafted():
    # Build a raster where bin activity is exactly known.
    # dt = 1.0, window [0, 6)
    # Bins: 0,1 active; 2 blank; 3,4 active; 5 blank
    #
    # Bin 0: ch0 fires
    # Bin 1: ch0 and ch1 fire
    # Bin 3: ch2 fires
    # Bin 4: ch2 fires
    r = Raster.empty(channels=[0, 1, 2])
    r.insert_timestamparray(0, [0.1, 1.1])   # bin 0 and 1
    r.insert_timestamparray(1, [1.2])        # bin 1
    r.insert_timestamparray(2, [3.1, 4.2])   # bins 3 and 4

    res1 = extract_avalanches(r, dt=1.0, tstart=0.0, tstop=6.0, size_definition=1)
    res2 = extract_avalanches(r, dt=1.0, tstart=0.0, tstop=6.0, size_definition=2)

    # Two avalanches: bins [0,2) and [3,5)
    assert res1.intervals_bins == [(0, 2), (3, 5)]
    assert np.array_equal(res1.lifetimes, np.array([2, 2], dtype=np.int64))
    assert np.array_equal(res2.lifetimes, np.array([2, 2], dtype=np.int64))

    # Size definition 1:
    # avalanche 1: bin0 has 1 active ch, bin1 has 2 => size 3
    # avalanche 2: bin3 has 1, bin4 has 1 => size 2
    assert np.array_equal(res1.sizes, np.array([3, 2], dtype=np.int64))

    # Size definition 2:
    # avalanche 1 uses channels {0,1} => 2
    # avalanche 2 uses channel {2} => 1
    assert np.array_equal(res2.sizes, np.array([2, 1], dtype=np.int64))


if __name__ == "__main__":
    test_extract_avalanches_handcrafted()
    print("All tests passed!")