from tad import Raster
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Quick demo / "neural-like" Poisson test on ~10 channels
# -------------------------------------------------------------------------

def test_demo_01() -> None:
    rng = np.random.default_rng(0)

    n_channels = 10
    duration = 2.0  # seconds
    rate_hz = 15.0  # spikes/s per channel (roughly)

    r = Raster.empty(channels=range(n_channels))

    # Homogeneous Poisson process per channel: N ~ Poisson(rate * T), times ~ Uniform(0, T)
    for ch in range(n_channels):
        n_spikes = rng.poisson(rate_hz * duration)
        times = rng.uniform(0.0, duration, size=n_spikes)
        r.insert_timestamparray(ch, times, assume_sorted=False)

    # Add one extra channel dynamically + then pop it to test channel management
    r.insert_channel("extra", times=[0.1, 0.4, 1.2], overwrite=False)
    popped = r.pop_channel("extra")
    print("Popped channel 'extra' had:", popped)

    # Plot
    r.plot(tstart=0.0, tstop=duration)

    plt.show()
