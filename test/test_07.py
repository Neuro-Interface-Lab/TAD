import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.rates import firing_rate_curve


def main():
    rng = np.random.default_rng(0)
    n_channels, duration, rate_hz = 10, 2.0, 15.0

    r = Raster.empty(channels=range(n_channels))


    for ch in range(n_channels):
        n = rng.poisson(rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)
    
    r.plot(tstart=0.0, tstop=duration)

    res = firing_rate_curve(r, dt=0.02, tstart=0.0, tstop=duration)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(res.t, res.fr_pop)
    ax.set_xlabel("Time")
    ax.set_ylabel("Population FR (Hz)")
    ax.set_title("Population firing-rate curve")
    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
