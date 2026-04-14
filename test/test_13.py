import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.isi import isi, isih


def make_poisson_raster(n_channels=8, duration=5.0, rate_hz=10.0, seed=0) -> Raster:
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))
    for ch in range(n_channels):
        n = rng.poisson(rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)
    return r


def main() -> None:
    r = make_poisson_raster(n_channels=8, duration=5.0, rate_hz=10.0, seed=0)

    # Per-channel ISIs
    res_ch = isi(r, mode="per_channel", tstart=0.0, tstop=5.0)
    assert res_ch.mode == "per_channel"
    assert isinstance(res_ch.isi, dict)
    for ch, d in res_ch.isi.items():
        assert np.all(d > 0.0)

    # Pooled ISIs
    res_pool = isi(r, mode="pooled", tstart=0.0, tstop=5.0)
    assert res_pool.mode == "pooled"
    assert res_pool.isi.ndim == 1
    assert np.all(res_pool.isi > 0.0)

    print("OK: ISI asserts passed.")
    print("Pooled ISI count:", res_pool.isi.size)

    # Plot: pooled ISI histogram in linear and log domain
    centers_lin, h_lin = isih(res_pool.isi, bins=60, density=True, log=False)
    centers_log, h_log = isih(res_pool.isi, bins=60, density=True, log=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(centers_lin, h_lin)
    axes[0].set_title("Pooled ISI density (linear)")
    axes[0].set_xlabel("ISI (s)")
    axes[0].set_ylabel("density")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    axes[1].plot(centers_log, h_log)
    axes[1].set_title("Pooled log10(ISI) density")
    axes[1].set_xlabel("log10(ISI)")
    axes[1].set_ylabel("density")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
