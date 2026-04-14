import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.rates import firing_rate_curve
from tad.metrics.synchrony import pearson_corr_firing_rate


def make_poisson_raster(n_channels=16, duration=2.0, rate_hz=10.0, seed=0) -> Raster:
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))
    for ch in range(n_channels):
        n = rng.poisson(rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)
    return r


def main() -> None:
    r = make_poisson_raster(n_channels=16, duration=2.0, rate_hz=10.0, seed=0)

    # Compute FR curves (choose dt a bit larger for smoother correlations)
    fr = firing_rate_curve(r, dt=0.02, tstart=0.0, tstop=2.0)

    syn = pearson_corr_firing_rate(fr, zscore=True, drop_constant=True)

    C = syn.corr

    # -------------------------
    # ASSERTS
    # -------------------------
    assert C.ndim == 2
    assert C.shape[0] == C.shape[1]
    assert C.shape[0] == len(syn.channels)

    # Symmetry and diagonal ~ 1
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.allclose(np.diag(C), 1.0, atol=1e-10)

    # Values should lie in [-1, 1]
    assert np.max(C) <= 1.0 + 1e-12
    assert np.min(C) >= -1.0 - 1e-12

    print("OK: Pearson FR synchrony asserts passed.")
    if syn.masked_channels:
        print("Masked constant channels:", syn.masked_channels)

    # -------------------------
    # PLOT: FR heatmap + corr matrix
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # FR heatmap (raw) for context
    im0 = axes[0].imshow(
        fr.fr_ch,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[fr.t[0], fr.t[-1], 0, fr.fr_ch.shape[0]],
    )
    axes[0].set_title("Firing rate heatmap (Hz)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Channel index")
    plt.colorbar(im0, ax=axes[0])

    # Correlation matrix
    im1 = axes[1].imshow(C, aspect="equal", origin="lower", interpolation="nearest", vmin=-1.0, vmax=1.0)
    axes[1].set_title("Pearson corr of z-scored FR")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
