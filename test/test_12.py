import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.rates import firing_rate_curve
from tad.metrics.synchrony import pearson_corr_firing_rate


def coupled_poisson_raster(
    *,
    duration: float = 10.0,
    n_channels: int = 16,
    # Two groups of channels with different shared drives
    group_a: range = range(0, 8),
    group_b: range = range(8, 16),
    # Rates (Hz)
    rate_ind: float = 3.0,
    rate_shared_a: float = 10.0,
    rate_shared_b: float = 10.0,
    # Probability a shared event is delivered to each channel in that group
    p_deliver: float = 0.7,
    seed: int = 0,
) -> Raster:
    """
    Build a coupled-Poisson raster with block structure.

    Each channel i has:
    - independent Poisson spikes at rate_ind
    - plus group-shared spikes driven by a Poisson process at rate_shared_{A/B}
      where each shared event is delivered to each channel independently with
      probability p_deliver.

    Parameters
    ----------
    duration
        Total duration (s).
    n_channels
        Number of channels.
    group_a, group_b
        Channel index sets defining two coupled groups.
    rate_ind
        Independent rate per channel (Hz).
    rate_shared_a, rate_shared_b
        Shared drive rates for groups A and B (Hz).
    p_deliver
        Delivery probability of a shared event to each channel (0..1).
        Higher => stronger coupling.
    seed
        RNG seed.

    Returns
    -------
    Raster
        A Raster instance with coupled Poisson activity.
    """
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))

    def _poisson_times(rate_hz: float) -> np.ndarray:
        n = rng.poisson(rate_hz * duration)
        return rng.uniform(0.0, duration, size=n) if n > 0 else np.asarray([], dtype=float)

    # Independent spikes for all channels
    for ch in range(n_channels):
        t = _poisson_times(rate_ind)
        if t.size:
            r.insert_timestamparray(ch, t, assume_sorted=False)

    # Shared drive for group A
    shared_a = _poisson_times(rate_shared_a)
    if shared_a.size:
        for ch in group_a:
            mask = rng.random(shared_a.size) < p_deliver
            if np.any(mask):
                r.insert_timestamparray(ch, shared_a[mask], assume_sorted=False)

    # Shared drive for group B
    shared_b = _poisson_times(rate_shared_b)
    if shared_b.size:
        for ch in group_b:
            mask = rng.random(shared_b.size) < p_deliver
            if np.any(mask):
                r.insert_timestamparray(ch, shared_b[mask], assume_sorted=False)

    return r


def main() -> None:
    duration = 10.0
    n_channels = 16
    group_a = np.arange(0, 8)
    group_b = np.arange(8, 16)

    r = coupled_poisson_raster(
        duration=duration,
        n_channels=n_channels,
        group_a=range(0, 8),
        group_b=range(8, 16),
        rate_ind=3.0,
        rate_shared_a=10.0,
        rate_shared_b=10.0,
        p_deliver=0.7,
        seed=0,
    )

    # Use moderate dt to smooth a bit (synchrony should pop out)
    dt = 0.05
    fr = firing_rate_curve(r, dt=dt, tstart=0.0, tstop=duration)

    syn = pearson_corr_firing_rate(fr, zscore=True, drop_constant=True)
    C = syn.corr

    # -------------------------
    # ASSERTS: block structure
    # -------------------------
    assert C.shape == (n_channels, n_channels)
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.allclose(np.diag(C), 1.0, atol=1e-10)

    # Compute mean within-group correlation (excluding diagonal) vs between-group
    def mean_offdiag(mat: np.ndarray) -> float:
        m = mat.copy()
        np.fill_diagonal(m, np.nan)
        return float(np.nanmean(m))

    C_A = C[np.ix_(group_a, group_a)]
    C_B = C[np.ix_(group_b, group_b)]
    C_AB = C[np.ix_(group_a, group_b)]

    mean_within_a = mean_offdiag(C_A)
    mean_within_b = mean_offdiag(C_B)
    mean_between = float(np.mean(C_AB))

    print("Mean corr within A:", mean_within_a)
    print("Mean corr within B:", mean_within_b)
    print("Mean corr between A-B:", mean_between)

    # These should hold for this coupled construction (with high probability)
    assert mean_within_a > mean_between
    assert mean_within_b > mean_between

    # -------------------------
    # PLOTS
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # (1) Raster snippet
    r.plot(ax=axes[0, 0], tstart=0.0, tstop=2.0, show=False)
    axes[0, 0].set_title("Coupled-Poisson raster (first 2 s)")

    # (2) FR heatmap (raw Hz)
    im0 = axes[0, 1].imshow(
        fr.fr_ch,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[fr.t[0], fr.t[-1], 0, fr.fr_ch.shape[0]],
    )
    axes[0, 1].set_title("FR heatmap (Hz)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Channel")
    plt.colorbar(im0, ax=axes[0, 1])

    # (3) FR heatmap (z-scored per channel) for synchrony visibility
    Y = fr.fr_ch.astype(float)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    sd = Yc.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    Z = Yc / sd
    im1 = axes[1, 0].imshow(
        Z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[fr.t[0], fr.t[-1], 0, Z.shape[0]],
        vmin=np.percentile(Z, 2),
        vmax=np.percentile(Z, 98),
    )
    axes[1, 0].set_title("FR heatmap (z-scored per channel)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[1, 0])

    # (4) Correlation matrix
    im2 = axes[1, 1].imshow(C, aspect="equal", origin="lower", interpolation="nearest", vmin=-1.0, vmax=1.0)
    axes[1, 1].set_title("Pearson corr of z-scored FR")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Channel")
    plt.colorbar(im2, ax=axes[1, 1])

    # Draw group boundary lines for readability
    for ax in (axes[1, 1],):
        ax.axhline(7.5, color="white", linewidth=1.0)
        ax.axvline(7.5, color="white", linewidth=1.0)

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
