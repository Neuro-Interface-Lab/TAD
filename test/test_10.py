import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.avalanches import extract_avalanches
from tad.metrics.rates import firing_rate_curve
from tad.metrics.powerlaw import fit_avalanche_powerlaw
from tad.plotting import plot_firing_rate_stack, plot_firing_rate_heatmap

def simulate_critical_branching_avalanches(rng, n_avalanches=800, max_steps=250):
    avalanches = []
    for _ in range(n_avalanches):
        a = []
        active = 1
        steps = 0
        while active > 0 and steps < max_steps:
            a.append(active)
            active = rng.poisson(lam=1.0, size=active).sum()
            steps += 1
        if a:
            avalanches.append(a)
    return avalanches


def avalanches_to_raster(avalanches, n_channels=16, dt=0.002, gap_bins=2, seed=1):
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))
    tbin = 0
    for a in avalanches:
        tbin += gap_bins
        for k, count in enumerate(a):
            if count <= 0:
                continue
            chs = rng.integers(0, n_channels, size=count)
            base = (tbin + k) * dt
            times = base + rng.uniform(0.0, dt * 0.999, size=count)
            for ch in np.unique(chs):
                r.insert_timestamparray(ch, times[chs == ch], assume_sorted=False)
        tbin += len(a)
    return r


def pmf_from_samples(samples):
    s = np.asarray(samples, dtype=int)
    s = s[s >= 1]
    if s.size == 0:
        return np.asarray([]), np.asarray([])
    x, c = np.unique(s, return_counts=True)
    p = c / c.sum()
    return x.astype(float), p.astype(float)


def main() -> None:
    rng = np.random.default_rng(0)
    dt = 0.002
    n_channels = 16

    avals = simulate_critical_branching_avalanches(rng, n_avalanches=800, max_steps=250)
    r = avalanches_to_raster(avals, n_channels=n_channels, dt=dt, gap_bins=2, seed=1)

    aval = extract_avalanches(r, dt=dt, size_definition=1)
    fr = firing_rate_curve(r, dt=dt)

    fit_s = fit_avalanche_powerlaw(aval, variable="size", exclude_unit=True, tail_fraction=0.01)
    fit_l = fit_avalanche_powerlaw(aval, variable="lifetime", exclude_unit=True, tail_fraction=0.01)

    # -------------------------
    # ASSERTS (sanity)
    # -------------------------
    assert np.isfinite(fit_s.rmse) or np.isnan(fit_s.rmse)
    assert np.isfinite(fit_l.rmse) or np.isnan(fit_l.rmse)

    # If we had enough points, we expect a negative exponent and positive prefactor.
    if np.isfinite(fit_s.b):
        assert fit_s.a > 0.0
        assert fit_s.b < 0.0
        assert fit_s.x_used.size >= 5

    if np.isfinite(fit_l.b):
        assert fit_l.a > 0.0
        assert fit_l.b < 0.0
        assert fit_l.x_used.size >= 5

    print("OK: power-law fit sanity checks passed.")
    print(f"Size fit:     a={fit_s.a:.4g}, b={fit_s.b:.3f}, rmse={fit_s.rmse:.3g}, points={fit_s.x_used.size}")
    print(f"Lifetime fit: a={fit_l.a:.4g}, b={fit_l.b:.3f}, rmse={fit_l.rmse:.3g}, points={fit_l.x_used.size}")

    # -------------------------
    # PLOTS
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # Raster snippet
    r.plot(ax=axes[0, 0], tstart=0.0, tstop=None, show=False)
    axes[0, 0].set_title("Raster snippet (SOC-like)")

    # Population FR
    axes[0, 1].plot(fr.t, fr.fr_pop, label="population FR")
    axes[0, 1].plot(fr.t, fr.fr_pop / n_channels, label="population / channel")
    axes[0, 1].legend()
    axes[0, 1].set_title("Population firing-rate curve")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Population FR (Hz)")

    # Size PMF + fit overlay
    xs, ps = pmf_from_samples(aval.sizes)
    if xs.size:
        axes[1, 0].loglog(xs, ps, marker="o", linestyle="none", label="empirical")
    if fit_s.x_used.size:
        axes[1, 0].loglog(fit_s.x_used, fit_s.p_hat_used, linestyle="-", label=f"fit b={fit_s.b:.2f}")
    axes[1, 0].set_title(f"Size PMF (RMSE={fit_s.rmse:.2g})")
    axes[1, 0].set_xlabel("size")
    axes[1, 0].set_ylabel("P(size)")
    axes[1, 0].legend()

    # Lifetime PMF + fit overlay
    xl, pl = pmf_from_samples(aval.lifetimes)
    if xl.size:
        axes[1, 1].loglog(xl, pl, marker="o", linestyle="none", label="empirical")
    if fit_l.x_used.size:
        axes[1, 1].loglog(fit_l.x_used, fit_l.p_hat_used, linestyle="-", label=f"fit b={fit_l.b:.2f}")
    axes[1, 1].set_title(f"Lifetime PMF (RMSE={fit_l.rmse:.2g})")
    axes[1, 1].set_xlabel("lifetime (bins)")
    axes[1, 1].set_ylabel("P(lifetime)")
    axes[1, 1].legend()
    plt.tight_layout()


    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    plot_firing_rate_stack(fr, ax=axes[0], mode="stack", normalize="none")     # EEG-like
    # or:
    plot_firing_rate_stack(fr, ax=axes[1], mode="stack", normalize="zscore")   # better for sync visually
    # or:
    plot_firing_rate_stack(fr, ax=axes[2], mode="overlay", normalize="max")    # “butterfly”
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    # Best default for “sync” patterns:
    plot_firing_rate_heatmap(fr, ax=axes[0], normalize="none", robust=True)
    # Best default when rare huge peaks dominate:
    plot_firing_rate_heatmap(fr, ax=axes[1], normalize="log1p", robust=True)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    # 1) Simple: order by correlation to population
    plot_firing_rate_heatmap(fr, ax=axes[0], normalize="zscore", order="population_corr")
    # 2) PCA-based ordering (often very clean)
    plot_firing_rate_heatmap(fr, ax=axes[1], normalize="zscore", order="pca1")
    # 3) Greedy correlation chain (clustering-like)
    plot_firing_rate_heatmap(fr, ax=axes[2], normalize="zscore", order="greedy_corr_chain")
    plt.tight_layout()

    plt.show()

    




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
