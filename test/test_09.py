import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.avalanches import extract_avalanches
from tad.metrics.rates import firing_rate_curve


def simulate_critical_branching_avalanches(
    rng: np.random.Generator,
    n_avalanches: int = 800,
    max_steps: int = 250,
) -> list[list[int]]:
    """
    Simulate avalanches using a critical branching process (mean offspring = 1).

    Returns a list of avalanches; each avalanche is a list A[k] = number of events in bin k.
    """
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


def avalanches_to_raster(
    avalanches: list[list[int]],
    *,
    n_channels: int = 16,
    dt: float = 0.002,
    gap_bins: int = 2,
    seed: int = 1,
) -> Raster:
    """
    Convert avalanche bin-count sequences into a Raster.

    Each bin k has A[k] events assigned uniformly to channels, with small jitter inside the bin.
    Avalanches are separated by `gap_bins` empty bins (blank frames).
    """
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))

    tbin = 0
    for a in avalanches:
        tbin += gap_bins  # blank bins before avalanche
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


def loglog_hist(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete histogram (PMF) for positive integer samples, for log-log plotting.
    """
    x = np.asarray(x, dtype=int)
    x = x[x >= 1]
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    vals, counts = np.unique(x, return_counts=True)
    pmf = counts / counts.sum()
    return vals.astype(float), pmf.astype(float)


def main() -> None:
    rng = np.random.default_rng(0)

    # FAST parameters
    dt = 0.002
    n_channels = 16
    avals = simulate_critical_branching_avalanches(rng, n_avalanches=800, max_steps=250)
    r = avalanches_to_raster(avals, n_channels=n_channels, dt=dt, gap_bins=2, seed=1)

    # Metrics
    res1 = extract_avalanches(r, dt=dt, size_definition=1)
    res2 = extract_avalanches(r, dt=dt, size_definition=2)
    fr = firing_rate_curve(r, dt=dt)  # same dt so FR bins align with avalanche bins

    # -------------------------
    # ASSERTS
    # -------------------------
    assert np.array_equal(res1.lifetimes, res2.lifetimes)

    if res1.lifetimes.size > 0:
        assert np.all(res1.lifetimes >= 1)
        assert np.all(res1.sizes >= 1)
        assert np.all(res2.sizes >= 1)
        assert np.all(res2.sizes <= n_channels)
        assert np.all(res1.sizes >= res2.sizes)

    # FR consistency: population FR equals sum of per-channel FR
    assert np.allclose(fr.fr_pop, fr.fr_ch.sum(axis=0))

    # Active bins should correspond to FR_pop > 0 (given same dt)
    # Note: fr.fr_pop is in Hz, so >0 iff at least 1 spike in bin.
    assert fr.fr_pop.shape == res1.active_bins.shape
    assert np.array_equal(res1.active_bins, fr.fr_pop > 0.0)

    print("OK: avalanche + firing rate curve invariants passed.")
    print("N avalanches:", res1.sizes.size)

    # -------------------------
    # PLOTS
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # (1) Raster snippet
    t_snip = 0.4
    r.plot(ax=axes[0, 0], tstart=0.0, tstop=t_snip, show=False)
    axes[0, 0].set_title("Raster snippet (SOC-like)")

    # (2) Population FR curve
    axes[0, 1].plot(fr.t, fr.fr_pop)
    axes[0, 1].set_title("Population firing-rate curve")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Population FR (Hz)")

    # (3) Size PMF log-log
    xs_s, ps_s = loglog_hist(res1.sizes)
    if xs_s.size:
        axes[1, 0].loglog(xs_s, ps_s, marker="o", linestyle="none")
    axes[1, 0].set_title("Avalanche size PMF (def1)")
    axes[1, 0].set_xlabel("size")
    axes[1, 0].set_ylabel("P(size)")

    # (4) Lifetime PMF log-log
    xs_l, ps_l = loglog_hist(res1.lifetimes)
    if xs_l.size:
        axes[1, 1].loglog(xs_l, ps_l, marker="o", linestyle="none")
    axes[1, 1].set_title("Avalanche lifetime PMF")
    axes[1, 1].set_xlabel("lifetime (bins)")
    axes[1, 1].set_ylabel("P(lifetime)")

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
