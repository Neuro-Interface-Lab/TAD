import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.avalanches import extract_avalanches


def simulate_critical_branching_avalanches(
    rng: np.random.Generator,
    n_avalanches: int = 5000,
    max_steps: int = 500,
) -> list[list[int]]:
    """
    Simulate avalanches using a critical branching process in discrete time.

    Each avalanche is a list A[t] = number of active "events" at time bin t.
    At criticality: each active event produces Poisson(1) offspring (mean=1).

    Parameters
    ----------
    rng
        RNG.
    n_avalanches
        How many avalanches to generate.
    max_steps
        Safety cap to avoid pathological long avalanches.

    Returns
    -------
    avalanches
        List of avalanches, each avalanche is a list of integer counts per step.
    """
    avalanches = []
    for _ in range(n_avalanches):
        a = []
        active = 1  # start with 1 event
        steps = 0
        while active > 0 and steps < max_steps:
            a.append(active)
            # critical branching: total offspring from 'active' parents
            active = rng.poisson(lam=1.0, size=active).sum()
            steps += 1
        if a:  # keep non-empty
            avalanches.append(a)
    return avalanches


def avalanches_to_raster(
    avalanches: list[list[int]],
    *,
    n_channels: int = 32,
    dt: float = 0.001,
    gap_bins: int = 3,
    seed: int = 0,
) -> Raster:
    """
    Convert discrete-time avalanche counts into a Raster.

    Strategy:
    - Time is binned with width dt.
    - Each bin t has A[t] events distributed across channels (uniform random).
    - Spike times are placed within the bin with small jitter to avoid ties.
    - Avalanches are separated by 'gap_bins' empty bins to enforce blank frames.

    Parameters
    ----------
    avalanches
        Output from simulate_critical_branching_avalanches.
    n_channels
        Number of channels for the raster.
    dt
        Bin width used later in extract_avalanches.
    gap_bins
        Number of empty bins between avalanches.
    seed
        RNG seed (local).

    Returns
    -------
    Raster
        Raster containing events encoding the avalanches.
    """
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))

    tbin = 0
    for a in avalanches:
        # ensure blank bins before avalanche (except maybe first)
        tbin += gap_bins

        for k, count in enumerate(a):
            if count <= 0:
                continue
            # choose channels with replacement (multiple events can land on same channel)
            chs = rng.integers(0, n_channels, size=count)

            # Put all events inside this bin with jitter
            base = (tbin + k) * dt
            jitter = rng.uniform(0.0, dt * 0.999, size=count)
            times = base + jitter

            # Insert events per channel (bulk)
            for ch in np.unique(chs):
                r.insert_timestamparray(ch, times[chs == ch], assume_sorted=False)

        # blank bins after avalanche
        tbin += len(a)

    return r


def loglog_hist(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple log-binning-free histogram in log-log coordinates (discrete x).

    Returns
    -------
    xs
        Unique x values (>=1).
    ps
        Probability mass P(x).
    """
    x = np.asarray(x, dtype=int)
    x = x[x >= 1]
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    vals, counts = np.unique(x, return_counts=True)
    ps = counts / counts.sum()
    return vals.astype(float), ps.astype(float)


def fit_powerlaw_slope(xs: np.ndarray, ps: np.ndarray, *, xmin: int = 2) -> float:
    """
    Fit slope in log-log using linear regression on log10(P) vs log10(x), for x>=xmin.
    (This is just a visualization helper, not the paper’s RMSE pipeline.)

    Returns
    -------
    slope
        Estimated slope.
    """
    mask = (xs >= xmin) & (ps > 0)
    xs2 = xs[mask]
    ps2 = ps[mask]
    if xs2.size < 3:
        return float("nan")
    X = np.log10(xs2)
    Y = np.log10(ps2)
    slope, intercept = np.polyfit(X, Y, 1)
    return float(slope)


def main() -> None:
    rng = np.random.default_rng(0)

    # 1) simulate SOC-like avalanches
    avals = simulate_critical_branching_avalanches(rng, n_avalanches=8000, max_steps=1000)

    # 2) convert to raster
    dt = 0.001
    r = avalanches_to_raster(avals, n_channels=32, dt=dt, gap_bins=3, seed=1)

    # 3) extract avalanches from raster
    # window inferred from data; dt must match conversion dt for clean recovery
    res1 = extract_avalanches(r, dt=dt, size_definition=1)
    res2 = extract_avalanches(r, dt=dt, size_definition=2)

    # 4) log-log distributions
    xs_s1, ps_s1 = loglog_hist(res1.sizes)
    xs_l, ps_l = loglog_hist(res1.lifetimes)

    slope_s1 = fit_powerlaw_slope(xs_s1, ps_s1, xmin=2)
    slope_l = fit_powerlaw_slope(xs_l, ps_l, xmin=2)

    print("Extracted avalanches:", res1.sizes.size)
    print("Size slope (def1, rough):", slope_s1)
    print("Lifetime slope (rough):", slope_l)

    # 5) plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].loglog(xs_s1, ps_s1, marker="o", linestyle="none")
    axes[0].set_title(f"Avalanche size (def1), slope~{slope_s1:.2f}")
    axes[0].set_xlabel("size")
    axes[0].set_ylabel("P(size)")

    axes[1].loglog(xs_l, ps_l, marker="o", linestyle="none")
    axes[1].set_title(f"Avalanche lifetime, slope~{slope_l:.2f}")
    axes[1].set_xlabel("lifetime (bins)")
    axes[1].set_ylabel("P(lifetime)")

    plt.tight_layout()
    plt.show()

def main_fast() -> None:
    rng = np.random.default_rng(0)

    avals = simulate_critical_branching_avalanches(
        rng,
        n_avalanches=1500,
        max_steps=300,
    )

    dt = 0.002
    r = avalanches_to_raster(avals, n_channels=16, dt=dt, gap_bins=2, seed=1)

    res1 = extract_avalanches(r, dt=dt, size_definition=1)

    xs_s, ps_s = loglog_hist(res1.sizes)
    xs_l, ps_l = loglog_hist(res1.lifetimes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].loglog(xs_s, ps_s, marker="o", linestyle="none")
    axes[0].set_title("Avalanche size (def1)")
    axes[0].set_xlabel("size"); axes[0].set_ylabel("P(size)")
    axes[1].loglog(xs_l, ps_l, marker="o", linestyle="none")
    axes[1].set_title("Avalanche lifetime")
    axes[1].set_xlabel("lifetime (bins)"); axes[1].set_ylabel("P(lifetime)")
    plt.tight_layout()
    plt.show()




def test_main_fast() -> None:
    main_fast()
if __name__ == "__main__":
    main_fast()
