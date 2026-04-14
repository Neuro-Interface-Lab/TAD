import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.psth import compute_psth
from tad.metrics.evoked import evoked_peak_metrics, response_probability


def make_evoked_raster(
    *,
    n_channels: int = 16,
    duration: float = 10.0,
    bg_rate_hz: float = 2.0,
    stim_period: float = 1.0,
    stim_start: float = 1.0,
    stim_count: int = 8,
    evoked_channels=range(0, 8),
    evoked_prob: float = 0.8,
    evoked_latency_mean: float = 0.020,
    evoked_latency_sd: float = 0.003,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    stim_times = stim_start + stim_period * np.arange(stim_count)
    r = Raster.empty(channels=range(n_channels))

    for ch in range(n_channels):
        n = rng.poisson(bg_rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)

    for ch in evoked_channels:
        for s in stim_times:
            if rng.random() < evoked_prob:
                lat = max(0.0, float(rng.normal(evoked_latency_mean, evoked_latency_sd)))
                t = s + lat
                if 0.0 <= t <= duration:
                    r.insert(ch, t)

    return r, stim_times


def main() -> None:
    r, stim_times = make_evoked_raster()

    psth = compute_psth(
        r, stim_times,
        dt=0.002,
        t_pre=0.050,
        t_post=0.100,
        tstart=0.0,
        tstop=10.0,
        inclusive_zero=True,
    )

    ev = evoked_peak_metrics(
        psth,
        baseline_window=(-0.050, 0.0),
        response_window=(0.0, 0.050),
        rectify=True,
    )

    chs, p = response_probability(
        r, stim_times,
        t0=0.0, t1=0.050,
        tstart=0.0, tstop=10.0,
    )

    # --- asserts ---
    assert ev.peak_latency_s.shape[0] == 16
    assert p.shape[0] == 16

    # evoked channels should have higher response probability (in expectation)
    evoked_idx = np.arange(0, 8)
    nonevoked_idx = np.arange(8, 16)
    assert p[evoked_idx].mean() > p[nonevoked_idx].mean()

    print("OK: evoked metrics asserts passed.")
    print("Mean peak latency evoked (s):", ev.peak_latency_s[evoked_idx].mean())
    print("Mean response prob evoked:", p[evoked_idx].mean())

    # --- plot raster and mark lines on stimulation
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    r.plot(ax=axes)
    ylim = axes.get_ylim()
    axes.vlines(stim_times, ylim[0], ylim[1], color='r')
    # --- plot quick summaries ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))

    axes[0].plot(ev.peak_latency_s, marker="o", linestyle="none")
    axes[0].set_title("Peak latency (s)")
    axes[0].set_xlabel("channel idx")

    axes[1].plot(ev.peak_minus_baseline_hz, marker="o", linestyle="none")
    axes[1].set_title("Peak - baseline (Hz)")
    axes[1].set_xlabel("channel idx")

    axes[2].plot(p, marker="o", linestyle="none")
    axes[2].set_title("Response probability")
    axes[2].set_xlabel("channel idx")

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
