import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics import compute_psth
from tad.plotting import plot_psth_lines, plot_psth_heatmap


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

    # background spikes
    for ch in range(n_channels):
        n = rng.poisson(bg_rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)

    # evoked spikes (one per stim with probability evoked_prob)
    for ch in evoked_channels:
        for s in stim_times:
            if rng.random() < evoked_prob:
                lat = rng.normal(evoked_latency_mean, evoked_latency_sd)
                lat = max(0.0, float(lat))
                t = s + lat
                if 0.0 <= t <= duration:
                    r.insert(ch, t)

    return r, stim_times


def main() -> None:
    r, stim_times = make_evoked_raster()

    psth = compute_psth(
        r,
        stim_times,
        dt=0.002,
        t_pre=0.050,
        t_post=0.100,
        channels=range(0, 16),
        tstart=0.0,
        tstop=10.0,
        inclusive_zero=True,
    )

    # -----------------
    # ASSERTS
    # -----------------
    assert psth.counts.shape == psth.rate_hz.shape
    assert psth.counts.shape[0] == 16
    assert psth.t.shape[0] == psth.counts.shape[1]
    assert psth.stim_times_used.size == stim_times.size  # all stimuli valid here

    # Evoked channels should tend to have larger post-stim response than pre-stim (in expectation)
    pre_mask = psth.t < 0.0
    post_mask = (psth.t >= 0.0) & (psth.t <= 0.050)

    evoked_idx = np.arange(0, 8)
    nonevoked_idx = np.arange(8, 16)

    ev_pre = psth.rate_hz[evoked_idx][:, pre_mask].mean()
    ev_post = psth.rate_hz[evoked_idx][:, post_mask].mean()
    nv_pre = psth.rate_hz[nonevoked_idx][:, pre_mask].mean()
    nv_post = psth.rate_hz[nonevoked_idx][:, post_mask].mean()

    print("Evoked pre/post:", ev_pre, ev_post)
    print("Non-evoked pre/post:", nv_pre, nv_post)
    assert ev_post >= ev_pre  # probabilistic but should hold strongly with these params

    print("OK: PSTH asserts passed.")

    # -----------------
    # PLOTS
    # -----------------
    print("If everything is right, there should be a peak in the PSTH around 20ms")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    plot_psth_lines(psth, ax=axes[0], channels=[0, 1, 2, 3, 8, 9], kind="rate", show=False)
    axes[0].set_title("PSTH lines (evoked vs non-evoked)")

    plot_psth_heatmap(psth, ax=axes[1], kind="rate", robust=True, show_colorbar=True, show=False)

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
