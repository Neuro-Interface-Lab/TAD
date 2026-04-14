import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics.scalar import (
    spike_count,
    firing_rate,
    mean_firing_rate_across_channels,
    mean_inter_event_interval,
    percent_random_spiking,
)


def make_poisson_raster(
    n_channels: int = 10,
    duration: float = 2.0,
    rate_hz: float = 15.0,
    seed: int = 0,
) -> Raster:
    """
    Generate a simple homogeneous Poisson raster:
    for each channel, N ~ Poisson(rate_hz * duration), spike times ~ Uniform(0, duration).
    """
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))

    for ch in range(n_channels):
        n_spikes = rng.poisson(rate_hz * duration)
        times = rng.uniform(0.0, duration, size=n_spikes)
        r.insert_timestamparray(ch, times, assume_sorted=False)

    return r


def main() -> None:
    duration = 2.0
    r = make_poisson_raster(n_channels=10, duration=duration, rate_hz=15.0, seed=0)

    # ------------------------------------------------------------
    # Create a simple "burst_intervals" example for %RS.
    # In the real pipeline this will come from your burst detector.
    #
    # Here: define two burst windows on channels 0 and 1.
    # Format: dict[channel] -> array of shape (B, 2) with (start, end).
    # ------------------------------------------------------------
    burst_intervals = {
        0: np.array([[0.20, 0.35], [1.10, 1.20]], dtype=float),
        1: np.array([[0.50, 0.65]], dtype=float),
        # channels without entries are treated as having no bursts
    }

    # ------------------------------------------------------------
    # Call ALL implemented scalar metrics
    # ------------------------------------------------------------
    # 1) spike_count
    counts_per_ch = spike_count(r, tstart=0.0, tstop=duration, per_channel=True)
    total_count = spike_count(r, tstart=0.0, tstop=duration, per_channel=False)

    # 2) firing_rate (FR = N/T)
    fr_per_ch = firing_rate(r, tstart=0.0, tstop=duration, per_channel=True)
    pooled_fr = firing_rate(r, tstart=0.0, tstop=duration, per_channel=False)

    # 3) mean_firing_rate_across_channels (optionally active channels)
    mean_fr_all = mean_firing_rate_across_channels(r, tstart=0.0, tstop=duration)
    mean_fr_active = mean_firing_rate_across_channels(
        r,
        tstart=0.0,
        tstop=duration,
        active_threshold_hz=5.0,  # arbitrary example threshold
    )

    # 4) mean_inter_event_interval (pooled IEI mean)
    mean_iei = mean_inter_event_interval(r, tstart=0.0, tstop=duration)

    # 5) % random spiking activity (%RS = fraction of spikes outside bursts)
    prs = percent_random_spiking(
        r,
        burst_intervals=burst_intervals,
        tstart=0.0,
        tstop=duration,
    )

    # ------------------------------------------------------------
    # Print results (sanity / inspection)
    # ------------------------------------------------------------
    print("=== Scalar metrics demo ===")
    print(f"Duration: {duration} s")
    print(f"Channels: {r.n_channels()}")
    print()

    print("Spike counts per channel:", counts_per_ch)
    print("Total spike count:", total_count)
    print()

    print("Firing rates per channel (Hz):", np.round(fr_per_ch, 3))
    print("Pooled firing rate (Hz):", round(pooled_fr, 3))
    print("Mean FR across channels (Hz):", round(mean_fr_all, 3))
    print("Mean FR across active channels (Hz, thr=5):", round(mean_fr_active, 3))
    print()

    print("Mean pooled IEI (s):", mean_iei)
    print("Percent random spiking (0..1):", prs)
    print()

    # ------------------------------------------------------------
    # Visualize: raster + burst windows on a couple channels
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    r.plot(ax=ax, tstart=0.0, tstop=duration, show=False)
    ax.set_title("Poisson raster + example burst windows (ch 0 and 1)")

    # Draw burst windows as shaded spans at the y positions of the channel ticks
    # Your plot uses sorted channel labels; here channels are ints 0..9 so this matches.
    yticks = ax.get_yticks()
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    ymap = {int(lbl): y for lbl, y in zip(ylabels, yticks) if lbl.isdigit()}

    for ch, intervals in burst_intervals.items():
        if ch not in ymap:
            continue
        y = ymap[ch]
        for (bs, be) in intervals:
            ax.axvspan(bs, be, ymin=(y - 0.4) / (yticks[-1] + 0.8), ymax=(y + 0.4) / (yticks[-1] + 0.8), alpha=0.2)

    plt.tight_layout()
    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
