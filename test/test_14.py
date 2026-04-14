import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics import detect_bursts
from tad.plotting import plot_bursts_on_raster, plot_bursts_spans


def make_bursty_raster(
    *,
    n_channels: int = 10,
    duration: float = 5.0,
    bg_rate_hz: float = 1.0,
    burst_channels=(2, 3, 7),
    burst_times=((1.0, 1.2), (3.0, 3.25)),
    burst_intra_hz: float = 120.0,
    seed: int = 0,
) -> Raster:
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))

    # background
    for ch in range(n_channels):
        n = rng.poisson(bg_rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)

    # inject bursts as high-rate Poisson inside windows
    for ch in burst_channels:
        for (a, b) in burst_times:
            win = b - a
            n = rng.poisson(burst_intra_hz * win)
            t = a + rng.uniform(0.0, win, size=n)
            r.insert_timestamparray(ch, t, assume_sorted=False)

    return r


def main() -> None:
    duration = 5.0
    r = make_bursty_raster()

    # Detect bursts: choose isi_th consistent with burst_intra_hz ~120 Hz => mean ISI ~ 8.3 ms
    # Set threshold to e.g. 20 ms.
    res = detect_bursts(r, method="fixed", isi_th=0.02, min_spikes=5, tstart=0.0, tstop=duration)

    # -------------------------
    # ASSERTS (sanity)
    # -------------------------
    assert set(res.per_channel.keys()) == set(r.channels())
    assert res.tstart == 0.0
    assert res.tstop == duration

    # Expect burst channels to have at least one burst, most others likely 0
    bursty = [ch for ch, cr in res.per_channel.items() if len(cr.bursts) > 0]
    print("Channels with detected bursts:", bursty)
    assert 2 in bursty or 3 in bursty or 7 in bursty

    # Burst structure sanity: start <= end, n_spikes>=min_spikes
    for ch, cr in res.per_channel.items():
        for b in cr.bursts:
            assert b.start <= b.end
            assert b.n_spikes >= 5
            assert b.duration >= 0.0

    print("OK: fixed-ISI burst detection asserts passed.")

    # -------------------------
    # PLOT: raster + burst intervals on a couple channels
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    r.plot(ax=ax, tstart=0.0, tstop=duration, show=False)
    plot_bursts_spans(res, ax=ax, alpha=0.2)
    ax.set_title("Raster with detected bursts (fixed ISI)")

    # Overlay bursts as red spans across the full y-range (simple)
    #for ch, cr in res.per_channel.items():
    #    for b in cr.bursts:
    #        ax.axvspan(b.start, b.end, alpha=0.15)

    plt.tight_layout()

    plt.show()




def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
