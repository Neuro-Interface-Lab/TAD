import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics import detect_bursts
from tad.metrics.isi import isi, isih
from tad.plotting import plot_bursts_on_raster  # from earlier helper
from tad.plotting.isi import plot_logisih_threshold


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

    for ch in range(n_channels):
        n = rng.poisson(bg_rate_hz * duration)
        t = rng.uniform(0.0, duration, size=n)
        r.insert_timestamparray(ch, t, assume_sorted=False)

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

    # Pooled ISIs
    #res_pool = isi(r, mode="pooled")
    # Plot: pooled ISI histogram in linear and log domain
    #centers_lin, h_lin = isih(res_pool.isi, bins=60, density=True, log=False)

    #plt.figure(figsize=(6, 4))
    #plt.plot(centers_lin, h_lin)
    #plt.title("Pooled ISI density (linear)")
    #plt.xlabel("ISI (s)")
    #plt.ylabel("density")
    #plt.grid(True, which="both", ls="--", alpha=0.5)
    #plt.tight_layout()

    res = detect_bursts(
        r,
        method="logisih",
        threshold_scope="per_channel",
        min_spikes=5,
        tstart=0.0,
        tstop=duration,
        logisih_bins=60,
        logisih_smooth_window=7,
        fallback_quantile=0.2,
    )

    print(res.per_channel[7].bursts)

    # ASSERTS: thresholds finite for channels with enough spikes
    bursty_channels = [2, 3, 7]
    for ch in bursty_channels:
        cr = res.per_channel[ch]
        assert cr.diagnostics is not None
        assert np.isfinite(cr.isi_th)


    # At least one burst detected in one burst channel (probabilistic, but should hold)
    detected = [ch for ch in bursty_channels if len(res.per_channel[ch].bursts) > 0]
    print("Detected burst channels:", detected)
    assert len(detected) >= 1

    # Plot raster + bursts
    plot_bursts_on_raster(r, res, tstart=0.0, tstop=duration, alpha=0.2, show=False)

    # Plot log-ISIH threshold diagnostics for one example channel
    ch = 2
    diag = res.per_channel[ch].diagnostics
    if diag is not None:
        plot_logisih_threshold(diag, show=False)

    # -------------------------
    # INSPECT: ISI distribution and threshold for one bursty channel
    # -------------------------
    ch = 2
    t0, t1 = 1.0, 1.2

    cr = res.per_channel[ch]  # BurstChannelResult
    print("ISI_th:", cr.isi_th)
    print("diagnostic method:", None if cr.diagnostics is None else cr.diagnostics.method)

    arr = r.events[ch]
    w = arr[(arr >= t0) & (arr <= t1)]
    print("spikes in window:", w.size)

    if w.size >= 2:
        isi = np.diff(np.sort(w))
        print("max ISI in window:", isi.max())
        print("ISIs > ISI_th:", np.sum(isi > cr.isi_th))
        print("largest few ISIs:", np.sort(isi)[-5:])

    plt.show()



def test_main() -> None:
    main()
if __name__ == "__main__":
    main()
