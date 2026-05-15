import numpy as np
import matplotlib.pyplot as plt

from tad.raster import Raster
from tad.metrics import detect_bursts
from tad.plotting import plot_bursts_on_raster  # from earlier helper
from tad.plotting.isi import plot_logisih_threshold


def make_bursty_raster(
    *,
    n_channels: int = 10,
    burst_channels=(2, 3, 7),
) -> Raster:
    r = Raster.empty(channels=range(n_channels))

    # Sparse deterministic background activity on all channels.
    background_times = np.array([0.15, 0.85, 2.1, 4.35], dtype=float)
    for ch in range(n_channels):
        offset = ch * 0.001
        r.insert_timestamparray(ch, background_times + offset, assume_sorted=True)

    # Two clear burst epochs with short ISIs inside each burst and long gaps outside.
    burst_blocks = np.array(
        [
            1.000,
            1.008,
            1.016,
            1.024,
            1.032,
            1.040,
            3.000,
            3.009,
            3.018,
            3.027,
            3.036,
            3.045,
        ],
        dtype=float,
    )

    for i, ch in enumerate(burst_channels):
        # Slight per-channel offset keeps channels distinct while preserving burst structure.
        offset = i * 0.0005
        r.insert_timestamparray(
            ch,
            burst_blocks + offset,
            assume_sorted=True,
        )

    return r


def main() -> None:
    duration = 5.0
    r = make_bursty_raster()

    res = detect_bursts(
        r,
        method="logisih",
        threshold_scope="per_channel",
        min_spikes=5,
        tstart=0.0,
        tstop=duration,
        logisih_bins=60,
        logisih_smooth_window=7,
        fallback=0.1,
    )

    print(res.per_channel[7].bursts)

    # ASSERTS: thresholds finite for channels with enough spikes
    bursty_channels = [2, 3, 7]
    for ch in bursty_channels:
        cr = res.per_channel[ch]
        assert cr.diagnostics is not None
        assert np.isfinite(cr.isi_th)


    # Deterministic raster: each bursty channel should show at least one detected burst.
    detected = [ch for ch in bursty_channels if len(res.per_channel[ch].bursts) > 0]
    print("Detected burst channels:", detected)
    assert detected == bursty_channels

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
    t0, t1 = 1.0, 1.05

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
