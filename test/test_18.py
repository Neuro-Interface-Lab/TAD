from tad.raster import Raster
from tad.metrics import detect_bursts


def test_detect_bursts_from_pooled_channels():
    raster = Raster.empty(channels=[1, 2])
    raster.insert_timestamparray(1, [0.00, 0.02], assume_sorted=True)
    raster.insert_timestamparray(2, [0.01, 0.03, 0.04], assume_sorted=True)

    per_channel = detect_bursts(
        raster,
        method="fixed",
        isi_th=0.011,
        min_spikes=5,
        tstart=0.0,
        tstop=0.05,
    )
    assert all(not result.bursts for result in per_channel.per_channel.values())

    per_channel_with_burst = detect_bursts(
        raster,
        method="fixed",
        isi_th=0.021,
        min_spikes=3,
        tstart=0.0,
        tstop=0.05,
    )
    assert per_channel_with_burst.per_channel[2].bursts[0].initial_channel == 2

    pooled = detect_bursts(
        raster,
        method="fixed",
        isi_th=0.011,
        min_spikes=5,
        tstart=0.0,
        tstop=0.05,
        detection_scope="pooled",
    )

    assert pooled.detection_scope == "pooled"
    assert pooled.per_channel == {}
    assert pooled.pooled is not None
    assert len(pooled.pooled.bursts) == 1
    burst = pooled.pooled.bursts[0]
    assert burst.start == 0.0
    assert burst.end == 0.04
    assert burst.n_spikes == 5
    assert burst.initial_channel == 1