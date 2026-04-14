from pathlib import Path

from tad import Raster
import numpy as np
import matplotlib.pyplot as plt



def _make_poisson_raster(
    n_channels: int = 10,
    duration: float = 2.0,
    rate_hz: float = 15.0,
    seed: int = 0,
) -> "Raster":
    rng = np.random.default_rng(seed)
    r = Raster.empty(channels=range(n_channels))
    for ch in range(n_channels):
        n_spikes = rng.poisson(rate_hz * duration)
        times = rng.uniform(0.0, duration, size=n_spikes)
        r.insert_timestamparray(ch, times, assume_sorted=False)
    return r


def _assert_equal_rasters(a: "Raster", b: "Raster") -> None:
    assert set(a.channels()) == set(b.channels())
    for ch in a.channels():
        if not np.array_equal(a.events[ch], b.events[ch]):
            raise AssertionError(f"Mismatch in channel {ch!r}")


def demo_save_load_plot(
    *,
    h5: bool,
    path: str,
    figure_path: str | None = None,
) -> None:
    duration = 2.0
    r = _make_poisson_raster(n_channels=10, duration=duration, rate_hz=15.0, seed=0)

    r.save(path, h5=h5)
    r2 = Raster.load(path, h5=h5)

    _assert_equal_rasters(r, r2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    r.plot(ax=axes[0], tstart=0.0, tstop=duration, show=False)
    axes[0].set_title(f"Original ({'H5' if h5 else 'JSON'})")
    r2.plot(ax=axes[1], tstart=0.0, tstop=duration, show=False)
    axes[1].set_title(f"Reloaded ({'H5' if h5 else 'JSON'})")
    plt.tight_layout()
    if figure_path is not None:
        fig.savefig(figure_path, format="pdf")
    plt.close(fig)



def test_demo_save_load_plot(tmp_path: Path) -> None:
    demo_save_load_plot(
        h5=True,
        path=str(tmp_path / "raster_demo.h5"),
        figure_path=str(tmp_path / "raster_demo_h5.pdf"),
    )
    demo_save_load_plot(
        h5=False,
        path=str(tmp_path / "raster_demo.json"),
        figure_path=str(tmp_path / "raster_demo_json.pdf"),
    )

if __name__ == "__main__":
    # HDF5 (default)
    demo_save_load_plot(h5=True, path="raster_demo.h5")
    # JSON
    demo_save_load_plot(h5=False, path="raster_demo.json")
