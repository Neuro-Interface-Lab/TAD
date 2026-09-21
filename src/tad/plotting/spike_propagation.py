from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import ArtistAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import Normalize

from tad.raster import Raster
from tad.metrics.scalar import spike_count

NULL_CHANNELS = {11, 18, 81, 88}

def plot_spike_gif(r : Raster, 
                   tstart, 
                   tstop, 
                   window_ms,
                   save :bool = False,
                   save_path = None,
                   show = False,
                   fps = 10,
                   ) -> ArtistAnimation:
    ch_ids = r.channels()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    frames = []
    ax.set_xticks([])
    ax.set_yticks([])

    # nsteps = np.int8(((tstop-tstart)*1000)/window_ms)
    # print(nsteps)
    # input()
    timesteps = np.arange(tstart, tstop, window_ms / 1000)

    # First determine the global vmax
    vmax = 0

    for i, t in enumerate(timesteps[:-1]):
        for ch in ch_ids:
            count = spike_count(
                r,
                channels=[ch],
                tstart=t,
                tstop=timesteps[i + 1],
                per_channel=True,
            )

            vmax = max(vmax, count)

    vmax = max(vmax, 1)
    norm = Normalize(vmin=0, vmax=vmax)

    im = ax.imshow(
        np.zeros((8, 8)),
        cmap="hot",
        interpolation="nearest",
        norm=norm,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spike count")

    for (i,t ) in enumerate(timesteps[:-1]):
        #print(t, i)
        spike_matrix = np.zeros((8,8), dtype = np.int64)
        for ch in ch_ids:
            id = int(ch[2:])
            row = id // 10 - 1
            col = id % 10 - 1
            #print(ch)
            if i < len(timesteps):
                count = spike_count(r, channels=[ch],  tstart = t, tstop= timesteps[i+1], per_channel=True)
            else:
                count = 0
            spike_matrix[row, col] = count
            #print(count)
            #print(col, row)
            #input()
               # plot one frame for this time point
        im = ax.imshow(
            spike_matrix,
            cmap="hot",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        title = ax.text(
            0.5,
            1.02,
            f"t = {t:.4f} s",
            transform=ax.transAxes,
            ha="center",
        )

        frames.append([im, title])
    fig.canvas.draw()

    ani = ArtistAnimation(fig, frames, interval=1000 / fps)

    if save:
        save_path = Path(save_path)

        if save_path.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps)
        elif save_path.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError("save_path must end with .mp4 or .gif")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        ani.save(
            str(save_path),
            writer=writer,
            dpi=200,
        )

    if show:
        plt.show()

    plt.close(fig)

    return ani