from __future__ import annotations
import numpy as np

from tad.metrics.correlation import CrossCorrelationResult


def compute_crosscorrelation(raster, tstart: float = 0, tstop: float = None, taumax: float = None, fsample: float = None,binsize: float = None) -> CrossCorrelationResult:
    """
    This function receives a raster object between tstart and tstop, and computes the cross-correlation between all pairs of electrodes in the raster object. The function returns a CrossCorrelationResult dataclass containing the cross-correlation matrix, the tau matrix, and additional information about the analysis.

    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - taumax: The maximum time-lag (in seconds) to consider for the cross-correlation analysis. The function will compute the cross-correlation for time-lags between -taumax and +taumax.
    - fsample: Sampling frequency of the data.
    """
    if tstop == None:
        tstop = max(np.max(times) for times in raster.events.values() if len(times)>0)

    # reduce events between tstart and tstop
    events = {
            channel_id: times[(times>=tstart) & (times <tstop)] 
            for channel_id, times in raster.events.items()    
        }

    events = dict(
        sorted(
            events.items(),
            key = lambda item: int(item[0][2:])
        )
    )

    channels = list(events.keys())

    if fsample == None:
        fsample = 20000 # Hz

    if binsize == None:
        binsize =200*1/fsample # s
    if taumax == None:
        taumax = 0.2 # 200 ms

    n_electrodes = len(list(events.keys()))
    cross_correlation = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)
    tau = np.zeros((n_electrodes,n_electrodes), dtype = np.float64)

    for i, ch_x in enumerate(events.keys()):
        for j, ch_y in enumerate(events.keys()):
            print(ch_x, ch_y)
            tau_range = np.arange(0, taumax, step=binsize)
            spike_times_x = events[ch_x]
            Nx = len(spike_times_x)
            spike_times_y = events[ch_y]
            Ny = len(spike_times_y)
            cc = np.zeros(len(tau_range))
            for k, tau_i in enumerate(tau_range):
                count = 0
                for ts in spike_times_x:
                    J = (spike_times_y >= ts - tau_i - binsize/2) & (spike_times_y < ts - tau_i + binsize/2)
                    count += np.sum(J)
                if Nx > 0 and Ny > 0:
                    cc[k] = count / np.sqrt(Nx * Ny)
                else:
                    cc[k] = 0
            cross_correlation[i,j] = np.max(cc)
            tau[i,j] = tau_range[np.argmax(cc)]

    info = {
        "binsize": binsize,
        "tstart": tstart,
        "tstop": tstop
    }

    return CrossCorrelationResult(
        correlation=cross_correlation,
        tau = tau,
        channels = channels,
        info=info
    )
