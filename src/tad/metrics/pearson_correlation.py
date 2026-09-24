from __future__ import annotations
import numpy as np

from tad.metrics.correlation import CorrelationResult
from tad.metrics.utils import bin_spike_trains

def compute_pearson_correlation(raster, tstart: float = 0, tstop: float = None, binsize : float = 0.001) -> CorrelationResult:
    """
    This function receives a raster object, and selects the events between tstart and tstop.
    Then, it computes the Pearson correlation coefficient between all pairs of electrodes in the raster object.
    
    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - binsize: The size of the bins (in seconds) for the spike train binning
    - Returns:
    - A CorrelationResult dataclass containing the correlation matrix, the correlated channels, and additional information about the analysis.
    
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
    n_electrodes = len(list(events.keys()))
    correlation_matrix = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)

    for i, ch_x in enumerate(events.keys()):
        for j, ch_y in enumerate(events.keys()):
            spike_times_x = events[ch_x]
            spike_times_y = events[ch_y]
            binned_x = bin_spike_trains({ch_x: spike_times_x}, tstart, tstop, binsize=binsize)[2]
            binned_y = bin_spike_trains({ch_y: spike_times_y}, tstart, tstop, binsize=binsize)[2]
            if np.std(binned_x) == 0 or np.std(binned_y) == 0:
                correlation_matrix[i, j] = 0
            else:
                correlation_matrix[i, j] = np.corrcoef(binned_x, binned_y)[0, 1]

    info = {
        "tstart": tstart,
        "tstop": tstop,
        "binsize": binsize
    }

    return CorrelationResult(
        correlation=correlation_matrix,
        tau=None,
        channels=channels,
        info=info
    )

def compute_delayed_pearson_correlation(raster, tstart: float = 0.0, tstop :float = None, binsize : float = 0.001, taumax : float = 0.2) -> CorrelationResult:
    """
    This function receives a raster object, and selects the events between tstart and tstop.
    Then, it computes the delayed Pearson correlation coefficient between all pairs of electrodes in the raster object.
    
    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - binsize: The size of the bins (in seconds) for the spike train binning
    - taumax: The maximum delay (in seconds) to consider for the delayed correlation
    - Returns:
    - A CorrelationResult dataclass containing the correlation matrix, the correlated channels, and additional information about the analysis.
    
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
    n_electrodes = len(list(events.keys()))
    correlation_matrix = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)
    tau_matrix = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)

    for i, ch_x in enumerate(events.keys()):
        for j, ch_y in enumerate(events.keys()):
            print(ch_x, ch_y)
            spike_times_x = events[ch_x]
            spike_times_y = events[ch_y]
            binned_x = bin_spike_trains({ch_x: spike_times_x}, tstart, tstop, binsize=binsize)[2]
            binned_y = bin_spike_trains({ch_y: spike_times_y}, tstart, tstop, binsize=binsize)[2]
            taus = np.arange(-taumax, taumax + binsize, binsize)
            corr = np.zeros(len(taus))
            for k, tau in enumerate(taus):
                shifted_binned_y = np.roll(binned_y, int(tau / binsize))
                if np.std(binned_x) == 0 or np.std(shifted_binned_y) == 0:
                    corr[k] = 0
                else:
                    corr[k] = np.corrcoef(binned_x, shifted_binned_y)[0, 1]
            correlation_matrix[i, j] = corr.flat[np.argmax(np.abs(corr))]
            tau_matrix[i, j] = taus[np.argmax(corr.flat[np.argmax(np.abs(corr))])]

    info = {
        "tstart": tstart,
        "tstop": tstop,
        "binsize": binsize,
        "taumax": taumax
    }

    return CorrelationResult(
        correlation=correlation_matrix,
        tau=tau_matrix,
        channels=channels,
        info=info
    )