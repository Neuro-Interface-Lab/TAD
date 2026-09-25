from __future__ import annotations

import numpy as np
from tad import raster
from tad import raster
from tad.metrics.correlation import CorrelationResult
from tad.raster import Raster, ChannelId
from tad.metrics.utils import bin_spike_trains
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

def compute_mutual_information(raster : Raster, tstart : float = 0, tstop : float = None, binsize : float = 0.005) -> CorrelationResult:
    """
    This function computes the mutual information between all pairs of electrodes in a raster object. The function returns a CorrelationResult dataclass containing the mutual information matrix and additional information about the analysis.

    Parameters:
    - raster: A Raster object containing the spike-train and channels data for the analysis.
    - tstart: The start time for the analysis.
    - tstop: The stop time for the analysis.
    - binsize: The size of the bins for the spike trains.

    Returns:
    - A CorrelationResult object containing the mutual information matrix and additional information about the analysis.
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
    mutual_information = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)
    for i, ch_x in enumerate(events.keys()):
        for j, ch_y in enumerate(events.keys()):
            spike_times_x = events[ch_x]
            spike_times_y = events[ch_y]
            binned_x = np.ravel(bin_spike_trains({ch_x: spike_times_x}, tstart, tstop, binsize)[2])
            binned_y = np.ravel(bin_spike_trains({ch_y: spike_times_y}, tstart, tstop, binsize)[2])
            mutual_information[i, j] = mutual_info_score(binned_x, binned_y)
    info = {
         'tstart': tstart,
         'tstop': tstop,
         'binsize': binsize
    }

    return CorrelationResult(
        correlation = mutual_information,
        tau = np.zeros((n_electrodes, n_electrodes), dtype = np.float64),
        channels = channels,
        info = info
    )

def compute_normalized_mutual_information(raster : Raster, tstart : float = 0, tstop : float = None, binsize : float = 0.005) -> CorrelationResult:
    """
    This function computes the normalized mutual information between all pairs of electrodes in a raster object. The function returns a CorrelationResult dataclass containing the normalized mutual information matrix and additional information about the analysis.

    Parameters:
    - raster: A Raster object containing the spike-train and channels data for the analysis.
    - tstart: The start time for the analysis.
    - tstop: The stop time for the analysis.
    - binsize: The size of the bins for the spike trains.

    Returns:
    - A CorrelationResult object containing the normalized mutual information matrix and additional information about the analysis.
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
    mutual_information = np.zeros((n_electrodes, n_electrodes), dtype = np.float64)
    for i, ch_x in enumerate(events.keys()):
        for j, ch_y in enumerate(events.keys()):
            spike_times_x = events[ch_x]
            spike_times_y = events[ch_y]
            binned_x = np.ravel(bin_spike_trains({ch_x: spike_times_x}, tstart, tstop, binsize)[2])
            binned_y = np.ravel(bin_spike_trains({ch_y: spike_times_y}, tstart, tstop, binsize)[2])
            mutual_information[i, j] = normalized_mutual_info_score(binned_x, binned_y, average_method='geometric')
    info = {
            'tstart': tstart,
            'tstop': tstop,
            'binsize': binsize
    }

    return CorrelationResult(
        correlation = mutual_information,
        tau = np.zeros((n_electrodes, n_electrodes), dtype = np.float64),
        channels = channels,
        info = info
    )

