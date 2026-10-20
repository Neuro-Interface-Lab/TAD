from __future__ import annotations

import numpy as np
from tad.metrics.correlation import CorrelationResult
from tad.raster import Raster
from tad.metrics.utils import bin_spike_trains
from sklearn.metrics import normalized_mutual_info_score

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
    n_electrodes = len(channels)
    mutual_information = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)

    # Bina todos os canais em uma única chamada, em vez de refazer isso para cada par.
    binned = bin_spike_trains(events, tstart, tstop, binsize)[2].astype(np.int64, copy=False)
    n_bins = binned.shape[1]

    if n_bins and n_electrodes:
        # As contagens inteiras de spikes em cada bin são as categorias da MI.
        n_categories = int(binned.max()) + 1
        marginal_entropy = np.zeros(n_electrodes, dtype=np.float64)
        for channel_index, counts_by_bin in enumerate(binned):
            counts = np.bincount(counts_by_bin, minlength=n_categories)
            probabilities = counts[counts > 0] / n_bins
            marginal_entropy[channel_index] = -np.sum(
                probabilities * np.log(probabilities)
            )
            # MI de um canal consigo mesmo é a entropia desse canal.
            mutual_information[channel_index, channel_index] = marginal_entropy[channel_index]

        # MI é simétrica; calcula cada par uma vez e copia para o triângulo oposto.
        for i in range(n_electrodes):
            for j in range(i + 1, n_electrodes):
                joint_codes = binned[i] * n_categories + binned[j]
                joint_counts = np.bincount(
                    joint_codes, minlength=n_categories * n_categories
                )
                joint_probabilities = joint_counts[joint_counts > 0] / n_bins
                joint_entropy = -np.sum(
                    joint_probabilities * np.log(joint_probabilities)
                )
                value = max(
                    0.0,
                    marginal_entropy[i] + marginal_entropy[j] - joint_entropy,
                )
                mutual_information[i, j] = value
                mutual_information[j, i] = value
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
