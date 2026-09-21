from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from tad import Raster


@dataclass(frozen=True)
class CrossCorrelationResult:
    """
    Store and manipulate Cross-Correlation results.
    The dataclass is intended to be used for neuroscience data analysis of spike-trains in neuronal cultures, however, it can be used for any other type of data that requires cross-correlation analysis.
    This dataclass will contain the following parameters:
    - cross_correlation: The n x m matrix of cross-correlation values, for n and m the number of rows and columns of the electrode array, respectively. The tau-cross-correlation values are computed for each pair of electrodes for a given time-window, and the result of the maximum value is stored in the matrix.
    - tau: The time-lag (in seconds) at which the maximum cross-correlation value was found for each pair of electrodes. The tau values are stored in a n x m matrix, where n and m are the number of rows and columns of the electrode array, respectively.
    - info: A dictionary containing additional information about the cross-correlation analysis, such as the time-window used for the analysis, the sampling frequency of the data, and any other relevant parameters.
    """
    cross_correlation: np.ndarray
    tau: np.ndarray
    info: dict

def compute_crosscorrelation(raster, tstart: float = None, tstop: float = None, taumax: float = 10.0) -> CrossCorrelationResult:
    """
    This function receives a raster object between tstart and tstop, and computes the cross-correlation between all pairs of electrodes in the raster object. The function returns a CrossCorrelationResult dataclass containing the cross-correlation matrix, the tau matrix, and additional information about the analysis.

    Parameters:
    - raster: A Raster object containing the spike-train data for the analysis.
    - tstart: The start time (in seconds) of the time-window for the analysis
    - tstop: The stop time (in seconds) of the time-window for the analysis
    - taumax: The maximum time-lag (in seconds) to consider for the cross-correlation analysis. The function will compute the cross-correlation for time-lags between -taumax and +taumax.
    """
    # get spike times from raster
    spike_times = raster.events['Ch63']
    print(spike_times)


    return()
