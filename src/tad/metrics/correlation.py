from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Any, Union
from tad.raster import Raster, ChannelId

@dataclass(frozen=True)
class CorrelationResult:
    """
    Store correlation results between channels.

    Parameters
    ----------
    correlation : np.ndarray
        n x n matrix containing the correlation values.

    tau : np.ndarray
        n x n matrix containing the time-lag values corresponding to the correlation values.

    channels : list
        Channel IDs corresponding to the rows and columns
        of the correlation matrix.

    info : dict
        Additional information about the analysis.
    """

    correlation: np.ndarray
    tau: np.ndarray
    channels: list[ChannelId]
    info: dict[str, Any]