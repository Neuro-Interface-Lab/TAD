# src/tad/data/cdata.py
from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Sequence

import numpy as np

from .adata import AData


class CData(AData, ABC):
    """
    Abstract base class for computational/simulation data sources.

    CData provides a place to store explicit time vectors and model outputs,
    but keeps the core contract defined by AData: export to a Raster.
    """

    def __init__(
        self,
        fsample: float,
        channel_ids: Sequence[Any],
        electrode_labels: Optional[Sequence[Any]] = None,
        mask: Optional[Sequence[bool]] = None,
        time_vector: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            fsample=fsample,
            channel_ids=channel_ids,
            electrode_labels=electrode_labels,
            mask=mask,
        )

        self.time_vector: Optional[np.ndarray] = None if time_vector is None else np.asarray(time_vector)
        if self.time_vector is not None and self.time_vector.ndim != 1:
            raise ValueError("time_vector must be a 1D numpy array (seconds).")

        self.model_name: Optional[str] = model_name

        # Optional container for arbitrary simulation outputs (trajectories, spikes, etc.)
        self.model_output: Any = None