# src/tad/data/edata.py
from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Sequence

from .adata import AData


class EData(AData, ABC):
    """
    Abstract base class for experimental data sources.

    EData adds minimal experimental-specific metadata (like a source filename),
    but remains backend-agnostic (it does not force SpikeInterface).
    """

    def __init__(
        self,
        fsample: float,
        channel_ids: Sequence[Any],
        electrode_labels: Optional[Sequence[Any]] = None,
        mask: Optional[Sequence[bool]] = None,
        fname: Optional[str] = None,
        recording_system: Optional[str] = None,
    ) -> None:
        super().__init__(
            fsample=fsample,
            channel_ids=channel_ids,
            electrode_labels=electrode_labels,
            mask=mask,
        )

        self.fname: Optional[str] = fname
        self.recording_system: Optional[str] = recording_system

        # Optional backend-specific object (SpikeInterface Recording, etc.)
        self.recording: Any = None