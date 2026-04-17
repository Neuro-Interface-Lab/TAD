# src/tad/data/adata.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import json
import csv

import numpy as np


Number = Union[int, float]
Interval = Tuple[float, float]


@dataclass(frozen=True)
class DataSelection:
    """
    Immutable selection descriptor used by AData.get_raster() implementations.

    Attributes
    ----------
    t0, t1 : float | None
        Optional time window bounds (seconds). If None, means "use full duration".
    mask : np.ndarray | None
        Optional channel mask override. If None, use self.mask.
    excluded_intervals : list[tuple[float, float]] | None
        Optional exclusion intervals override. If None, use self.excluded_intervals.
    """
    t0: Optional[float] = None
    t1: Optional[float] = None
    mask: Optional[np.ndarray] = None
    excluded_intervals: Optional[List[Interval]] = None


class AData(ABC):
    """
    Abstract base class for all TAD data sources (experimental and computational).

    This class is intentionally small: it defines the common *selection state* and
    the contract to export a Raster (or an equivalent downstream event container).

    Concrete children are expected to:
      - provide a consistent channel axis (channel_ids) and optional labels
      - maintain a boolean mask for channel inclusion
      - maintain excluded time intervals
      - implement get_raster()

    Notes
    -----
    - This class does *not* assume any particular backend (SpikeInterface, Brian2, etc.).
    - It does *not* assume traces exist; only that events can be exported to a Raster.
    """

    def __init__(
        self,
        fsample: float,
        channel_ids: Sequence[Any],
        electrode_labels: Optional[Sequence[Any]] = None,
        mask: Optional[Sequence[bool]] = None,
    ) -> None:
        self.history: list[dict] = []
        self.fsample: float = float(fsample)

        self.channel_ids: np.ndarray = np.asarray(list(channel_ids))
        if self.channel_ids.ndim != 1:
            raise ValueError("channel_ids must be a 1D sequence.")

        self.electrode_labels: Optional[np.ndarray]
        if electrode_labels is None:
            self.electrode_labels = None
        else:
            labels = np.asarray(list(electrode_labels))
            if labels.ndim != 1:
                raise ValueError("electrode_labels must be a 1D sequence.")
            if len(labels) != len(self.channel_ids):
                raise ValueError(
                    "electrode_labels must have the same length as channel_ids."
                )
            self.electrode_labels = labels

        if mask is None:
            self.mask: np.ndarray = np.ones(len(self.channel_ids), dtype=bool)
        else:
            m = np.asarray(list(mask), dtype=bool)
            if m.ndim != 1:
                raise ValueError("mask must be a 1D sequence.")
            if len(m) != len(self.channel_ids):
                raise ValueError("mask must have the same length as channel_ids.")
            self.mask = m

        # list of (t0, t1) intervals to exclude from analysis (seconds)
        self.excluded_intervals: List[Interval] = []

        # Optional: start/stop times can be provided by children.
        # Keep as soft attributes to avoid forcing a storage convention.
        self.t_start: Optional[float] = None
        self.t_stop: Optional[float] = None

    # --------- compatibility helpers (so MCSData can be refactored gradually) ---------

    @property
    def ch_ids(self) -> np.ndarray:
        """Backward-compatible alias for channel_ids."""
        return self.channel_ids

    @ch_ids.setter
    def ch_ids(self, value) -> None:
        if value is None:
            self.channel_ids = np.asarray([], dtype=object)
            return
        self.channel_ids = np.asarray(list(value))

    # ------------------------- selection utilities -------------------------

    def set_mask(self, mask: Sequence[bool]) -> None:
        m = np.asarray(list(mask), dtype=bool)
        if m.ndim != 1:
            raise ValueError("mask must be a 1D sequence.")
        if len(m) != len(self.channel_ids):
            raise ValueError("mask must have the same length as channel_ids.")
        self.mask = m

    def enable_channels(self, channels: Iterable[Any]) -> None:
        """Enable a subset of channels by id (others unchanged)."""
        ids = set(channels)
        for i, cid in enumerate(self.channel_ids):
            if cid in ids:
                self.mask[i] = True

    def disable_channels(self, channels: Iterable[Any]) -> None:
        """Disable a subset of channels by id (others unchanged)."""
        ids = set(channels)
        for i, cid in enumerate(self.channel_ids):
            if cid in ids:
                self.mask[i] = False

    def exclude_interval(self, t0: Number, t1: Number) -> None:
        """Exclude a time interval [t0, t1] (seconds) from downstream analysis."""
        a, b = float(t0), float(t1)
        if b < a:
            a, b = b, a
        self.excluded_intervals.append((a, b))
        self.excluded_intervals = self._normalize_intervals(self.excluded_intervals)

    def clear_exclusions(self) -> None:
        self.excluded_intervals.clear()

    @staticmethod
    def _normalize_intervals(intervals: Sequence[Interval]) -> List[Interval]:
        """Merge overlapping/adjacent intervals and sort."""
        if not intervals:
            return []
        xs = sorted((float(a), float(b)) if a <= b else (float(b), float(a)) for a, b in intervals)
        merged: List[Interval] = []
        cur_a, cur_b = xs[0]
        for a, b in xs[1:]:
            if a <= cur_b:  # overlap or touch
                cur_b = max(cur_b, b)
            else:
                merged.append((cur_a, cur_b))
                cur_a, cur_b = a, b
        merged.append((cur_a, cur_b))
        return merged

    # ------------------------- mask/labels IO -------------------------

    def save_mask_and_labels(self, fname: str, csv_format: bool = False) -> int:
        """
        Save current mask, channel_ids and electrode_labels to a JSON or CSV file.

        Parameters
        ----------
        fname : str
            Output filename.
        csv_format : bool
            If True, write a CSV with 3 columns: channel_id, electrode_label, mask.
            If False, write JSON.

        Returns
        -------
        int
            0 if OK.
        """
        path = Path(fname)

        channel_ids = self.channel_ids.tolist()
        labels = (
            [None] * len(channel_ids)
            if self.electrode_labels is None
            else self.electrode_labels.tolist()
        )
        mask = self.mask.astype(bool).tolist()

        if len(labels) != len(channel_ids) or len(mask) != len(channel_ids):
            raise ValueError("Internal state inconsistent: channel arrays lengths differ.")

        if csv_format or path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["channel_id", "electrode_label", "keep"])
                for cid, lab, m in zip(channel_ids, labels, mask):
                    w.writerow([cid, "" if lab is None else lab, int(bool(m))])
        else:
            payload = {
                "channel_ids": channel_ids,
                "electrode_labels": labels,
                "keep": mask,
            }
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        return 0

    def load_mask_and_labels(self, fname: str) -> int:
        """
        Load mask, channel_ids and electrode_labels from a JSON or CSV file.

        The file is expected to contain the same number of entries as the current
        instance (same channel count). This method does NOT reorder channels;
        it assumes the file lines/order match your current channel ordering.

        Parameters
        ----------
        fname : str
            Input filename (.json or .csv)

        Returns
        -------
        int
            0 if OK.
        """
        path = Path(fname)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            channel_ids: List[Any] = []
            labels: List[Any] = []
            mask: List[bool] = []

            with path.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                expected = {"channel_id", "electrode_label", "keep"}
                if r.fieldnames is None or not expected.issubset(set(r.fieldnames)):
                    raise ValueError(
                        f"CSV must have columns {sorted(expected)} "
                        f"(got {r.fieldnames})."
                    )
                for row in r:
                    channel_ids.append(row["channel_id"])
                    lab = row.get("electrode_label", "")
                    labels.append(None if lab is None or str(lab).strip() == "" else lab)
                    m = row.get("keep", "0")
                    mask.append(bool(int(str(m).strip())))

        else:
            # default to JSON
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                raise ValueError("JSON content must be a dict-like object.")
            for key in ("channel_ids", "electrode_labels", "keep"):
                if key not in payload:
                    raise ValueError(f"JSON is missing required key '{key}'.")

            channel_ids = list(payload["channel_ids"])
            labels = list(payload["electrode_labels"])
            mask = [bool(x) for x in payload["keep"]]

        n = len(self.channel_ids)
        if len(channel_ids) != n or len(labels) != n or len(mask) != n:
            raise ValueError(
                "File does not match current instance channel count: "
                f"expected {n}, got channel_ids={len(channel_ids)}, "
                f"labels={len(labels)}, mask={len(mask)}."
            )

        # Important: do NOT reorder silently. Validate that IDs match exactly.
        # If you want reordering later, we can add a 'reorder=True' option.
        if list(map(str, channel_ids)) != list(map(str, self.channel_ids.tolist())):
            raise ValueError(
                "Channel IDs in file do not match current instance ordering. "
                "Refusing to apply to avoid silent misalignment."
            )

        self.mask = np.asarray(mask, dtype=bool)
        self.electrode_labels = np.asarray(labels) if labels is not None else None
        return 0

    # ------------------------- core contract -------------------------

    @abstractmethod
    def get_raster(self, selection: Optional[DataSelection] = None, **kwargs) -> Any:
        """
        Export data as a Raster-like object.

        Each backend decides how to produce event times per channel, but must apply:
          - channel mask (selection.mask or self.mask)
          - excluded intervals (selection.excluded_intervals or self.excluded_intervals)
          - optional time window [t0, t1]

        Returns
        -------
        Any
            Usually tad.raster.Raster, but kept as Any to avoid circular imports here.
        """
        raise NotImplementedError
