from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ChannelId = Union[int, str]


@dataclass
class Raster:
    """
    Store and manipulate rasterized (event-based) signals across channels.

    This class is primarily designed for neuroscience-style spike rasters, but it
    generalizes to any multi-channel event timestamp data in dynamical systems.

    Internally, timestamps are stored per channel as **sorted 1D NumPy arrays**
    (dtype typically float). This makes time-window slicing efficient using
    ``np.searchsorted`` and keeps representation compact.

    Parameters
    ----------
    events
        Mapping from channel IDs to 1D arrays (or array-like) of event times.
        Arrays will be coerced to ``dtype``, flattened, and sorted.
    dtype
        NumPy dtype used to store timestamps.
    allow_new_channels
        If True, missing channels referenced by methods (e.g., ``insert``) are
        created automatically. If False, referencing a missing channel raises
        ``KeyError``.
    """
    events: Dict[ChannelId, np.ndarray] = field(default_factory=dict)
    dtype: np.dtype = np.dtype(np.float64)
    allow_new_channels: bool = True

    def __post_init__(self) -> None:
        """
        Normalize internal storage after dataclass initialization.

        Coerces each channel's timestamps to a 1D NumPy array of ``self.dtype``
        and sorts them in ascending order.

        Notes
        -----
        - If ``events`` contains non-1D array-likes, they are flattened.
        - Empty channels are represented by empty 1D arrays.
        """
        normalized: Dict[ChannelId, np.ndarray] = {}
        for ch, ts in self.events.items():
            arr = np.asarray(ts, dtype=self.dtype).ravel()
            if arr.size:
                arr.sort()
            normalized[ch] = arr
        self.events = normalized

    # ---------------------------------------------------------------------
    # Constructors / basic accessors
    # ---------------------------------------------------------------------
    @classmethod
    def empty(
        cls,
        channels: Optional[Iterable[ChannelId]] = None,
        *,
        dtype: np.dtype = np.dtype(np.float64),
        allow_new_channels: bool = True,
    ) -> "Raster":
        """
        Create an empty raster, optionally pre-defining channels.

        Parameters
        ----------
        channels
            Iterable of channel IDs to create with no events.
        dtype
            NumPy dtype used to store timestamps.
        allow_new_channels
            If True, missing channels referenced by methods can be created
            automatically.

        Returns
        -------
        Raster
            A new empty Raster instance.
        """
        r = cls(events={}, dtype=dtype, allow_new_channels=allow_new_channels)
        if channels is not None:
            for ch in channels:
                r.events[ch] = np.asarray([], dtype=r.dtype)
        return r

    def channels(self) -> List[ChannelId]:
        """
        Return the list of channel IDs currently stored.

        Returns
        -------
        list
            List of channel IDs.
        """
        return list(self.events.keys())

    def n_channels(self) -> int:
        """
        Return the number of channels currently stored.

        Returns
        -------
        int
            Number of channels.
        """
        return len(self.events)

    def copy(self) -> "Raster":
        """
        Return a deep copy of this Raster (arrays are copied).

        Returns
        -------
        Raster
            Deep copy of the raster.
        """
        out = Raster.empty(channels=self.channels(), dtype=self.dtype, allow_new_channels=self.allow_new_channels)
        for ch, arr in self.events.items():
            out.events[ch] = arr.copy()
        return out

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _require_channel(self, channel: ChannelId) -> None:
        """
        Ensure a channel exists in the raster.

        If the channel is absent and ``allow_new_channels`` is True, a new empty
        channel is created. Otherwise, a ``KeyError`` is raised.

        Parameters
        ----------
        channel
            Channel ID to ensure.

        Raises
        ------
        KeyError
            If the channel does not exist and ``allow_new_channels`` is False.
        """
        if channel not in self.events:
            if not self.allow_new_channels:
                raise KeyError(f"Channel {channel!r} does not exist and allow_new_channels=False.")
            self.events[channel] = np.asarray([], dtype=self.dtype)

    def _validate_time(self, t: float) -> float:
        """
        Validate and coerce a scalar time value.

        Parameters
        ----------
        t
            Candidate timestamp.

        Returns
        -------
        float
            Validated timestamp as Python float.

        Raises
        ------
        ValueError
            If ``t`` is not finite (NaN or +/-inf).
        """
        tf = float(t)
        if not np.isfinite(tf):
            raise ValueError(f"Timestamp must be finite, got {t!r}.")
        return tf

    def _validate_time_array(self, t: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
        """
        Validate and coerce an array-like of timestamps.

        Parameters
        ----------
        t
            Array-like of timestamps.

        Returns
        -------
        ndarray
            1D NumPy array of dtype ``self.dtype``.

        Raises
        ------
        ValueError
            If any timestamp is not finite.
        """
        arr = np.asarray(list(t) if not isinstance(t, np.ndarray) else t, dtype=self.dtype).ravel()
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError("All timestamps must be finite (no NaN/+/-inf).")
        return arr

    def _window_indices(
        self,
        arr: np.ndarray,
        tstart: float,
        tstop: float,
        inclusive_stop: bool,
    ) -> Tuple[int, int]:
        """
        Compute slicing indices for a time window on a sorted array.

        Parameters
        ----------
        arr
            Sorted 1D array of timestamps.
        tstart
            Window start time.
        tstop
            Window stop time.
        inclusive_stop
            If True, include events at exactly ``tstop``. If False, treat the
            interval as half-open.

        Returns
        -------
        left, right
            Indices such that ``arr[left:right]`` are events in the window.

        Notes
        -----
        - Uses ``np.searchsorted`` for O(log n) boundary location.
        - Requires ``arr`` to be sorted ascending.
        """
        left = np.searchsorted(arr, tstart, side="left")
        right_side = "right" if inclusive_stop else "left"
        right = np.searchsorted(arr, tstop, side=right_side)
        return int(left), int(right)

    # ---------------------------------------------------------------------
    # Channel management (NEW)
    # ---------------------------------------------------------------------
    def insert_channel(
        self,
        channel: ChannelId,
        times: Optional[Union[np.ndarray, Iterable[float]]] = None,
        *,
        overwrite: bool = False,
        sort: bool = True,
    ) -> None:
        """
        Insert a new channel (optionally with initial events).

        Parameters
        ----------
        channel
            Channel ID to insert.
        times
            Optional array-like of timestamps for the channel. If None, the
            channel is created empty.
        overwrite
            If False (default), raises if the channel already exists.
            If True, replaces the channel's events with ``times`` (or empty).
        sort
            If True (default), sorts provided times.

        Raises
        ------
        KeyError
            If channel exists and ``overwrite=False``.
        ValueError
            If any timestamp is not finite.
        """
        if (channel in self.events) and not overwrite:
            raise KeyError(f"Channel {channel!r} already exists (overwrite=False).")

        if times is None:
            arr = np.asarray([], dtype=self.dtype)
        else:
            arr = self._validate_time_array(times)
            if sort and arr.size:
                arr.sort()

        self.events[channel] = arr

    def pop_channel(self, channel: ChannelId) -> np.ndarray:
        """
        Remove an entire channel and return its timestamps.

        Parameters
        ----------
        channel
            Channel ID to remove.

        Returns
        -------
        ndarray
            Sorted 1D array of timestamps that were stored in the channel.

        Raises
        ------
        KeyError
            If the channel does not exist.
        """
        if channel not in self.events:
            raise KeyError(f"Channel {channel!r} does not exist.")
        return self.events.pop(channel)

    # ---------------------------------------------------------------------
    # Core event manipulation
    # ---------------------------------------------------------------------
    def insert(self, channel: ChannelId, t: float) -> None:
        """
        Insert a timestamp into a channel, preserving sorted order.

        Parameters
        ----------
        channel
            Channel ID.
        t
            Timestamp to insert.

        Notes
        -----
        - Insertion into a NumPy array is O(n) due to shifting; for very frequent
          online insertion at large scale, a buffered strategy can be added later.
        - Duplicates are allowed.
        """
        tf = self._validate_time(t)
        self._require_channel(channel)
        arr = self.events[channel]
        idx = np.searchsorted(arr, tf, side="right")
        self.events[channel] = np.insert(arr, idx, np.asarray(tf, dtype=self.dtype))

    def insert_timestamparray(
        self,
        channel: ChannelId,
        times: Union[np.ndarray, Iterable[float]],
        *,
        assume_sorted: bool = False,
        sort_result: bool = True,
    ) -> None:
        """
        Insert multiple timestamps into a channel.

        Parameters
        ----------
        channel
            Channel ID.
        times
            Array-like timestamps to insert (NumPy array or iterable).
        assume_sorted
            If True, assumes ``times`` is already sorted ascending (saves a sort).
        sort_result
            If True (default), ensures the channel remains sorted after insertion.

        Raises
        ------
        ValueError
            If any timestamp is not finite.

        Notes
        -----
        - Uses concatenation + sort as a simple, robust approach.
        - Complexity is O((n+m) log(n+m)) due to sorting; for very large n with small
          m, a merge approach could be implemented later.
        """
        self._require_channel(channel)
        new_arr = self._validate_time_array(times)

        if new_arr.size == 0:
            return

        if not assume_sorted:
            new_arr.sort()

        old = self.events[channel]
        merged = np.concatenate([old, new_arr]).astype(self.dtype, copy=False)

        if sort_result and merged.size:
            merged.sort()

        self.events[channel] = merged

    def pop(self, channel: ChannelId, index: int = -1) -> float:
        """
        Remove and return one timestamp from a channel.

        Parameters
        ----------
        channel
            Channel ID.
        index
            Index of the event to remove (default: -1, last event).

        Returns
        -------
        float
            The removed timestamp.

        Raises
        ------
        IndexError
            If the channel has no events.
        """
        self._require_channel(channel)
        arr = self.events[channel]
        if arr.size == 0:
            raise IndexError(f"Cannot pop from empty channel {channel!r}.")
        t = float(arr[index])
        self.events[channel] = np.delete(arr, index)
        return t

    def clear(self, channel: Optional[ChannelId] = None) -> None:
        """
        Clear events from one channel or from all channels.

        Parameters
        ----------
        channel
            If provided, clears only that channel. If None, clears all channels.
        """
        if channel is None:
            for ch in list(self.events.keys()):
                self.events[ch] = np.asarray([], dtype=self.dtype)
        else:
            self._require_channel(channel)
            self.events[channel] = np.asarray([], dtype=self.dtype)

    # ---------------------------------------------------------------------
    # Time-window operations
    # ---------------------------------------------------------------------
    def between(self, tstart: float, tstop: float, *, inclusive_stop: bool = False) -> "Raster":
        """
        Return a new Raster containing events in a time interval.

        Parameters
        ----------
        tstart
            Start time of the interval.
        tstop
            Stop time of the interval.
        inclusive_stop
            If False (default), returns events in [tstart, tstop).
            If True, returns events in [tstart, tstop].

        Returns
        -------
        Raster
            A new Raster with events restricted to the interval.

        Raises
        ------
        ValueError
            If ``tstop < tstart`` or either boundary is not finite.
        """
        tstart_f = self._validate_time(tstart)
        tstop_f = self._validate_time(tstop)
        if tstop_f < tstart_f:
            raise ValueError(f"tstop ({tstop_f}) must be >= tstart ({tstart_f}).")

        out = Raster.empty(channels=self.channels(), dtype=self.dtype, allow_new_channels=self.allow_new_channels)
        for ch, arr in self.events.items():
            left, right = self._window_indices(arr, tstart_f, tstop_f, inclusive_stop)
            out.events[ch] = arr[left:right].copy()
        return out

    def shift(self, dt: float, *, in_place: bool = False) -> "Raster":
        """
        Shift all timestamps by a constant offset.

        Parameters
        ----------
        dt
            Time offset to add to every timestamp (can be negative).
        in_place
            If True, modifies and returns self. If False, returns a shifted copy.

        Returns
        -------
        Raster
            Shifted raster.

        Notes
        -----
        This operation preserves sorted order within each channel because adding
        a constant is order-preserving.
        """
        dt_f = self._validate_time(dt)
        target = self if in_place else self.copy()
        for ch, arr in target.events.items():
            if arr.size:
                target.events[ch] = (arr + dt_f).astype(target.dtype, copy=False)
        return target

    # ---------------------------------------------------------------------
    # Channel operations
    # ---------------------------------------------------------------------
    def relabel_channels(
        self,
        mapping: Mapping[ChannelId, ChannelId],
        *,
        on_conflict: str = "raise",
        in_place: bool = False,
    ) -> "Raster":
        """
        Relabel channels according to a mapping.

        Parameters
        ----------
        mapping
            Dictionary-like mapping {old_channel: new_channel}.
            Channels not present in mapping are left unchanged.
        on_conflict
            What to do if multiple old channels map to the same new channel, or
            if a new channel already exists.
            - "raise": raise ValueError
            - "merge": merge events into the target channel (and keep sorted)
        in_place
            If True, modifies and returns self. If False, returns a relabeled copy.

        Returns
        -------
        Raster
            Raster with relabeled channels.

        Raises
        ------
        ValueError
            If conflicts occur and ``on_conflict="raise"``.
        """
        if on_conflict not in {"raise", "merge"}:
            raise ValueError(f"on_conflict must be 'raise' or 'merge', got {on_conflict!r}.")

        src = self if in_place else self.copy()

        new_events: Dict[ChannelId, np.ndarray] = {}
        for old_ch, arr in src.events.items():
            new_ch = mapping.get(old_ch, old_ch)
            if new_ch not in new_events:
                new_events[new_ch] = arr.copy()
            else:
                if on_conflict == "raise":
                    raise ValueError(f"Channel relabel conflict on target {new_ch!r}.")
                merged = np.concatenate([new_events[new_ch], arr])
                if merged.size:
                    merged.sort()
                new_events[new_ch] = merged.astype(src.dtype, copy=False)

        src.events = new_events
        return src

    # ---------------------------------------------------------------------
    # Multi-raster operations
    # ---------------------------------------------------------------------
    def merge(
        self,
        other: "Raster",
        *,
        on_missing_channels: str = "union",
        in_place: bool = False,
    ) -> "Raster":
        """
        Merge another Raster into this one (concatenate events per channel).

        Parameters
        ----------
        other
            Raster to merge into this raster.
        on_missing_channels
            Channel set handling:
            - "union": result contains union of channels from both rasters
            - "intersection": result contains only channels present in both
        in_place
            If True, modifies and returns self. If False, returns a merged copy.

        Returns
        -------
        Raster
            Merged raster.

        Raises
        ------
        ValueError
            If ``on_missing_channels`` is not recognized.

        Notes
        -----
        - Event times are concatenated and then sorted within each channel.
        - ``other`` is coerced to ``self.dtype`` in the result.
        """
        if on_missing_channels not in {"union", "intersection"}:
            raise ValueError(
                f"on_missing_channels must be 'union' or 'intersection', got {on_missing_channels!r}."
            )

        target = self if in_place else self.copy()

        if on_missing_channels == "union":
            ch_set = set(target.events.keys()) | set(other.events.keys())
        else:
            ch_set = set(target.events.keys()) & set(other.events.keys())

        for ch in ch_set:
            a = target.events.get(ch, np.asarray([], dtype=target.dtype))
            b = other.events.get(ch, np.asarray([], dtype=other.dtype))
            b = np.asarray(b, dtype=target.dtype).ravel()
            if a.size == 0:
                merged = b.copy()
            elif b.size == 0:
                merged = a.copy()
            else:
                merged = np.concatenate([a, b])
                merged.sort()
            target.events[ch] = merged

        if on_missing_channels == "intersection":
            target.events = {ch: target.events[ch] for ch in ch_set}

        return target

    def as_arrays(
        self,
        *,
        channels: Optional[Sequence[ChannelId]] = None,
        sort_by_time: bool = True,
        channel_dtype: Optional[np.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export events as concatenated (times, channels) arrays.

        Parameters
        ----------
        channels
            Subset/order of channels to export. If None, exports all channels in
            insertion order.
        sort_by_time
            If True, sort the concatenated output by time (stable for equal times).
            If False, output is grouped by channel in the provided order.
        channel_dtype
            Optional dtype for returned channel array. If None:
            - uses int64 if all channel IDs are ints
            - otherwise uses object dtype

        Returns
        -------
        times : ndarray, shape (N,)
            Concatenated event times.
        ch_ids : ndarray, shape (N,)
            Channel identifiers per event.
        """
        ch_list = list(channels) if channels is not None else self.channels()

        if channel_dtype is None:
            if all(isinstance(ch, (int, np.integer)) for ch in ch_list):
                channel_dtype = np.dtype(np.int64)
            else:
                channel_dtype = np.dtype(object)

        times_list: List[np.ndarray] = []
        ch_list_rep: List[np.ndarray] = []

        for ch in ch_list:
            self._require_channel(ch)
            ts = self.events[ch]
            if ts.size == 0:
                continue
            times_list.append(ts.astype(self.dtype, copy=False))
            ch_list_rep.append(np.full(ts.shape, ch, dtype=channel_dtype))

        if not times_list:
            return (np.asarray([], dtype=self.dtype), np.asarray([], dtype=channel_dtype))

        times = np.concatenate(times_list)
        ch_ids = np.concatenate(ch_list_rep)

        if sort_by_time and times.size:
            order = np.argsort(times, kind="stable")
            times = times[order]
            ch_ids = ch_ids[order]

        return times, ch_ids

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    def plot(
        self,
        *,
        ax: Optional[plt.Axes] = None,
        channels: Optional[Sequence[ChannelId]] = None,
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        inclusive_stop: bool = False,
        y_gap: float = 1.0,
        tick_halfheight: Optional[float] = None,
        linewidth: float = 1.0,
        sort_channels: bool = True,
        show: bool = False,
    ) -> plt.Axes:
        """
        Plot the raster using matplotlib.

        Each event is drawn as a vertical tick at its timestamp, arranged by
        channel along the y-axis.

        Parameters
        ----------
        ax
            Matplotlib Axes to draw on. If None, creates a new figure and axes.
        channels
            Subset/order of channels to plot. If None, plot all channels.
        tstart
            Optional time window start. If provided with ``tstop``, the plot
            is restricted to that interval.
        tstop
            Optional time window stop.
        inclusive_stop
            If True, include events at exactly ``tstop`` when windowing.
        y_gap
            Vertical spacing between channels.
        tick_halfheight
            Half-height of the vertical tick marks. If None, defaults to
            ``0.4 * y_gap``.
        linewidth
            Line width of tick marks.
        sort_channels
            If True, try to sort channel IDs for display.
        show
            If True, calls ``plt.show()`` before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()

        ch_list = list(channels) if channels is not None else self.channels()
        if sort_channels:
            try:
                ch_list = sorted(ch_list)
            except TypeError:
                pass

        raster = self
        if tstart is not None or tstop is not None:
            all_times = np.concatenate(
                [v for v in self.events.values() if v.size] or [np.asarray([], dtype=self.dtype)]
            )
            if tstart is None:
                tstart = float(all_times.min()) if all_times.size else 0.0
            if tstop is None:
                tstop = float(all_times.max()) if all_times.size else (tstart + 1.0)
            raster = self.between(float(tstart), float(tstop), inclusive_stop=inclusive_stop)

        if tick_halfheight is None:
            tick_halfheight = 0.4 * y_gap

        y_positions = np.arange(len(ch_list), dtype=float) * y_gap

        for i, ch in enumerate(ch_list):
            raster._require_channel(ch)
            ts = raster.events[ch]
            if ts.size == 0:
                continue
            y = y_positions[i]
            ax.vlines(ts, y - tick_halfheight, y + tick_halfheight, linewidth=linewidth)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(ch) for ch in ch_list])
        ax.set_ylabel("Channel")
        ax.set_xlabel("Time")
        ax.set_title("Raster")

        if tstart is not None and tstop is not None:
            ax.set_xlim(float(tstart), float(tstop))

        ax.margins(x=0.01)

        if show:
            plt.show()

        return ax



