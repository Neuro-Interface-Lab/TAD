"""
mcs_data.py

Utilities to load MCS .h5 recordings with SpikeInterface, detect spikes, interpret
digital trigger channels, and build raster representations.

This file is a *cleanup* of an existing working implementation:
- No intentional algorithmic changes
- Improved organization, naming, typing, and docstrings (NumPy format)
- Removed unused imports and clarified responsibilities
"""

from __future__ import annotations

import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as pre
import spikeinterface.widgets as sw
import json
import csv
from pathlib import Path

from typing import Any, Callable, Dict, Optional, List
from matplotlib.widgets import CheckButtons
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from .processing_history import DatasetInfo, ProcessingHistory


def on_delta_t(digital_recording, triggers, fsample: float, delta_t: float) -> None:
    """
    Interpret each rising edge as a trigger starting a fixed-duration interval.

    Parameters
    ----------
    digital_recording : array-like
        Digital signal samples.
    triggers : object
        Triggers container with `add_interval_slot(start, end, ...)`.
    fsample : float
        Sampling frequency (Hz).
    delta_t : float
        Duration (seconds) to extend each trigger interval after the rising edge.

    Notes
    -----
    This function follows the existing implementation, including its rounding
    strategy for aligning to sample time.
    """
    for i in range(1, len(digital_recording)):
        if digital_recording[i] > digital_recording[i - 1]:
            start = round(i) / fsample
            end = round((round(i) / fsample + round(delta_t * fsample) / fsample) * fsample) / fsample
            triggers.add_interval_slot(start=start, end=end, ID="stim_ON", blank=True)


def on_off_interpretor(digital_recording, triggers, fsample: float) -> None:
    """
    Interpret odd/even rising edges as start/end of stimulus intervals.

    Parameters
    ----------
    digital_recording : array-like
        Digital signal samples.
    triggers : object
        Triggers container with `add_interval_slot(start, end, ...)`.
    fsample : float
        Sampling frequency (Hz).

    Notes
    -----
    Rising edges are detected by simple sample-to-sample increase.
    Intervals are formed as (rising_edges[0] -> rising_edges[1]),
    (rising_edges[2] -> rising_edges[3]), ...
    """
    rising_edges = []
    for i in range(1, len(digital_recording)):
        if digital_recording[i] > digital_recording[i - 1]:
            rising_edges.append(i)

    for i in range(1, len(rising_edges), 2):
        triggers.add_interval_slot(
            start=rising_edges[i - 1] / fsample,
            end=rising_edges[i] / fsample,
            ID="stim_ON",
            blank=False,
        )

def tracked_operation(
    name: Optional[str] = None,
    *,
    track: bool = True,
    include_result_artifacts: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Callable:
    """
    Decorator to record a method call into `self.history`.

    Parameters
    ----------
    name : str, optional
        Operation name. Defaults to function name.
    track : bool, default=True
        Whether to track this operation.
    include_result_artifacts : callable, optional
        Function called with the method return value; must return JSON-friendly dict
        to attach as `artifacts`.
    """
    def _decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        def _wrapped(self, *args, **kwargs):
            if (not track) or (getattr(self, "history", None) is None):
                return func(self, *args, **kwargs)

            before_snapshot = self.history.snapshot_state()
            before_hash = self.history.state_hash(before_snapshot)

            result = func(self, *args, **kwargs)

            after_snapshot = self.history.snapshot_state()
            after_hash = self.history.state_hash(after_snapshot)

            # Capture parameters in a conservative, explicit way
            params = {"args": args, "kwargs": kwargs}

            artifacts = {}
            if include_result_artifacts is not None:
                try:
                    artifacts = include_result_artifacts(result) or {}
                except Exception:
                    artifacts = {"note": "artifact extraction failed"}

            # Optional summary can be derived from after_snapshot if desired
            summary = {
                "mask_n_kept": int(after_snapshot.get("mask_n_kept", -1)) if isinstance(after_snapshot, dict) else -1,
                "excluded_intervals_n": int(after_snapshot.get("excluded_intervals_n", -1)) if isinstance(after_snapshot, dict) else -1,
            }

            self.history.record(
                op_name,
                params=params,
                state_before=before_hash,
                state_after=after_hash,
                summary=summary,
                artifacts=artifacts,
            )
            return result

        _wrapped.__name__ = func.__name__
        _wrapped.__doc__ = func.__doc__
        _wrapped.__qualname__ = func.__qualname__
        return _wrapped

    return _decorator

def _raster_artifacts(r) -> Dict[str, Any]:
    # Best-effort, does not assume Raster internals
    artifacts = {}
    try:
        artifacts["raster_channels_n"] = int(len(getattr(r, "channels", [])))  # if exists
    except Exception:
        pass
    return artifacts


class MCSData:
    """
    Load an MCS `.h5` recording and provide basic signal processing utilities.

    Parameters
    ----------
    fname : str
        Path to the `.h5` file containing the MCS recording.
    fsample : float, optional
        Sampling frequency (Hz). If None, it is read from the recording.
    load_recording : bool, default=True
        Whether to load the analog recording stream into SpikeInterface.
    load_digital : bool, default=False
        Whether to also read a digital channel from the HDF5 file.
        The current implementation reads:
        `Data/Recording_0/AnalogStream/Stream_0/ChannelData[0]`.
    generate_probe : bool, default=False
        Whether to generate a probe object from `probe_data`.
    probe_data : dict, optional
        Probe configuration. Expected keys:
        - "positions"
        - "contact_shape"
        - "shape_params"
        - "ndims"
    """

    def __init__(
        self,
        fname: str,
        fsample: Optional[float] = None,
        load_recording: bool = True,
        load_digital: bool = False,
        generate_probe: bool = False,
        probe_data: Optional[dict] = None,
    ) -> None:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")

        self.fname: str = fname
        self.fsample: Optional[float] = fsample
        self.load_digital: bool = load_digital

        # Core data
        self.recording = None
        self.traces: Optional[np.ndarray] = None
        self.peaks = None

        # Channel/probe metadata
        self.ch_ids = None
        self.electrode_labels = None
        self.mask: Optional[np.ndarray] = None

        self.probe = None
        self.probe_positions = None
        self.probe_contact_shape = None
        self.probe_shape_params = None
        self.probe_ndims = None

        # Time/selection masks
        self.time_vector: Optional[np.ndarray] = None
        self.temporal_mask: Optional[np.ndarray] = None
        self.excluded_intervals = []  

        # Digital/triggers
        self.digital_recording = None
        self.triggers = None

        # State
        self.artifact_removal_status: bool = False

        if load_recording:
            self._load_recording()
            self.time_vector = np.arange(self.recording.get_total_samples()) / float(self.fsample)
            self.ch_ids = self.recording.channel_ids
            self.electrode_labels = self.recording.get_property("electrode_labels")
            self.mask = np.ones(self.recording.get_num_channels(), dtype=bool)
            self.temporal_mask = np.ones_like(self.time_vector, dtype=bool)

        if generate_probe:
            self._generate_probe(probe_data)
        
        # Initialize history,to record metadata
        if load_recording and self.recording is not None:
            ds = DatasetInfo.from_path(
                fname=self.fname,
                sampling_frequency=float(self.fsample) if self.fsample is not None else None,
                stream_id=1,
                channel_ids=list(self.ch_ids) if self.ch_ids is not None else None,
                electrode_labels=list(self.electrode_labels) if self.electrode_labels is not None else None,
            )
            self.history = ProcessingHistory(dataset=ds)
            self.history.set_state_getter(self._history_snapshot)  # NEW callback
        else:
            self.history = None

    # ----------------------------- History handling -----------------------
    def _history_snapshot(self) -> Dict[str, Any]:
        """
        Create a JSON-friendly snapshot of relevant processing state (C).

        Returns
        -------
        dict
            Snapshot dict suitable for hashing and JSON serialization.
        """
        snap: Dict[str, Any] = {
            "fname": self.fname,
            "sampling_frequency": float(self.fsample) if self.fsample is not None else None,
        }

        # Channel mask summary + compact representation (store list of kept channel ids)
        if self.mask is not None and self.ch_ids is not None:
            kept = list(self.ch_ids[self.mask])
            snap["mask_n_kept"] = int(len(kept))
            snap["mask_kept_channel_ids"] = kept

        # Temporal exclusions: store intervals list rather than full per-sample boolean
        snap["excluded_intervals"] = list(self.excluded_intervals) if hasattr(self, "excluded_intervals") else []
        snap["excluded_intervals_n"] = int(len(snap["excluded_intervals"]))

        # Trigger summary (store the slots, not the digital signal)
        if self.triggers is not None and hasattr(self.triggers, "slots"):
            snap["triggers"] = [
                {"start": float(s.start), "end": float(s.end), "id": getattr(s, "ID", None), "blank": getattr(s, "blank", None)}
                for s in self.triggers.slots
            ]
            snap["triggers_n"] = int(len(snap["triggers"]))

        # Note: filter parameters are not explicitly stored by SpikeInterface here;
        # you can store them at call-time in the operation log (B), which is enough.
        # Same for spike detection params: store them in operation params.

        return snap


    # ----------------------------- IO / setup -----------------------------

    def _load_recording(self) -> None:
        """
        Load the MCS recording via SpikeInterface and optionally load digital signal.

        Notes
        -----
        - Uses `stream_id=1` exactly as in the original code.
        - Renames channels based on `electrode_labels` to `Ch{label}`.
        """
        try:
            self.recording = se.read_mcsh5(self.fname, stream_id=1)
        except Exception as exc:
            print(f"Error loading recording: {exc}")
            sys.exit(1)

        if self.load_digital:
            try:
                with h5py.File(self.fname, "r") as f:
                    stream = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
                    self.digital_recording = stream[0]
            except Exception as exc:
                print(f"Error loading digital recording: {exc}")

        electrode_labels = self.recording.get_property("electrode_labels")
        self.recording = self.recording.rename_channels([f"Ch{lab}" for lab in electrode_labels])

        if self.fsample is None:
            self.fsample = self.recording.get_sampling_frequency()

    def _generate_probe(self, probe_data: Optional[dict]) -> None:
        """
        Create a probe object (if probe configuration is provided).

        Parameters
        ----------
        probe_data : dict, optional
            Probe configuration dictionary.

        Notes
        -----
        Keeps the original behavior, including the placeholder path when
        `probe_data` is None.
        """
        from .mea_probe import MEAProbe  # avoid circular import

        if probe_data is None:
            print("probe_data was not provided, implementing 60MEA100/10iR geometry")
            # Placeholder (unchanged behavior)
            return

        self.probe_positions = probe_data.get("positions")
        self.probe_contact_shape = probe_data.get("contact_shape")
        self.probe_shape_params = probe_data.get("shape_params")
        self.probe_ndims = probe_data.get("ndims")
        self.probe = MEAProbe(
            self.probe_positions,
            self.probe_contact_shape,
            self.probe_shape_params,
            self.probe_ndims,
        )

    # ----------------------------- History export ------------------------
    def export_history_json(self, path: str, indent: int = 2) -> None:
        """
        Export processing history to a JSON file.

        Parameters
        ----------
        path : str
            Output path for the JSON file.
        indent : int, default=2
            JSON indentation level.
        """
        if self.history is None:
            raise ValueError("No history available to export.")
        self.history.save_json(path, indent=indent)


    # ----------------------------- Basic API -----------------------------
    @tracked_operation("set_mask")
    def set_mask(self, mask: np.ndarray) -> int:
        """
        Set the channel mask used by downstream operations.

        Parameters
        ----------
        mask : np.ndarray of bool
            Boolean array aligned with `self.recording` channel order.

        Returns
        -------
        int
            Always returns 1 (kept for backward compatibility).
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if len(mask) != self.recording.get_num_channels():
            raise ValueError("Mask length must match number of channels.")
        self.mask = mask
        return 1

    @tracked_operation("save_mask_and_labels")
    def save_mask_and_labels(self, fname: str, csv_format: bool = False) -> int:
        """
        Save the current channel mask, channel IDs, and electrode labels.

        Parameters
        ----------
        fname : str
            Output filepath. If `csv_format` is True, this should typically end with
            ".csv"; otherwise ".json".
        csv_format : bool, default=False
            If True, save as CSV (editable in a text editor). If False, save as JSON.

        Returns
        -------
        int
            Always returns 1 if the file is written successfully.

        Raises
        ------
        ValueError
            If `fname` is empty or if required attributes are not initialized.
        """
        if not isinstance(fname, str) or len(fname.strip()) == 0:
            raise ValueError("`fname` must be a non-empty string.")

        if self.mask is None or self.ch_ids is None or self.electrode_labels is None:
            raise ValueError("mask/ch_ids/electrode_labels are not initialized.")

        out_path = Path(fname)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_list = np.asarray(self.mask, dtype=bool).tolist()
        ch_ids_list = np.asarray(self.ch_ids).tolist()
        labels_list = np.asarray(self.electrode_labels).tolist()

        if csv_format:
            # One row per channel: channel_id, electrode_label, keep(0/1)
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["channel_id", "electrode_label", "keep"])
                for ch, lab, keep in zip(ch_ids_list, labels_list, mask_list):
                    writer.writerow([ch, lab, int(bool(keep))])
            return 1

        payload: Dict[str, Any] = {
            "mask": mask_list,
            "channel_ids": ch_ids_list,
            "electrode_labels": labels_list,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return 1

    @tracked_operation("load_mask_and_labels")
    def load_mask_and_labels(self, fname: str) -> int:
        """
        Loads a mask and channel labels from a JSON or CSV file into this instance.

        The loader auto-detects format:
        - JSON: expects keys {"mask", "channel_ids", "electrode_labels"}
        - CSV : expects header with columns including:
        - channel_id
        - electrode_label
        - keep   (0/1 or true/false)

        Parameters
        ----------
        fname : str
            Path to the JSON/CSV file.

        Returns
        -------
        int
            Always returns 1 on success.

        Raises
        ------
        FileNotFoundError
            If `fname` does not exist.
        ValueError
            If the file format is unsupported, contents are invalid, or lengths mismatch.
        """
        if not isinstance(fname, str) or len(fname.strip()) == 0:
            raise ValueError("`fname` must be a non-empty string.")

        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {fname}")

        suffix = path.suffix.lower()

        # -------------------------
        # Helpers
        # -------------------------
        def _parse_keep(val) -> bool:
            if isinstance(val, bool):
                return val
            if val is None:
                raise ValueError("Missing 'keep' value.")
            s = str(val).strip().lower()
            if s in {"1", "true", "t", "yes", "y"}:
                return True
            if s in {"0", "false", "f", "no", "n"}:
                return False
            raise ValueError(f"Invalid keep value: {val!r} (expected 0/1 or true/false)")

        def _validate_lengths(mask, ch_ids, labels) -> None:
            if mask is None or ch_ids is None or labels is None:
                raise ValueError("Loaded data missing one of: mask, channel_ids, electrode_labels.")
            n = len(mask)
            if len(ch_ids) != n or len(labels) != n:
                raise ValueError(
                    f"Length mismatch: mask={len(mask)}, channel_ids={len(ch_ids)}, electrode_labels={len(labels)}"
                )

            # If recording already loaded, enforce expected number of channels
            if getattr(self, "recording", None) is not None:
                expected = int(self.recording.get_num_channels())
                if n != expected:
                    raise ValueError(f"File contains {n} channels but recording has {expected} channels.")

            # If ch_ids already exist, also enforce alignment length
            if getattr(self, "ch_ids", None) is not None:
                expected = len(self.ch_ids)
                if n != expected:
                    raise ValueError(f"File contains {n} channels but this instance has {expected} channels.")

        # -------------------------
        # Load JSON
        # -------------------------
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                raise ValueError("JSON file must contain an object with keys: mask, channel_ids, electrode_labels.")

            try:
                mask = payload["mask"]
                ch_ids = payload["channel_ids"]
                labels = payload["electrode_labels"]
            except KeyError as exc:
                raise ValueError(f"Missing key in JSON file: {exc}") from exc

            _validate_lengths(mask, ch_ids, labels)

            self.mask = np.asarray(mask, dtype=bool)
            self.ch_ids = np.asarray(ch_ids)
            self.electrode_labels = np.asarray(labels)
            return 1

        # -------------------------
        # Load CSV
        # -------------------------
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV file has no header row.")

                # Normalize fieldnames
                fields = {name.strip().lower(): name for name in reader.fieldnames}

                required = {"channel_id", "electrode_label", "keep"}
                if not required.issubset(fields.keys()):
                    raise ValueError(
                        f"CSV must contain columns {sorted(required)}; found {sorted(fields.keys())}"
                    )

                ch_ids = []
                labels = []
                mask = []

                for row_idx, row in enumerate(reader, start=2):  # header = line 1
                    ch = row.get(fields["channel_id"])
                    lab = row.get(fields["electrode_label"])
                    keep_raw = row.get(fields["keep"])

                    if ch is None or lab is None or keep_raw is None:
                        raise ValueError(f"Missing values at CSV row {row_idx}.")

                    ch_ids.append(ch)
                    labels.append(lab)
                    mask.append(_parse_keep(keep_raw))

            _validate_lengths(mask, ch_ids, labels)

            self.mask = np.asarray(mask, dtype=bool)
            self.ch_ids = np.asarray(ch_ids)
            self.electrode_labels = np.asarray(labels)
            return 1

        # -------------------------
        # Unknown format
        # -------------------------
        raise ValueError(f"Unsupported file type {suffix!r}. Use .json or .csv.")


    @tracked_operation("apply_filter")
    def apply_filter(self, bandpass=None, btype: str = "bandpass"):
        """
        Apply a SpikeInterface filter to the recording.

        Parameters
        ----------
        bandpass : array-like or float
            For 'bandpass', provide [low_freq, high_freq].
            For 'highpass', provide a single cutoff or the format expected by SpikeInterface.
        btype : {'bandpass', 'highpass'}, default='bandpass'
            Filter type.

        Returns
        -------
        recording : spikeinterface.BaseRecording
            The filtered recording.

        Raises
        ------
        ValueError
            If recording is not loaded or `bandpass` is not provided.

        Notes
        -----
        The original code had a check against the builtin `filter`; this cleanup
        removes that non-functional check without changing behavior of the actual filtering.
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if bandpass is None:
            raise ValueError("Bandpass frequencies must be provided.")

        self.recording = pre.filter(self.recording, band=bandpass, btype=btype)
        return self.recording

    # ----------------------------- Traces -----------------------------

    def get_traces(
        self,
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        channel_ids=None,
        return_in_uV: bool = True,
    ) -> np.ndarray:
        """
        Retrieve traces for a time window and channel selection.

        Parameters
        ----------
        tstart : float, optional
            Start time in seconds. Defaults to the beginning of the recording.
        tstop : float, optional
            Stop time in seconds. Defaults to the end of the recording.
        channel_ids : list-like, optional
            Channels to extract. Defaults to masked channels.
        return_in_uV : bool, default=True
            Whether to return traces in microvolts.

        Returns
        -------
        traces : np.ndarray
            Array of shape (num_samples, num_channels).
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if self.time_vector is None:
            raise ValueError("Time vector not initialized.")

        if tstart is None:
            tstart = float(self.time_vector[0])
        if tstop is None:
            tstop = float(self.time_vector[-1])
        if channel_ids is None:
            channel_ids = self.ch_ids[self.mask]

        start_frame = int(tstart * float(self.fsample))
        end_frame = int(tstop * float(self.fsample))

        self.traces = self.recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=channel_ids,
            return_in_uV=return_in_uV,
        )
        return self.traces

    def plot_traces_in_grid(self, tmin: float = 0, tmax: float = 10, n_subsample: Optional[int] = None, show: bool = True) -> None:
        """
        Plot the traces in the MEA grid

        Parameters
        ----------
        tmin : float, default=0
            Start time (s) for trace preview.
        tmax : float, default=10
            Stop time (s) for trace preview.
        n_subsample: Optional[int]
            1/n_subsample factor, represents the number of time samples skipped in plotting.
        show: bool, defauls = True
            Boolean option for showing the plot.
            
        Notes
        -----
        This method keeps the original 8x8 grid layout and electrode label mapping:
        - col = lab // 10 - 1
        - row = lab % 10 - 1
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.ch_ids is None or self.electrode_labels is None:
            raise ValueError("Channel metadata not initialized.")

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for ax in axes.flat:
            ax.axis("off")

        n = len(self.ch_ids)
        
        lines = [None] * n
        checks = [None] * n

        # def make_toggle(i: int):
        #     def _toggle(_label):
        #         mask[i] = not mask[i]

        #         ln = lines[i]
        #         if ln is not None:
        #             ln.set_color("C0" if mask[i] else "0.7")

        #         cb = checks[i]
        #         if cb is not None:
        #             try:
        #                 cb.rectangles[0].set_facecolor("white" if mask[i] else "0.9")
        #             except Exception:
        #                 pass

        #         fig.canvas.draw_idle()

        #     return _toggle

        for i, (ch, lab) in enumerate(zip(self.ch_ids, self.electrode_labels)):
            lab = int(lab)
            col = lab // 10 - 1
            row = lab % 10 - 1
            ax = axes[row, col]

            traces = self.recording.get_traces(
                start_frame=int(tmin * float(self.fsample)),
                end_frame=int(tmax * float(self.fsample)),
                channel_ids=[ch],
                return_in_uV=True,
            )
            local_time_vector = np.arange(traces.shape[0]) / float(self.fsample) + tmin

            # option for subsampling
            if n_subsample is None:
                (ln,) = ax.plot(local_time_vector, traces, lw=0.8, color="C0")
                lines[i] = ln

                ax.set_title(lab, fontsize=6)
                ax.set_xlim(tmin, tmax)
                ax.set_ylim(-50, 50)
                ax.axis("off")
            else:
                # check if n_subsample is >1
                if n_subsample > 1:
                    local_time_vector_sub = local_time_vector[::n_subsample]
                    traces_sub = traces[::n_subsample]

                    (ln,) = ax.plot(local_time_vector_sub, traces_sub, lw=0.8, color="C0")
                    lines[i] = ln

                    ax.set_title(lab, fontsize=6)
                    ax.set_xlim(tmin, tmax)
                    ax.set_ylim(-50, 50)
                    ax.axis("off")
                else:
                    raise ValueError("n_subsample must be greater than 1.")

        if show:
            plt.show()

        

    # ----------------------------- Spike detection -----------------------------
    @tracked_operation("detect_spikes")
    def detect_spikes(
        self,
        method: str = "by_channel",
        peak_sign: str = "neg",
        detect_threshold: float = 5,
        exclude_sweep_ms: float = 0.2,
    ) -> int:
        """
        Detect spikes (peaks) in the recording.

        Parameters
        ----------
        method : str, default='by_channel'
            Peak detection method passed to `detect_peaks`.
        peak_sign : str, default='neg'
            'neg' or 'pos' depending on spike polarity.
        detect_threshold : float, default=5
            Detection threshold (units as expected by SpikeInterface detector).
        exclude_sweep_ms : float, default=0.2
            Refractory/exclusion window around detected events, in ms.

        Returns
        -------
        int
            Always returns 1 (kept for backward compatibility).
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")

        self.peaks = detect_peaks(
            recording=self.recording,
            method=method,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
        )
        return 1

    # ----------------------------- Visualization -----------------------------

    def plot_raster(self, ax) -> int:
        """
        Plot a raster of detected spikes into an existing axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes to draw into.

        Returns
        -------
        int
            Always returns 1 (kept for backward compatibility).
        """
        if self.peaks is None:
            raise ValueError("Spikes not detected.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")

        peaks_sc = np.column_stack((self.peaks["sample_index"], self.peaks["channel_index"]))
        ax.scatter(peaks_sc[:, 0] / float(self.fsample), peaks_sc[:, 1], s=1)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Channel Index")
        ax.set_title("Spike Raster Plot")
        return 1

    @tracked_operation("choose_mask")
    def choose_mask(self, tmin: float = 0, tmax: float = 10, show: bool = True) -> None:
        """
        Open a GUI to select channels to include (updates `self.mask`).

        Parameters
        ----------
        tmin : float, default=0
            Start time (s) for trace preview.
        tmax : float, default=10
            Stop time (s) for trace preview.
        show: bool, defauls = True
            Boolean option for showing the plot.
            
        Notes
        -----
        This method keeps the original 8x8 grid layout and electrode label mapping:
        - col = lab // 10 - 1
        - row = lab % 10 - 1
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.ch_ids is None or self.electrode_labels is None:
            raise ValueError("Channel metadata not initialized.")

        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for ax in axes.flat:
            ax.axis("off")

        n = len(self.ch_ids)
        mask = np.ones(n, dtype=bool)
        lines = [None] * n
        checks = [None] * n

        def make_toggle(i: int):
            def _toggle(_label):
                mask[i] = not mask[i]

                ln = lines[i]
                if ln is not None:
                    ln.set_color("C0" if mask[i] else "0.7")

                cb = checks[i]
                if cb is not None:
                    try:
                        cb.rectangles[0].set_facecolor("white" if mask[i] else "0.9")
                    except Exception:
                        pass

                fig.canvas.draw_idle()

            return _toggle

        for i, (ch, lab) in enumerate(zip(self.ch_ids, self.electrode_labels)):
            lab = int(lab)
            col = lab // 10 - 1
            row = lab % 10 - 1
            ax = axes[row, col]

            traces = self.recording.get_traces(
                start_frame=int(tmin * float(self.fsample)),
                end_frame=int(tmax * float(self.fsample)),
                channel_ids=[ch],
                return_in_uV=True,
            )
            local_time_vector = np.arange(traces.shape[0]) / float(self.fsample) + tmin

            (ln,) = ax.plot(local_time_vector, traces, lw=0.8, color="C0")
            lines[i] = ln

            ax.set_title(lab, fontsize=6)
            ax.set_xlim(tmin, tmax)
            ax.set_ylim(-50, 50)
            ax.axis("off")

            bbox = ax.get_position()
            w = bbox.width * 0.18
            h = bbox.height * 0.18
            x0 = bbox.x0 + bbox.width * 0.02
            y0 = bbox.y1 - h - bbox.height * 0.02
            cax = fig.add_axes([x0, y0, w, h])
            cax.set_xticks([])
            cax.set_yticks([])
            for spine in cax.spines.values():
                spine.set_visible(False)

            cb = CheckButtons(cax, labels=[""], actives=[True])
            for txt in getattr(cb, "labels", []):
                txt.set_visible(False)

            line_groups = getattr(cb, "lines", None) or getattr(cb, "lines_", None)
            if line_groups is not None:
                for pair in line_groups:
                    try:
                        pair[0].set_linewidth(1.0)
                        pair[1].set_linewidth(1.0)
                    except Exception:
                        pass

            cb.on_clicked(make_toggle(i))
            checks[i] = cb
        if show:
            plt.show()
        self.mask = mask

    # ----------------------------- Temporal masking -----------------------------
    @tracked_operation("blank_period")
    def blank_period(self, tstart: float, tstop: float) -> None:
        """
        Exclude a time interval from spike inclusion (updates `self.temporal_mask`).

        Parameters
        ----------
        tstart : float
            Start time in seconds.
        tstop : float
            Stop time in seconds.

        Raises
        ------
        ValueError
            If the time vector is not initialized or if tstart >= tstop.
        """
        if self.time_vector is None or self.temporal_mask is None:
            raise ValueError("Time vector not initialized.")
        if tstart >= tstop:
            raise ValueError("tstart must be less than tstop.")

        mask = (self.time_vector < tstart) | (self.time_vector > tstop)
        self.temporal_mask &= mask
        self.excluded_intervals.append((float(tstart), float(tstop)))

    # ----------------------------- Digital signal utilities -----------------------------

    def convert_digital(self) -> np.ndarray:
        """
        Convert raw digital recording to a small integer state representation.

        Returns
        -------
        np.ndarray
            Converted digital signal as int32.

        Notes
        -----
        Preserves the original transformation:
        - anchor by first sample
        - log2(abs(x - a + 1))
        - values > 2 are set to 0
        """
        a = self.digital_recording[0]
        self.digital_recording = np.log2(np.abs(self.digital_recording - a + 1))
        self.digital_recording[self.digital_recording > 2] = 0
        self.digital_recording = np.asarray(self.digital_recording, dtype=np.int32)
        return self.digital_recording

    def detect_digital_rising_edge(self) -> list[int]:
        """
        Detect rising edges (index positions) in the converted digital signal.

        Returns
        -------
        list of int
            Sample indices where digital_recording[i] > digital_recording[i-1].
        """
        edges: list[int] = []
        for i in range(1, len(self.digital_recording)):
            if self.digital_recording[i] > self.digital_recording[i - 1]:
                edges.append(i)
        return edges

    def detect_digital_falling_edge(self) -> list[int]:
        """
        Detect falling edges (index positions) in the converted digital signal.

        Returns
        -------
        list of int
            Sample indices where digital_recording[i] < digital_recording[i-1].
        """
        edges: list[int] = []
        for i in range(1, len(self.digital_recording)):
            if self.digital_recording[i] < self.digital_recording[i - 1]:
                edges.append(i)
        return edges

    @tracked_operation("get_triggers")
    def get_triggers(
        self,
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        interpretor: Optional[Callable] = None,
        dt_after_trigger: Optional[float] = None,
    ):
        """
        Build `Triggers` from the digital recording in a given time window.

        Parameters
        ----------
        tstart : float, optional
            Start time in seconds. Defaults to start of recording.
        tstop : float, optional
            Stop time in seconds. Defaults to end of recording.
        interpretor : callable, optional
            Custom function to interpret digital signal into trigger slots.
            If None, a default rising/falling pairing is used.
        dt_after_trigger : float, optional
            If provided, passed as the fourth argument to `interpretor`.

        Returns
        -------
        Triggers
            Triggers object containing interval slots.

        Raises
        ------
        ValueError
            If `interpretor` is not None and not callable.
        """
        if self.time_vector is None:
            raise ValueError("Time vector not initialized.")
        if self.digital_recording is None:
            raise ValueError("Digital recording not loaded. Set load_digital=True when constructing MCSData.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")

        if tstart is None:
            tstart = float(self.time_vector[0])
        if tstop is None:
            tstop = float(self.time_vector[-1])

        from .Triggers import Triggers  # avoid circular import

        self.triggers = Triggers(slots=[])

        # Window digital samples to match the requested time interval
        window_mask = (self.time_vector >= tstart) & (self.time_vector <= tstop)
        self.digital_recording = self.digital_recording[window_mask]
        self.digital_recording = self.convert_digital()

        if interpretor is None:
            rising_edges = self.detect_digital_rising_edge()
            falling_edges = self.detect_digital_falling_edge()
            for start, end in zip(rising_edges, falling_edges):
                self.triggers.add_interval_slot(
                    start=round(start * float(self.fsample)) / float(self.fsample),
                    end=round(end * float(self.fsample)) / float(self.fsample),
                )
        elif callable(interpretor):
            if dt_after_trigger is None:
                interpretor(self.digital_recording, self.triggers, float(self.fsample))
            else:
                interpretor(self.digital_recording, self.triggers, float(self.fsample), dt_after_trigger)
        else:
            raise ValueError(
                "interpretor must be a callable function that defines how to interpret the digital signal into triggers."
            )

        return self.triggers

    @tracked_operation("remove_artifacts_from_trigger")
    def remove_artifacts_from_trigger(
        self,
        ms_before: float = 0.1,
        ms_after: float = 0.4,
        mode: str = "zeros",
    ):
        """
        Remove stimulation artifacts around triggers using SpikeInterface.

        Parameters
        ----------
        ms_before : float, default=0.1
            Time (ms) before each trigger to remove.
        ms_after : float, default=0.4
            Time (ms) after each trigger to remove.
        mode : str, default='zeros'
            Artifact replacement mode (as supported by `pre.remove_artifacts`).

        Returns
        -------
        recording : spikeinterface.BaseRecording
            Recording after artifact removal.

        Raises
        ------
        ValueError
            If artifact removal has already been performed or triggers are missing.
        """
        if self.artifact_removal_status:
            raise ValueError("Artifact removal already performed.")
        if self.triggers is None:
            raise ValueError("Triggers not defined. Run get_triggers() first.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.recording is None:
            raise ValueError("Recording not loaded.")

        list_triggers = [int(slot.start * float(self.fsample)) for slot in self.triggers.slots]

        self.recording = pre.remove_artifacts(
            self.recording,
            list_triggers=list_triggers,
            ms_before=ms_before,
            ms_after=ms_after,
            mode=mode,
        )
        self.artifact_removal_status = True
        return self.recording

    # ----------------------------- Raster export -----------------------------
    @tracked_operation("get_raster", include_result_artifacts=_raster_artifacts)
    def get_raster(self, tstart: Optional[float] = None, tstop: Optional[float] = None):
        """
        Build a Raster object for spikes on the currently selected channels.

        Parameters
        ----------
        tstart : float, optional
            Start time (s). Defaults to start of recording.
        tstop : float, optional
            Stop time (s). Defaults to end of recording.

        Returns
        -------
        Raster
            Raster containing per-channel spike time arrays.

        Notes
        -----
        Preserves the original channel-index logic:
        - `peaks['channel_index']` is assumed to align with a 0..N-1 indexing
          compatible with iterating `enumerate(self.ch_ids[self.mask])`.
        """
        if self.peaks is None:
            raise ValueError("Spikes not detected.")
        if self.time_vector is None or self.temporal_mask is None:
            raise ValueError("Time vector / temporal mask not initialized.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.mask is None:
            raise ValueError("Channel mask not initialized.")

        if tstart is None:
            tstart = float(self.time_vector[0])
        if tstop is None:
            tstop = float(self.time_vector[-1])

        from .raster import Raster  # avoid circular import

        r = Raster.empty(channels=self.ch_ids[self.mask])

        peaks_sc = np.column_stack((self.peaks["sample_index"], self.peaks["channel_index"]))
        for k, ch in enumerate(self.ch_ids[self.mask]):
            this_channel_times = peaks_sc[peaks_sc[:, 1] == k][:, 0] / float(self.fsample)

            idx = (this_channel_times * float(self.fsample)).astype(int)
            idx = np.clip(idx, 0, len(self.temporal_mask) - 1)

            keep_spikes = (
                (this_channel_times >= tstart)
                & (this_channel_times <= tstop)
                & self.temporal_mask[idx]
            )
            r.insert_timestamparray(ch, this_channel_times[keep_spikes], assume_sorted=True)
        # Attach provenance snapshot directly to the raster 
        if getattr(self, "history", None) is not None:
                try:
                    r.provenance = self.history.to_dict()
                except Exception:
                    pass
        return r