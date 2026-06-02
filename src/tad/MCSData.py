"""
MCSData.py

Experimental data loader/handler for Multi Channel Systems (.h5) recordings.

This version is a *minimal-API-change* refactor of your previous MCSData to fit the
new architecture:

- MCSData now inherits from EData (and thus AData).
- All existing public methods + prototypes are preserved.
- Methods already provided by AData/EData are kept as thin wrappers for backward
  compatibility (so user code/tests keep working).
- tracked_operation / ProcessingHistory live in processing_history.py (imported).

No intentional changes to user-facing behavior or default parameters.
"""

from __future__ import annotations


import os
import sys

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as pre
import spikeinterface as si

from matplotlib.widgets import CheckButtons
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from .edata import EData
from .processing_history import DatasetInfo, ProcessingHistory, tracked_operation


# ---------------------------------------------------------------------
# Helper functions (kept public; re-exported from tad.__init__)
# ---------------------------------------------------------------------
def on_delta_t(digital_recording, triggers, fsample: float, delta_t: float) -> None:
    """
    Create stimulation ON triggers at each rising edge + delta_t.

    Parameters
    ----------
    digital_recording : np.ndarray
        Digital trace (samples).
    triggers : Triggers
        Trigger container (in-place updates).
    fsample : float
        Sampling frequency (Hz).
    delta_t : float
        Time after each rising edge at which to create a stimulation ON slot (s).
    """
    rising_edges = np.where(np.diff(digital_recording.astype(int)) == 1)[0] + 1
    for idx in rising_edges:
        start = idx / fsample
        end = start + float(delta_t)
        triggers.add_trigger(start=start, end=end, ID="stim_ON", blank=False)


def on_off_interpretor(digital_recording, triggers, fsample: float) -> None:
    """
    Interpret digital trace as ON/OFF blocks and populate triggers.

    Parameters
    ----------
    digital_recording : np.ndarray
        Digital trace (samples).
    triggers : Triggers
        Trigger container (in-place updates).
    fsample : float
        Sampling frequency (Hz).
    """
    rising_edges = np.where(np.diff(digital_recording.astype(int)) == 1)[0] + 1
    falling_edges = np.where(np.diff(digital_recording.astype(int)) == -1)[0] + 1

    if len(rising_edges) == 0 or len(falling_edges) == 0:
        return

    # Align edges: each rising should have a following falling
    if falling_edges[0] < rising_edges[0]:
        falling_edges = falling_edges[1:]
    n = min(len(rising_edges), len(falling_edges))
    rising_edges = rising_edges[:n]
    falling_edges = falling_edges[:n]

    for i in range(n):
        triggers.add_trigger(
            start=rising_edges[i] / fsample,
            end=falling_edges[i] / fsample,
            ID="stim_ON",
            blank=False,
        )


def _raster_artifacts(raster_obj: Any) -> Dict[str, Any]:
    """
    Optional artifact extractor for tracked_operation on get_raster().
    Kept small and JSON-friendly.
    """
    artifacts: Dict[str, Any] = {}
    try:
        if raster_obj is None:
            return artifacts
        for key in ("tstart", "tstop", "n_spikes", "n_channels"):
            if hasattr(raster_obj, key):
                v = getattr(raster_obj, key)
                if isinstance(v, (int, float, str, bool)) or v is None:
                    artifacts[key] = v
    except Exception:
        return {"note": "artifact extraction failed"}
    return artifacts


# ---------------------------------------------------------------------
# MCSData class
# ---------------------------------------------------------------------
class MCSData(EData):
    """
    MCS recording handler.

    Parameters
    ----------
    fname : str
        Path to the `.h5` file.
    fsample : float, optional
        Sampling frequency (Hz). If None, read from the recording.
    load_recording : bool, default=True
        Load the analog recording stream into SpikeInterface.
    load_digital : bool, default=False
        Load a digital channel from the HDF5 file.
    generate_probe : bool, default=False
        Generate a probe object from probe_data (if you use this in your pipeline).
    probe_data : dict, optional
        Probe configuration.
    """

    def __init__(
        self,
        fname: Optional[str] = None,
        fsample: Optional[float] = None,
        load_recording: bool = True,
        load_digital: bool = False,
        generate_probe: bool = False,
        probe_data: Optional[dict] = None,
        recording: Optional[si.BaseRecording] = None,
    ) -> None:
        if recording is None:
            if fname is None:
                raise ValueError("Either fname or recording must be provided.")
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} does not exist.")
        else:
            if fname is None:
                fname = "<recording>"
            if load_digital:
                raise ValueError(
                    "Cannot load digital data when constructing from an existing recording object."
                )

        # Keep these as before (public/state)
        self.fname: str = str(fname)
        self.fsample: Optional[float] = fsample
        self.load_digital: bool = load_digital

        # Core data
        self.recording = None
        self.traces: Optional[np.ndarray] = None
        self.peaks = None

        # Channel/probe metadata (historical public attributes)
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
        self.excluded_intervals: List[Tuple[float, float]] = []

        # Digital/triggers
        self.digital_recording = None
        self.triggers = None

        # State
        self.artifact_removal_status: bool = False

        # Load recording first (to preserve original behaviour/attributes)
        if recording is not None:
            self.recording = recording
        elif load_recording:
            self._load_recording()

        if self.recording is not None:
            if self.fsample is None:
                self.fsample = self.recording.get_sampling_frequency()
            self.time_vector = np.arange(self.recording.get_total_samples()) / float(self.fsample)
            self.ch_ids = self.recording.channel_ids
            self.electrode_labels = self.recording.get_property("electrode_labels")
            self.mask = np.ones(self.recording.get_num_channels(), dtype=bool)
            self.temporal_mask = np.ones_like(self.time_vector, dtype=bool)

        if generate_probe:
            self._generate_probe(probe_data)

        # Now initialize EData/AData with the loaded channel axis (minimal change)
        if self.recording is not None:
            rec = self.recording  # preserve handle; EData sets self.recording = None

            super().__init__(
                fsample=float(self.fsample) if self.fsample is not None else 0.0,
                channel_ids=list(self.ch_ids),
                electrode_labels=None if self.electrode_labels is None else list(self.electrode_labels),
                mask=None if self.mask is None else list(self.mask.astype(bool)),
                fname=fname,
                recording_system="MCS",
            )
            self.recording = rec  # restore

            # Keep historical attributes aligned to base storage
            # (AData defines channel_ids/electrode_labels/mask; we keep old names too)
            self.ch_ids = self.channel_ids
            self.electrode_labels = self.electrode_labels  # already set by base
            self.mask = self.mask  # already set by base

        else:
            super().__init__(
                fsample=float(self.fsample) if self.fsample is not None else 0.0,
                channel_ids=[],
                electrode_labels=None,
                mask=None,
                fname=self.fname,
                recording_system="MCS",
            )

        # History:
        # - AData already has `self.history: list[dict]` (lightweight)
        # - Here we keep a rich ProcessingHistory for export_history_json
        if self.recording is not None:
            ds = DatasetInfo.from_path(
                fname=self.fname,
                sampling_frequency=float(self.fsample) if self.fsample is not None else None,
                stream_id=1,
                channel_ids=self.channel_ids.tolist() if getattr(self, "channel_ids", None) is not None else None,
                electrode_labels=self.electrode_labels.tolist() if getattr(self, "electrode_labels", None) is not None else None,
            )
            self.processing_history = ProcessingHistory(dataset=ds)
            self.processing_history.set_state_getter(self._history_snapshot)
        else:
            self.processing_history = None

    # -----------------------------------------------------------------
    # History handling
    # -----------------------------------------------------------------
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
            kept = list(np.asarray(self.ch_ids)[np.asarray(self.mask, dtype=bool)])
            snap["mask_n_kept"] = int(len(kept))
            snap["mask_kept_channel_ids"] = kept

        # Temporal exclusions: store intervals list rather than full per-sample boolean
        snap["excluded_intervals"] = list(self.excluded_intervals) if hasattr(self, "excluded_intervals") else []
        snap["excluded_intervals_n"] = int(len(snap["excluded_intervals"]))

        # Trigger summary (store the slots, not the digital signal)
        if self.triggers is not None and hasattr(self.triggers, "slots"):
            snap["triggers"] = [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "id": getattr(s, "ID", None),
                    "blank": getattr(s, "blank", None),
                }
                for s in self.triggers.slots
            ]
            snap["triggers_n"] = int(len(snap["triggers"]))

        return snap

    def export_history_json(self, path: str, indent: int = 2) -> None:
        """
        Export processing history to a JSON file.

        Parameters
        ----------
        path : str
            Output path.
        indent : int, default=2
            JSON indentation.
        """
        if self.processing_history is None:
            raise ValueError("No processing history available to export.")
        self.processing_history.save_json(path, indent=indent)

    # -----------------------------------------------------------------
    # IO / setup
    # -----------------------------------------------------------------
    def _load_recording(self) -> None:
        """
        Load the MCS recording via SpikeInterface and optionally load digital signal.

        Notes
        -----
        - Uses stream_id=1.
        - Renames channels based on electrode_labels to "Ch{label}".
        """

        if self.load_digital:
            try:
                self.recording = se.read_mcsh5(self.fname, stream_id=1)
            except Exception as exc:
                print(f"Error loading recording: {exc}. load_digital is {self.load_digital}, set it to {not(self.load_digital)}")
                sys.exit(1)
            try:
                with h5py.File(self.fname, "r") as f:
                    stream = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
                    self.digital_recording = stream[0]
            except Exception as exc:
                print(f"Error loading digital recording: {exc}")
        else:
            try:
                self.recording = se.read_mcsh5(self.fname, stream_id=0)
            except Exception as exc:
                print(f"Error loading recording: {exc}. load_digital is {self.load_digital}, set it to {not(self.load_digital)}")
                sys.exit(1)

        electrode_labels = self.recording.get_property("electrode_labels")
        self.recording = self.recording.rename_channels([f"Ch{lab}" for lab in electrode_labels])

        if self.fsample is None:
            self.fsample = self.recording.get_sampling_frequency()

    def _serialize_triggers(self) -> Optional[List[Dict[str, Any]]]:
        if self.triggers is None:
            return None
        out: List[Dict[str, Any]] = []
        for slot in getattr(self.triggers, "slots", []):
            out.append(
                {
                    "start": float(getattr(slot, "start", 0.0)),
                    "end": float(getattr(slot, "end", 0.0)),
                    "ID": getattr(slot, "ID", None),
                    "blank": getattr(slot, "blank", None),
                }
            )
        return out

    def _generate_probe(self, probe_data: Optional[dict]) -> None:
        """
        Generate a probe object from probe_data.

        Parameters
        ----------
        probe_data : dict, optional
            Probe configuration dictionary.
        """
        # Keep your previous import style to avoid circular imports
        from .mea_probe import MEAProbe

        if probe_data is None:
            raise ValueError("probe_data must be provided when generate_probe=True.")

        self.probe_positions = probe_data.get("positions", None)
        self.probe_contact_shape = probe_data.get("contact_shape", "circle")
        self.probe_shape_params = probe_data.get("shape_params", {"radius": 5})
        self.probe_ndims = probe_data.get("ndims", 2)

        self.probe = MEAProbe(
            positions=self.probe_positions,
            contact_shape=self.probe_contact_shape,
            shape_params=self.probe_shape_params,
            ndims=self.probe_ndims,
        )

    @classmethod
    def from_recording(
        cls,
        recording: si.BaseRecording,
        fname: Optional[str] = None,
        fsample: Optional[float] = None,
        generate_probe: bool = False,
        probe_data: Optional[dict] = None,
        time_vector: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        excluded_intervals: Optional[List[Tuple[float, float]]] = None,
        triggers: Optional[Any] = None,
    ) -> "MCSData":
        """
        Create an MCSData object from an existing SpikeInterface recording.

        Parameters
        ----------
        recording : spikeinterface.BaseRecording
            Existing recording object (for example, a FrameSliceRecording).
        fname : str, optional
            Optional source filename for provenance.
        fsample : float, optional
            Sampling frequency in Hz. If None, read from the recording.
        generate_probe : bool, default=False
            Generate a probe object from probe_data.
        probe_data : dict, optional
            Probe configuration for probe generation.
        time_vector : np.ndarray, optional
            Time vector for the recording. If None, a 0-based vector is created.
        mask : np.ndarray, optional
            Channel mask. If None, all channels are kept.
        excluded_intervals : list of tuple, optional
            Excluded intervals for the new object.
        triggers : Triggers, optional
            Trigger container for the new object.

        Returns
        -------
        MCSData
            New MCSData instance wrapping the provided recording.
        """
        if fsample is None:
            fsample = recording.get_sampling_frequency()

        obj = cls(
            fname=fname,
            fsample=fsample,
            load_recording=False,
            load_digital=False,
            generate_probe=generate_probe,
            probe_data=probe_data,
            recording=recording,
        )

        obj.time_vector = (
            np.asarray(time_vector, dtype=float)
            if time_vector is not None
            else np.arange(recording.get_total_samples()) / float(obj.fsample)
        )
        obj.temporal_mask = np.ones_like(obj.time_vector, dtype=bool)

        if mask is not None:
            obj.mask = np.asarray(mask, dtype=bool)

        if excluded_intervals is not None:
            obj.excluded_intervals = list(excluded_intervals)

        if triggers is not None:
            obj.triggers = triggers

        return obj
    
    def save_recording_h5(self, path, fname, 
                          h5_group: str = "/Data/Recording_0/AnalogStream/Stream_", 
                          stream_id: int = 0
                          ):
        """
        Saves a MCSdata recording object into an h5 file at 'path' and 'fname' with the given h5_struc in the given 'stream_id'

        path: str
            Path where the recording will be saved
        fname: str
            File name 
        h5_group: str
            Structure of the h5 groups.
        stream_id: int
            Stream id value
        """

        filename = path + "/" + fname

        with h5py.File(filename, "w") as f:
            f.require_group(h5_group+str(stream_id))
            

    # -----------------------------------------------------------------
    # Backward-compatible wrappers for AData methods
    # -----------------------------------------------------------------
    def set_mask(self, mask) -> None:
        return super().set_mask(mask)

    def save_mask_and_labels(self, fname: str, csv_format: bool = False) -> int:
        return super().save_mask_and_labels(fname=fname, csv_format=csv_format)

    def load_mask_and_labels(self, fname: str) -> int:
        return super().load_mask_and_labels(fname=fname)

    # -----------------------------------------------------------------
    # Processing methods (kept with same signatures/defaults)
    # -----------------------------------------------------------------
    @tracked_operation("apply_filter")
    def apply_filter(self, bandpass: Tuple[float, float] = (300, 3000), btype: str = "bandpass") -> int:
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

    def get_traces(
        self,
        tstart: float = 0,
        tstop: float = 10,
        channel_ids: Optional[List[int]] = None,
        return_in_uV: bool = True,
    ) -> np.ndarray:
        """
        Retrieve traces from the recording.

        Parameters
        ----------
        tstart : float, default=0
            Start time in seconds.
        tstop : float, default=10
            Stop time in seconds.
        channel_ids : list, optional
            Subset of channel ids to extract.
        return_in_uV : bool, default=True
            Convert to microvolts.

        Returns
        -------
        np.ndarray
            Traces array of shape (n_samples, n_channels).
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

        traces = self.recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=channel_ids,
            return_in_uV=return_in_uV,
        )
        return traces

    def plot_traces_in_grid(
        self,
        tmin: float = 0,
        tmax: float = 10,
        n_subsample: Optional[int] = None,
        show: bool = True,
    ) -> int:
        """
        Plot channel traces in a grid.

        Parameters
        ----------
        tmin : float, default=0
            Start time (s).
        tmax : float, default=10
            Stop time (s).
        n_subsample: Optional[int]
            1/n_subsample factor, represents the number of time samples skipped in plotting.
        show : bool, default=True
            Show plot.

        Returns
        -------
        int
            Always returns 1 (backward compatible).
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.ch_ids is None or self.electrode_labels is None:
            raise ValueError("Channel metadata not initialized.")

        _, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for ax in axes.flat:
            ax.axis("off")

        n = len(self.ch_ids)
        
        lines = [None] * n
        
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
        return 1

    def plot_PSTH_in_grid(
        self,
        psth,
        kind: str = "rate",
        show: bool = True,
    ) -> int:
        """
        Plot PSTH for each channel in an 8x8 grid layout.

        Parameters
        ----------
        psth : tad.metrics.psth.PSTHResult
            Per-channel PSTH result from `tad.metrics.psth.compute_psth`.
        kind : {'rate', 'counts', 'total_counts'}, default='rate'
            Which PSTH quantity to plot.
        show : bool, default=True
            Show plot.

        Returns
        -------
        int
            Always returns 1 (backward compatible).

        Notes
        -----
        The layout follows the same 8x8 electrode grid used by
        :meth:`plot_traces_in_grid`. Channels that are not present in the PSTH
        result remain empty in the grid.
        """
        if self.ch_ids is None or self.electrode_labels is None:
            raise ValueError("Channel metadata not initialized.")
        if getattr(psth, "mode", None) != "per_channel":
            raise ValueError("plot_PSTH_in_grid requires a per-channel PSTHResult.")

        if kind == "rate":
            values = psth.rate_hz
        elif kind == "counts":
            values = psth.counts
        elif kind == "total_counts":
            values = psth.total_counts
        else:
            raise ValueError("kind must be 'rate', 'counts', or 'total_counts'.")

        def _normalize_channel_id(channel_id):
            if isinstance(channel_id, str):
                if channel_id.startswith("Ch"):
                    try:
                        return int(channel_id[2:])
                    except ValueError:
                        pass
                digits = "".join([c for c in channel_id if c.isdigit()])
                if digits:
                    return int(digits)
            return channel_id

        channel_to_index = {}
        for idx, ch in enumerate(psth.channels):
            channel_to_index[ch] = idx
            normalized = _normalize_channel_id(ch)
            if normalized != ch:
                channel_to_index[normalized] = idx

        _, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for ax in axes.flat:
            ax.axis("off")

        for ch, lab in zip(self.ch_ids, self.electrode_labels):
            lab = int(lab)
            col = lab // 10 - 1
            row = lab % 10 - 1
            ax = axes[row, col]

            psth_idx = None
            if ch in channel_to_index:
                psth_idx = channel_to_index[ch]
            else:
                normalized = _normalize_channel_id(ch)
                psth_idx = channel_to_index.get(normalized, None)

            if psth_idx is not None:
                y = values[psth_idx]
                ax.plot(psth.t, y, lw=0.8, color="C0")
                ax.axvline(0.0, linestyle="--", alpha=0.6)
                ax.set_xlim(float(psth.t[0]), float(psth.t[-1]))
                if y.size:
                    y_min = 0
                    y_max = float(np.max(y))+0.2*float(np.max(y))
                    print(y_min, y_max)
                    ax.set_ylim(y_min, y_max)
                ax.axis("off")

            ax.set_title(lab, fontsize=6)

        if show:
            plt.show()
        return 1

    @tracked_operation("detect_spikes")
    def detect_spikes(
        self,
        recording: Optional[si.BaseRecording] = None,
        method: str = "by_channel",
        peak_sign: str = "neg",
        detect_threshold: float = 5,
        exclude_sweep_ms: float = 1.0,
        noise_levels=None,
        detect_noise_levels: Optional[bool] = None,
        job_kwargs = None
    ) -> int:
        """
        Detect spikes (peaks) in the recording.

        Parameters
        ----------
        recording : spikeinterface.BaseRecording, optional
            Recording object to use for detect spikes. If None, uses `self.recording`. This allows using an artifact-removed recording if desired.
        method : str, default='by_channel'
            Peak detection method passed to `detect_peaks`.
        peak_sign : str, default='neg'
            'neg' or 'pos' depending on spike polarity.
        detect_threshold : float, default=5
            Detection threshold.
        exclude_sweep_ms : float, default=0.2
            Exclusion window in ms.
        noise_levels: array or None
            array of noise levels array precomputed in uV.
        detect_noise_levels: bool,
            if no array of noise level is provided, they can be computed, if set to True.
        job_kwargs
            if not None, explicitly uses parallel computing, e.g. {'n_jobs': n}.

        Returns
        -------
        int
            Always returns 1 (kept for backward compatibility).
        """

        if recording is None:
            recording = self.recording
        else:
            recording = recording  # use provided recording (e.g., artifact-removed)

        method_kwargs = {
            'peak_sign': peak_sign,
            'detect_threshold': detect_threshold,
            'exclude_sweep_ms': exclude_sweep_ms,
        }

        if noise_levels is not None:
            method_kwargs['noise_levels'] = noise_levels
            if not isinstance(noise_levels, np.ndarray):
                raise ValueError("noise_levels must be a numpy array.")
            self.peaks = detect_peaks(
                recording=recording,
                method=method,
                method_kwargs = method_kwargs,
                job_kwargs = job_kwargs
            )
        else:
            if detect_noise_levels is None:
                self.peaks = detect_peaks(
                    recording=recording,
                    method=method,
                    method_kwargs = method_kwargs,
                    job_kwargs = job_kwargs
                )
            else:
                noise_levels = si.get_noise_levels(recording, return_in_uV =False)
                method_kwargs['noise_levels'] = noise_levels
                self.peaks = detect_peaks(
                    recording=recording,
                    method=method,
                    method_kwargs = method_kwargs,
                    job_kwargs = job_kwargs
                )            
        return 1

    def plot_raster(self, ax) -> int:
        """
        Plot a raster of detected spikes into an existing axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.

        Returns
        -------
        int
            Always returns 1 (backward compatible).
        """
        if self.peaks is None:
            raise ValueError("Spikes not detected.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")

        ch_ids = np.asarray(self.ch_ids)
        y = ch_ids[self.peaks["channel_index"]]
        y_vals = np.array([int(ch.replace("Ch", "")) for ch in y])

        ax.scatter(self.peaks["sample_index"] / float(self.fsample), y_vals, s=1)
        ax.set_yticks(y_vals)
        ax.set_yticklabels(y)

        # peaks_sc = np.column_stack((self.peaks["sample_index"], self.peaks["channel_index"]))
        # ax.scatter(peaks_sc[:, 0] / float(self.fsample), peaks_sc[:, 1], s=1)
        # ax.set_xlabel("Sample Index")
        # ax.set_ylabel("Channel Index")
        # ax.set_title("Spike Raster Plot")
        return 1
    
    def get_probe(self):
        """
        Get the probe object.

        Returns
        -------
        MEAProbe
            The probe object if generated, else None.
        """

        print(self.recording.get_probe())

        return self.recording.get_probe() if self.recording is not None else None

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
        show : bool, default=True
            Show plot.
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

        def make_toggle(i: int, checks):
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

            cb.on_clicked(make_toggle(i, checks))
            checks[i] = cb
        if show:
            plt.show()
        self.mask = mask

    @tracked_operation("blank_period")
    def blank_period(self, tstart: float, tstop: float) -> None:
        """
        Exclude (blank) a time window from analysis.

        Parameters
        ----------
        tstart : float
            Start time (s).
        tstop : float
            Stop time (s).
        """
        if self.time_vector is None or self.temporal_mask is None:
            raise ValueError("Time vector not initialized.")

        a = float(min(tstart, tstop))
        b = float(max(tstart, tstop))
        self.excluded_intervals.append((a, b))

        self.temporal_mask &= ~((self.time_vector >= a) & (self.time_vector <= b))

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
        
        # find the most common peak value in the entire digital recording and set all other non-zero values to zero
        data = np.round(self.digital_recording).astype(int) # first round, then truncate
        most_common_value = np.bincount(data)[1:].argmax() + 1 # find the most common nonzero value (+1 to find the int not the position)
        self.digital_recording[data != most_common_value] = 0 # zero all others
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
        method: str = "artifact",
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        interpretor: Optional[Callable] = None,
        dt_after_trigger: Optional[float] = None,
        artifact_threshold: Optional[float] = None,
        mean_noise_level: Optional[float] = None,
        refractory_trigger_period: Optional[float] = None,
        stim_on_time: Optional[float] = None,
        moving_avg_window: Optional[float] = 0.005

    ):
        """
        Build `Triggers` from the digital recording in a given time window.

        Parameters
        ----------
        method : str, default='artifact'
            Method to obtain triggers.
            - 'artifact': detect artifact blocks from the analog recording using a threshold.
            - 'first_passage': detect the first threshold crossing of each event and generate a fixed-duration slot.
            - 'digital_trigger': use the digital recording directly, optionally with a custom `interpretor`.
        tstart : float, optional
            Start time in seconds. Defaults to the start of the recording.
        tstop : float, optional
            Stop time in seconds. Defaults to the end of the recording.
        interpretor : callable, optional
            Custom function to interpret the (windowed) digital signal into trigger slots.
            It must accept `(digital_recording, triggers, fsample)` or
            `(digital_recording, triggers, fsample, dt_after_trigger)`.
            Trigger times created by the custom interpreter are assumed to be
            relative to `tstart`; `get_triggers` will shift them to absolute time.
        dt_after_trigger : float, optional
            Extra argument forwarded to `interpretor` when provided.
        artifact_threshold : float, optional
            Threshold multiplier used by `artifact` and `first_passage` methods.
        mean_noise_level : float, optional
            Mean noise level used by `artifact` and `first_passage` methods.
        refractory_trigger_period : float, optional
            Required for `first_passage`. Minimum time in seconds between consecutive triggers.
        stim_on_time : float, optional
            Required for `first_passage`. Duration in seconds for the generated trigger slot.
        moving_avg_window : float, optional
            Time in ms of the moving average window when artifact or first passage methods are chosen.

        Returns
        -------
        Triggers
            Trigger container with the detected interval slots.

        Raises
        ------
        TypeError
            If `method` is not a string.
        ValueError
            If required arguments for the selected method are missing, invalid,
            or if the requested time window is outside the recording range.
        """
        if self.time_vector is None:
            raise ValueError("Time vector not initialized.")
        if self.digital_recording is None:
            raise ValueError(
                "Digital recording not loaded. Set load_digital=True when constructing MCSData."
            )
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")

        if tstart is None:
            tstart = float(self.time_vector[0])
        if tstop is None:
            tstop = float(self.time_vector[-1])

        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method = method.strip().lower()

        if tstart < float(self.time_vector[0]) or tstop > float(self.time_vector[-1]):
            raise ValueError(
                "tstart and tstop must fall within the available recording interval."
            )
        if tstart >= tstop:
            raise ValueError("tstart must be strictly less than tstop.")

        if method not in {"artifact", "first_passage", "digital_trigger"}:
            raise ValueError(
                "Unknown method. Supported values are 'artifact', 'first_passage', and 'digital_trigger'."
            )

        if method in {"artifact", "first_passage"}:
            if artifact_threshold is None:
                raise ValueError(
                    "artifact_threshold is required for method='artifact' or method='first_passage'."
                )
            if mean_noise_level is None:
                raise ValueError(
                    "mean_noise_level is required for method='artifact' or method='first_passage'."
                )
            if artifact_threshold <= 0 or mean_noise_level <= 0:
                raise ValueError(
                    "artifact_threshold and mean_noise_level must be positive values."
                )

        if method == "first_passage":
            if refractory_trigger_period is None:
                raise ValueError(
                    "refractory_trigger_period is required for method='first_passage'."
                )
            if stim_on_time is None:
                raise ValueError("stim_on_time is required for method='first_passage'.")
            if refractory_trigger_period <= 0:
                raise ValueError("refractory_trigger_period must be positive.")
            if stim_on_time <= 0:
                raise ValueError("stim_on_time must be positive.")

        if method == "digital_trigger" and interpretor is not None and not callable(interpretor):
            raise ValueError(
                "interpretor must be a callable function that defines how to interpret the digital signal into triggers."
            )

        from .Triggers import Triggers  # avoid circular import

        self.triggers = Triggers(slots=[])

        start_frame = int(tstart * float(self.fsample))
        end_frame = int(tstop * float(self.fsample))
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("Invalid start/stop sample frame range for trigger extraction.")

        if method in {"artifact", "first_passage"}:
            traces = self.recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                return_in_uV=False,
            )
            valid_channels = np.where(self.mask)[0]
            if len(valid_channels) == 0:
                raise ValueError("No valid channels found in mask for artifact detection.")
            channel_index = int(valid_channels[0])

            trace = np.abs(traces[:, channel_index])
            n = len(trace)
            if n == 0:
                raise ValueError("No samples available in the selected recording window.")
            rect_trace = trace * ((-1) ** np.arange(n))

            window = max(1, int(float(self.fsample) * moving_avg_window))
            if n < window:
                raise ValueError(
                    "Selected recording window is too short for artifact detection."
                )

            moving_avg = np.convolve(np.abs(rect_trace), np.ones(window), "valid") / window
            stim_on = np.flatnonzero(moving_avg > artifact_threshold * mean_noise_level)

            stim_status = np.zeros_like(moving_avg, dtype=np.int64)
            stim_status[stim_on] = 1


            rising_edges = np.flatnonzero(np.diff(stim_status) == 1) + window
            if method == "artifact":
                print(method)
                falling_edges = np.flatnonzero(np.diff(stim_status) == -1) + window
                for start_idx, end_idx in zip(rising_edges, falling_edges):
                    self.triggers.add_interval_slot(
                        start=float((start_frame + start_idx) / float(self.fsample)),
                        end=float((start_frame + end_idx) / float(self.fsample)),
                    )
            else:
                last_trigger_time = -np.inf
                for start_idx in rising_edges:
                    start_time = float((start_frame + start_idx) / float(self.fsample))
                    if start_time - last_trigger_time < refractory_trigger_period:
                        continue
                    self.triggers.add_interval_slot(
                        start=start_time,
                        end=float(start_time + stim_on_time),
                    )
                    last_trigger_time = start_time

        else:
            window_mask = (self.time_vector >= tstart) & (self.time_vector <= tstop)
            digital_window = np.asarray(self.digital_recording[window_mask], dtype=np.float64)
            if digital_window.size == 0:
                raise ValueError("No digital samples found in the requested time window.")

            backup_digital = self.digital_recording
            self.digital_recording = digital_window
            self.convert_digital()

            if interpretor is None:
                rising_edges = self.detect_digital_rising_edge()
                falling_edges = self.detect_digital_falling_edge()
                self.digital_recording = backup_digital
                for start_idx, end_idx in zip(rising_edges, falling_edges):
                    self.triggers.add_interval_slot(
                        start=float((start_frame + start_idx) / float(self.fsample)),
                        end=float((start_frame + end_idx) / float(self.fsample)),
                    )
            else:
                digital_signal = np.asarray(self.digital_recording, dtype=np.int32)
                self.digital_recording = backup_digital
                if dt_after_trigger is None:
                    interpretor(digital_signal, self.triggers, float(self.fsample))
                else:
                    interpretor(digital_signal, self.triggers, float(self.fsample), dt_after_trigger)
                offset = float(start_frame) / float(self.fsample)
                for slot in self.triggers.slots:
                    slot.start += offset
                    slot.end += offset

        self.triggers.sort_slots()
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
        recording_clean : spikeinterface.BaseRecording
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
        return 1

    @tracked_operation("get_raster", include_result_artifacts=_raster_artifacts)
    def get_raster(self, 
                   tstart: Optional[float] = None,
                   tstop: Optional[float] = None,
                   include_amplitudes: bool = False, 
                   include_triggers: bool = False):
        """
        Export a Raster object using detected peaks and current selection.

        Parameters
        ----------
        tstart : float, optional
            Start time in seconds. Defaults to start of recording.
        tstop : float, optional
            Stop time in seconds. Defaults to end of recording.
        include_amplitudes : bool, default=True
            If True, include spike amplitudes in the raster in uV.
        include_triggers : bool, default=True
            If True, include triggers from the recording in the raster.

        Returns
        -------
        Raster
            Raster object with amplitudes and triggers fields populated.
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

        channel_ids = np.asarray(self.ch_ids)
        kept_channel_ids = channel_ids[self.mask]
        kept_channel_indices = np.arange(len(channel_ids))[self.mask]

        r = Raster.empty(channels=kept_channel_ids)
        amplitudes: Dict[Union[int, str], np.ndarray] = {}

        for orig_idx, ch in zip(kept_channel_indices, kept_channel_ids):
            ch_peaks = self.peaks[self.peaks["channel_index"] == orig_idx]
            this_channel_times = self.peaks["sample_index"][self.peaks["channel_index"] == orig_idx] / float(self.fsample)

            idx = (this_channel_times * float(self.fsample)).astype(int)
            idx = np.clip(idx, 0, len(self.temporal_mask) - 1)

            keep_spikes = (
                (this_channel_times >= tstart)
                & (this_channel_times <= tstop)
                & self.temporal_mask[idx]
            )
            r.insert_timestamparray(ch, this_channel_times[keep_spikes], assume_sorted=True)
            if include_amplitudes:
                kept_samples = ch_peaks["sample_index"].astype(int)[keep_spikes]
                if kept_samples.size:
                    start = int(max(0, kept_samples.min()))
                    end = int(kept_samples.max()) + 1
                    trace = self.recording.get_traces(
                        start_frame=start,
                        end_frame=end,
                        channel_ids=[ch],
                        return_in_uV=True,
                    ).flatten()
                    amplitudes[ch] = trace[kept_samples - start]
                else:
                    amplitudes[ch] = np.asarray([], dtype=np.float64)
        if include_amplitudes:
            r.amplitudes = amplitudes
        if include_triggers:
            r.triggers = self._serialize_triggers()

        # Attach provenance snapshot directly to the raster 
        if getattr(self, "history", None) is not None:
            try:
                r.provenance = self.history.to_dict()
            except Exception:
                pass
        return r
    
    def frame_slice(self, start_frame: int, end_frame: int) -> si.BaseRecording:
        """
        Get a slice of the recording between specified frame indices.

        Parameters
        ----------
        start_frame : int
            Starting frame index (inclusive).
        end_frame : int
            Ending frame index (exclusive).

        Returns
        -------
        spikeinterface.BaseRecording
            Sliced recording object.

        Raises
        ------
        ValueError
            If recording is not loaded or if frame indices are invalid.
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("Invalid start/end frame indices.")
        
        return self.recording.frame_slice(start_frame, end_frame)
    
    def split_mcs_data(self, split_times: list[float]):
        """
        Split the MCSData object into multiple MCSData objects based on split times.
        
        split_times: list of float
            List of split times in seconds. The recording will be split at these times.
        
        returns: list of MCSData
            List of MCSData objects corresponding to the split segments.        
        """

        if self.recording is None:
            raise ValueError("Recording not loaded.")
        if self.fsample is None:
            raise ValueError("Sampling frequency not initialized.")
        if self.time_vector is None:
            raise ValueError("Time vector not initialized.")

        split_samples = [int(t * float(self.fsample)) for t in split_times]
        split_samples = [0] + split_samples + [len(self.time_vector)]
        split_samples = sorted(set(split_samples))

        from .Triggers import Triggers

        segments = []
        segment_start_time = 0.0
        for i in range(len(split_samples) - 1):
            start_sample = split_samples[i]
            end_sample = split_samples[i + 1]
            segment_duration = end_sample - start_sample
            segment_start_time = float(start_sample) / float(self.fsample)
            segment_end_time = float(end_sample) / float(self.fsample)

            segment_recording = self.recording.frame_slice(start_sample, end_sample)
            segment_time_vector = np.arange(segment_duration) / float(self.fsample)

            segment_data = MCSData.from_recording(
                recording=segment_recording,
                fname=self.fname,
                fsample=self.fsample,
                time_vector=segment_time_vector,
                mask=self.mask.copy() if self.mask is not None else None,
            )
            segment_data.temporal_mask = (
                self.temporal_mask[start_sample:end_sample].copy()
                if self.temporal_mask is not None
                else np.ones(segment_duration, dtype=bool)
            )
            segment_data.excluded_intervals = []
            for a, b in self.excluded_intervals:
                if b <= segment_start_time or a >= segment_end_time:
                    continue
                segment_data.excluded_intervals.append(
                    (max(0.0, a - segment_start_time), max(0.0, b - segment_start_time))
                )

            if self.triggers is not None:
                segment_data.triggers = Triggers(slots=[])
                for slot in self.triggers.slots:
                    if slot.end <= segment_start_time or slot.start >= segment_end_time:
                        continue
                    segment_data.triggers.add_interval_slot(
                        start=max(0.0, slot.start - segment_start_time),
                        end=max(0.0, slot.end - segment_start_time),
                        ID=getattr(slot, "ID", None),
                        description=getattr(slot, "description", None),
                        blank=getattr(slot, "blank", None),
                    )
                segment_data.triggers.sort_slots()

            segment_data.history = list(self.history)
            segment_data.artifact_removal_status = self.artifact_removal_status
            segment_data.probe = self.probe
            segment_data.probe_positions = self.probe_positions
            segment_data.probe_contact_shape = self.probe_contact_shape
            segment_data.probe_shape_params = self.probe_shape_params
            segment_data.probe_ndims = self.probe_ndims

            segments.append(segment_data)

        return segments
