import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import List, Optional
import spikeinterface.extractors as se
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

import os
import sys


def define_MCS_probe():
    return None

#### class MCSData:
class MCSData:
    def __init__(self, 
                 fname,
                 fsample=None,
                 load_recording=True,
                 load_digital = False
                 ):
        """
        Opens an MCS .h5 file and prepares it for analysis.
        
        args:
        -----
         fname: str, path to the .h5 file containing the MCS recording.
        """
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File {fname} does not exist.")
        self.fname = fname  
        self.recording = None
        self.peaks = None
        self.probe = define_MCS_probe()
        self.fsample = fsample
        self.time_vector = None
        self.mask = None
        self.ch_ids = None
        self.electrode_labels = None
        self.temporal_mask = None
        self.triggers = None
        self.digital_recording = None
        self.load_digital = load_digital
        if load_recording:
            self.__load_recording()
            self.time_vector = np.arange(self.recording.get_total_samples()) / self.fsample
            self.ch_ids = self.recording.channel_ids
            self.electrode_labels = self.recording.get_property('electrode_labels')
            self.mask = np.ones(self.recording.get_num_channels(), dtype=bool)
            self.temporal_mask = np.ones_like(self.time_vector, dtype=bool)

    def __load_recording(self):
        """
        Loads the MCS recording using spikeinterface and renames channels based on electrode labels.
        """
        try:
            self.recording = se.read_mcsh5(self.fname, stream_id=1)
        except Exception as e:
            print(f"Error loading recording: {e}")
            sys.exit(1)
        if self.load_digital:
            try:
                with h5py.File(self.fname, "r") as f:
                    stream = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
                    self.digital_recording = stream[0]
            except Exception as e:
                print(f"Error loading digital recording: {e}")
        electrode_labels = self.recording.get_property('electrode_labels')
        self.recording = self.recording.rename_channels([f"Ch{lab}" for lab in electrode_labels])
        if self.fsample is None:
            self.fsample = self.recording.get_sampling_frequency()
        return 1
    
    ### basic methods
    def set_mask(self, mask):
        """
        Sets the channel mask for analysis.
        """
        if len(mask) != self.recording.get_num_channels():
            raise ValueError("Mask length must match number of channels.")
        self.mask = mask
        return 1

    ### processing methods
    def detect_spikes(self,
                      method='by_channel',
                      peak_sign='neg',
                      detect_threshold=5,
                      exclude_sweep_ms=0.2
                    ):
        """
        Detects spikes in the recording using the specified parameters.
        """
        if self.recording is None:
            raise ValueError("Recording not loaded.")
        
        self.peaks = detect_peaks(
            recording=self.recording,
            method=method,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms
        )
        return 1

    ### visualization methods
    def plot_raster(self, ax):
        """
        Plots a raster of detected spikes.
        """
        if self.peaks is None:
            raise ValueError("Spikes not detected.")
        
        peaks_sc = np.column_stack((self.peaks['sample_index'], self.peaks['channel_index']))
        ax.scatter(peaks_sc[:, 0]/self.fsample, peaks_sc[:, 1], s=1)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Channel Index')
        ax.set_title('Spike Raster Plot')
        return 1
    
    """
    def chose_mask(self, tmin=0, tmax=10):
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # limpa tudo
        for ax in axes.flat:
             ax.axis("off")

        for ch, lab in zip(self.ch_ids, self.electrode_labels):
            lab = int(lab)
            col = lab // 10 - 1   # coluna 0–7
            row = lab % 10 - 1    # linha 0–7
            ax = axes[row,col]
            traces = self.recording.get_traces(start_frame = tmin*self.fsample , end_frame = tmax*self.fsample, channel_ids = [ch], return_in_uV = True)
            local_time_vector = np.arange(traces.shape[0]) / self.fsample + tmin
            ax.plot(local_time_vector, traces)
            #ax.text(0.5, 0.5, ch, ha="center", va="center", fontsize=7)
            ax.set_title(lab, fontsize=6)
            ax.set_xlim(0, 10)
            ax.set_ylim(-50, 50)
            ax.axis("off")


        plt.show()
    """
    def chose_mask(self, tmin=0, tmax=10):
        """
        Opens a GUI to choose which channels to include in the analysis.
        Returns a boolean mask aligned with self.ch_ids (True = keep, False = exclude).
        """
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # turn everything off first
        for ax in axes.flat:
            ax.axis("off")

        n = len(self.ch_ids)
        mask = np.ones(n, dtype=bool)
        lines = [None] * n
        checks = [None] * n

        def make_toggle(i):
            def _toggle(_label):
                mask[i] = not mask[i]

                ln = lines[i]
                if ln is not None:
                    ln.set_color("C0" if mask[i] else "0.7")

                cb = checks[i]
                if cb is not None:
                    # Light tint when unchecked
                    try:
                        cb.rectangles[0].set_facecolor("white" if mask[i] else "0.9")
                    except Exception:
                        pass

                fig.canvas.draw_idle()
            return _toggle

        for i, (ch, lab) in enumerate(zip(self.ch_ids, self.electrode_labels)):
            lab = int(lab)
            col = lab // 10 - 1   # col 0–7
            row = lab % 10 - 1    # row 0–7
            ax = axes[row, col]

            traces = self.recording.get_traces(
                start_frame=int(tmin * self.fsample),
                end_frame=int(tmax * self.fsample),
                channel_ids=[ch],
                return_in_uV=True
            )
            local_time_vector = np.arange(traces.shape[0]) / self.fsample + tmin

            ln, = ax.plot(local_time_vector, traces, lw=0.8, color="C0")
            lines[i] = ln

            ax.set_title(lab, fontsize=6)
            ax.set_xlim(tmin, tmax)
            ax.set_ylim(-50, 50)
            ax.axis("off")

            # Tiny checkbox inside the subplot (figure-relative coordinates)
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

            # Hide the label text (keep widget clickable)
            for txt in getattr(cb, "labels", []):
                txt.set_visible(False)

            # Some matplotlib versions use `lines`, others `lines_` (or neither); handle safely.
            line_groups = getattr(cb, "lines", None) or getattr(cb, "lines_", None)
            if line_groups is not None:
                for pair in line_groups:  # each entry is typically a (line1, line2) tuple
                    try:
                        pair[0].set_linewidth(1.0)
                        pair[1].set_linewidth(1.0)
                    except Exception:
                        pass

            cb.on_clicked(make_toggle(i))
            checks[i] = cb

        plt.show()
        self.mask = mask

    def blank_period(self, tstart, tstop):
        """
        Blanks out (excludes) the specified time periods from analysis.
        """
        if self.time_vector is None:
            raise ValueError("Time vector not initialized.")
        if tstart >= tstop:
            raise ValueError("tstart must be less than tstop.")
        
        mask = (self.time_vector < tstart) | (self.time_vector > tstop)
        print(mask)
        self.temporal_mask &= mask

    def convert_digital(self):
        a = self.digital_recording[0]
        self.digital_recording = np.log2(np.abs(self.digital_recording - a + 1))
        self.digital_recording[self.digital_recording > 2] = 0
        self.digital_recording = np.asarray(self.digital_recording, dtype=np.int32)
        return self.digital_recording

    def detect_digital_rising_edge(self):
        edges = []
        for i in range(1, len(self.digital_recording)):
            if self.digital_recording[i] > self.digital_recording[i-1]:
                edges.append(i)
        return edges
    
    def detect_digital_falling_edge(self):
        edges = []
        for i in range(1, len(self.digital_recording)):
            if self.digital_recording[i] < self.digital_recording[i-1]:
                edges.append(i)
        return edges
    
    def get_triggers(self, tstart=None, tstop=None, interpretor=None, dt_after_trigger: Optional[float] =  None):

        if tstart is None:
            tstart = self.time_vector[0]
        if tstop is None:
            tstop = self.time_vector[-1]

        from .Triggers import Triggers  # avoid circular import

        self.triggers = Triggers(slots=[])
        
        self.digital_recording = self.digital_recording[(self.time_vector >= tstart) & (self.time_vector <= tstop)]
        self.digital_recording = self.convert_digital()
        if interpretor is None:
            # treat each rising as the start of a time slot and each falling as the end of a time slot:
            rising_edges = self.detect_digital_rising_edge()
            falling_edges = self.detect_digital_falling_edge()
            for start, end in zip(rising_edges, falling_edges):
                self.triggers.add_interval_slot(start=start/self.fsample, end = end / self.fsample)
        elif callable(interpretor):
            if dt_after_trigger is None:
                interpretor(self.digital_recording, self.triggers, self.fsample)
            else:
                interpretor(self.digital_recording, self.triggers, self.fsample, dt_after_trigger)
        else:
            raise ValueError("interpretor must be a callable function that defines how to interpret the digital signal into triggers.")

        return self.triggers
        

    def get_raster(self, tstart=None, tstop=None):
        """
        Returns a Raster object containing the spike times for the selected channels.
        """
        if self.peaks is None:
            raise ValueError("Spikes not detected.")
        if tstart is None:
            tstart = self.time_vector[0]
        if tstop is None:
            tstop = self.time_vector[-1]

        from .raster import Raster  # avoid circular import
        r = Raster.empty(channels=self.ch_ids[self.mask])

        peaks_sc = np.column_stack((self.peaks['sample_index'], self.peaks['channel_index']))
        for k, ch in enumerate(self.ch_ids[self.mask]):
            # r.insert_channel(ch)
            this_channel_times = peaks_sc[peaks_sc[:, 1] == k][:, 0] / self.fsample
            keep_spikes = (this_channel_times >= tstart) & (this_channel_times <= tstop) & self.temporal_mask[(this_channel_times * self.fsample).astype(int)]
            r.insert_timestamparray(ch, this_channel_times[keep_spikes], assume_sorted=True)
        return r
