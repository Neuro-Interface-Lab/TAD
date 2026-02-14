import h5py
import numpy as np
import matplotlib.pyplot as plt
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
        if load_recording:
            self.__load_recording()
            self.time_vector = np.arange(self.recording.get_total_samples()) / self.fsample


    def __load_recording(self):
        """
        Loads the MCS recording using spikeinterface and renames channels based on electrode labels.
        """
        try:
            self.recording = se.read_mcsh5(self.fname, stream_id=1)
        except Exception as e:
            print(f"Error loading recording: {e}")
            sys.exit(1)
        electrode_labels = self.recording.get_property('electrode_labels')
        self.recording = self.recording.rename_channels([f"Ch{lab}" for lab in electrode_labels])
        if self.fsample is None:
            self.fsample = self.recording.get_sampling_frequency()
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