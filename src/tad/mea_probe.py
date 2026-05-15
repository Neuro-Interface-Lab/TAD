# recording_probe.py

from probeinterface import Probe
from probeinterface.plotting import plot_probe
from typing import List, Optional, Tuple, Dict

class MEAProbe:
    """
    Wrapper around probeinterface.Probe to handle MEA probes
    and allow filtering/removal of channels, setting shapes, etc.
    """
    def __init__(self,
                 positions: List[Tuple[float, float]],
                 contact_shapes: Optional[List[str]] = None,
                 shape_params: Optional[List[Dict]] = None,
                 name: str = "MEA_probe"):
        """
        Initialize a RecordingProbe.

        Parameters
        ----------
        positions : list of (x, y)
            Coordinates of each contact.
        contact_shapes : list of str, optional
            Shape of each contact ('circle', 'square', etc.)
        shape_params : list of dict, optional
            Parameters for each contact shape (radius, width, etc.)
        name : str
            Name of the probe
        """
        self.positions = positions
        self.contact_shapes = contact_shapes or ["circle"] * len(positions)
        self.shape_params = shape_params or [{}] * len(positions)
        self.name = name

        # The internal probeinterface object
        self._pi_probe = Probe(ndim=2, si_units='um', contact_positions=positions,
                               contact_shapes=self.contact_shapes,
                               shape_params=self.shape_params,
                               name=name)

        # Mask to mark channels that should be excluded from analysis
        self.masked_channels: List[int] = []

    # -------------------------
    # Placeholder methods
    # -------------------------

    def mask_channels(self, channels: List[int]) -> None:
        """
        Mark channels to be ignored in analysis (dead, noisy, stim, reference, etc.)
        """
        self.masked_channels.extend(channels)

    def unmask_channels(self, channels: List[int]) -> None:
        """
        Remove channels from the masked list
        """
        self.masked_channels = [ch for ch in self.masked_channels if ch not in channels]

    def get_active_channels(self) -> List[int]:
        """
        Return a list of channel indices that are not masked
        """
        all_channels = list(range(len(self.positions)))
        return [ch for ch in all_channels if ch not in self.masked_channels]

    def plot(self, show_mask: bool = True):
        """
        Plot the probe. Optionally indicate masked channels.
        """
        plot_probe(self._pi_probe)
        if show_mask and self.masked_channels:
            print(f"Masked channels: {self.masked_channels}")
            # Could add visual markers here

    # Placeholder for future integration with raster or spike detection
    def apply_to_recording(self, recording):
        """
        Apply the probe settings to a recording (mask channels, etc.)
        """
        pass

    def save(self, filepath: str):
        """
        Save probe information to a JSON or other format
        """
        pass

    def load(self, filepath: str):
        """
        Load probe information from a saved file
        """
        pass
