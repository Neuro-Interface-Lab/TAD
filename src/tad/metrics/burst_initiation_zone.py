from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tad.metrics.burst import BurstDetectionResult

@dataclass
class BurstInitiationPoint:
    """
    A dataclass that holds the list of ($x_0$, $y_0$) coordinates (on electrode array matrix) of the first spike of the bursts between t_start and t_end, along with the channel ID and time of the first spike."""
    burst_initiation_points: tuple[int, int]
    channel_id: str
    time: float

@dataclass
class BurstInitiationPoints:
    """
    A dataclass that holds the list of BurstInitiationPoint objects for all bursts detected between t_start and t_end."""
    burst_initiation_points: list[BurstInitiationPoint]

def channel_position(channel_id: str) -> tuple[int, int]:
    channel = int(channel_id[2:])  # "Ch12" -> 12
    row = channel // 10 - 1
    col = channel % 10 - 1
    return (row,col)

def compute_burst_initiation_points(burst_detection_result: BurstDetectionResult) -> BurstInitiationPoints:
    """
    This function receives a BurstDetectionResult object and computes the list of BurstInitiationPoint objects for all bursts detected between t_start and t_end.

    Parameters:
    - burst_detection_result: A BurstDetectionResult object containing the burst detection results for the analysis.

    Returns:
    - A BurstInitiationPoints object containing the list of BurstInitiationPoint objects for all bursts detected between t_start and t_end.
    """
    burst_initiation_points = []
    
    return BurstInitiationPoints(burst_initiation_points=burst_initiation_points)

@dataclass
class BurstInitiationProbabilityResult:
    """
    A dataclass to hold the probability density for burst initiation across channels.

    Parameters:
    - burst_initiation_probabilities: A 1D array where each entry represents the probability of burst initiation at each channel.
    - channel_ids: A list of channel identifiers corresponding to the rows and columns of the burst_initiation_probabilities matrix.
    """
    
    burst_initiation_probabilities: np.ndarray
    channel_ids: list[str]

def 