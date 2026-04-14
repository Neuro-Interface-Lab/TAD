"""
Example 1: Creating and manipulating Raster objects
=====================================================

This example shows how to create a Raster object, add spike times,
and perform basic queries.
"""

import numpy as np
from tad import Raster

# Create an empty raster with channels 0-9
raster = Raster.empty(channels=range(10))

# Add spike times for each channel (in seconds)
rng = np.random.default_rng(seed=42)
duration = 5.0  # 5 second recording

for ch in range(10):
    # Generate random spike times (Poisson-like)
    n_spikes = rng.poisson(lam=50)  # ~50 spikes per channel
    spike_times = rng.uniform(0.0, duration, size=n_spikes)
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

# Query the raster
print(f"Number of channels: {raster.n_channels()}")
print(f"Channels: {raster.channels()}")

# Get spike times on a specific channel
ch0_spikes = raster.events[0]
print(f"Channel 0 has {len(ch0_spikes)} spikes")
print(f"First 5 spike times: {ch0_spikes[:5]}")

# Add a new channel dynamically
new_spike_times = [0.5, 1.2, 2.3, 3.1]
raster.insert_channel("extra_ch", times=new_spike_times, overwrite=False)
print(f"Total channels after adding 'extra_ch': {raster.n_channels()}")

# Remove a channel
raster.pop_channel("extra_ch")
print(f"Total channels after removing 'extra_ch': {raster.n_channels()}")
