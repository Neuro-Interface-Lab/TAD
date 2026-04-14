"""
Example 2: Saving and loading Raster data
==========================================

This example demonstrates how to save and load Raster objects
using both JSON and HDF5 formats.
"""

import numpy as np
from tad import Raster

# Create a simple raster
raster = Raster.empty(channels=range(5))

rng = np.random.default_rng(seed=42)
for ch in range(5):
    spike_times = rng.uniform(0.0, 10.0, size=rng.poisson(50))
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

print(f"Original raster: {raster.n_channels()} channels")

# Save to JSON format
json_path = "my_raster.json"
raster.save(json_path, h5=False)
print(f"Saved to {json_path}")

# Save to HDF5 format
h5_path = "my_raster.h5"
raster.save(h5_path, h5=True)
print(f"Saved to {h5_path}")

# Load from JSON
raster_from_json = Raster.load(json_path, h5=False)
print(f"Loaded from JSON: {raster_from_json.n_channels()} channels")

# Load from HDF5
raster_from_h5 = Raster.load(h5_path, h5=True)
print(f"Loaded from HDF5: {raster_from_h5.n_channels()} channels")

# Verify the data matches
for ch in raster.channels():
    if np.array_equal(raster.events[ch], raster_from_h5.events[ch]):
        print(f"Channel {ch}: data matches!")
    else:
        print(f"Channel {ch}: data mismatch!")
