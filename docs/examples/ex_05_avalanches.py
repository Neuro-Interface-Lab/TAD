"""
Example 5: Extracting and analyzing avalanches
===============================================

This example demonstrates avalanche extraction from neural recordings.
Avalanches are periods of consecutive active bins with intermittent silent bins.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster
from tad.metrics.avalanches import extract_avalanches

# Create a handcrafted raster for illustration
raster = Raster.empty(channels=[0, 1, 2])

# Add spike patterns that will create distinct avalanches
# Avalanche 1: bins 0-1 active
raster.insert_timestamparray(0, [0.1, 1.1])  # channel 0 in bins 0 and 1
raster.insert_timestamparray(1, [1.2])       # channel 1 in bin 1

# Avalanche 2: bins 3-4 active (bin 2 is silent)
raster.insert_timestamparray(2, [3.1, 4.2])  # channel 2 in bins 3 and 4

# Extract avalanches using 1-second time bins
dt = 1.0
result = extract_avalanches(raster, dt=dt, tstart=0.0, tstop=6.0, size_definition=1)

print(f"Number of avalanches: {len(result.sizes)}")
print(f"Avalanche sizes: {result.sizes}")
print(f"Avalanche lifetimes (bins): {result.lifetimes}")
print(f"Avalanche intervals (bins): {result.intervals_bins}")

# Show which bins had activity
print(f"Active bins: {result.active_bins}")

# Extract using a second size definition (number of unique channels per avalanche)
result2 = extract_avalanches(raster, dt=dt, tstart=0.0, tstop=6.0, size_definition=2)
print(f"Sizes (n_channels per avalanche): {result2.sizes}")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(10, 6))

# Raster
raster.plot(ax=axes[0], tstart=0.0, tstop=6.0, show=False)
axes[0].set_title("Spike Raster")

# Active bins
axes[1].bar(range(len(result.active_bins)), result.active_bins.astype(int))
axes[1].set_xlabel("Time bin")
axes[1].set_ylabel("Active")
axes[1].set_title("Active Bins")

# Avalanche sizes
axes[2].bar(range(len(result.sizes)), result.sizes)
axes[2].set_xlabel("Avalanche index")
axes[2].set_ylabel("Size (definition 1)")
axes[2].set_title("Avalanche Sizes")

fig.savefig("avalanche_extraction.pdf")
print("Saved figure to avalanche_extraction.pdf")
