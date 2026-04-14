"""
Example 10: Visualizing rasters and neural data
================================================

This example demonstrates various plotting capabilities
for visualizing neural spike data.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster

# Create a raster with interesting structure
raster = Raster.empty(channels=range(20))

rng = np.random.default_rng(seed=42)
duration = 5.0

# Create some spatial structure: neurons at different rates
for ch in range(20):
    # Firing rate varies across channels (e.g., different neuron types)
    base_rate = 20 + ch * 2
    n_spikes = rng.poisson(lam=base_rate * duration)
    spike_times = rng.uniform(0.0, duration, size=n_spikes)
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

# Create figure with multiple viewing modes
fig = plt.figure(figsize=(14, 10))

# Plot 1: Full raster
ax1 = plt.subplot(2, 2, 1)
raster.plot(ax=ax1, tstart=0.0, tstop=duration, show=False)
ax1.set_title(\"Full Raster (all 20 channels)\")
ax1.set_ylabel(\"Channel\")

# Plot 2: Zoomed view
ax2 = plt.subplot(2, 2, 2)
raster.plot(ax=ax2, tstart=1.0, tstop=2.0, show=False)
ax2.set_title(\"Zoomed View (1-2s)\")
ax2.set_ylabel(\"Channel\")

# Plot 3: Subset of channels
ax3 = plt.subplot(2, 2, 3)
subset = Raster.empty(channels=range(5))
for ch in range(5):
    subset.insert_timestamparray(ch, raster.events[ch], assume_sorted=True)
subset.plot(ax=ax3, tstart=0.0, tstop=duration, show=False)
ax3.set_title(\"Subset: Channels 0-4\")
ax3.set_ylabel(\"Channel\")

# Plot 4: Spike count per channel (bar plot)
ax4 = plt.subplot(2, 2, 4)
spike_counts = np.array([len(raster.events[ch]) for ch in raster.channels()])
ax4.bar(raster.channels(), spike_counts, color='steelblue', edgecolor='black')
ax4.set_xlabel(\"Channel\")
ax4.set_ylabel(\"Spike Count\")
ax4.set_title(\"Spikes per Channel\")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(\"raster_visualization.pdf\")
print(\"Saved visualization to raster_visualization.pdf\")

# Print summary statistics
print(f\"\\nRaster Summary:\")
print(f\"  Channels: {raster.n_channels()}\")
print(f\"  Duration: {duration} seconds\")
print(f\"  Total spikes: {sum(len(raster.events[ch]) for ch in raster.channels())}\")
print(f\"  Spikes per channel (mean ± std): {spike_counts.mean():.1f} ± {spike_counts.std():.1f}\")
