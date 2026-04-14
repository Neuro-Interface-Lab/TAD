"""
Example 9: Inter-spike interval (ISI) analysis
===============================================

This example shows how to analyze inter-spike intervals (ISIs),
which reveal the temporal structure of neural firing.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster
from tad.metrics.isi import compute_isi

# Create a raster with different firing patterns
raster = Raster.empty(channels=range(2))

rng = np.random.default_rng(seed=42)
duration = 20.0

# Channel 0: Regular firing (low ISI variance)
# Channel 1: Irregular firing (high ISI variance)

# Regular channel
regular_times = np.arange(0.1, duration, 0.02)  # ~20 ms regular interval
raster.insert_timestamparray(0, regular_times, assume_sorted=True)

# Irregular channel (Poisson-like)
n_spikes = rng.poisson(lam=100)
irregular_times = np.sort(rng.uniform(0.0, duration, size=n_spikes))
raster.insert_timestamparray(1, irregular_times, assume_sorted=True)

# Compute ISIs
print(\"ISI Analysis:\")
print(\"=\" * 50)

for ch in raster.channels():
    isi_result = compute_isi(raster, channels=[ch], tstart=0.0, tstop=duration)
    isis = isi_result.isis[ch]
    
    if len(isis) > 0:
        print(f\"\\nChannel {ch}:\")
        print(f\"  N spikes: {len(raster.events[ch])}\")
        print(f\"  N ISIs: {len(isis)}\")
        print(f\"  Mean ISI: {np.mean(isis)*1000:.2f} ms\")
        print(f\"  Std ISI: {np.std(isis)*1000:.2f} ms\")
        print(f\"  Min ISI: {np.min(isis)*1000:.2f} ms\")
        print(f\"  Max ISI: {np.max(isis)*1000:.2f} ms\")
        print(f\"  CV (Std/Mean): {np.std(isis) / np.mean(isis):.3f}\")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ch in raster.channels():
    isi_result = compute_isi(raster, channels=[ch], tstart=0.0, tstop=duration)
    isis = isi_result.isis[ch] * 1000  # Convert to ms
    
    # Histogram
    axes[0, ch].hist(isis, bins=20, edgecolor='black', alpha=0.7)
    axes[0, ch].set_xlabel(\"ISI (ms)\")
    axes[0, ch].set_ylabel(\"Count\")
    axes[0, ch].set_title(f\"Channel {ch}: ISI Histogram\")
    axes[0, ch].axvline(np.mean(isis), color='r', linestyle='--', label=f\"Mean: {np.mean(isis):.2f} ms\")
    axes[0, ch].legend()
    
    # Log-log plot
    if len(isis) > 2:
        vals, counts = np.unique(np.round(isis, 1), return_counts=True)
        axes[1, ch].loglog(vals, counts, 'o-')
        axes[1, ch].set_xlabel(\"ISI (ms, log scale)\")
        axes[1, ch].set_ylabel(\"Count (log scale)\")
        axes[1, ch].set_title(f\"Channel {ch}: ISI Log-Log Distribution\")

fig.savefig(\"isi_analysis.pdf\")
print(\"\\nSaved figure to isi_analysis.pdf\")
