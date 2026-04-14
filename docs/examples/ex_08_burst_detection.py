"""
Example 8: Burst detection and analysis
========================================

This example shows how to detect bursts (periods of high spike rate)
in neural recordings.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster
from tad.metrics.burst import detect_burst_logISIh

# Create a raster with burst-like structure
raster = Raster.empty(channels=range(3))

rng = np.random.default_rng(seed=42)
duration = 10.0

for ch in range(3):
    spike_times = []
    
    # Define burst and non-burst periods
    burst_periods = [(1.0, 2.0), (4.5, 5.5), (7.0, 8.5)]
    
    for tstart, tstop in burst_periods:
        # High firing rate during burst
        n_burst = rng.poisson(lam=200 * (tstop - tstart))
        burst_times = rng.uniform(tstart, tstop, size=n_burst)
        spike_times.extend(burst_times)
    
    # Low firing rate during non-burst
    n_base = rng.poisson(lam=30 * duration)
    base_times = rng.uniform(0.0, duration, size=n_base)
    spike_times.extend(base_times)
    
    spike_times = np.sort(np.array(spike_times))
    raster.insert_timestamparray(ch, spike_times, assume_sorted=True)

# Detect bursts using log-ISI threshold method
result = detect_burst_logISIh(
    raster,
    tstart=0.0,
    tstop=duration,
    log_threshold=1.0,  # log10(ISI threshold in seconds)
)

print(f\"Burst detection results:\")
for ch in raster.channels():
    if ch in result.burst_intervals:
        bursts = result.burst_intervals[ch]
        print(f\"  Channel {ch}: {len(bursts)} bursts detected\")
        for i, (bs, be) in enumerate(bursts[:3]):  # Show first 3
            print(f\"    Burst {i+1}: {bs:.3f}s - {be:.3f}s (duration {be-bs:.3f}s)\")

# Visualize raster with burst intervals highlighted
fig, ax = plt.subplots(figsize=(12, 4))
raster.plot(ax=ax, tstart=0.0, tstop=duration, show=False)

# Highlight burst regions
yticks = ax.get_yticks()
for ch, bursts in result.burst_intervals.items():
    if ch in [0, 1, 2]:  # Our channels
        ch_idx = list(raster.channels()).index(ch)
        y = yticks[ch_idx]
        for bs, be in bursts:
            ax.axvspan(bs, be, alpha=0.2, color='red')

ax.set_title(\"Raster with Burst Intervals Highlighted (red)\")

fig.savefig(\"burst_detection.pdf\")
print(\"\\nSaved figure to burst_detection.pdf\")
