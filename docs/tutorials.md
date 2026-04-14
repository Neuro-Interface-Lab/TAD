# Tutorials

Learn TAD through practical examples. Each tutorial is a standalone Python script you can run to understand how to use TAD features.

## Beginner Tutorials

### 1. Creating and Manipulating Rasters

Learn how to create a `Raster` object, add spike data, and perform basic queries.

```python
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
```

**Key concepts:**
- Creating a Raster with channels
- Adding spike times with `insert_timestamparray()`
- Querying channels and spike data
- Dynamically adding/removing channels

---

### 2. Saving and Loading Data

Learn how to persist raster data to disk in JSON or HDF5 format.

```python
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
```

**Key concepts:**
- Saving rasters to JSON (text-based, portable)
- Saving rasters to HDF5 (efficient binary format)
- Loading data back and verifying integrity

---

### 3. Computing Basic Spike Statistics

Compute common spike metrics like spike counts, firing rates, and inter-spike intervals.

```python
"""
Example 3: Computing scalar spike metrics
==========================================

This example shows how to compute basic spike statistics:
spike count, firing rate, and inter-spike intervals.
"""

import numpy as np
from tad import Raster
from tad.metrics.scalar import (
    spike_count,
    firing_rate,
    mean_firing_rate_across_channels,
    mean_inter_event_interval,
)

# Create a small raster
raster = Raster.empty(channels=range(5))

rng = np.random.default_rng(seed=42)
duration = 10.0

for ch in range(5):
    n_spikes = rng.poisson(lam=100)  # ~100 spikes per channel
    spike_times = rng.uniform(0.0, duration, size=n_spikes)
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

# Compute spike count per channel
spike_counts = spike_count(raster, tstart=0.0, tstop=duration, per_channel=True)
total_spikes = spike_count(raster, tstart=0.0, tstop=duration, per_channel=False)

print(f"Spike counts per channel: {spike_counts}")
print(f"Total spikes: {total_spikes}")

# Compute firing rates (spikes/second)
firing_rates = firing_rate(raster, tstart=0.0, tstop=duration, per_channel=True)
pooled_fr = firing_rate(raster, tstart=0.0, tstop=duration, per_channel=False)

print(f"Firing rates (Hz) per channel: {np.round(firing_rates, 2)}")
print(f"Pooled firing rate (Hz): {pooled_fr:.2f}")

# Mean firing rate across channels
mean_fr = mean_firing_rate_across_channels(raster, tstart=0.0, tstop=duration)
print(f"Mean FR across channels (Hz): {mean_fr:.2f}")

# Mean inter-spike interval (in seconds)
mean_isi = mean_inter_event_interval(raster, tstart=0.0, tstop=duration)
print(f"Mean ISI (seconds): {mean_isi:.4f}")
```

**Key concepts:**
- Spike count (total and per-channel)
- Firing rate in spikes/second
- Mean firing rate across channels
- Mean inter-spike intervals

---

## Intermediate Tutorials

### 4. Time-Varying Firing Rates

Compute firing rate curves that show how firing rate changes over time.

```python
"""
Example 4: Computing firing rate curves
========================================

This example shows how to compute time-varying firing rates
for individual channels and the population.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster
from tad.metrics.rates import firing_rate_curve

# Create a raster with variable firing rate
raster = Raster.empty(channels=range(10))

rng = np.random.default_rng(seed=42)
duration = 20.0

for ch in range(10):
    n_spikes = rng.poisson(lam=200)
    spike_times = rng.uniform(0.0, duration, size=n_spikes)
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

# Compute firing rate curve with 50 ms time bins
dt = 0.05  # 50 ms
result = firing_rate_curve(raster, dt=dt, tstart=0.0, tstop=duration)

print(f"Time bins: shape {result.t.shape}")
print(f"Population FR: shape {result.fr_pop.shape}")
print(f"Per-channel FR: shape {result.fr_ch.shape}")
print(f"Population FR range: {result.fr_pop.min():.2f} - {result.fr_pop.max():.2f} Hz")

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Plot population firing rate
axes[0].plot(result.t, result.fr_pop, linewidth=2)
axes[0].set_ylabel("Population FR (Hz)")
axes[0].set_title("Population Firing Rate Curve")
axes[0].grid(True, alpha=0.3)

# Plot per-channel firing rates as a heatmap
im = axes[1].imshow(
    result.fr_ch,
    aspect="auto",
    origin="lower",
    extent=[result.t[0], result.t[-1], 0, result.fr_ch.shape[0]],
    cmap="viridis",
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Channel")
axes[1].set_title("Per-Channel Firing Rate")
plt.colorbar(im, ax=axes[1], label="FR (Hz)")

fig.savefig("firing_rate_curve.pdf")
print("Saved figure to firing_rate_curve.pdf")
```

**Key concepts:**
- Sliding window firing rate
- Per-channel firing rates
- Population (pooled) firing rate
- Time-frequency visualization

---

### 5. Avalanche Extraction and Analysis

Extract and analyze neuronal avalanches—bursts of activity separated by quiet periods.

```python
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
```

**Key concepts:**
- Defining time bins
- Avalanche size definitions
- Avalanche lifetime
- Identifying active bins

---

### 6. Neural Synchrony (Pairwise Correlation)

Measure how synchronized firing is between pairs of neurons.

```python
"""
Example 6: Computing neural synchrony
======================================

This example shows how to compute pairwise synchrony (correlation)
between firing rate time series of different channels.
"""

import numpy as np
import matplotlib.pyplot as plt
from tad import Raster
from tad.metrics.rates import firing_rate_curve
from tad.metrics.synchrony import pearson_corr_firing_rate

# Create a raster with some correlated structure
raster = Raster.empty(channels=range(8))

rng = np.random.default_rng(seed=42)
duration = 30.0
dt = 0.1  # 100 ms time bins for smoothing

# Create channels with different correlation patterns
# Channels 0-2: highly correlated (shared input)
# Channels 3-5: another correlated group
# Channels 6-7: independent

common_event_1 = rng.choice([0, 1], size=300, p=[0.7, 0.3])  # shared events
common_event_2 = rng.choice([0, 1], size=300, p=[0.7, 0.3])

for ch in range(3):
    # Channels subscribe to common_event_1 + own spikes
    base = common_event_1 * 50
    own = rng.poisson(30, size=300)
    total = np.convolve(base + own, np.ones(3)/3, mode='same')
    spike_times = []
    for i, rate in enumerate(total):
        n = rng.poisson(lam=rate * dt)
        times = rng.uniform(i*dt, (i+1)*dt, size=n)
        spike_times.extend(times)
    raster.insert_timestamparray(ch, np.array(spike_times), assume_sorted=False)

for ch in range(3, 6):
    # Independent channels
    n_spikes = rng.poisson(lam=100)
    spike_times = rng.uniform(0.0, duration, size=n_spikes)
    raster.insert_timestamparray(ch, spike_times, assume_sorted=False)

# Compute firing rates
fr = firing_rate_curve(raster, dt=dt, tstart=0.0, tstop=duration)

# Compute Pearson correlation
sync = pearson_corr_firing_rate(fr, zscore=True, drop_constant=True)

corr_matrix = sync.corr

print(f"Correlation matrix shape: {corr_matrix.shape}")
print(f"Channels: {sync.channels}")
print(f"Diagonal (should be ~1): {np.diag(corr_matrix)}")
print(f"Max off-diagonal correlation: {np.max(np.abs(corr_matrix - np.eye(len(sync.channels))))}")

# Visualize
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Channel")
ax.set_ylabel("Channel")
ax.set_title("Firing Rate Correlation Matrix")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_xticks(range(len(sync.channels)))
ax.set_yticks(range(len(sync.channels)))
ax.set_xticklabels(sync.channels)
ax.set_yticklabels(sync.channels)

fig.savefig("synchrony_matrix.pdf")
print("Saved figure to synchrony_matrix.pdf")
```

**Key concepts:**
- Firing rate normalization (z-scoring)
- Pearson correlation between channels
- Correlation matrices
- Identifying correlated neural groups

---

## Advanced Tutorials

### 7. Working with Triggers and Events

Create, save, and manage event markers (e.g., stimulus onset, behavior).

```python
"""
Example 7: Working with Triggers and TimeSlots
===============================================

This example shows how to create, manipulate, and save trigger events
(e.g., stimulation times, behavioral events).
"""

from tad import Triggers, TimeSlot, load_triggers_from_json

# Create an empty trigger object
triggers = Triggers(slots=[])

# Add triggers using different approaches

# Method 1: Add a trigger using tstart and duration
triggers.add_timed_slot(
    tstart=1.0,
    duration=0.5,
    ID="stim_1",
    description="First stimulation",
)

# Method 2: Add a trigger using start and end times
triggers.add_interval_slot(
    start=3.0,
    end=3.2,
    ID="stim_2",
    description="Second stimulation",
)

# Method 3: Add a trigger manually via TimeSlot
triggers.slots.append(
    TimeSlot(
        start=5.0,
        end=5.1,
        ID="stim_3",
        description="Third stimulation",
    )
)

# Display all triggers
print(f"Total triggers: {len(triggers.slots)}")
for i, slot in enumerate(triggers.slots):
    print(f"  Trigger {i}: {slot.start:.2f}s - {slot.end:.2f}s, ID={slot.ID}")

# Save triggers to JSON
triggers.save2json("my_triggers.json")
print("\nSaved triggers to my_triggers.json")

# Load triggers back from JSON
loaded = load_triggers_from_json("my_triggers.json")
print(f"Loaded {len(loaded.slots)} triggers from file")
assert len(loaded.slots) == len(triggers.slots), "Trigger count mismatch!"

# Query triggers in a time window
window_start, window_end = 0.5, 4.0
triggers_in_window = [
    s for s in triggers.slots
    if s.start < window_end and s.end > window_start
]
print(f"\nTriggers in window [{window_start}, {window_end}]: {len(triggers_in_window)}")
```

**Key concepts:**
- Creating `Triggers` and `TimeSlot` objects
- Adding triggers by duration or interval
- Saving/loading triggers to JSON
- Querying triggers in time windows

---

### 8. Burst Detection

Automatically detect periods of high activity (bursts) in neural recordings.

```python
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

print(f"Burst detection results:")
for ch in raster.channels():
    if ch in result.burst_intervals:
        bursts = result.burst_intervals[ch]
        print(f"  Channel {ch}: {len(bursts)} bursts detected")
        for i, (bs, be) in enumerate(bursts[:3]):  # Show first 3
            print(f"    Burst {i+1}: {bs:.3f}s - {be:.3f}s (duration {be-bs:.3f}s)")

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

ax.set_title("Raster with Burst Intervals Highlighted (red)")

fig.savefig("burst_detection.pdf")
print("\nSaved figure to burst_detection.pdf")
```

**Key concepts:**
- Log-ISI threshold method
- Per-channel burst intervals
- Visualizing bursts on rasters
- Interpreting burst statistics

---

### 9. Inter-Spike Interval (ISI) Analysis

Analyze the distribution of time intervals between consecutive spikes.

```python
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
print("ISI Analysis:")
print("=" * 50)

for ch in raster.channels():
    isi_result = compute_isi(raster, channels=[ch], tstart=0.0, tstop=duration)
    isis = isi_result.isis[ch]
    
    if len(isis) > 0:
        print(f"\nChannel {ch}:")
        print(f"  N spikes: {len(raster.events[ch])}")
        print(f"  N ISIs: {len(isis)}")
        print(f"  Mean ISI: {np.mean(isis)*1000:.2f} ms")
        print(f"  Std ISI: {np.std(isis)*1000:.2f} ms")
        print(f"  Min ISI: {np.min(isis)*1000:.2f} ms")
        print(f"  Max ISI: {np.max(isis)*1000:.2f} ms")
        print(f"  CV (Std/Mean): {np.std(isis) / np.mean(isis):.3f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ch in raster.channels():
    isi_result = compute_isi(raster, channels=[ch], tstart=0.0, tstop=duration)
    isis = isi_result.isis[ch] * 1000  # Convert to ms
    
    # Histogram
    axes[0, ch].hist(isis, bins=20, edgecolor='black', alpha=0.7)
    axes[0, ch].set_xlabel("ISI (ms)")
    axes[0, ch].set_ylabel("Count")
    axes[0, ch].set_title(f"Channel {ch}: ISI Histogram")
    axes[0, ch].axvline(np.mean(isis), color='r', linestyle='--', label=f"Mean: {np.mean(isis):.2f} ms")
    axes[0, ch].legend()
    
    # Log-log plot
    if len(isis) > 2:
        vals, counts = np.unique(np.round(isis, 1), return_counts=True)
        axes[1, ch].loglog(vals, counts, 'o-')
        axes[1, ch].set_xlabel("ISI (ms, log scale)")
        axes[1, ch].set_ylabel("Count (log scale)")
        axes[1, ch].set_title(f"Channel {ch}: ISI Log-Log Distribution")

fig.savefig("isi_analysis.pdf")
print("\nSaved figure to isi_analysis.pdf")
```

**Key concepts:**
- Computing ISIs per channel
- Regular vs. irregular firing
- Coefficient of variation (CV)
- Log-log ISI distributions

---

### 10. Visualization and Plotting

Visualize spike rasters using various plotting modes and perspectives.

```python
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
ax1.set_title("Full Raster (all 20 channels)")
ax1.set_ylabel("Channel")

# Plot 2: Zoomed view
ax2 = plt.subplot(2, 2, 2)
raster.plot(ax=ax2, tstart=1.0, tstop=2.0, show=False)
ax2.set_title("Zoomed View (1-2s)")
ax2.set_ylabel("Channel")

# Plot 3: Subset of channels
ax3 = plt.subplot(2, 2, 3)
subset = Raster.empty(channels=range(5))
for ch in range(5):
    subset.insert_timestamparray(ch, raster.events[ch], assume_sorted=True)
subset.plot(ax=ax3, tstart=0.0, tstop=duration, show=False)
ax3.set_title("Subset: Channels 0-4")
ax3.set_ylabel("Channel")

# Plot 4: Spike count per channel (bar plot)
ax4 = plt.subplot(2, 2, 4)
spike_counts = np.array([len(raster.events[ch]) for ch in raster.channels()])
ax4.bar(raster.channels(), spike_counts, color='steelblue', edgecolor='black')
ax4.set_xlabel("Channel")
ax4.set_ylabel("Spike Count")
ax4.set_title("Spikes per Channel")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("raster_visualization.pdf")
print("Saved visualization to raster_visualization.pdf")

# Print summary statistics
print(f"\nRaster Summary:")
print(f"  Channels: {raster.n_channels()}")
print(f"  Duration: {duration} seconds")
print(f"  Total spikes: {sum(len(raster.events[ch]) for ch in raster.channels())}")
print(f"  Spikes per channel (mean ± std): {spike_counts.mean():.1f} ± {spike_counts.std():.1f}")
```

**Key concepts:**
- Plotting full and zoomed rasters
- Plotting channel subsets
- Spike count bar plots
- Exporting figures to PDF

---

## Running the Examples

You can copy and paste each code block above into a Python script or Jupyter notebook to run them interactively.

## Next Steps

- Check the **[API Reference](api/index.rst)** for detailed function documentation
- Join the community on GitHub for issues and discussions
