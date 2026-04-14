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
