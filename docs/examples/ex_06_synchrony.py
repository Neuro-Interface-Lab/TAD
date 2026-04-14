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
