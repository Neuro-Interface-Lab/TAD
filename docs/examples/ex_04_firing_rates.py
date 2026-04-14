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
