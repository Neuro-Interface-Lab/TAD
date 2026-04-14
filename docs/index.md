# TAD - They All Die

**TAD** is a Python package for analyzing neural spike recordings. It provides tools for spike detection, raster manipulation, burst analysis, and various spike metrics.

## Features

- **Flexible Raster Handling**: Create, manipulate, and save spike rasters in JSON or HDF5 formats
- **Spike Metrics**: Compute firing rates, spike counts, inter-spike intervals, and more
- **Avalanche Analysis**: Extract and analyze neuronal avalanches
- **Synchrony Analysis**: Compute pairwise neural synchrony 
- **Burst Detection**: Detect and characterize burst periods
- **Flexible Triggers**: Define and manage event markers (stimulations, behavioral events)
- **MCS Data Support**: Load and process Multi-Channel Systems recordings

## Quick Start

```python
import numpy as np
from tad import Raster

# Create a raster
raster = Raster.empty(channels=range(10))

# Add spike times
for ch in range(10):
    raster.insert_timestamparray(
        ch, 
        np.random.uniform(0, 10, 100)
    )

# Save to file
raster.save("my_raster.h5", h5=True)
```

## Documentation

```{toctree}
:maxdepth: 2
:hidden:

installation
tutorials
api/index
```

- **[Installation](installation.md)** - Installation and setup
- **[Tutorials](tutorials.md)** - Step-by-step examples
- **[API Reference](api/index.rst)** - Complete API documentation

## License

TAD is licensed under the GPL-3.0 license. See LICENSE for details.
