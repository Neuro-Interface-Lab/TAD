# TAD - They All Die

![TAD logo](https://raw.githubusercontent.com/Neuro-Interface-Lab/TAD/main/docs/tad_logo_2_monogram_transparent.png)

**TAD** is a Python package for handling and analyzing neural spike recordings.
It provides tools for spike detection, raster manipulation, burst analysis,
synchrony, avalanche analysis, and trigger-based workflows.

## Development

TAD is developed jointly at:

- **IMS laboratory, University of Bordeaux**  
  Florian Kolbl, Bioelectronics group:  
  <https://www.ims-bordeaux.fr/research-groups/bioelectronics/>
- **Universidade Federal do Parana (UFPR)**  
  Jaderson Polli, Department of Physics:  
  <http://fisica.ufpr.br/pagina_ppgf_english/>

## Scientific Background

At this stage, TAD is mainly inspired by the methods and workflows described in:

1. Pasquale, V., Massobrio, P., Bologna, L. L., Chiappalone, M., & Martinoia, S. (2008). *Self-organization and neuronal avalanches in networks of dissociated cortical neurons*. Neuroscience, 153(4), 1354-1369.
2. Bologna, L. L., Pasquale, V., Garofalo, M., Gandolfo, M., Baljon, P. L., Maccione, A., ... & Chiappalone, M. (2010). *Investigating neuronal activity by SPYCODE multi-channel data analyzer*. Neural Networks, 23(6), 685-697.

## Features

- **Flexible Raster Handling**: Create, manipulate, and save spike rasters in JSON or HDF5 formats
- **Spike Metrics**: Compute firing rates, spike counts, inter-spike intervals, and more
- **Avalanche Analysis**: Extract and analyze neuronal avalanches
- **Synchrony Analysis**: Compute pairwise neural synchrony 
- **Burst Detection**: Detect and characterize burst periods
- **Flexible Triggers**: Define and manage event markers (stimulations, behavioral events)
- **MCS Data Support**: Load and process Multi-Channel Systems recordings

## Installation

For a standard user installation:

```bash
pip install tad-py
```

For development:

```bash
git clone https://github.com/Neuro-Interface-Lab/TAD.git
cd TAD
pip install -e ".[dev]"
```

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
