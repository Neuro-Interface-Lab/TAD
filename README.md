# TAD

![TAD logo](https://raw.githubusercontent.com/Neuro-Interface-Lab/TAD/main/docs/tad_logo_2_monogram_transparent.png)

TAD ("They All Die") is a Python package for handling and analyzing neural
recordings, with a focus on spike rasters, burst detection, synchrony, firing
rates, avalanches, and event-triggered analyses.

TAD is developed jointly at:

- IMS laboratory, University of Bordeaux
  Florian Kolbl, Bioelectronics group:
  https://www.ims-bordeaux.fr/research-groups/bioelectronics/
- Universidade Federal do Parana (UFPR)
  Jaderson Polli, Department of Physics:
  http://fisica.ufpr.br/pagina_ppgf_english/

## Scientific context

At this stage, TAD is primarily grounded in the analysis workflows described in:

1. Pasquale, V., Massobrio, P., Bologna, L. L., Chiappalone, M., & Martinoia,
   S. (2008). Self-organization and neuronal avalanches in networks of
   dissociated cortical neurons. *Neuroscience*, 153(4), 1354-1369.
2. Bologna, L. L., Pasquale, V., Garofalo, M., Gandolfo, M., Baljon, P. L.,
   Maccione, A., ... & Chiappalone, M. (2010). Investigating neuronal activity
   by SPYCODE multi-channel data analyzer. *Neural Networks*, 23(6), 685-697.

## Features

- Flexible raster creation, manipulation, and serialization
- Spike metrics such as counts, rates, and inter-spike intervals
- Burst detection and burst-related visualizations
- Avalanche analysis and synchrony metrics
- Trigger and event-window handling
- Support for Multi Channel Systems recordings

## Installation

For a standard user installation, install TAD from PyPI:

```bash
pip install tad-py
```

If you want to work on TAD locally in development mode:

```bash
git clone https://github.com/Neuro-Interface-Lab/TAD.git
cd TAD
pip install -e ".[dev]"
```

If you prefer a conda-based environment, you can also use:

```bash
conda env create -f environment.yml
conda activate tad-env
```

## Quick start

```python
import numpy as np
from tad import Raster

raster = Raster.empty(channels=range(10))

for ch in range(10):
    raster.insert_timestamparray(
        ch,
        np.random.uniform(0, 10, 100),
    )

raster.save("my_raster.h5", h5=True)
```

## Documentation

- Installation guide:
  https://tad.readthedocs.io/en/latest/installation.html
- Tutorials:
  https://tad.readthedocs.io/en/latest/tutorials.html
- API reference:
  https://tad.readthedocs.io/en/latest/api/index.html

## License

TAD is distributed under the GPL-3.0 license.
