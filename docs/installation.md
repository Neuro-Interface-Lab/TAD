# Installation

## Prerequisites

- Python 3.12 or higher
- conda or mamba (recommended)

## Standard Installation

The recommended user installation is through PyPI:

```bash
pip install tad-py
```

This installs the published version of TAD and its runtime dependencies.

## Development Installation

If you want to work on TAD locally, clone the repository and install it in
editable mode:

```bash
git clone https://github.com/Neuro-Interface-Lab/TAD.git
cd TAD
pip install -e ".[dev]"
```

This installs:

- TAD in editable mode
- Runtime dependencies such as `numpy`, `matplotlib`, `spikeinterface`, `probeinterface`, and `h5py`
- Development tools such as `pytest`, `flake8`, and documentation dependencies

## Installing with conda/mamba

The easiest way to install TAD with all dependencies is to use conda:

```bash
# Download the environment file
curl -O https://raw.githubusercontent.com/Neuro-Interface-Lab/TAD/main/environment.yml

# Create and activate the environment
conda env create -f environment.yml
conda activate tad-env
```

This installs:
- TAD package in editable mode (`-e .`)
- All runtime dependencies: numpy, matplotlib, spikeinterface, probeinterface, h5py
- Development tools: pytest, sphinx, black, flake8
- Documentation dependencies: myst-parser, furo

## Project Context

TAD is developed jointly at:

- **IMS laboratory, University of Bordeaux**  
  Florian Kolbl, Bioelectronics group:  
  <https://www.ims-bordeaux.fr/research-groups/bioelectronics/>
- **Universidade Federal do Parana (UFPR)**  
  Jaderson Polli, Department of Physics:  
  <http://fisica.ufpr.br/pagina_ppgf_english/>

The current package is mainly inspired by the analysis strategies presented in:

1. Pasquale, V., Massobrio, P., Bologna, L. L., Chiappalone, M., & Martinoia, S. (2008). *Self-organization and neuronal avalanches in networks of dissociated cortical neurons*. Neuroscience, 153(4), 1354-1369.
2. Bologna, L. L., Pasquale, V., Garofalo, M., Gandolfo, M., Baljon, P. L., Maccione, A., ... & Chiappalone, M. (2010). *Investigating neuronal activity by SPYCODE multi-channel data analyzer*. Neural Networks, 23(6), 685-697.

## Verifying Installation

To verify that TAD is installed correctly, run:

```python
import tad
from tad import Raster, Triggers, TimeSlot, MCSData
print("Installation successful!")
```

## System Requirements

- **macOS**: Intel and Apple Silicon (arm64) supported
- **Linux**: Ubuntu 20.04 or newer
- **Windows**: Not officially tested
- **Memory**: Minimum 2GB RAM (depends on dataset size)

## Troubleshooting

### ImportError: No module named 'tad'

Make sure you installed the package in the active environment:

```bash
pip install tad-py
```

For development, reinstall in editable mode:

```bash
pip install -e .
```

### Build errors with spikeinterface

This is usually because you need to install build tools. On macOS:
```bash
xcode-select --install
```

On Ubuntu/Linux:
```bash
sudo apt-get install build-essential
```
