# Installation

## Prerequisites

- Python 3.12 or higher
- conda or mamba (recommended)

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

## Installing with pip

If you prefer pip, you can install TAD after setting up a Python 3.12 environment:

```bash
pip install git+https://github.com/Neuro-Interface-Lab/TAD.git
```

This installs only the core runtime dependencies.

## Verifying Installation

To verify that TAD is installed correctly, run:

```python
import tad
print(tad.__version__)

# Try importing main classes
from tad import Raster, Triggers, TimeSlot, MCSData
print("Installation successful!")
```

## Development Setup

If you want to contribute or modify TAD, clone the repository and install in editable mode:

```bash
git clone https://github.com/Neuro-Interface-Lab/TAD.git
cd TAD
pip install -e ".[dev]"
```

Or using the environment file:

```bash
conda env create -f environment.yml
conda activate tad-env
# Package is automatically installed in editable mode
```

Then run tests to verify:

```bash
pytest test/
```

## System Requirements

- **macOS**: Intel and Apple Silicon (arm64) supported
- **Linux**: Ubuntu 20.04 or newer
- **Windows**: Not officially tested
- **Memory**: Minimum 2GB RAM (depends on dataset size)

## Troubleshooting

### ImportError: No module named 'tad'

Make sure you've activated the conda environment:
```bash
conda activate tad-env
```

Or reinstall in editable mode:
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
