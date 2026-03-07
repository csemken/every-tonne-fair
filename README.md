# Every tonne matters: temperature response

This is the code repository to replicate the state-dependent temperature impulse response functions for “Every tonne matters: marginal emission reductions have human-scale benefits”.

Note: all shell commands/scripts should be run from the project’s root directory. Notebooks should be run from the `notebooks/` folder.

## Pre-requirements

- Conda ([Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Miniforge](https://conda-forge.org/miniforge/))

Tested with Miniconda 24.7.1 on Ubuntu 24.04.

## Setup

Note: the setup takes about 5 minutes on a "normal" desktop computer.

1. Create the conda environment and install all required packages:
```shell
conda env create --prefix .conda --file environment.lock.yml
conda activate ./.conda
```

2. Create jupyter notebooks:
```shell
jupytext --sync notebooks/*
```

## Replicate

Execute the notebooks in the following order:
1. `alternative-scenarios.ipynb`
2. `alternative-scenarios-pulse.ipynb`
3. `correlate_rho_T.ipynb`

The legacy notebook that runs the SSPs is included but no longer part of the workflow.

Replication requires at least 16GB of RAM and takes approximately 10 minutes.

To re-export the notebooks, run:
```
jupytext --to py:percent notebooks/*.ipynb
```

## Development

### Adding/updating requirements

Add required packages to `environment.yml`. Then run:
```shell
conda activate ./.conda
conda env update --file environment.yml --prune
conda env export --no-build --prefix ./.conda > environment.lock.yml
```
and replace the `channels` settings in `environment.lock.yml` with those in `environment.yml`.
