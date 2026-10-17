# pyPlumeRise

pyPlumeRise is a Python implementation of the PlumeRise model for wind-blown
volcanic plumes. It includes thermodynamics for dry eruptions, wet eruptions with condensation and for ice formation.
The package includes meteorological data handling, and allows for ensemble simulations.

## Requirements

- Python 3.10 or newer.

## Installation

Install pyPlumeRise in a dedicated virtual environment or Conda environment;
avoid installing it into a system or base Python environment.

Using `venv`:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install plumerise
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Using Conda:

```sh
conda create --name plumerise python=3.12
conda activate plumerise
python -m pip install plumerise
```

To build from a source checkout, activate a virtual or Conda environment,
then run:

```sh
python -m pip install --upgrade build
python -m build
python -m pip install dist/plumerise-1.0.0-py3-none-any.whl
```

The project code is licensed under BSD-3-Clause. Some bundled datasets and
assets have separate terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before redistribution.

## Run the example notebook

Launch the included Jupyter notebook example with:

```sh
plumerise-example 1
```

The command takes the example number as a positional integer argument. Example
`1` is currently available; launching it starts Jupyter Notebook.

## Python API

The package exposes the solver and setup helpers from `plumerise`. For example,
run its small initialization check with:

```python
import plumerise

plumerise.initialize(verbose=True)
```

Initialization downloads a radiosonde profile from the University of Wyoming,
then solves a small test problem. It therefore requires an internet connection.

For a package overview, meteorological data setup, simulation workflow,
ensemble usage, and troubleshooting, see the [package guide](docs/guide.md).

## Troubleshooting

- If the example command cannot find Jupyter, verify it is installed in the
  active environment, then try `python -m jupyter notebook` from the project
  directory.
- The initialization check uses online meteorological data; if its download
  fails, check your internet connection and try again.
- If installation fails, confirm the active virtual or Conda environment uses
  Python 3.10 or newer and install with that environment's `python -m pip`.
