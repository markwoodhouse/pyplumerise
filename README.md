# pyPlumeRise

pyPlumeRise is a Python implementation of the PlumeRise model for wind-blown
volcanic plumes. It includes thermodynamics for dry eruptions, wet eruptions with condensation and for ice formation.
The package includes meteorological data handling, and allows for ensemble simulations.

## Requirements

- Python 3.10 or newer.
- [Poetry](https://python-poetry.org/) for dependency and environment management

## Installation

Clone the repository, change into the project directory, and install the
dependencies:

```sh
git clone https://github.com/markwoodhouse/pyplumerise.git
cd pyplumerise
poetry install
```

## Run the example notebook

Launch the included Jupyter notebook example with:

```sh
poetry run plumerise-example 1
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

- If Poetry is not installed, install it using the
  [official instructions](https://python-poetry.org/docs/#installation).
- If the example command cannot find Jupyter, verify installation completed
  with `poetry install`, then try `poetry run jupyter notebook` from the project
  directory.
- The initialization check uses online meteorological data; if its download
  fails, check your internet connection and try again.
