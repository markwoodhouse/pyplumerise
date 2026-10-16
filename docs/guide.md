# pyPlumeRise package guide

pyPlumeRise simulates wind-blown volcanic eruption columns. A model run combines
an atmospheric profile, physical parameters, solver settings, and eruption
source conditions. The model integrates the plume equations along the plume
centreline and returns both the calculated solution and summary plume
properties.

## Install and launch

Install the package and its dependencies from the repository root:

```sh
poetry install
```

Launch the included Jupyter notebook example:

```sh
poetry run plumerise-example 1
```

Example 1 demonstrates Wyoming radiosonde data, model configuration, solving,
and reading output variables. The command opens Jupyter Notebook; run it from
the project environment.

## Main package components

| Module | Purpose |
| --- | --- |
| `plumerise` | Top-level exports for meteorological data, model setup, solving, and package version. |
| `plumerise.model` | Meteorology conversion, physical and solver parameters, initial conditions, dry/wet/ice equations, integration, and ensemble solving. |
| `plumerise.data` | IGRA2 radiosonde station list, exposed as `StationList`. |
| `plumerise.notebooks` | Launches packaged Jupyter examples. |
| `plumerise.app` | Dashboard application. |
| `plumerise.utilities` | Numeric, plotting, and string utility functions. |

The most common workflow uses the top-level functions `MetData`, `set_params`,
`set_solver_params`, `set_IC`, `solve`, and `get_soln`. For the broader set of
model and ensemble functions, import from `plumerise.model`.

## Meteorological profiles

The solver expects a two-dimensional NumPy array with one atmospheric
observation per row and six columns in this order:

| Column | Quantity | Units |
| --- | --- | --- |
| 1 | Altitude | m |
| 2 | Temperature | K |
| 3 | Pressure | Pa |
| 4 | Wind direction | radians |
| 5 | Wind speed | m/s |
| 6 | Relative humidity | fraction from 0 to 1 |

Use `MetData` to retrieve and convert an atmospheric profile. Dates use
`YYYY-MM-DD`; times use `HH:MM` in UTC.

### Wyoming radiosonde

Retrieve an archived sounding with its station identifier:

```python
import plumerise as pr

met_data = pr.MetData(
    "2010-04-14",
    "12:00",
    "radiosonde",
    source="wyoming",
    stationID="04018",
)
met = met_data.to_array()
```

The station identifier can be looked up in `plumerise.StationList`. Retrieval
requires an internet connection and availability of the Wyoming Weather Web
service.

### ERA5 reanalysis

`MetData` also accepts `type="era5"` with `latitude` and `longitude` in decimal
degrees:

```python
met_data = pr.MetData(
    "2020-01-01",
    "12:00",
    "era5",
    latitude=64.0,
    longitude=-22.0,
)
met = met_data.to_array()
```

ERA5 retrieval uses the Copernicus Climate Data Store API and requires a
configured CDS API account and credentials. The retrieval can take time and
writes an `era5_data.nc` file in the current working directory.

You can also build the six-column array from your own profile data, provided
the columns and units match the table above.

## Configure and run a simulation

Create physical parameters, solver settings, and vent initial conditions, then
pass them with the meteorological array to `solve`:

```python
import plumerise as pr

params = pr.set_params(ks=0.09, kw=0.9)
solver_params = pr.set_solver_params(eqn="ice", atol=1e-8, rtol=1e-8)
initial_conditions = pr.set_IC(
    L0=100,       # vent radius [m]
    U0=100,       # exit velocity [m/s]
    n0=0.03,      # gas mass fraction
    T0=1000,      # vent temperature [K]
    z0=1666,      # vent altitude [m]
)

soln, plume_state, solver_flag = pr.solve(
    Params=params,
    SolParams=solver_params,
    IC=initial_conditions,
    Met=met,
)
```

`solve()` returns three values:

- `soln`: the solution along the plume centreline.
- `plume_state`: summary state values, including plume heights and collapse
  information.
- `solver_flag`: the solver status code.

Use `get_soln()` to extract a named series from the solution:

```python
height = pr.get_soln(soln, "z")
temperature = pr.get_soln(soln, "T")
vertical_velocity = pr.get_soln(soln, "Uz")
```

`get_soln()` accepts the following series names:

| Name | Quantity | Units / notes |
| --- | --- | --- |
| `s` | Distance along the plume centreline | m |
| `x`, `y` | Horizontal position coordinates | m; zero in the 2-D formulation |
| `z` | Altitude | m |
| `theta` | Plume inclination | radians |
| `psi` | Horizontal plume direction | radians; zero in the 2-D formulation |
| `L` | Plume radius | m |
| `Lgaussian` | Gaussian plume radius (`L / sqrt(2)`) | m |
| `U` | Plume speed | m/s |
| `Ugaussian` | Gaussian plume speed (`2 * U`) | m/s |
| `Uz` | Vertical plume velocity (`U * sin(theta)`) | m/s |
| `Uzgaussian` | Gaussian vertical velocity (`2 * Uz`) | m/s |
| `Q` | Mass flux | kg/s; `with_pi=True` (the default) multiplies the returned values by `pi` |
| `M` | Momentum flux | kg m/s^2; `with_pi=True` (the default) multiplies the returned values by `pi` |
| `E` | Energy flux | J/s; `with_pi=True` (the default) multiplies the returned values by `pi` |
| `logQ` | Natural logarithm of the internally stored mass flux | Dimensionless |
| `logM` | Natural logarithm of the internally stored momentum flux | Dimensionless |
| `phia` | Dry-air mass fraction in the plume | Fraction |
| `phiv` | Water-vapour mass fraction in the plume | Fraction |
| `phis` | Solid mass fraction in the plume | Fraction |
| `phiw` | Liquid-water mass fraction in the plume | Fraction; zero in the dry formulation |
| `phii` | Ice mass fraction in the plume | Fraction; zero unless enabled by the ice formulation |
| `n` | Gas mass fraction (`phia + phiv`) | Fraction |
| `T` | Plume temperature | K |
| `rho` | Plume density | kg/m^3 |
| `Cp` | Plume mixture specific heat capacity | J/(kg K) |
| `Pa` | Ambient atmospheric pressure | Pa |
| `Ta` | Ambient atmospheric temperature | K |
| `RH` | Ambient relative humidity | Fraction, from 0 to 1 |
| `phivA` | Ambient water-vapour mass fraction | Fraction |
| `rhoAtm` | Ambient atmospheric density | kg/m^3 |
| `V` | Ambient wind speed | m/s |
| `psia` | Ambient wind direction | radians |

Each call returns a one-dimensional NumPy series, one value per saved solution
row. Names are case-sensitive; an unsupported name raises `ValueError`. The
`Q`, `M`, and `E` values returned by default include the `pi` factor, while
`logQ` and `logM` are the unscaled logarithmic state variables. To retrieve
fluxes without the factor, pass `with_pi=False`, for example:

```python
mass_flux = pr.get_soln(soln, "Q", with_pi=False)
```

#### Plotting the solution

The arrays returned by `get_soln()` are aligned by solution row, so they can
be plotted directly against each other. For example, use Matplotlib to show
plume altitude and temperature along the centreline:

```python
import matplotlib.pyplot as plt

s = pr.get_soln(soln, "s")
z = pr.get_soln(soln, "z")
temperature = pr.get_soln(soln, "T")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(s / 1000, z / 1000)
axes[0].set(xlabel="Centreline distance (km)", ylabel="Altitude (km)")
axes[1].plot(temperature, z / 1000)
axes[1].set(xlabel="Plume temperature (K)", ylabel="Altitude (km)")

for ax in axes:
    ax.grid(True)

fig.tight_layout()
plt.show()
```

For a 3-D solver run, plot the plume trajectory using the horizontal
coordinates and altitude:

```python
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(
    pr.get_soln(soln, "x") / 1000,
    pr.get_soln(soln, "y") / 1000,
    pr.get_soln(soln, "z") / 1000,
)
ax.set(xlabel="East/west position (km)", ylabel="North/south position (km)", zlabel="Altitude (km)")
plt.show()
```

The 3-D trajectory is meaningful when `threeD=True`; in a 2-D run, `x` and
`y` are zero. Matplotlib is an optional plotting dependency and must be
installed separately if it is not already available in your environment.

### Initial conditions: `set_IC()`

`set_IC()` returns the vent initial conditions as a mapping. Supply exactly
two of `L0`, `U0`, `Q0`, and `M0`; the model derives the other two after
combining the source conditions with the atmospheric state at `z0`.

| Argument | Default | Meaning and units |
| --- | --- | --- |
| `L0` | `None` | Vent radius [m]. Must be positive when supplied. |
| `U0` | `None` | Exit speed [m/s]. Must be positive when supplied. |
| `Q0` | `None` | Mass flux [kg/s]. Must be positive when supplied. |
| `M0` | `None` | Momentum flux [kg m/s^2] (N). Must be positive when supplied. |
| `T0` | `None` | Source mixture temperature [K]. Supply this or `E0`, not both; must be positive. |
| `E0` | `None` | Source energy flux [J/s]. Supply this or `T0`, not both; must be positive. |
| `z0` | `0.0` | Vent altitude [m] above the altitude datum used by the meteorological profile. |
| `theta0` | `pi/2` | Initial plume inclination [radians], measured from the horizontal; `pi/2` is vertical. |
| `n0` | `None` | Convenience gas mass fraction. Sets `phia0=0`, `phiv0=n0`, `phis0=1-n0`, and both water fractions to zero. |
| `phia0` | `None` | Dry-air mass fraction at the vent. |
| `phiv0` | `None` | Water-vapour mass fraction at the vent. |
| `phis0` | `None` | Solid mass fraction at the vent. |
| `phiw0` | `None` (zero when omitted) | Liquid-water mass fraction at the vent. |
| `phii0` | `None` (zero when omitted) | Ice mass fraction at the vent. |
| `with_pi` | `True` | Flux convention for `Q0`, `M0`, and `E0`; see below. |

Choose one composition form:

- Set `n0` to describe a dry-air-free gas/solid mixture. For example,
  `n0=0.03` gives 3% water vapour and 97% solids by mass at the vent.
- Otherwise, provide `phia0`, `phiv0`, and `phis0`. `phiw0` and `phii0`
  default to zero. The five fractions are intended to sum to one.

There is a current implementation limitation: explicitly supplied `phiw0`
and `phii0` values are checked but are not copied into the returned mapping.
Leave them unset to use their zero defaults; non-zero initial liquid-water or
ice fractions are therefore not currently supported through `set_IC()`.

With the default `with_pi=True`, flux values supplied as `Q0`, `M0`, or `E0`
use the convention that includes the circular-area factor `pi`. Set
`with_pi=False` when supplying those fluxes without that factor; the setter
converts them to the model's internal convention. This option does not affect
`L0` or `U0`.

Example using radius and speed, plus temperature and gas fraction:

```python
initial_conditions = pr.set_IC(
    L0=100.0,
    U0=100.0,
    T0=1000.0,
    z0=1666.0,
    n0=0.03,
)
```

### Solver parameters: `set_solver_params()`

All arguments are optional keyword settings; the returned structured NumPy
array is passed to `solve()` as `SolParams`. Distances are in metres and
speeds in metres per second.

| Argument | Default | Meaning |
| --- | --- | --- |
| `sMax` | `100000.0` | Maximum centreline distance [m]; the forward integration stops when it is reached. |
| `Umin` | `0.001` | Minimum vertical plume velocity [m/s]; integration stops when the vertical component falls to or below this value. |
| `MaxHorizDistance` | `100000.0` | Maximum horizontal distance [m]. It is stored in the solver settings, but the corresponding forward-solver stopping check is currently inactive. |
| `atol` | `1e-8` | Absolute error tolerance used by the adaptive Runge–Kutta integrator. |
| `rtol` | `1e-8` | Relative error tolerance used by the adaptive Runge–Kutta integrator. |
| `MinStep` | `1e-8` | Minimum integration step [m]. |
| `MaxStep` | `10.0` | Maximum integration step [m]. |
| `InitStep` | `0.001` | Initial integration step [m]. |
| `OutStep` | `100.0` | Minimum centreline-distance increment [m] between saved output rows. A negative value requests output at every accepted integration step. |
| `MaxSolnPnts` | `100000` | Maximum number of solution rows allocated. |
| `invsolMaxIts` | `100` | Maximum iterations for inverse-solver routines. |
| `invsolrtol` | `1e-4` | Relative tolerance for inverse-solver routines. |
| `invsolatol` | `0.01` | Absolute tolerance for inverse-solver routines. |
| `invsolxtol` | `1e-14` | Interval-width tolerance for inverse-solver routines. |
| `threeD` | `True` | Enable the 3-D plume formulation and horizontal position/direction components. Set `False` for the 2-D formulation. |
| `eqn` | `"ice"` | Equation set: `"dry"` disables water phase change, `"wet"` includes liquid-water phase change, and `"ice"` includes ice phase change. |

The integrator's adaptive step sizes are distinct from `OutStep`: the former
control numerical integration, while the latter controls how frequently rows
are retained in the returned solution. The forward solver stops on `sMax` or
the `Umin` condition.

### Physical parameters: `set_params()`

`set_params()` returns a structured NumPy array of physical constants and
model coefficients. Override any values by keyword; all values below are the
defaults.

| Argument | Default | Meaning and units |
| --- | --- | --- |
| `g` | `9.81` | Gravitational acceleration [m/s^2]. |
| `rhos` | `1200.0` | Solid-particle density [kg/m^3]. |
| `rhow` | `1000.0` | Liquid-water density [kg/m^3]. |
| `rhoi` | `900.0` | Ice density [kg/m^3]. |
| `Ca` | `998.0` | Specific heat capacity of dry air [J/(kg K)]. |
| `Cs` | `1617.0` | Specific heat capacity of solids [J/(kg K)]. |
| `Cv` | `1850.0` | Specific heat capacity of water vapour [J/(kg K)]. |
| `Cw` | `4200.0` | Specific heat capacity of liquid water [J/(kg K)]. |
| `Ci` | `1850.0` | Specific heat capacity of ice [J/(kg K)]. |
| `Ra` | `285.0` | Specific gas constant of dry air [J/(kg K)]. |
| `Rv` | `462.0` | Specific gas constant of water vapour [J/(kg K)]. |
| `Lvt0` | `2.50e6` | Latent heat of vaporization at 0 °C [J/kg]. |
| `Lmt0` | `3.34e6` | Latent heat of melting at 0 °C [J/kg]. |
| `ks` | `0.09` | No-wind entrainment coefficient [dimensionless]. |
| `kw` | `0.9` | Wind entrainment coefficient [dimensionless]. |
| `Ti0` | `263.15` | Upper ice-transition temperature [K]; above this temperature the ice fraction is zero in the ice formulation. |
| `Ti00` | `233.15` | Lower ice-transition temperature [K]; below this temperature the condensed-water fraction is all ice in the ice formulation. |
| `Devenish_m` | `1.5` | Exponent parameter for the Devenish entrainment formulation. |
| `gas_thrust_entrainment` | `True` | Allow the entrainment coefficient to vary in the gas-thrust region. |

The returned parameter structure also contains `Lv0` and `Lm0`, derived from
the supplied latent heats and heat capacities using the model's Kirchhoff
correction; they are not direct `set_params()` arguments.

## Ensemble simulations

`solve_ensemble()` runs one simulation per sampled parameter combination and
returns a pandas DataFrame with the input settings and plume summary outputs.
Import it and the related sampling helper from `plumerise.model`:

```python
from plumerise.model import make_ensemble, solve_ensemble
```

The settings dictionary must contain all eight keys below. Their values can
be fixed numbers or frozen continuous distributions from `scipy.stats`.
Distributions need the usual `ppf()` method for Latin-hypercube sampling (and
`rvs()` for Monte Carlo sampling).

| Key | Meaning | Example units |
| --- | --- | --- |
| `"vent-elevation"` | Vent altitude | m |
| `"vent-radius"` | Vent radius | m |
| `"gas-mass-fraction"` | Source gas mass fraction | Fraction |
| `"source-temperature"` | Source temperature | K |
| `"source-velocity"` | Source exit velocity | m/s |
| `"pyroclast-density"` | Solid-particle density | kg/m^3 |
| `"no-wind-entrainment-coef"` | No-wind entrainment coefficient (`ks`) | Dimensionless |
| `"wind-entrainment-coef"` | Wind entrainment coefficient (`kw`) | Dimensionless |

For uncertainty ranges, provide frozen SciPy distributions in place of the
numeric values. For example, `uniform(loc=40, scale=20)` samples vent radii
from 40 to 60 m (SciPy's `scale` is the range width):

```python
from scipy.stats import truncnorm, uniform
from plumerise.model import solve_ensemble

# A truncated normal with mean 1000 K and standard deviation 100 K,
# restricted to temperatures from 800 K to 1200 K.
a, b = (800 - 1000) / 100, (1200 - 1000) / 100
temperature = truncnorm(a, b, loc=1000, scale=100)

settings = {
    "vent-elevation": 10.0,
    "vent-radius": uniform(loc=40, scale=20),
    "gas-mass-fraction": 0.03,
    "source-temperature": temperature,
    "source-velocity": 100.0,
    "pyroclast-density": 1200.0,
    "no-wind-entrainment-coef": 0.09,
    "wind-entrainment-coef": 0.9,
}
results = solve_ensemble(settings, ensemble_size=100, met_data=met_data)
```

### How `make_ensemble()` samples

`make_ensemble(settings, ensemble_size, design="LHC")` creates the input
DataFrame only; it does not run the plume solver. It has two sampling designs:

- `"LHC"` (the default) generates a Latin-hypercube design. For each
  distributed setting, the unit interval is divided into `ensemble_size`
  strata, with one sample drawn from each stratum. Each column is then
  transformed through that distribution's percent-point function (`ppf`).
  This spreads each parameter's samples across its distribution while
  retaining randomized combinations between columns.
- `"MC"` draws `ensemble_size` independent random values from each
  distribution using its `rvs()` method.

Numeric settings are repeated unchanged in every row. Each row represents one
model run, and the DataFrame has one column per setting. For example, generate
an MC input design to inspect or summarize before running anything:

```python
from plumerise.model import make_ensemble

inputs = make_ensemble(settings, ensemble_size=100, design="MC")
print(inputs.head())
print(inputs.describe())
```

The `design` option belongs to `make_ensemble()`. The current
`solve_ensemble()` signature does not expose it: calling `solve_ensemble()`
uses the default LHC design. `solve_ensemble()` processes the generated runs
with a multiprocessing pool, reserving one CPU core, and appends these output
columns to the sampled inputs:

| Output column | Meaning | Units / values |
| --- | --- | --- |
| `Q0` | Total source mass flux | kg/s |
| `Qs0` | Solid source mass flux | kg/s |
| `Hn` | Neutral-buoyancy height | m |
| `Htop` | Height at the plume top | m |
| `HtopL` | Plume-top height plus plume radius at the top | m |
| `HtopLg` | Plume-top height plus Gaussian plume radius at the top | m |
| `HCondense` | Height where condensation begins | m |
| `Collapse` | Whether plume collapse was detected | `1` for yes, `0` for no |

As an example, inspect a few sampled runs and their predicted top heights:

```python
print(results[["vent-radius", "source-temperature", "Htop", "Collapse"]].head())
```

## Dashboard

Start the interactive dashboard from the project environment with:

```sh
poetry run python -m plumerise
```

The dashboard binds to `127.0.0.1` with debug mode and hot reload disabled.
Stop it with `Ctrl+C` when finished. At startup, it also runs
`plumerise.initialize(verbose=False)` to warm up the JIT-compiled solver. This
downloads an example Wyoming radiosonde profile, so startup requires network
access to the Wyoming Weather Web service and may take longer on the first run.
The dashboard prints a short initialization-start and completion message to
the terminal. Ensemble runs suppress per-step solver diagnostics and do not
print the complete results table.

## Initialization check

`plumerise.initialize()` runs a small dry-plume test after retrieving an
example Wyoming radiosonde profile. It is useful for checking the installation
and warming up the JIT-compiled solver, but it needs network access. The
dashboard runs this check automatically at startup; run it manually for a
standalone check:

```python
pr.initialize(verbose=True)
```

## Troubleshooting and practical notes

- If the Wyoming download fails, confirm internet access and that the remote
  sounding service is available.
- ERA5 downloads require valid CDS API credentials and may wait in the
  provider's processing queue.
- The meteorological array must use the documented column order and units.
- JIT compilation can make the first solve slower than subsequent solves.  Running the `initialize()` function performs this compilation on a small problem, so is recommended after package import.
- The top-level package version is available as `plumerise.__version__`.
