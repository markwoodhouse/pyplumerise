import requests

from .ensemble_solve import make_ensemble, solve_ensemble
from .IC import set_IC
from .MetDataSet import Met_interp, MetData, MetDataArray
from .ParamSet import set_params
from .sampling_variables import Params
from .solver import get_soln, integrate, integrateICs, solve
from .SolverParams import set_solver_params


def initialize(verbose: bool=False) -> None:
    """
    Perform an initialization of pyPlumeRise.

    This creates and solves a small test problem, checking that parameters and meteorological data can be loaded
    and that the main `solve` command works.

    Running initialize also ensures that jit-compilation is completed on a small problem,
    reducing the jit-compilation overhead that would be encountered on a larger problem.

    :param verbose: flag to print messages during the initialization steps, defaults to False
    :type verbose: bool, optional
    """

    if verbose:
        print("Initializing...")

    try:
        met_data = MetData(
            "2010-04-14",
            "12:00",
            "radiosonde",
            source="wyoming",
            stationID="04018",
            verbose=verbose,
        )
    except requests.RequestException as exc:
        if not verbose:
            status = (
                f" (HTTP {exc.response.status_code})"
                if exc.response is not None
                else f" ({type(exc).__name__})"
            )
            print(f"Initialization stopped: Wyoming download failed{status}.")
        return
    met = met_data.to_array()

    params = set_params()
    solver_params = set_solver_params(eqn="dry", atol=1e-6, rtol=1e-6)
    initial_conditions = set_IC(L0=100, U0=10, n0=0.03, T0=800, z0=1666)
    try:
        _, _, solver_flag = solve(
            Params=params,
            SolParams=solver_params,
            IC=initial_conditions,
            Met=met,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"Initialization solve could not run: {exc}")
        return

    if verbose:
        print(f"    solve ran (solver flag: {solver_flag})")
        print("...Initialization complete")
