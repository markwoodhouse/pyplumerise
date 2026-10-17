import pytest

from plumerise.model.IC import set_IC
from plumerise.model.ParamSet import set_params
from plumerise.model.SolverParams import set_solver_params


def test_set_ic_expands_gas_mass_fraction():
    initial_conditions = set_IC(L0=50, U0=75, n0=0.03, T0=1200)

    assert initial_conditions["phia0"] == 0
    assert initial_conditions["phiv0"] == pytest.approx(0.03)
    assert initial_conditions["phis0"] == pytest.approx(0.97)
    assert initial_conditions["phiw0"] == 0
    assert initial_conditions["phii0"] == 0


def test_set_ic_requires_two_source_flux_inputs():
    with pytest.raises(ValueError, match="Only two"):
        set_IC(L0=50, U0=75, Q0=100, n0=0.03, T0=1200)


def test_set_params_contains_derived_latent_heats():
    params = set_params()

    assert params["Lv0"][0] == pytest.approx(
        params["Lvt0"][0] - 273.15 * (params["Cw"][0] - params["Cv"][0])
    )
    assert params["Lm0"][0] == pytest.approx(
        params["Lmt0"][0] - 273.15 * (params["Ci"][0] - params["Cw"][0])
    )


@pytest.mark.parametrize(("equation", "stored_value"), [("dry", 0), ("wet", 1), ("ice", 2)])
def test_set_solver_params_encodes_equation(equation, stored_value):
    solver_params = set_solver_params(eqn=equation)

    assert solver_params["eqn"][0] == stored_value


def test_set_solver_params_rejects_unknown_equation():
    with pytest.raises(ValueError, match="not recognized"):
        set_solver_params(eqn="unknown")
