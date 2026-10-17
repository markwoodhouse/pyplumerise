from scipy.stats import uniform

from plumerise.model.ensemble_solve import make_ensemble, make_ensemble_app
from plumerise.model.sampling_variables import Params


def _dashboard_settings():
    return {
        key: {
            "fixed": True,
            "dist": "Uniform",
            "dist-param-0": spec["default"],
            "dist-param-1": None,
            "dist-param-2": None,
            "dist-param-3": None,
        }
        for key, spec in Params.items()
    }


def test_make_ensemble_app_samples_dashboard_distribution():
    settings = _dashboard_settings()
    settings["vent-radius"].update(
        {
            "fixed": False,
            "dist-param-0": 40,
            "dist-param-1": 60,
        }
    )

    samples = make_ensemble_app(settings, ensemble_size=12)

    assert samples.shape == (12, len(Params))
    assert samples["vent-radius"].between(40, 60).all()
    assert (samples["source-velocity"] == Params["source-velocity"]["default"]).all()


def test_make_ensemble_samples_scipy_distribution():
    settings = {key: spec["default"] for key, spec in Params.items()}
    settings["vent-radius"] = uniform(loc=40, scale=20)

    samples = make_ensemble(settings, ensemble_size=12)

    assert samples.shape == (12, len(Params))
    assert samples["vent-radius"].between(40, 60).all()
