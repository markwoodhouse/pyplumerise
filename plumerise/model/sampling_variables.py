Params = {
    "vent-elevation": {
        "name": "Vent elevation",
        "unit": "m",
        "default": 10.0,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    "vent-radius": {
        "name": "Vent radius",
        "unit": "m",
        "default": 50.0,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    "gas-mass-fraction": {
        "name": "Gas mass fraction",
        "unit": "",
        "default": 0.03,
        "min": 0,
        "max": 1,
        "type": "IC",
    },
    "source-temperature": {
        "name": "Source temperature",
        "unit": "K",
        "default": 1200,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    "source-velocity": {
        "name": "Source velocity",
        "unit": "m/s",
        "default": 75,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    "pyroclast-density": {
        "name": "Pyroclast density",
        "unit": "kg/m\u00b3",
        "default": 1200,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    "no-wind-entrainment-coef": {
        "name": "No-wind entrainment coef",
        "unit": "",
        "default": 0.09,
        "min": 0,
        "max": None,
        "type": "Param",
    },
    "wind-entrainment-coef": {
        "name": "Wind entrainment coef",
        "unit": "",
        "default": 0.9,
        "min": 0,
        "max": None,
        "type": "Param",
    },
}

# for p in Params:
#     p.update({"label": p["name"].lower().replace(" ", "-")})
