Params = [
    {
        "name": "Vent elevation",
        "unit": "m",
        "default": 10.0,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    {
        "name": "Vent radius",
        "unit": "m",
        "default": 50.0,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    {
        "name": "Gas mass fraction",
        "unit": "",
        "default": 0.03,
        "min": 0,
        "max": 1,
        "type": "IC",
    },
    {
        "name": "Source temperature",
        "unit": "K",
        "default": 1200,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    {
        "name": "Source velocity",
        "unit": "m/s",
        "default": 75,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    {
        "name": "Pyroclast density",
        "unit": "kg/m\u00b3",
        "default": 1200,
        "min": 0,
        "max": None,
        "type": "IC",
    },
    {
        "name": "No-wind entrainment coef",
        "unit": "",
        "default": 0.09,
        "min": 0,
        "max": None,
        "type": "Param",
    },
    {
        "name": "Wind entrainment coef",
        "unit": "",
        "default": 0.9,
        "min": 0,
        "max": None,
        "type": "Param",
    },
]

for p in Params:
    p.update({"label": p["name"].lower().replace(" ", "-")})
